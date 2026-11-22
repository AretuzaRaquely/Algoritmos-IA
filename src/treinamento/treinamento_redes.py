from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SequenciaPoseDataset(Dataset):
    def __init__(self, entradas: np.ndarray, alvos: np.ndarray, mascaras: Optional[np.ndarray] = None):
        if len(entradas) != len(alvos):
            raise ValueError('Entradas e alvos devem ter o mesmo número de amostras.')
        self.entradas = torch.as_tensor(entradas, dtype=torch.float32)
        self.alvos = torch.as_tensor(alvos, dtype=torch.float32)
        self.mascaras = (
            torch.as_tensor(mascaras, dtype=torch.float32)
            if mascaras is not None
            else torch.ones_like(self.alvos)
        )
        if len(self.mascaras) != len(self.alvos):
            raise ValueError('Máscaras e alvos devem ter o mesmo número de amostras.')

    def __len__(self) -> int:
        return len(self.entradas)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.entradas[idx], self.alvos[idx], self.mascaras[idx]


def _predizer(model: nn.Module, entradas: torch.Tensor, horizonte: int) -> torch.Tensor:
    if callable(getattr(model, 'prever', None)):
        return model.prever(entradas, horizonte=horizonte)
    return model(entradas)


def _perda_mascarada(pred: torch.Tensor, alvo: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    if mascara.shape[-1] == 17 and pred.shape[-1] == 34:
        mascara = mascara.repeat_interleave(2, dim=-1)
    if mascara.shape != pred.shape:
        raise ValueError(f'Máscara {tuple(mascara.shape)} incompatível com previsões {tuple(pred.shape)}.')
    peso = mascara.to(dtype=pred.dtype)
    return ((pred - alvo).square() * peso).sum() / peso.sum().clamp_min(1.0)


def treinar_modelo(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = 'cpu',
    validation_loader: DataLoader | None = None,
    early_stopping_patience: int | None = None,
    seed: int = 42,
    caminho_checkpoint: str | Path | None = None,
    caminho_historico: str | Path | None = None,
    configuracao: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if epochs < 1:
        raise ValueError('epochs deve ser pelo menos 1.')
    if lr <= 0:
        raise ValueError('lr deve ser positivo.')
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, list[float]] = {'perda_treino': [], 'perda_validacao': []}
    best_loss = float('inf')
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    sem_melhora = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for entradas, alvos, mascaras in loader:
            entradas, alvos, mascaras = entradas.to(device), alvos.to(device), mascaras.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = _predizer(model, entradas, alvos.shape[1])
            if pred.shape != alvos.shape:
                raise ValueError(f'Saída {tuple(pred.shape)} não corresponde aos alvos {tuple(alvos.shape)}.')
            loss = _perda_mascarada(pred, alvos, mascaras)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
        if not train_losses:
            raise ValueError('O DataLoader de treinamento está vazio.')
        train_loss = float(np.mean(train_losses))
        history['perda_treino'].append(train_loss)

        validation_loss = train_loss
        if validation_loader is not None:
            model.eval()
            validation_losses = []
            with torch.no_grad():
                for entradas, alvos, mascaras in validation_loader:
                    entradas, alvos, mascaras = entradas.to(device), alvos.to(device), mascaras.to(device)
                    pred = _predizer(model, entradas, alvos.shape[1])
                    validation_losses.append(float(_perda_mascarada(pred, alvos, mascaras).item()))
            if not validation_losses:
                raise ValueError('O DataLoader de validação está vazio.')
            validation_loss = float(np.mean(validation_losses))
        history['perda_validacao'].append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            sem_melhora = 0
        else:
            sem_melhora += 1
        if early_stopping_patience is not None and sem_melhora >= early_stopping_patience:
            break

    model.load_state_dict(best_state)
    resultado: Dict[str, Any] = {
        **history,
        'perdas': history['perda_treino'],
        'ultima_perda': history['perda_treino'][-1],
        'melhor_perda_validacao': best_loss,
        'melhor_epoca': best_epoch,
        'epocas_executadas': len(history['perda_treino']),
        'seed': seed,
        'configuracao': configuracao or {},
    }
    if caminho_checkpoint is not None:
        caminho = Path(caminho_checkpoint)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {'estado': best_state, 'configuracao': configuracao or {}, 'seed': seed, 'melhor_epoca': best_epoch},
            caminho,
        )
    if caminho_historico is not None:
        caminho = Path(caminho_historico)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    return resultado


def avaliar_modelo(model: nn.Module, loader: DataLoader, device: str = 'cpu') -> np.ndarray:
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for entradas, alvos, _ in loader:
            entradas = entradas.to(device)
            pred = _predizer(model, entradas, alvos.shape[1])
            preds.append(pred.cpu().numpy())
    if preds:
        return np.concatenate(preds, axis=0)
    return np.empty((0, 0), dtype=np.float32)
