from __future__ import annotations

from typing import Dict

import numpy as np


def _normalizar_mascara(predicoes: np.ndarray, alvo: np.ndarray, mascara: np.ndarray | None) -> np.ndarray:
    if predicoes.shape != alvo.shape:
        raise ValueError(f'Predições {predicoes.shape} e alvo {alvo.shape} devem ter o mesmo formato.')
    if mascara is None:
        return np.ones_like(alvo, dtype=bool)

    mascara = np.asarray(mascara, dtype=bool)
    if mascara.shape == predicoes.shape:
        return mascara

    if predicoes.shape[-1] == 34 and mascara.shape == (*predicoes.shape[:-1], 17):
        return np.repeat(mascara, 2, axis=-1)

    if predicoes.shape[-2:] == (17, 2) and mascara.shape == predicoes.shape[:-1]:
        return np.repeat(mascara[..., None], 2, axis=-1)

    raise ValueError(f'Máscara {mascara.shape} incompatível com as métricas para formato {predicoes.shape}.')


def mae(predicoes: np.ndarray, alvo: np.ndarray, mascara: np.ndarray | None = None) -> float:
    pred = np.asarray(predicoes, dtype=np.float32)
    alvo = np.asarray(alvo, dtype=np.float32)
    mascara = _normalizar_mascara(pred, alvo, mascara)
    dif = np.abs(pred[mascara] - alvo[mascara])
    if not dif.size:
        raise ValueError('Não há valores válidos para calcular MAE.')
    return float(np.mean(dif))


def rmse(predicoes: np.ndarray, alvo: np.ndarray, mascara: np.ndarray | None = None) -> float:
    pred = np.asarray(predicoes, dtype=np.float32)
    alvo = np.asarray(alvo, dtype=np.float32)
    mascara = _normalizar_mascara(pred, alvo, mascara)
    dif = pred[mascara] - alvo[mascara]
    if not dif.size:
        raise ValueError('Não há valores válidos para calcular RMSE.')
    return float(np.sqrt(np.mean(dif ** 2)))


def erro_euclidiano(predicoes: np.ndarray, alvo: np.ndarray, mascara: np.ndarray | None = None) -> np.ndarray:
    pred = np.asarray(predicoes, dtype=np.float32)
    alvo = np.asarray(alvo, dtype=np.float32)
    if pred.shape[-1] == 34:
        pred = pred.reshape(*pred.shape[:-1], 17, 2)
        alvo = alvo.reshape(*alvo.shape[:-1], 17, 2)
    if pred.shape[-2:] != (17, 2) or alvo.shape != pred.shape:
        raise ValueError('Predições e alvo devem terminar em 34 ou em (17, 2).')
    erro = np.linalg.norm(pred - alvo, axis=-1)
    if mascara is not None:
        mascara = np.asarray(mascara, dtype=bool)
        if mascara.shape != erro.shape:
            if mascara.size != erro.size:
                raise ValueError(f'Máscara {mascara.shape} incompatível com erro {erro.shape}.')
            mascara = mascara.reshape(erro.shape)
        erro = erro[mascara]
    return erro


def calcular_metricas(predicoes: np.ndarray, alvo: np.ndarray, mascara: np.ndarray | None = None) -> Dict[str, float]:
    erro = erro_euclidiano(predicoes, alvo, mascara)
    if not erro.size:
        raise ValueError('Não há articulações válidas para calcular as métricas.')
    return {
        'mae': mae(predicoes, alvo, mascara),
        'rmse': rmse(predicoes, alvo, mascara),
        'erro_euclidiano_medio': float(np.mean(erro)) if erro.size else 0.0,
    }
