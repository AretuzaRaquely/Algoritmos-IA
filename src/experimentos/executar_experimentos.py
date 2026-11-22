from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.avaliacao.metricas import calcular_metricas
from src.modelos.ekf import EkfPose
from src.modelos.gru import GruPose
from src.modelos.lstm import LstmPose
from src.modelos.tgnn import TgnnPose
from src.modelos.transformer_temporal import TransformerTemporalPose
from src.treinamento.treinamento_redes import SequenciaPoseDataset, avaliar_modelo, treinar_modelo

HORIZONTES_PADRAO = (5, 10, 20)
_ALGORITMOS = {'ekf', 'lstm', 'gru', 'transformer', 'tgnn'}


def _criar_modelo(nome: str, configuracao: Dict):
    entrada_dim = 51
    saida_dim = 34
    horizonte_maximo = int(configuracao.get('horizonte_maximo', 20))
    oculto_dim = int(configuracao.get('dimensao_oculta', 64))
    if nome == 'lstm':
        return LstmPose(entrada_dim, oculto_dim, saida_dim, horizonte_maximo)
    if nome == 'gru':
        return GruPose(entrada_dim, oculto_dim, saida_dim, horizonte_maximo)
    if nome == 'transformer':
        return TransformerTemporalPose(entrada_dim, oculto_dim, saida_dim=saida_dim, horizonte_maximo=horizonte_maximo)
    if nome == 'tgnn':
        return TgnnPose(entrada_dim, int(configuracao.get('dimensao_oculta_tgnn', 32)), saida_dim, horizonte_maximo)
    raise ValueError(f'Algoritmo inválido: {nome}')


def _medir_condicao(predicao: np.ndarray, alvo: np.ndarray, mascara: np.ndarray) -> Dict:
    quantidade = int(np.count_nonzero(mascara))
    if quantidade == 0:
        return {'quantidade_articulacoes': 0, 'mae': None, 'rmse': None, 'erro_euclidiano_medio': None}
    metricas = calcular_metricas(predicao, alvo, mascara)
    return {'quantidade_articulacoes': quantidade, **metricas}


def _avaliar_horizontes(
    predicoes: np.ndarray,
    amostras: Dict[str, np.ndarray],
    horizontes: List[int],
) -> Dict[str, Dict]:
    resultado = {}
    for horizonte in horizontes:
        indice = horizonte - 1
        if indice >= predicoes.shape[1]:
            raise ValueError(f'Previsão não contém horizonte {horizonte}.')
        pred = predicoes[:, indice]
        alvo = amostras['alvos'][:, indice]
        valido = amostras['mascaras_alvo'][:, indice]
        ocluido = amostras['condicoes_oclusao'][:, indice] & valido
        resultado[str(horizonte)] = {
            'global': _medir_condicao(pred, alvo, valido),
            'ocluida': _medir_condicao(pred, alvo, ocluido),
        }
    return resultado


def executar_algoritmo(
    nome: str,
    treino: Dict[str, np.ndarray],
    validacao: Dict[str, np.ndarray],
    teste: Dict[str, np.ndarray],
    configuracao: Dict,
    diretorio_saida: str | Path | None = None,
) -> Dict:
    if nome not in _ALGORITMOS:
        raise ValueError(f'Algoritmo inválido: {nome}')
    if not len(treino['entradas']) or not len(validacao['entradas']) or not len(teste['entradas']):
        raise ValueError('Treino, validação e teste precisam conter janelas de previsão.')
    device = str(configuracao.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    seed = int(configuracao.get('seed', 42))
    diretorio = Path(diretorio_saida) if diretorio_saida is not None else None
    diretorio_modelo = diretorio / 'modelos' if diretorio else None
    historico_path = diretorio / 'historicos' / f'{nome}.json' if diretorio else None
    horizons = [int(h) for h in configuracao.get('horizontes_previsao', HORIZONTES_PADRAO)]

    if nome == 'ekf':
        modelo = EkfPose(dt=1.0 / float(configuracao.get('fps', 30)))
        modelo.fit(treino['entradas'], treino['mascaras_historico'])
        previsoes = modelo.prever(
            teste['entradas'],
            horizonte=max(horizons),
            mascaras=teste['mascaras_historico'],
        )
        historico = {
            'tipo': 'calibracao_ekf',
            'noise_process': modelo.noise_process,
            'noise_medicao': modelo.noise_medicao,
            'dt': modelo.dt,
            'seed': seed,
            'observacao': 'Calibrado apenas com históricos de treino; sem otimização por épocas.',
        }
        if historico_path is not None:
            historico_path.parent.mkdir(parents=True, exist_ok=True)
            historico_path.write_text(json.dumps(historico, indent=2), encoding='utf-8')
    else:
        modelo = _criar_modelo(nome, configuracao)
        batch_size = int(configuracao.get('batch_size', 32))
        generator = torch.Generator().manual_seed(seed)
        treino_loader = DataLoader(
            SequenciaPoseDataset(treino['entradas'], treino['alvos'], treino['mascaras_alvo']),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        validacao_loader = DataLoader(
            SequenciaPoseDataset(validacao['entradas'], validacao['alvos'], validacao['mascaras_alvo']),
            batch_size=batch_size,
            shuffle=False,
        )
        teste_loader = DataLoader(
            SequenciaPoseDataset(teste['entradas'], teste['alvos'], teste['mascaras_alvo']),
            batch_size=batch_size,
            shuffle=False,
        )
        hist = treinar_modelo(
            modelo,
            treino_loader,
            epochs=int(configuracao.get('qtde_episodios', 30)),
            lr=float(configuracao.get('taxa_aprendizado', 1e-3)),
            device=device,
            validation_loader=validacao_loader,
            early_stopping_patience=configuracao.get('patiencia_early_stopping', 10),
            seed=seed,
            caminho_checkpoint=diretorio_modelo / f'{nome}.pt' if diretorio_modelo else None,
            caminho_historico=historico_path,
            configuracao={
                'algoritmo': nome,
                'entrada_dim': 51,
                'saida_dim': 34,
                'historico': int(configuracao.get('janela_temporal', 10)),
                'horizonte_maximo': max(horizons),
                'batch_size': batch_size,
                'taxa_aprendizado': float(configuracao.get('taxa_aprendizado', 1e-3)),
                'epocas': int(configuracao.get('qtde_episodios', 30)),
                'seed': seed,
            },
        )
        previsoes = avaliar_modelo(modelo, teste_loader, device=device)
        historico = hist
    resultado = {
        'nome': nome,
        'metricas': _avaliar_horizontes(previsoes, teste, horizons),
        'historico': historico,
        'previsoes': previsoes,
    }
    return resultado


def executar_experimentos(
    algoritmos: List[str],
    treino: Dict[str, np.ndarray],
    validacao: Dict[str, np.ndarray],
    teste: Dict[str, np.ndarray],
    configuracao: Dict,
    diretorio_saida: str | Path | None = None,
) -> List[Dict]:
    invalidos = set(algoritmos) - _ALGORITMOS
    if invalidos:
        raise ValueError(f'Algoritmos inválidos: {sorted(invalidos)}')
    resultados = []
    for algoritmo in algoritmos:
        resultados.append(
            executar_algoritmo(algoritmo, treino, validacao, teste, configuracao, diretorio_saida)
        )
    return resultados


def salvar_resultados(resultado: List[Dict], caminho_saida: str | Path) -> None:
    caminho = Path(caminho_saida)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    publicos = [{'nome': item['nome'], 'metricas': item['metricas'], 'historico': item['historico']} for item in resultado]
    caminho.write_text(json.dumps(publicos, ensure_ascii=False, indent=2), encoding='utf-8')
    caminho_csv = caminho.with_suffix('.csv')
    with caminho_csv.open('w', newline='', encoding='utf-8-sig') as arquivo:
        colunas = ['metodo', 'horizonte_quadros', 'condicao', 'quantidade_articulacoes', 'mae', 'rmse', 'erro_euclidiano_medio']
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        for metodo in publicos:
            for horizonte, por_condicao in metodo['metricas'].items():
                for condicao, metricas in por_condicao.items():
                    escritor.writerow({
                        'metodo': metodo['nome'],
                        'horizonte_quadros': horizonte,
                        'condicao': condicao,
                        **metricas,
                    })
