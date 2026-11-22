from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.dados.dados_sinteticos import gerar_dados_sinteticos
from src.dados.oclusoes import TIPOS_OCLUSAO, gerar_dados_com_oclusao
from src.dados.preparacao_dados import criar_janelas_previsao, dividir_sequencias
from src.experimentos.executar_experimentos import executar_experimentos, salvar_resultados
from src.visualizacao.gerar_graficos import gerar_grafico_metricas
from src.visualizacao.gerar_videos import gerar_video_comparativo, gerar_video_esqueleto


def carregar_configuracao(caminho: str | Path) -> dict:
    config = yaml.safe_load(Path(caminho).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Configuração YAML inválida: {caminho}')
    return config


def _criar_registros_com_oclusao(configuracao: dict, base: Path, geracao_id: str) -> list[dict]:
    dados = gerar_dados_sinteticos(
        base,
        quantidade_sequencias=int(configuracao.get('quantidade_sequencias', 128)),
        comprimento_sequencia=int(configuracao.get('comprimento_sequencia', 60)),
        tipos=configuracao.get('movimentos'),
        seed=int(configuracao.get('seed', 42)),
        fps=float(configuracao.get('fps', 30)),
    )
    diretorio_entradas = base / 'dados' / 'sinteticos' / 'entradas_oclusao' / geracao_id
    diretorio_entradas.mkdir(parents=True, exist_ok=True)
    registros = []
    duracoes = (int(configuracao.get('oclusao_curta', 5)), int(configuracao.get('oclusao_longa', 20)))
    for indice, item in enumerate(dados):
        referencia = np.asarray(item['pose'], dtype=np.float32)
        tipo_oclusao = TIPOS_OCLUSAO[indice % len(TIPOS_OCLUSAO)]
        duracao = duracoes[indice % len(duracoes)]
        seed = int(configuracao.get('seed', 42)) + indice
        oclusao = gerar_dados_com_oclusao(
            referencia,
            duracao=duracao,
            seed=seed,
            tipo=tipo_oclusao,
            quadros_iniciais=int(configuracao.get('quadros_iniciais_visiveis', 5)),
        )
        observada = np.asarray(oclusao['observada'], dtype=np.float32)
        mascara = np.asarray(oclusao['mascara'], dtype=bool)
        entrada_path = diretorio_entradas / f'{item["id"]}_entrada.npz'
        np.savez_compressed(
            entrada_path,
            observada=observada,
            mascara=mascara,
            timestamps=np.asarray(item['timestamps'], dtype=np.float32),
            id=np.asarray(item['id']),
            tipo_movimento=np.asarray(item['tipo']),
            tipo_oclusao=np.asarray(tipo_oclusao),
            seed=np.asarray(seed),
        )
        registros.append({
            'id': item['id'],
            'tipo': item['tipo'],
            'tipo_oclusao': tipo_oclusao,
            'referencia': referencia,
            'visibilidade_referencia': np.asarray(item['visibilidade'], dtype=bool),
            'observada': observada,
            'mascara': mascara,
            'arquivo_referencia': item['arquivo_referencia'],
            'arquivo_entrada': str(entrada_path),
        })
    return registros


def _salvar_predicoes(resultados: list[dict], teste: dict, caminho: Path) -> None:
    valores = {
        f'pred_{item["nome"]}': np.asarray(item['previsoes'], dtype=np.float32)
        for item in resultados
    }
    valores.update({
        'alvos': teste['alvos'],
        'mascaras_alvo': teste['mascaras_alvo'],
        'condicoes_oclusao': teste['condicoes_oclusao'],
        'ids_sequencia': teste['ids_sequencia'],
        'indices_inicio': teste['indices_inicio'],
    })
    np.savez_compressed(caminho, **valores)


def _gerar_visualizacoes(
    registros: list[dict],
    teste: dict,
    resultados: list[dict],
    caminho_saida: Path,
    fps: float,
) -> None:
    videos = caminho_saida / 'videos'
    videos.mkdir(parents=True, exist_ok=True)
    movimentos = {}
    for registro in registros:
        movimentos.setdefault(registro['tipo'], registro)
    for tipo, registro in movimentos.items():
        nome = re.sub(r'[^a-zA-Z0-9_-]+', '_', tipo).strip('_')
        gerar_video_esqueleto(
            registro['referencia'],
            videos / f'sintetico_{nome}.mp4',
            fps=fps,
        )
        gerar_video_esqueleto(
            registro['observada'],
            videos / f'oclusao_{nome}.mp4',
            visibilidade=registro['mascara'],
            fps=fps,
        )

    if not len(teste['entradas']):
        return
    identificador = str(teste['ids_sequencia'][0])
    registro = next(item for item in registros if item['id'] == identificador)
    inicio = int(teste['indices_inicio'][0])
    quadro_alvo = inicio + teste['entradas'].shape[1]
    quadro_fim = quadro_alvo + teste['alvos'].shape[1]
    referencia_futura = registro['referencia'][quadro_alvo:quadro_fim]
    for resultado in resultados:
        previsao = np.asarray(resultado['previsoes'][0]).reshape(-1, 17, 2)
        gerar_video_comparativo(
            referencia_futura,
            previsao,
            videos / f'comparacao_{resultado["nome"]}.mp4',
            fps=fps,
        )


def executar_pipeline(configuracao: dict, experimento_pequeno: bool = False) -> dict:
    config = dict(configuracao)
    if experimento_pequeno:
        config.update({
            'quantidade_sequencias': 6,
            'comprimento_sequencia': 35,
            'qtde_episodios': 1,
            'patiencia_early_stopping': 1,
            'batch_size': 4,
            'oclusao_curta': 25,
            'oclusao_longa': 25,
            'horizontes_previsao': [5, 10, 20],
            'algoritmos': ['ekf', 'lstm', 'gru', 'transformer', 'tgnn'],
        })
    raiz = Path(__file__).resolve().parents[1]
    historico = int(config.get('janela_temporal', 10))
    horizontes = [int(h) for h in config.get('horizontes_previsao', [5, 10, 20])]
    maior_horizonte = max(horizontes)
    if int(config.get('comprimento_sequencia', 60)) < historico + maior_horizonte:
        raise ValueError(
            'comprimento_sequencia deve ser pelo menos janela_temporal + maior horizonte '
            f'({historico + maior_horizonte} quadros).'
        )
    if sorted(horizontes) != sorted(set(horizontes)) or min(horizontes) < 1:
        raise ValueError('horizontes_previsao deve conter inteiros positivos sem repetição.')

    execucao_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    diretorio_saida = raiz / 'resultados' / 'experimentos' / execucao_id
    diretorio_saida.mkdir(parents=True, exist_ok=False)
    (diretorio_saida / 'configuracao.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    registros = _criar_registros_com_oclusao(config, raiz, execucao_id)
    treino_registros, validacao_registros, teste_registros = dividir_sequencias(
        registros,
        frac_treino=float(config.get('proporcao_treinamento', 0.70)),
        frac_validacao=float(config.get('proporcao_validacao', 0.15)),
        seed=int(config.get('seed', 42)),
    )
    split = {
        'treino': [item['id'] for item in treino_registros],
        'validacao': [item['id'] for item in validacao_registros],
        'teste': [item['id'] for item in teste_registros],
    }
    (diretorio_saida / 'divisao_sequencias.json').write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    amostras = [
        criar_janelas_previsao(grupo, historico, maior_horizonte)
        for grupo in (treino_registros, validacao_registros, teste_registros)
    ]
    nomes = ('treino', 'validação', 'teste')
    for nome, grupo in zip(nomes, amostras):
        if not len(grupo['entradas']):
            raise ValueError(f'Não foi possível gerar janelas de {nome}; aumente a duração das sequências.')
    config['seed'] = int(config.get('seed', 42))
    config['fps'] = float(config.get('fps', 30))
    config['horizontes_previsao'] = horizontes
    config['horizonte_maximo'] = maior_horizonte
    config['janela_temporal'] = historico

    resultados = executar_experimentos(
        list(config.get('algoritmos', ['ekf', 'lstm', 'gru', 'transformer', 'tgnn'])),
        amostras[0],
        amostras[1],
        amostras[2],
        config,
        diretorio_saida,
    )
    caminho_metricas = diretorio_saida / 'resultados.json'
    salvar_resultados(resultados, caminho_metricas)
    _salvar_predicoes(resultados, amostras[2], diretorio_saida / 'previsoes_teste.npz')
    gerar_grafico_metricas(resultados, diretorio_saida / 'metricas_por_horizonte.png')
    _gerar_visualizacoes(registros, amostras[2], resultados, diretorio_saida, config['fps'])

    resumo = {
        'id': execucao_id,
        'diretorio': str(diretorio_saida),
        'quantidade_sequencias': len(registros),
        'amostras': {nome: len(grupo['entradas']) for nome, grupo in zip(nomes, amostras)},
        'algoritmos': list(config.get('algoritmos', [])),
        'resultados': [
            {'nome': item['nome'], 'metricas': item['metricas'], 'historico': item['historico']}
            for item in resultados
        ],
    }
    (diretorio_saida / 'resumo.json').write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description='Pipeline causal de previsão de pose humana sob oclusões.')
    parser.add_argument('--configuracao', default='configuracoes/configuracao.yaml')
    parser.add_argument(
        '--experimento-pequeno',
        action='store_true',
        help='Executa um fluxo curto com as cinco arquiteturas para validação, não o treinamento completo.',
    )
    args = parser.parse_args()
    resumo = executar_pipeline(carregar_configuracao(args.configuracao), args.experimento_pequeno)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
