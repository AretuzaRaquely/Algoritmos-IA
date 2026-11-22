import cv2
import numpy as np
import torch

from src.avaliacao.metricas import calcular_metricas
from src.dados.dados_sinteticos import gerar_dados_sinteticos, gerar_sequencia_movimento
from src.dados.oclusoes import aplicar_oclusao, criar_mascara_oclusao
from src.dados.preparacao_dados import (
    criar_janelas_previsao,
    dividir_sequencias,
)
from src.execucao import _criar_registros_com_oclusao
from src.modelos.ekf import EkfPose
from src.modelos.gru import GruPose
from src.modelos.lstm import LstmPose
from src.modelos.tgnn import TgnnPose
from src.modelos.transformer_temporal import TransformerTemporalPose
from src.visualizacao.gerar_videos import gerar_video_comparativo


def test_sintetico_tem_coco17_continuo_e_registro_separado(tmp_path):
    sequencias = gerar_dados_sinteticos(
        tmp_path,
        quantidade_sequencias=3,
        comprimento_sequencia=35,
        tipos=['caminhada'],
        seed=7,
    )
    assert len(sequencias) == 3
    assert all(np.asarray(item['pose']).shape == (35, 17, 2) for item in sequencias)
    for item in sequencias:
        with np.load(item['arquivo_referencia']) as arquivo:
            assert arquivo['origem'].item() == 'sintetica'
            assert arquivo['pose'].shape == (35, 17, 2)
            assert arquivo['visibilidade'].all()


def test_arquivos_de_referencia_e_entrada_ocluida_sao_independentes(tmp_path):
    registros = _criar_registros_com_oclusao(
        {
            'quantidade_sequencias': 3,
            'comprimento_sequencia': 35,
            'movimentos': ['caminhada'],
            'seed': 17,
            'oclusao_curta': 8,
            'oclusao_longa': 12,
            'quadros_iniciais_visiveis': 5,
        },
        tmp_path,
        'execucao_teste',
    )
    assert len(registros) == 3
    for registro in registros:
        with np.load(registro['arquivo_referencia']) as arquivo_referencia:
            referencia = arquivo_referencia['pose']
            assert referencia.shape == (35, 17, 2)
            assert arquivo_referencia['visibilidade'].all()
        with np.load(registro['arquivo_entrada']) as arquivo_entrada:
            assert {'observada', 'mascara', 'timestamps', 'id'} <= set(arquivo_entrada.files)
            assert not {'pose', 'referencia', 'real'} & set(arquivo_entrada.files)
            observada = arquivo_entrada['observada']
            mascara = arquivo_entrada['mascara']
            assert mascara[:5].all()
            np.testing.assert_array_equal(observada[mascara], referencia[mascara])
            assert np.all(observada[~mascara] == 0)
        np.testing.assert_array_equal(registro['referencia'], referencia)


def test_oclusao_progressiva_e_preservacao_da_referencia():
    referencia = np.arange(20 * 17 * 2, dtype=np.float32).reshape(20, 17, 2)
    mascara = criar_mascara_oclusao(20, 9, seed=2, tipo='bracos', inicio=5)
    observada, preservada = aplicar_oclusao(referencia, mascara)

    assert mascara[:5].all()
    assert not mascara[5, 9] and not mascara[5, 10]
    assert mascara[5, 7] and mascara[5, 5]
    assert not mascara[8, 7] and not mascara[8, 8]
    assert mascara[8, 5] and mascara[8, 6]
    assert not mascara[11, 5] and not mascara[11, 6]
    np.testing.assert_array_equal(preservada, referencia)
    assert np.all(observada[~mascara] == 0)
    np.testing.assert_array_equal(observada[mascara], referencia[mascara])


def test_janelas_separam_historico_e_alvos_sem_vazamento():
    referencia = gerar_sequencia_movimento('caminhada', 35, seed=4)
    mascara = np.ones((35, 17), dtype=bool)
    mascara[5:15, 9] = False
    observada, _ = aplicar_oclusao(referencia, mascara)
    registro = {'id': 'seq-a', 'referencia': referencia, 'observada': observada, 'mascara': mascara}
    original = criar_janelas_previsao([registro], historico=10, horizonte_maximo=20)

    alterada = referencia.copy()
    alterada[10:] += 100
    registro_alterado = {**registro, 'referencia': alterada}
    mutada = criar_janelas_previsao([registro_alterado], historico=10, horizonte_maximo=20)
    np.testing.assert_array_equal(original['entradas'], mutada['entradas'])
    assert not np.array_equal(original['alvos'], mutada['alvos'])
    assert original['entradas'].shape == (6, 10, 51)
    assert original['alvos'].shape == (6, 20, 34)
    np.testing.assert_array_equal(
        original['alvos'][0],
        referencia[10:30].reshape(20, 34),
    )
    assert original['indices_inicio'][0] == 0
    assert not original['entradas'][0, 5, 34 + 9]
    assert np.all(original['entradas'][0, 5, 18:20] == 0)


def test_divisao_de_dados_e_por_sequencia():
    registros = [{'id': f'seq-{i}'} for i in range(10)]
    treino, validacao, teste = dividir_sequencias(registros, seed=12)
    ids = [{item['id'] for item in grupo} for grupo in (treino, validacao, teste)]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i + 1, 3))
    assert set.union(*ids) == {item['id'] for item in registros}


def test_cinco_metodos_produzem_previsao_futura_de_20_quadros():
    entrada = torch.randn(2, 10, 51)
    modelos = [
        LstmPose(51, 32, 34, 20),
        GruPose(51, 32, 34, 20),
        TransformerTemporalPose(51, 32, saida_dim=34, horizonte_maximo=20),
        TgnnPose(51, 16, 34, 20),
    ]
    for modelo in modelos:
        modelo.eval()
        with torch.no_grad():
            assert modelo.prever(entrada, horizonte=20).shape == (2, 20, 34)

    observacoes = entrada.numpy()
    ekf = EkfPose(dt=1 / 30).fit(observacoes)
    assert ekf.prever(observacoes, horizonte=20).shape == (2, 20, 34)


def test_metricas_mae_rmse_e_euclidiana_com_mascara():
    alvo = np.ones((2, 17, 2), dtype=np.float32)
    pred = np.zeros_like(alvo)
    mascara = np.zeros((2, 17), dtype=bool)
    mascara[:, 0] = True
    metricas = calcular_metricas(pred, alvo, mascara)
    assert metricas['mae'] == 1.0
    assert metricas['rmse'] == 1.0
    assert np.isclose(metricas['erro_euclidiano_medio'], np.sqrt(2))


def test_video_comparativo_mp4_e_reproduzivel(tmp_path):
    sequencia = gerar_sequencia_movimento('caminhada', 3, seed=1)
    caminho = tmp_path / 'comparacao.mp4'
    assert gerar_video_comparativo(sequencia, sequencia, caminho) == str(caminho)
    captura = cv2.VideoCapture(str(caminho))
    assert captura.isOpened()
    lido, quadro = captura.read()
    captura.release()
    assert lido and quadro.shape[:2] == (640, 640)
