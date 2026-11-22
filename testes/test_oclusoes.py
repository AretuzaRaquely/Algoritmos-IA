import numpy as np

from src.dados.oclusoes import aplicar_oclusao, criar_mascara_oclusao


def test_mascara_oclusao_shape():
    mascara = criar_mascara_oclusao(30, 5, seed=0)
    assert mascara.shape == (30, 17)


def test_aplicar_oclusao():
    seq = np.ones((10, 17, 2), dtype=np.float32)
    mascara = np.ones((10, 17), dtype=bool)
    mascara[2:5, 0:3] = False
    observada, real = aplicar_oclusao(seq, mascara)
    assert observada.shape == seq.shape
    assert real.shape == seq.shape
    assert np.all(observada[2:5, 0:3] == 0.0)
