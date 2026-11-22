import numpy as np

from src.avaliacao.metricas import mae, rmse


def test_metricas_basicas():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    alvo = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    assert mae(pred, alvo) >= 0.0
    assert rmse(pred, alvo) >= 0.0
