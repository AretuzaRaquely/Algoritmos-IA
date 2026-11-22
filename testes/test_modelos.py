import torch

from src.modelos.gru import GruPose
from src.modelos.lstm import LstmPose
from src.modelos.transformer_temporal import TransformerTemporalPose


def test_modelos_retorna_formato():
    x = torch.randn(2, 10, 34)
    model_lstm = LstmPose(34, 32)
    model_gru = GruPose(34, 32)
    model_tf = TransformerTemporalPose(34, 32)

    assert model_lstm(x).shape == (2, 10, 34)
    assert model_gru(x).shape == (2, 10, 34)
    assert model_tf(x).shape == (2, 10, 34)
