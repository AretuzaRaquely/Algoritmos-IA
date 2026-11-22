import numpy as np
from torch.utils.data import DataLoader

from src.modelos.gru import GruPose
from src.treinamento.treinamento_redes import SequenciaPoseDataset, treinar_modelo


def test_treinamento_pequeno_dataset():
    entradas = np.random.randn(4, 10, 34).astype(np.float32)
    alvos = np.random.randn(4, 10, 34).astype(np.float32)
    dataset = SequenciaPoseDataset(entradas, alvos)
    loader = DataLoader(dataset, batch_size=2)
    modelo = GruPose(34, 16)
    resultado = treinar_modelo(modelo, loader, epochs=1, lr=1e-3, device='cpu')
    assert 'ultima_perda' in resultado
    assert resultado['ultima_perda'] >= 0
