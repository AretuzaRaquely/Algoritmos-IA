from __future__ import annotations

import torch
import torch.nn as nn


class LstmPose(nn.Module):
    def __init__(self, entrada_dim: int = 34, oculto_dim: int = 64, saida_dim: int = 34, horizonte_maximo: int = 20):
        super().__init__()
        self.entrada_dim = entrada_dim
        self.saida_dim = saida_dim
        self.horizonte_maximo = horizonte_maximo
        self.lstm = nn.LSTM(
            input_size=entrada_dim,
            hidden_size=oculto_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.saida = nn.Linear(oculto_dim, saida_dim)
        self.projecao_futura = nn.Linear(oculto_dim, horizonte_maximo * saida_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.saida(out)

    def prever(self, historico: torch.Tensor, horizonte: int = 20) -> torch.Tensor:
        if not 1 <= horizonte <= self.horizonte_maximo:
            raise ValueError(f'horizonte deve estar entre 1 e {self.horizonte_maximo}.')
        out, _ = self.lstm(historico)
        futuro = self.projecao_futura(out[:, -1]).reshape(-1, self.horizonte_maximo, self.saida_dim)
        return futuro[:, :horizonte]
