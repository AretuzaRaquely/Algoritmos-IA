from __future__ import annotations

import torch
import torch.nn as nn


class TransformerTemporalPose(nn.Module):
    def __init__(
        self,
        entrada_dim: int = 34,
        oculto_dim: int = 64,
        cabecas: int = 4,
        saida_dim: int = 34,
        horizonte_maximo: int = 20,
        historico_maximo: int = 128,
    ):
        super().__init__()
        if oculto_dim % cabecas != 0:
            cabecas = max(cabeca for cabeca in range(1, min(cabecas, oculto_dim) + 1) if oculto_dim % cabeca == 0)
        self.saida_dim = saida_dim
        self.horizonte_maximo = horizonte_maximo
        self.historico_maximo = historico_maximo
        self.projecao = nn.Linear(entrada_dim, oculto_dim)
        self.posicoes = nn.Parameter(torch.zeros(1, historico_maximo, oculto_dim))
        nn.init.normal_(self.posicoes, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=oculto_dim,
            nhead=cabecas,
            dim_feedforward=max(oculto_dim, 32),
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.saida = nn.Linear(oculto_dim, saida_dim)
        self.projecao_futura = nn.Linear(oculto_dim, horizonte_maximo * saida_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.historico_maximo:
            raise ValueError(f'O histórico não pode exceder {self.historico_maximo} quadros.')
        x_emb = self.projecao(x)
        encoded = self.encoder(x_emb + self.posicoes[:, :x.shape[1]])
        return self.saida(encoded)

    def prever(self, historico: torch.Tensor, horizonte: int = 20) -> torch.Tensor:
        if not 1 <= horizonte <= self.horizonte_maximo:
            raise ValueError(f'horizonte deve estar entre 1 e {self.horizonte_maximo}.')
        encoded = self.forward_features(historico)
        futuro = self.projecao_futura(encoded[:, -1]).reshape(-1, self.horizonte_maximo, self.saida_dim)
        return futuro[:, :horizonte]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.historico_maximo:
            raise ValueError(f'O histórico não pode exceder {self.historico_maximo} quadros.')
        x_emb = self.projecao(x)
        return self.encoder(x_emb + self.posicoes[:, :x.shape[1]])
