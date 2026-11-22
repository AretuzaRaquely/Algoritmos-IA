from __future__ import annotations

import torch
import torch.nn as nn

from src.dados.coco17 import NUMERO_ARTICULACOES, obter_adjacencias


class TgnnPose(nn.Module):
    def __init__(
        self,
        entrada_dim: int = 34,
        oculto_dim: int = 16,
        saida_dim: int = 34,
        horizonte_maximo: int = 20,
    ):
        super().__init__()
        if entrada_dim not in (NUMERO_ARTICULACOES * 2, NUMERO_ARTICULACOES * 3):
            raise ValueError('TgnnPose espera 34 coordenadas ou 34 coordenadas + 17 visibilidades.')
        self.entrada_dim = entrada_dim
        self.saida_dim = saida_dim
        self.horizonte_maximo = horizonte_maximo
        adjacency = torch.tensor(obter_adjacencias(), dtype=torch.float32)
        adjacency += torch.eye(NUMERO_ARTICULACOES)
        adjacency /= adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer('adjacencia', adjacency)
        self.projecao = nn.Linear(3, oculto_dim)
        self.temporal = nn.GRU(oculto_dim, oculto_dim, batch_first=True)
        self.saida = nn.Linear(oculto_dim, 2)
        self.projecao_futura = nn.Linear(oculto_dim, horizonte_maximo * 2)

    def _features_espaciais(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[..., :NUMERO_ARTICULACOES * 2].reshape(*x.shape[:-1], NUMERO_ARTICULACOES, 2)
        if x.shape[-1] == NUMERO_ARTICULACOES * 3:
            visibilidade = x[..., NUMERO_ARTICULACOES * 2:].unsqueeze(-1)
        else:
            visibilidade = torch.ones_like(coords[..., :1])
        features = self.projecao(torch.cat((coords, visibilidade), dim=-1))
        return torch.einsum('ij,btjd->btid', self.adjacencia, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        espacial = self._features_espaciais(x)
        return self.saida(espacial).reshape(x.shape[0], x.shape[1], self.saida_dim)

    def prever(self, historico: torch.Tensor, horizonte: int = 20) -> torch.Tensor:
        if not 1 <= horizonte <= self.horizonte_maximo:
            raise ValueError(f'horizonte deve estar entre 1 e {self.horizonte_maximo}.')
        espacial = self._features_espaciais(historico)
        batch, tempo, articulacoes, dimensao = espacial.shape
        por_articulacao = espacial.permute(0, 2, 1, 3).reshape(batch * articulacoes, tempo, dimensao)
        temporal, _ = self.temporal(por_articulacao)
        contexto = temporal[:, -1]
        futuro = self.projecao_futura(contexto).reshape(batch, articulacoes, self.horizonte_maximo, 2)
        return futuro[:, :, :horizonte].permute(0, 2, 1, 3).reshape(batch, horizonte, self.saida_dim)
