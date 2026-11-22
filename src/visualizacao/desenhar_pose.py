from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.dados.coco17 import CONEXOES_COCO17


def desenhar_esqueleto(
    pose: Sequence[Sequence[float]],
    ax=None,
    cor: str = 'green',
    mostrar_pontos: bool = True,
    visibilidade: Sequence[bool] | None = None,
):
    arr = np.asarray(pose, dtype=np.float32).reshape(17, 2)
    vis = np.ones(17, dtype=bool) if visibilidade is None else np.asarray(visibilidade, dtype=bool)
    if vis.shape != (17,):
        raise ValueError('visibilidade deve conter 17 valores.')
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    for i, j in CONEXOES_COCO17:
        if vis[i] and vis[j]:
            ax.plot([arr[i, 0], arr[j, 0]], [arr[i, 1], arr[j, 1]], color=cor, linewidth=2)
    if mostrar_pontos:
        ax.scatter(arr[vis, 0], arr[vis, 1], color=cor, s=18)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    return ax
