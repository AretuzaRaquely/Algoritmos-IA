from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def gerar_grafico_metricas(resultados: list[dict] | dict, caminho_saida: str | Path) -> str:
    caminho = Path(caminho_saida)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    if isinstance(resultados, dict):
        labels = list(resultados)
        values = [resultados[k]['mae'] for k in labels]
        ax.bar(labels, values, color='steelblue')
        ax.set_title('MAE por algoritmo')
        ax.set_ylabel('MAE')
    else:
        algoritmos = [item['nome'] for item in resultados]
        horizontes = sorted({int(h) for item in resultados for h in item['metricas']})
        x = np.arange(len(horizontes))
        width = 0.8 / max(len(algoritmos), 1)
        for indice, item in enumerate(resultados):
            values = [item['metricas'][str(h)]['global']['mae'] for h in horizontes]
            ax.bar(x + indice * width, values, width, label=item['nome'])
        ax.set_xticks(x + width * max(len(algoritmos) - 1, 0) / 2, [str(h) for h in horizontes])
        ax.set_xlabel('Horizonte (quadros)')
        ax.set_ylabel('MAE')
        ax.set_title('MAE por algoritmo e horizonte')
        ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return str(caminho)
