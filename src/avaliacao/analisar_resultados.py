from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def ler_resultados(caminho: str | Path) -> List[Dict]:
    dados = json.loads(Path(caminho).read_text(encoding='utf-8'))
    return dados if isinstance(dados, list) else [dados]


def comparar_algoritmos(resultados: List[Dict]) -> Dict[str, Dict[str, float]]:
    comparacao = {}
    for item in resultados:
        comparacao[item['nome']] = item['metricas']
    return comparacao
