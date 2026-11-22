from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def salvar_modelo(model: torch.nn.Module, caminho: str | Path, configuracao: Dict[str, Any] | None = None):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'estado': model.state_dict(), 'configuracao': configuracao or {}}, caminho)


def carregar_modelo(model: torch.nn.Module, caminho: str | Path):
    checkpoint = torch.load(caminho, map_location='cpu')
    model.load_state_dict(checkpoint['estado'])
    return model


def salvar_configuracao(caminho: str | Path, configuracao: Dict[str, Any]):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(configuracao, ensure_ascii=False, indent=2), encoding='utf-8')
