from __future__ import annotations

from pathlib import Path

import numpy as np

from src.dados.coco17 import CONEXOES_COCO17, NUMERO_ARTICULACOES

TAMANHO_QUADRO = 640


def _normalizar_sequencia(sequencia: np.ndarray) -> np.ndarray:
    dados = np.asarray(sequencia, dtype=np.float32)
    if dados.ndim == 2 and dados.shape[-1] == NUMERO_ARTICULACOES * 2:
        dados = dados.reshape(-1, NUMERO_ARTICULACOES, 2)
    if dados.ndim != 3 or dados.shape[1:] != (NUMERO_ARTICULACOES, 2):
        raise ValueError('A sequência deve ter formato (quadros, 17, 2) ou (quadros, 34).')
    return dados


def _desenhar(cv2, frame: np.ndarray, pose: np.ndarray, cor: tuple[int, int, int], visibilidade: np.ndarray) -> None:
    pontos = np.rint(np.clip(pose, 0.0, 1.0) * (TAMANHO_QUADRO - 1)).astype(int)
    for origem, destino in CONEXOES_COCO17:
        if visibilidade[origem] and visibilidade[destino]:
            cv2.line(frame, tuple(pontos[origem]), tuple(pontos[destino]), cor, 3, cv2.LINE_AA)
    for indice, ponto in enumerate(pontos):
        if visibilidade[indice]:
            cv2.circle(frame, tuple(ponto), 5, cor, -1, cv2.LINE_AA)


def _abrir_video(cv2, caminho: Path, fps: float):
    if caminho.suffix.lower() not in ('.mp4', '.avi'):
        raise ValueError('Use uma extensão de vídeo .mp4 ou .avi.')
    codec = 'mp4v' if caminho.suffix.lower() == '.mp4' else 'XVID'
    escritor = cv2.VideoWriter(
        str(caminho),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (TAMANHO_QUADRO, TAMANHO_QUADRO),
    )
    if not escritor.isOpened():
        escritor.release()
        raise RuntimeError(f'Não foi possível iniciar a gravação do vídeo: {caminho}')
    return escritor


def gerar_video_esqueleto(
    sequencia: np.ndarray,
    caminho_saida: str | Path,
    visibilidade: np.ndarray | None = None,
    fps: float = 15.0,
) -> str:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError('A geração de vídeos requer OpenCV: instale opencv-python.') from exc
    poses = _normalizar_sequencia(sequencia)
    mask = np.ones(poses.shape[:2], dtype=bool) if visibilidade is None else np.asarray(visibilidade, dtype=bool)
    if mask.shape != poses.shape[:2]:
        raise ValueError('A visibilidade deve ter formato (quadros, 17).')
    if fps <= 0:
        raise ValueError('fps deve ser positivo.')
    caminho = Path(caminho_saida)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    escritor = _abrir_video(cv2, caminho, fps)
    try:
        for idx, pose in enumerate(poses):
            frame = np.full((TAMANHO_QUADRO, TAMANHO_QUADRO, 3), 255, dtype=np.uint8)
            _desenhar(cv2, frame, pose, (0, 150, 0), mask[idx])
            cv2.putText(frame, f'Quadro {idx}', (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            escritor.write(frame)
    finally:
        escritor.release()
    return str(caminho)


def gerar_video_comparativo(
    sequencia_real: np.ndarray,
    sequencia_prevista: np.ndarray,
    caminho_saida: str | Path,
    sequencia_observada: np.ndarray | None = None,
    mascara_observada: np.ndarray | None = None,
    fps: float = 15.0,
) -> str:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError('A geração de vídeos requer OpenCV: instale opencv-python.') from exc
    real = _normalizar_sequencia(sequencia_real)
    prevista = _normalizar_sequencia(sequencia_prevista)
    if real.shape != prevista.shape:
        raise ValueError('Referência e previsão devem ter o mesmo formato e número de quadros.')
    observada = _normalizar_sequencia(sequencia_observada) if sequencia_observada is not None else None
    if observada is not None and observada.shape != real.shape:
        raise ValueError('A sequência observada deve corresponder à referência.')
    observada_mask = (
        np.ones(real.shape[:2], dtype=bool)
        if mascara_observada is None
        else np.asarray(mascara_observada, dtype=bool)
    )
    if observada_mask.shape != real.shape[:2]:
        raise ValueError('A máscara observada deve ter formato (quadros, 17).')
    if fps <= 0:
        raise ValueError('fps deve ser positivo.')

    caminho = Path(caminho_saida)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    escritor = _abrir_video(cv2, caminho, fps)
    completa = np.ones((NUMERO_ARTICULACOES,), dtype=bool)
    try:
        for idx in range(len(real)):
            frame = np.full((TAMANHO_QUADRO, TAMANHO_QUADRO, 3), 255, dtype=np.uint8)
            _desenhar(cv2, frame, real[idx], (0, 145, 0), completa)
            _desenhar(cv2, frame, prevista[idx], (0, 0, 220), completa)
            if observada is not None:
                _desenhar(cv2, frame, observada[idx], (220, 100, 0), observada_mask[idx])
            cv2.putText(frame, f'Quadro {idx}', (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            escritor.write(frame)
    finally:
        escritor.release()
    return str(caminho)
