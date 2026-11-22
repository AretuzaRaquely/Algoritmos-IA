from __future__ import annotations

from typing import Optional

import numpy as np


class EkfPose:
    def __init__(self, num_articulacoes: int = 17, dt: float = 1.0):
        if dt <= 0:
            raise ValueError('dt deve ser positivo.')
        self.num_articulacoes = num_articulacoes
        self.dt = dt
        self.estado = np.zeros((num_articulacoes, 4), dtype=np.float64)
        self.covariancia = np.repeat(np.eye(4, dtype=np.float64)[None] * 1e3, num_articulacoes, axis=0)
        self.noise_process = 1e-3
        self.noise_medicao = 1e-2
        self._inicializado = np.zeros(num_articulacoes, dtype=bool)

    def fit(self, observacoes: np.ndarray, mascaras: Optional[np.ndarray] = None) -> 'EkfPose':
        data = np.asarray(observacoes, dtype=np.float64)
        if data.ndim == 2:
            data = data[None]
        if data.shape[-1] == self.num_articulacoes * 3:
            if mascaras is None:
                mascaras = data[..., self.num_articulacoes * 2:].astype(bool)
            data = data[..., :self.num_articulacoes * 2]
        if data.ndim != 3 or data.shape[-1] != self.num_articulacoes * 2:
            raise ValueError('Observações devem ter formato (sequências, quadros, 34).')
        visible = np.ones((*data.shape[:2], self.num_articulacoes), dtype=bool) if mascaras is None else np.asarray(mascaras, dtype=bool)
        if visible.ndim == 2:
            visible = visible[None]
        if visible.shape != (*data.shape[:2], self.num_articulacoes):
            raise ValueError('Máscaras incompatíveis com observações.')

        velocidades, aceleracoes = [], []
        for seq, valid in zip(data, visible):
            frames = seq.reshape(-1, self.num_articulacoes, 2)
            for joint in range(self.num_articulacoes):
                valores = frames[valid[:, joint], joint]
                if len(valores) > 1:
                    deltas = np.diff(valores, axis=0)
                    velocidades.extend(deltas)
                    if len(deltas) > 1:
                        aceleracoes.extend(np.diff(deltas, axis=0))
        if velocidades:
            delta = np.asarray(velocidades)
            self.noise_medicao = max(float(np.mean(np.var(delta, axis=0))) * 0.05, 1e-4)
        if aceleracoes:
            self.noise_process = max(float(np.mean(np.square(aceleracoes))), 1e-5)
        return self

    def _matrizes(self) -> tuple[np.ndarray, np.ndarray]:
        damping = 0.98
        f = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, damping, 0],
            [0, 0, 0, damping],
        ], dtype=np.float64)
        h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        return f, h

    def predizer(self) -> np.ndarray:
        f, _ = self._matrizes()
        q = self.noise_process * np.array([
            [self.dt**4 / 4, 0, self.dt**3 / 2, 0],
            [0, self.dt**4 / 4, 0, self.dt**3 / 2],
            [self.dt**3 / 2, 0, self.dt**2, 0],
            [0, self.dt**3 / 2, 0, self.dt**2],
        ])
        for joint in range(self.num_articulacoes):
            self.estado[joint] = f @ self.estado[joint]
            self.covariancia[joint] = f @ self.covariancia[joint] @ f.T + q
        return self.estado

    def atualizar(self, medida: np.ndarray, visibilidade: np.ndarray | None = None) -> np.ndarray:
        z = np.asarray(medida, dtype=np.float64).reshape(self.num_articulacoes, 2)
        visible = np.ones(self.num_articulacoes, dtype=bool) if visibilidade is None else np.asarray(visibilidade, dtype=bool)
        _, h = self._matrizes()
        r = np.eye(2) * self.noise_medicao
        for joint in np.flatnonzero(visible):
            if not self._inicializado[joint]:
                self.estado[joint, :2] = z[joint]
                self.covariancia[joint] = np.eye(4) * 1e-2
                self._inicializado[joint] = True
                continue
            p = self.covariancia[joint]
            innovation_cov = h @ p @ h.T + r
            gain = p @ h.T @ np.linalg.inv(innovation_cov)
            residual = z[joint] - h @ self.estado[joint]
            self.estado[joint] += gain @ residual
            identity = np.eye(4)
            self.covariancia[joint] = (identity - gain @ h) @ p @ (identity - gain @ h).T + gain @ r @ gain.T
        return self.estado

    @staticmethod
    def _separar_entrada(historico: np.ndarray, mascaras: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        data = np.asarray(historico, dtype=np.float64)
        if data.ndim == 2:
            data = data[None]
        if data.shape[-1] == 51:
            if mascaras is None:
                mascaras = data[..., 34:].astype(bool)
            data = data[..., :34]
        frames = data.reshape(data.shape[0], data.shape[1], 17, 2)
        valid = np.ones(frames.shape[:-1], dtype=bool) if mascaras is None else np.asarray(mascaras, dtype=bool)
        if valid.ndim == 2:
            valid = valid[None]
        if valid.shape != frames.shape[:-1]:
            raise ValueError('Máscaras incompatíveis com o histórico.')
        return frames, valid

    def estimar_sequencia(self, observacoes: np.ndarray, mascaras: Optional[np.ndarray] = None) -> np.ndarray:
        frames, valid = self._separar_entrada(observacoes, mascaras)
        saida = np.zeros((frames.shape[0], frames.shape[1], self.num_articulacoes, 2), dtype=np.float32)
        for seq_idx in range(frames.shape[0]):
            self.estado.fill(0)
            self.covariancia[:] = np.eye(4) * 1e3
            self._inicializado.fill(False)
            for t in range(frames.shape[1]):
                if t:
                    self.predizer()
                self.atualizar(frames[seq_idx, t], valid[seq_idx, t])
                saida[seq_idx, t] = self.estado[:, :2]
        return saida.reshape(frames.shape[0], frames.shape[1], -1)

    def prever(self, historico: np.ndarray, horizonte: int = 20, mascaras: np.ndarray | None = None) -> np.ndarray:
        if horizonte < 1:
            raise ValueError('horizonte deve ser positivo.')
        frames, valid = self._separar_entrada(historico, mascaras)
        estimado = self.estimar_sequencia(historico, mascaras)
        previsoes = []
        for seq_idx in range(frames.shape[0]):
            self.estado[:, :2] = estimado[seq_idx, -1].reshape(self.num_articulacoes, 2)
            self.estado[:, 2:] = 0.0
            if len(frames[seq_idx]) > 1:
                for joint in range(self.num_articulacoes):
                    observados = np.flatnonzero(valid[seq_idx, :, joint])
                    if len(observados) > 1:
                        anterior, ultimo = observados[-2:]
                        delta = frames[seq_idx, ultimo, joint] - frames[seq_idx, anterior, joint]
                        intervalo = max((ultimo - anterior) * self.dt, self.dt)
                        self.estado[joint, 2:] = delta / intervalo
            futuro = []
            for _ in range(horizonte):
                self.predizer()
                futuro.append(self.estado[:, :2].copy())
            previsoes.append(np.asarray(futuro, dtype=np.float32).reshape(horizonte, -1))
        return np.asarray(previsoes, dtype=np.float32)
