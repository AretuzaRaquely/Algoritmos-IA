from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dados.coco17 import CONEXOES_COCO17
from src.modelos.ekf import EkfPose
from src.modelos.gru import GruPose
from src.modelos.lstm import LstmPose
from src.modelos.tgnn import TgnnPose
from src.modelos.transformer_temporal import TransformerTemporalPose
from src.treinamento.gerenciamento_modelos import carregar_modelo


ALGORITMOS = ("ekf", "lstm", "gru", "transformer", "tgnn")
HORIZONTES = (5, 10, 20)


def metricas(predicao: np.ndarray, alvo: np.ndarray, mascara: np.ndarray) -> dict:
    valida = np.asarray(mascara, dtype=bool)
    if predicao.shape != alvo.shape:
        raise ValueError("Previsão e alvo têm formatos diferentes.")
    if valida.ndim == 1:
        valida = np.repeat(valida[None, :], len(predicao), axis=0)
    if valida.shape[-1] == 17 and predicao.shape[-1] == 34:
        valida = valida.repeat(2, axis=-1)
    mae = float(np.abs(predicao[valida] - alvo[valida]).mean()) if np.any(valida) else None
    rmse = float(np.sqrt(np.mean(np.square(predicao[valida] - alvo[valida])))) if np.any(valida) else None
    pred = predicao.reshape(predicao.shape[0], -1, 2)
    ref = alvo.reshape(alvo.shape[0], -1, 2)
    if valida.shape[-1] == pred.shape[-1]:
        mask_euclidiana = valida
    else:
        mask_euclidiana = valida[:, ::2]
    if mask_euclidiana.shape != pred.shape[:2]:
        mask_euclidiana = np.broadcast_to(
            np.asarray(valida, dtype=bool).reshape(pred.shape[0], pred.shape[1]),
            pred.shape[:2],
        )
    erros = np.linalg.norm(pred - ref, axis=2)
    euclidiano = float(erros[mask_euclidiana].mean()) if np.any(mask_euclidiana) else None
    return {
        "quantidade_pontos": int(np.count_nonzero(valida)),
        "mae": mae,
        "rmse": rmse,
        "erro_euclidiano_medio": euclidiano,
    }


def criar_modelo(nome: str, diretorio_experimento: Path):
    if nome == "ekf":
        ekf = EkfPose(dt=1 / 30)
        resumo = json.loads((diretorio_experimento / "resultados.json").read_text(encoding="utf-8"))
        dados = next(item for item in resumo if item["nome"] == "ekf")
        ekf.noise_process = float(dados["historico"]["noise_process"])
        ekf.noise_medicao = float(dados["historico"]["noise_medicao"])
        return ekf
    configuracao = json.loads((diretorio_experimento / "configuracao.json").read_text(encoding="utf-8"))
    if nome == "lstm":
        modelo = LstmPose(51, 64, 34, 20)
    elif nome == "gru":
        modelo = GruPose(51, 64, 34, 20)
    elif nome == "transformer":
        modelo = TransformerTemporalPose(51, 64, saida_dim=34, horizonte_maximo=20)
    else:
        modelo = TgnnPose(51, 32, 34, 20)
    carregar_modelo(modelo, diretorio_experimento / "modelos" / f"{nome}.pt")
    modelo.eval()
    return modelo


def montar_entradas(coordenadas: np.ndarray, visibilidade: np.ndarray, historico: int = 10):
    entradas = []
    mascaras = []
    for fim in range(historico, len(coordenadas)):
        entrada = coordenadas[fim - historico:fim].copy()
        mascara = visibilidade[fim - historico:fim].copy()
        # A máscara é separada das coordenadas como indicado pela representação 51 do projeto.
        entrada_flat = entrada.reshape(historico, -1)
        entradas.append(np.concatenate((entrada_flat, mascara.reshape(historico, -1).astype(np.float32)), axis=1))
        mascaras.append(mascara)
    observadas = np.asarray(entradas, dtype=np.float32).reshape(-1, historico, 51)
    mascaras_historico = np.asarray(mascaras, dtype=bool)
    return observadas, mascaras_historico


def prever(algoritmo: str, modelo, entradas: np.ndarray, horizonte: int):
    if algoritmo == "ekf":
        return modelo.prever(entradas, horizonte=horizonte)
    with torch.no_grad():
        tensor = torch.as_tensor(entradas, dtype=torch.float32)
        return modelo.prever(tensor, horizonte=horizonte).cpu().numpy()


def desenhar_pose(frame: np.ndarray, pose: np.ndarray, visibilidade: np.ndarray, cor: tuple, espessura=3):
    altura, largura = frame.shape[:2]
    pontos = np.rint(pose * np.array([largura, altura])).astype(int)
    pontos = np.clip(pontos, 0, [largura - 1, altura - 1])
    for origem, destino in CONEXOES_COCO17:
        if visibilidade[origem] and visibilidade[destino]:
            cv2.line(frame, tuple(pontos[origem]), tuple(pontos[destino]), cor, espessura, cv2.LINE_AA)
    for ponto, valido in zip(pontos, visibilidade):
        if valido:
            cv2.circle(frame, tuple(ponto), 4, cor, -1, cv2.LINE_AA)


def gerar_grafico_sintetico(diretorio_experimento: Path, saida: Path):
    resumo = json.loads((diretorio_experimento / "resultados.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for eixo, condicao in zip(axes, ("global", "ocluida")):
        for metodo in resumo:
            nome = metodo["nome"]
            horizontes = []
            valores = []
            for horizonte in HORIZONTES:
                dados = metodo["metricas"][str(horizonte)][condicao]
                horizontes.append(horizonte)
                valores.append(dados["erro_euclidiano_medio"] if dados["erro_euclidiano_medio"] is not None else np.nan)
            eixo.plot(horizontes, valores, marker="o", label=nome.upper())
        eixo.set_title("Global" if condicao == "global" else "Ocluído")
        eixo.set_xlabel("Horizonte (quadros)")
        eixo.set_xticks(HORIZONTES)
        eixo.grid(alpha=0.25)
    axes[0].set_ylabel("Erro euclidiano médio")
    axes[1].legend(loc="best")
    fig.suptitle("Comparação sintética dos algoritmos")
    fig.tight_layout()
    fig.savefig(saida, dpi=180, bbox_inches="tight")
    plt.close(fig)


def gerar_grafico_real(metricas_por_algoritmo: dict, saida: Path):
    fig, eixo = plt.subplots(figsize=(9, 5.5))
    for nome, dados in metricas_por_algoritmo.items():
        eixo.plot(
            HORIZONTES,
            [dados[str(h)]["erro_euclidiano_medio"] for h in HORIZONTES],
            marker="o",
            label=nome.upper(),
        )
    eixo.set_xlabel("Horizonte (quadros)")
    eixo.set_ylabel("Erro euclidiano médio")
    eixo.set_xticks(HORIZONTES)
    eixo.grid(alpha=0.25)
    eixo.legend()
    fig.tight_layout()
    fig.savefig(saida, dpi=180, bbox_inches="tight")
    plt.close(fig)


def gerar_videos(
    video_path: Path,
    coordenadas: np.ndarray,
    visibilidade: np.ndarray,
    previsoes: dict,
    saida: Path,
):
    captura = cv2.VideoCapture(str(video_path))
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(captura.get(cv2.CAP_PROP_FPS)) or 30.0
    saida.mkdir(parents=True, exist_ok=True)

    cores = {
        "ekf": (255, 128, 0),
        "lstm": (220, 0, 220),
        "gru": (0, 180, 180),
        "transformer": (0, 0, 220),
        "tgnn": (100, 40, 180),
    }
    escritores = {}
    caminhos = {}
    tamanho_combinado = (largura * 3, altura * 2)
    combined_writer = cv2.VideoWriter(
        str(saida / "comparacao_real_todos_algoritmos.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        tamanho_combinado,
    )

    for nome in ALGORITMOS:
        caminho = saida / f"comparacao_real_{nome}.mp4"
        escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))
        escritores[nome] = escritor
        caminhos[nome] = str(caminho)

    quadro = 0
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        combinado = np.zeros((altura * 2, largura * 3, 3), dtype=np.uint8)
        for linha, nome in enumerate(("original",) + ALGORITMOS):
            x = (linha % 3) * largura
            y = (linha // 3) * altura
            painel = frame.copy()
            desenhar_pose(painel, coordenadas[quadro], visibilidade[quadro], (0, 220, 0), 2)
            if nome != "original" and quadro >= 30:
                indice = quadro - 30
                if indice < len(previsoes[nome]["20"]):
                    prevista = previsoes[nome]["20"][indice].reshape(17, 2)
                    desenhar_pose(painel, prevista, np.ones(17, dtype=bool), cores[nome], 3)
            cv2.rectangle(painel, (10, 10), (280, 48), (20, 20, 20), -1)
            titulo = "YOLO Pose observado" if nome == "original" else f"{nome.upper()} - previsao H20"
            cv2.putText(painel, titulo, (18, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            combinado[y:y + altura, x:x + largura] = painel
            if nome != "original":
                escritores[nome].write(painel)
        combined_writer.write(combinado)
        quadro += 1

    captura.release()
    for escritor in escritores.values():
        escritor.release()
    combined_writer.release()
    return caminhos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--experimento", required=True)
    parser.add_argument("--saida", required=True)
    args = parser.parse_args()

    video = Path(args.video)
    poses_path = Path(args.poses)
    experimento = Path(args.experimento)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    with np.load(poses_path) as dados:
        coordenadas = np.asarray(dados["coordenadas"], dtype=np.float32)
        visibilidade = np.asarray(dados["visibilidade"], dtype=bool)

    entradas, mascaras_historico = montar_entradas(coordenadas, visibilidade)
    previsoes_por_algoritmo = {}
    metricas_por_algoritmo = {}
    historicos_reais = {}

    for nome in ALGORITMOS:
        modelo = criar_modelo(nome, experimento)
        historico_resultados = {}
        previsoes_por_horizonte = {}
        for horizonte in HORIZONTES:
            predicoes = prever(nome, modelo, entradas, horizonte)
            predicoes = predicoes[:, horizonte - 1] if predicoes.ndim == 3 else predicoes
            indices = np.arange(10, len(coordenadas))[0:len(predicoes)] + horizonte - 1
            if indices[-1] >= len(coordenadas):
                tamanho = np.count_nonzero(indices < len(coordenadas))
                predicoes = predicoes[:tamanho]
                indices = indices[:tamanho]
            alvo = coordenadas[indices]
            mascara = visibilidade[indices]
            dados = predicoes.reshape(len(predicoes), -1)
            alvo_flat = alvo.reshape(len(alvo), -1)
            mascara_flat = mascara.reshape(len(mascara), -1)
            historico_resultados[str(horizonte)] = metricas(dados, alvo_flat, mascara_flat)
            if horizonte == 20:
                previsoes_por_horizonte["20"] = predicoes
        metricas_por_algoritmo[nome] = historico_resultados
        previsoes_por_algoritmo[nome] = previsoes_por_horizonte
        historicos_reais[nome] = historico_resultados

    (saida / "metricas_reais.json").write_text(
        json.dumps(metricas_por_algoritmo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    gerar_grafico_real(metricas_por_algoritmo, saida / "grafico_comparativo_real.png")
    gerar_grafico_sintetico(experimento, saida / "grafico_comparativo_sintetico.png")
    caminhos_videos = gerar_videos(
        video,
        coordenadas,
        visibilidade,
        previsoes_por_algoritmo,
        saida,
    )
    print(json.dumps({
        "metricas": metricas_por_algoritmo,
        "graficos": [
            str(saida / "grafico_comparativo_real.png"),
            str(saida / "grafico_comparativo_sintetico.png"),
        ],
        "videos": caminhos_videos,
        "video_combinado": str(saida / "comparacao_real_todos_algoritmos.mp4"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
