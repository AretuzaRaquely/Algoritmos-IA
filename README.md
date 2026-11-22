# Previsão temporal de pose humana sob oclusões

Projeto acadêmico para comparar EKF, LSTM, GRU, Transformer temporal e T-GNN na previsão causal de poses humanas COCO-17.

## Requisitos e instalação

- Python 3.10 ou superior;
- Dependências listadas em `requirements.txt`;
- OpenCV com suporte ao codec MP4V para gerar vídeos MP4;
- `ffmpeg` não é necessário;
- O peso YOLO Pose pode ser baixado pelo Ultralytics na primeira extração de vídeo real.

No PowerShell:

```powershell
cd "C:\Users\Aretuza\Algoritmos IA\projeto_pose_oclusao"
python -m pip install -r requirements.txt
```

## Estrutura

```text
configuracoes/                   Configuração do experimento oficial
dados/sinteticos/referencias/    Poses sintéticas completas (NPZ, sem entradas ocluídas)
dados/sinteticos/entradas_oclusao/ Coordenadas observadas e máscaras (NPZ, sem referências)
dados/reais/videos/              Vídeos de entrada fornecidos pela usuária
dados/reais/poses_extraidas/     Pseudo-referências extraídas pelo YOLO
dados/reais/visualizacoes/       Vídeos reais com poses sobrepostas
modelos_treinados/               Reservado para modelos exportados
resultados/experimentos/         Execuções versionadas, métricas e checkpoints
src/dados/                       Geração, preparação e extração de dados
src/modelos/                     Os cinco métodos
src/treinamento/                 Dataset, treinamento e checkpoints
src/avaliacao/                   Métricas e análise
src/visualizacao/                Gráficos e vídeos
testes/                          Testes automatizados
```

Cada execução recebe um identificador exclusivo. Arquivos antigos em `dados/` e `resultados/` não são removidos nem sobrescritos.

## Representação e dados sintéticos

As poses seguem a ordem COCO-17 definida em `src/dados/coco17.py`, com coordenadas `(x, y)` normalizadas para a imagem. São geradas sequências temporalmente suaves de caminhada, corrida, sentar/levantar, movimentos dos braços e mudança de direção.

O gerador preserva as referências completas em `dados/sinteticos/referencias/`. As entradas com oclusões ficam em `dados/sinteticos/entradas_oclusao/`; esse arquivo contém coordenadas zeradas nos pontos ausentes, máscara de visibilidade, tempos e metadados, mas não contém as coordenadas de referência. Os quadros iniciais permanecem visíveis. Oclusões distais, de braços, pernas ou membros completos progridem da extremidade para as articulações proximais, com início e duração variáveis.

A representação fornecida aos modelos tem 51 valores por quadro: 34 coordenadas observadas e 17 indicadores de visibilidade. Os alvos supervisionados contêm apenas as 34 coordenadas de referência dos quadros futuros.

## Experimento pequeno de ponta a ponta

Execute primeiro este fluxo curto. Ele gera dados, separa sequências, treina brevemente as quatro redes, calibra o EKF, calcula as métricas e grava vídeos de verificação:

```powershell
python -m src.execucao --experimento-pequeno
```

O experimento pequeno usa seis sequências, uma época e oclusões de duração ampliada para que a partição de teste cubra as condições ocluídas nos horizontes avaliados; serve para validar instalação e integração, não para tirar conclusões científicas. Os artefatos ficam numa nova pasta em `resultados/experimentos/`.

## Treinamento, previsão e avaliação oficiais

O comando abaixo executa a configuração completa de `configuracoes/configuracao.yaml` (128 sequências, 60 quadros, histórico de 10 quadros, horizontes de 5/10/20 e até 30 épocas):

```powershell
python -m src.execucao --configuracao configuracoes/configuracao.yaml
```

O treinamento/avaliação faz a separação 70/15/15 por sequência antes de criar janelas, evitando que quadros da mesma sequência apareçam em partições diferentes. Redes treinam com entradas ocluídas e alvos completos, usam validação com early stopping e salvam o melhor estado, histórico, configuração e semente. O EKF é calibrado apenas com históricos de treino; não é submetido a épocas de treinamento.

Todos os modelos usam somente o histórico passado. As previsões diretas de 20 quadros são comparadas nos índices correspondentes aos horizontes 5, 10 e 20. A referência futura é usada somente como alvo de treinamento ou para avaliação; a entrada do modelo é construída exclusivamente com o histórico observado e seus indicadores de visibilidade.

Cada execução guarda:

- `resultados.json` e `resultados.csv`: MAE, RMSE e erro euclidiano médio por método, horizonte e condição (global/ocluída);
- `previsoes_teste.npz`: previsões e metadados da partição de teste;
- `modelos/` e `historicos/`: melhores checkpoints e calibração/históricos;
- `metricas_por_horizonte.png`;
- vídeos sintéticos, de oclusão e comparações simultâneas referência/previsão.

O treinamento oficial ainda não foi iniciado automaticamente. Valide primeiro a execução pequena e inicie o comando oficial somente quando quiser executar o experimento completo.

## Vídeos reais e YOLO Pose

Forneça vídeos sem oclusão, com uma pessoa de corpo inteiro visível e movimento contínuo. Formatos sugeridos: MP4 (`.mp4`), AVI (`.avi`) ou MOV (`.mov`), legíveis pelo OpenCV. Use nomes únicos e simples, por exemplo:

```text
dados/reais/videos/sujeito_001.mp4
dados/reais/videos/sujeito_002.mp4
```

Evite cortes frequentes, pessoas sobrepostas e câmera em movimento. O YOLO retorna estimativas automáticas, não ground truth; confiança baixa e articulações ausentes são marcadas como inválidas. O vídeo original não é modificado.

Extraia e visualize um vídeo:

```powershell
python -m src.dados.dados_reais --video dados/reais/videos/sujeito_001.mp4
```

Por padrão, poses/confianças/visibilidade são gravadas em `dados/reais/poses_extraidas/sujeito_001_poses_yolo.npz`, e a sobreposição é gravada separadamente em `dados/reais/visualizacoes/sujeito_001_pose_yolo.mp4`. Para usar outro modelo ou limiar:

```powershell
python -m src.dados.dados_reais --video dados/reais/videos/sujeito_001.mp4 --modelo yolo11n-pose.pt --confianca-minima 0.30
```

Os vídeos reais não são necessários para gerar os dados sintéticos, treinar ou avaliar o pipeline atual. Antes de utilizá-los como pseudo-referências, revise as sobreposições e limiares de confiança.

## Testes

```powershell
pytest -q
```

Os testes cobrem a forma COCO-17, período inicial sem oclusão, oclusão progressiva, preservação da referência, entradas mascaradas, divisão por sequência, separação causal entre histórico e alvos, formatos de previsão dos cinco métodos, métricas e geração/leitura de MP4.
