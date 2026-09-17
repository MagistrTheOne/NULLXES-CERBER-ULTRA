# Спринт 1 и стек

A100 в Colab — обучение. RTX 2080 Super — захват и runtime. Qwen3-VL — не в этом спринте.

## Спринт

1. CUDA torch и камера на локалке.
2. Цикл `python -m cerber --config configs/runtime.yaml` (`yolo26n-seg.pt`), FPS и p95.
3. Colab: smoke `coco8-seg`, train VisDrone, train Seraphim — **сделано**.
4. Метрики Seraphim, затем probe семи готовых голов без train.
5. Пять отдельных smoke: OBB, pose, semantic, depth, classify.
6. Профили `runtime-ground` / `runtime-indoor` / `runtime-air` / `runtime-drone`. Aux включать после замера VRAM.
7. Следующий полный train — по найденной ошибке (VisDrone recall: val 640 vs 960, затем n vs s). OBB не чинит пропуски людей.

Не смешивать словари классов. Чекпойнты в `outputs/<name>/`. Smoke-веса не на борт, пока val не лучше предобученных.

## Стек

| Слой | Выбор |
| --- | --- |
| Seg борт | готовый `yolo26n-seg.pt` (COCO) |
| Aerial detect | `yolo26n.pt` + VisDrone |
| Drone detect | `yolo26n.pt` + Seraphim subset |
| Track | ByteTrack |
| Train | Colab Pro A100 |
| Данные | `data/` + Drive, не git |

Подробности: [DATASETS.md](DATASETS.md), [COLAB.md](COLAB.md), [YOLO26-VISION-STACK.md](YOLO26-VISION-STACK.md).
