# Спринт 1 и стек

A100 в Colab — обучение. RTX 2080 Super — захват и runtime. Qwen3-VL — не в этом спринте.

## Спринт

1. CUDA torch и камера на локалке.
2. Цикл `python -m cerber --config configs/runtime.yaml` (`yolo26n-seg.pt`), FPS и p95.
3. Colab: smoke `coco8-seg`, затем train `yolo26n.pt` на VisDrone.
4. Отдельный train: подвыборка Seraphim, класс `drone`, val из train, test не трогать.
5. Выбранные веса локально: ByteTrack → scene → events.
6. Позже: 150–300 кадров своей камеры для переноса; FRED; SAM-маски по боксам.

Не смешивать словари классов. Чекпойнты в `outputs/<name>/`.

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
