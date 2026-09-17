# Датасеты CERBER

Три независимые ветки. Не дообучать одну модель последовательно COCO → VisDrone → drone: словари классов и ID разные.

| Датасет | Разметка | Модель | Роль |
| --- | --- | --- | --- |
| COCO 2017 / COCO-Seg | боксы + маски, 80 классов | `yolo26n-seg.pt` | готовое локальное восприятие; полный train только на A100 |
| coco8-seg / coco128-seg | маски, Ultralytics | `yolo26n-seg.pt` | smoke обучения, не качество |
| VisDrone2019-DET | только боксы, 10 классов | `yolo26n.pt` | мелкие люди/транспорт с воздуха |
| Seraphim Drone Detection | YOLO-боксы, класс `drone`, 640×640 | `yolo26n.pt` | отдельный эксперимент по дронам |
| FRED | RGB + events | — | позже, трекинг/время |

Маски нужны только для `*-seg`. VisDrone и Seraphim — detect. Прямоугольник бокса не маска; SAM по боксам — отдельный этап.

## Ultralytics yaml

| Конфиг | Назначение |
| --- | --- |
| `coco8.yaml` | smoke detect |
| `coco8-seg.yaml` | smoke segment |
| `coco128-seg.yaml` | отладка seg |
| `coco.yaml` | полный COCO (и seg-аннотации) |
| `VisDrone.yaml` | загрузка и конвертация VisDrone DET |

Готовые COCO-веса уже знают `person`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `boat`. Класса `drone` нет; `airplane` его не заменяет.

## Seraphim

Карточка: [lgrzybowski/seraphim-drone-detection-dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset).

Изображения уже 640×640 (resize+pad). Есть фото, реклама и синтетика. Дубли чистили, ручную переразметку автор не обещает. Берём подвыборку батчей train, val режем из train, официальный test не качаем и не мешаем в val.

```bash
python -m cerber.experiments.prepare_seraphim --batches 1 --max-images 4000 --val-fraction 0.1
```

## Команды train

```bash
python -m cerber.experiments.train --config configs/experiments/coco8-seg.yaml
python -m cerber.experiments.train --config configs/experiments/visdrone-n.yaml
python -m cerber.experiments.train --config configs/experiments/seraphim-subset.yaml
python -m cerber.experiments.val --config configs/experiments/visdrone-n.yaml
python -m cerber.experiments.export --config configs/experiments/visdrone-n.yaml --format onnx
```

Чекпойнты: `outputs/<name>/weights/best.pt`, `last.pt`, `cerber-experiment.json`. После выбора весов — ONNX и сверка с `.pt` на тех же кадрах.
