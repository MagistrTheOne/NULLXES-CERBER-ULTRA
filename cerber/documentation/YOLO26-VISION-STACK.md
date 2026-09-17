# YOLO26 как зрение CERBER-ULTRA

CERBER — бортовой конвейер восприятия для машин, UAV и роботов. YOLO26 — нейросеть внутри него, не сам продукт.

Официальные источники:

- [YOLO26](https://docs.ultralytics.com/ru/models/yolo26/)
- [Train](https://docs.ultralytics.com/ru/modes/train/)
- [Predict](https://docs.ultralytics.com/ru/modes/predict/)
- [Track](https://docs.ultralytics.com/ru/modes/track/)
- [Detect](https://docs.ultralytics.com/ru/tasks/detect/)
- [Segment](https://docs.ultralytics.com/ru/tasks/segment/)
- [Export](https://docs.ultralytics.com/ru/modes/export/)

## Два контура

| Контур | Режимы Ultralytics | Где живёт |
| --- | --- | --- |
| Борт | `predict`, `track` | `cerber/core`, `cerber/features` |
| Офлайн | `train`, `val`, `export` | отдельные скрипты / ноутбук, не pipeline |

На роботе не вызывать `model.train()`. Обучение — на ПК с GPU, затем выгрузка весов/движка на борт.

## Задачи и веса

Файл весов задаёт голову. Сегментация экземпляров уже отдаёт боксы; semantic — нет. Track работает только на detect / instance-seg / pose / OBB.

Разбор — адаптеры в `cerber/core/adapters.py`, контейнер кадра — `FrameResult` (`instances`, `semantic`, `depth`, `classification`). Semantic читается из `result.semantic_mask.data` (карта классов HxW), не из instance-масок. Depth — `result.depth.data` в метрах; масштаб на своей камере нужно калибровать.

| Задача | Веса | Выход | Зачем на платформе |
| --- | --- | --- | --- |
| Detect | `yolo26n.pt` | `result.boxes` | препятствия, люди, ТС |
| Instance seg | `yolo26n-seg.pt` | `masks` + `boxes` | форма, проезжаемость |
| Semantic | `yolo26n-sem.pt` | `semantic_mask` | дорога / небо / грунт без id |
| Depth | `yolo26n-depth.pt` | глубина, м | грубая дистанция |
| Pose | `yolo26n-pose.pt` | keypoints | человек, оператор |
| OBB | `yolo26n-obb.pt` | повёрнутые рамки | надир; не чинит recall VisDrone |
| Classify | `yolo26n-cls.pt` | `probs` | тип сцены, ImageNet-классы |

Порядок лаборатории: готовые веса (`python -m cerber.experiments.probe`) → пять smoke без detect/seg → профили `runtime-ground` / `runtime-indoor` / `runtime-air` / `runtime-drone`. Aux-модули в профилях выключены, пока probe не показал VRAM и p95.

Масштаб: начать с `n`, сравнить с `s` на том же видео. `m/l/x` — офлайн, пока n/s не упрутся в качество.

## Каркас кода

Ultralytics импортировать только в детекторе (и тонко в трекинге). Остальное работает с нейтральными структурами: id, class, conf, xyxy, mask.

| Файл | Делает |
| --- | --- |
| `cerber/config.py` | model, imgsz, conf, iou, device, tracker, source |
| `cerber/core/capture.py` | камера / RTSP / файл → BGR кадр + timestamp |
| `cerber/core/detector.py` | `YOLO(weights)`, `infer` → `FrameResult` |
| `cerber/core/adapters.py` | разбор boxes / masks / obb / keypoints / semantic / depth / cls |
| `cerber/core/result.py` | `FrameResult`, `Detection`, `Classification` |
| `cerber/core/pipeline.py` | цикл кадра, тайминг FPS/p95 |
| `cerber/features/tracking.py` | `persist=True`, выбор YAML трекера |
| `cerber/features/scene.py` | краткая память объектов по id |
| `cerber/features/events.py` | появился / пропал / близость |
| `cerber/inference/local_model.py` | очередь, не блокирует CV-цикл |
| `cerber/__main__.py` | точка входа |

Конфиг — YAML в `configs/`. Веса не хардкодить.

## Инференс

Модель создать один раз при старте.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # или yolo26n-seg.pt
results = model.predict(
    frame,  # np.ndarray HWC BGR
    verbose=False,
    imgsz=640,
    conf=0.25,
    device=0,
)
result = results[0]
xyxy = result.boxes.xyxy
cls = result.boxes.cls
conf = result.boxes.conf
```

Треки (движущаяся камера UAV/машины — `botsort.yaml`; максимум FPS — `bytetrack.yaml`):

```python
results = model.track(
    frame,
    persist=True,
    tracker="botsort.yaml",
    verbose=False,
    device=0,
)
track_id = result.boxes.id
```

Для длинного файла/RTSP через API Ultralytics: `stream=True`. В своём `cap.read()` — один кадр за вызов.

Не делать: `yolo` CLI на каждый кадр, `show=True` как API робота, второй экземпляр YOLO в events.

`nms=False` — голова без NMS; сначала замерить точность, потом экспорт.

## Обучение

Всегда с предобученных `.pt`, не с голого YAML.

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
model.train(
    data="configs/domain-seg.yaml",
    epochs=100,
    imgsz=640,
    device=0,
    batch=-1,
    optimizer="MuSGD",
    project="outputs",
    name="seg-n",
)
metrics = model.val()
model.export(format="onnx")
```

Windows: код обучения под `if __name__ == "__main__":`. Несколько GPU на Windows с новым torch не использовать — `device=0`.

Датасет — формат YOLO ([датасеты](https://docs.ultralytics.com/ru/datasets/)). COCO8 / COCO8-seg только для проверки, что train вообще идёт. Для CERBER нужен домен: дорога, воздух, цех, те же ракурсы, что на борту.

После val: тот же `imgsz` и пороги на инференсе. TensorRT (`format="engine"`) — когда `.pt` на RTX 2080 Super уже не укладывается в бюджет.

## Порядок работ

0. CUDA-сборка torch, `yolo` в PATH, камера `predict`.
1. `config` + `capture` + `detector` (detect).
2. `pipeline` с метриками задержки.
3. `track` + `scene`.
4. `events` → `outputs/`.
5. Те же классы на `*-seg`.
6. Probe семи готовых голов, затем smoke OBB/pose/semantic/depth/cls.
7. Свой датасет, `train` / `val`.
8. Сравнение n/s, затем export.

## Лицензия

Код и веса Ultralytics: AGPL-3.0 или Enterprise. CERBER, использующий YOLO как библиотеку, наследует ограничения AGPL при распространении.
