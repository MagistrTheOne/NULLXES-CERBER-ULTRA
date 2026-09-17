# Colab A100: обучение YOLO26

Локалка — камера и runtime. Colab — train / val / export. Не гонять `python -m cerber` как прод с вебкамеры ноутбука Colab.

Ноутбук: `notebooks/NULLXES_YOLO26.ipynb`.

## 1. Среда

Runtime → A100 (High-RAM). Первая ячейка:

```python
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available(), "подключи A100, не запускай train на CPU"
```

Если «Не удалось подключиться к среде выполнения»: Disconnect and reconnect, снова A100. Квота Pro может быть занята.

## 2. Пакеты

```python
%pip install -U ultralytics huggingface_hub pyyaml
```

Не ставить `opencv-python-headless` поверх GUI-сборки, если позже понадобится `show` на локалке. В Colab headless нормален.

Проверка:

```python
import ultralytics
from huggingface_hub import hf_hub_download
print("ultralytics", ultralytics.__version__)
```

## 3. Hugging Face

Seraphim публичный; токен всё равно полезен против лимитов.

```python
from huggingface_hub import login
import os

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
else:
    print("HF_TOKEN не задан — публичные репо всё равно скачиваются")
```

В Colab: Secrets → `HF_TOKEN`. Либо:

```bash
huggingface-cli login
```

Датасет: `lgrzybowski/seraphim-drone-detection-dataset` (`hf_hub_download`, zip-батчи). Полный dump ~9 ГБ не нужен — `prepare_seraphim --batches 1`.

## 4. Код CERBER на Drive

Залей репозиторий (без `data/`, `outputs/`, `*.pt`) на Google Drive, затем:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd "/content/drive/MyDrive/NULLXES CERBER ULTRA"
```

`sys.path` должен видеть пакет `cerber`. Рабочая папка = корень репо.

Веса и прогоны пиши на Drive: `outputs/` уже в `.gitignore`.

## 5. Порядок экспериментов (отдельные веса)

Не делать `COCO → VisDrone → Seraphim` на одном `best.pt`.

1. Smoke: `configs/experiments/coco8-seg.yaml` (несколько эпох, проверка train).
2. VisDrone detect: `configs/experiments/visdrone-n.yaml`. Первый раз Ultralytics скачает и сконвертирует VisDrone.
3. Seraphim subset: `python -m cerber.experiments.prepare_seraphim ...` затем `configs/experiments/seraphim-subset.yaml`.
4. `val` и `export onnx` для выбранного `--config`.
5. Скачай `outputs/<name>/weights/best.pt` на ПК.

```bash
python -m cerber.experiments.train --config configs/experiments/coco8-seg.yaml
python -m cerber.experiments.train --config configs/experiments/visdrone-n.yaml
python -m cerber.experiments.prepare_seraphim --batches 1 --max-images 4000
python -m cerber.experiments.train --config configs/experiments/seraphim-subset.yaml
python -m cerber.experiments.val --config configs/experiments/visdrone-n.yaml
python -m cerber.experiments.export --config configs/experiments/visdrone-n.yaml --format onnx
```

Windows-обёртка `if __name__ == "__main__"` в скриптах есть; в Colab вызывай так же через `python -m`.

## 6. На локалку после Colab

- COCO-seg борт: `configs/runtime.yaml` + `yolo26n-seg.pt` (можно без своего train).
- VisDrone: `best.pt` → `outputs/visdrone-n/weights/best.pt`, запуск `configs/runtime-visdrone.yaml`.
- Drone: `outputs/seraphim-n/weights/best.pt` + `configs/runtime-drone.yaml`.

```bash
python -m cerber --config configs/runtime.yaml
python -m cerber --config configs/runtime.yaml --source 0 --show
```

После export сверь ONNX и `.pt` на одних кадрах (`yolo predict` / тот же `source`).
