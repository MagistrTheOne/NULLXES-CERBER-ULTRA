# Colab A100: с нуля

Runtime: **A100, большой объём ОЗУ, Python 3**. Не запускай train, пока ячейка GPU не печатает `NVIDIA A100`.

Эпохи в конфигах:

| Эксперимент | Конфиг | Эпохи | Зачем |
| --- | --- | --- | --- |
| smoke seg | `coco8-seg.yaml` | **3** | проверка пайплайна |
| VisDrone detect | `visdrone-n.yaml` | **50** | люди/транспорт с воздуха |
| Seraphim drone | `seraphim-subset.yaml` | **50** | класс `drone`, подвыборка |

Не дообучай VisDrone поверх COCO-seg и Seraphim поверх VisDrone — разные class id.

Репозиторий: `https://github.com/MagistrTheOne/NULLXES-CERBER-ULTRA.git`

---

## Ячейка 0 — GPU

```python
import torch
print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0), "выбери A100"
```

## Ячейка 1 — пакеты + Hugging Face Transfer

```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

%pip install -U "huggingface_hub[hf_transfer]" ultralytics pyyaml hf_transfer
```

```python
import os
from huggingface_hub import login

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
except Exception:
    token = os.environ.get("HF_TOKEN")

if token:
    login(token=token, add_to_git_credential=False)
    print("HF login ok, hf_transfer on")
else:
    print("HF_TOKEN нет — публичный Seraphim всё равно качается, лимиты могут резать скорость")
```

Colab → Secrets → `HF_TOKEN` (write access не нужен для публичных датасетов).

## Ячейка 2 — clone + Drive под outputs

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!git clone https://github.com/MagistrTheOne/NULLXES-CERBER-ULTRA.git
%cd /content/NULLXES-CERBER-ULTRA

import pathlib, sys
root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
print(root)
assert (root / "cerber" / "__main__.py").is_file()

# веса и прогоны на Drive, чтобы не потерять при отвале runtime
!mkdir -p "/content/drive/MyDrive/CERBER-outputs"
!ln -sfn "/content/drive/MyDrive/CERBER-outputs" /content/NULLXES-CERBER-ULTRA/outputs
```

Если репозиторий **private**:

```python
from google.colab import userdata
gh = userdata.get("GITHUB_TOKEN")
%cd /content
!git clone https://{gh}@github.com/MagistrTheOne/NULLXES-CERBER-ULTRA.git
%cd /content/NULLXES-CERBER-ULTRA
```

## Ячейка 3 — smoke COCO-seg, 3 эпохи

Ultralytics сам тянет `coco8-seg` и `yolo26n-seg.pt`.

```python
!python -m cerber.experiments.train --config configs/experiments/coco8-seg.yaml
```

## Ячейка 4 — VisDrone, 50 эпох

Датасет качает и конвертирует Ultralytics (`VisDrone.yaml`), не Hugging Face. Боксы, модель `yolo26n.pt`. На A100 `batch: -1` (autobatch).

```python
!python -m cerber.experiments.train --config configs/experiments/visdrone-n.yaml
!python -m cerber.experiments.val --config configs/experiments/visdrone-n.yaml
!python -m cerber.experiments.export --config configs/experiments/visdrone-n.yaml --format onnx
```

Ожидай десятки минут–пара часов на 50 эпох. `patience: 20` может остановить раньше.

## Ячейка 5 — Seraphim через HF Transfer, 50 эпох

Качает zip-батчи с Hugging Face, val режется из train, **test не трогаем**.

```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

!python -m cerber.experiments.prepare_seraphim --batches 1 --max-images 4000 --val-fraction 0.1 --output data/seraphim
!python -m cerber.experiments.train --config configs/experiments/seraphim-subset.yaml
!python -m cerber.experiments.val --config configs/experiments/seraphim-subset.yaml
!python -m cerber.experiments.export --config configs/experiments/seraphim-subset.yaml --format onnx
```

Больше мяса (осторожно с диском Colab):

```python
!python -m cerber.experiments.prepare_seraphim --batches 3 --max-images 12000 --val-fraction 0.1
```

Карточка: [lgrzybowski/seraphim-drone-detection-dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset). Картинки уже 640×640, `imgsz=640` не вернёт детали.

## Ячейка 6 — что скачать на ПК

С Drive / `outputs/`:

- `outputs/visdrone-n/weights/best.pt` → локально тот же путь, `python -m cerber --config configs/runtime-visdrone.yaml`
- `outputs/seraphim-n/weights/best.pt` → `configs/runtime-drone.yaml`
- COCO-seg борт без своего train: `python -m cerber --config configs/runtime.yaml`

```python
!ls -lh outputs/coco8-seg-smoke/weights/ outputs/visdrone-n/weights/ outputs/seraphim-n/weights/
```

---

## После smoke (pip и clone уже сделаны)

Не повторяй `pip install`. `HF_HUB_ENABLE_HF_TRANSFER` не ставить — у тебя `hf-xet`.

### A. Пути, Drive, оба скачивания одним блоком

```python
import os, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
import yaml
from google.colab import drive, userdata

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

from huggingface_hub import HfApi
print("HF:", HfApi().whoami()["name"])

drive.mount("/content/drive", force_remount=False)

repo = Path("/content/NULLXES-CERBER-ULTRA")
assert repo.exists(), "нет клона репо"
os.chdir(repo)
sys.path.insert(0, str(repo))

run_root = Path("/content/drive/MyDrive/CERBER-outputs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
config_dir = run_root / "configs"
config_dir.mkdir(parents=True, exist_ok=True)

sources = {
    "visdrone": "configs/experiments/visdrone-n.yaml",
    "seraphim": "configs/experiments/seraphim-subset.yaml",
}
configs = {}
for key, source in sources.items():
    cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    cfg.update(project=str(run_root), optimizer="auto", seed=42, save=True)
    if key == "seraphim":
        cfg["data"] = str(repo / "data/seraphim/subset/data.yaml")
    dest = config_dir / f"{key}.yaml"
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    configs[key] = dest
    print(key, cfg["epochs"], "эпох", cfg["model"])

def execute(module, *args):
    subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], cwd=repo, check=True)

from ultralytics.data.utils import check_det_dataset
print("качаю VisDrone через Ultralytics…")
check_det_dataset("VisDrone.yaml")

seraphim_dir = repo / "data/seraphim"
if not (seraphim_dir / "subset" / "data.yaml").exists():
    print("качаю Seraphim через Hugging Face (hf-xet), 1 батч ≤4000…")
    execute(
        "cerber.experiments.prepare_seraphim",
        "--batches", 1,
        "--max-images", 4000,
        "--val-fraction", 0.1,
        "--seed", 42,
        "--output", str(seraphim_dir),
    )
else:
    print("Seraphim subset уже есть, download skip")

print("готово. run_root =", run_root)
```

### B. VisDrone — до 50 эпох

```python
execute("cerber.experiments.train", "--config", configs["visdrone"])
execute("cerber.experiments.val", "--config", configs["visdrone"])
execute("cerber.experiments.export", "--config", configs["visdrone"], "--format", "onnx")
```

### C. Seraphim — до 50 эпох

```python
execute("cerber.experiments.train", "--config", configs["seraphim"])
execute("cerber.experiments.val", "--config", configs["seraphim"])
execute("cerber.experiments.export", "--config", configs["seraphim"], "--format", "onnx")
```

### D. Веса

```python
for weights in sorted(run_root.glob("*/weights/*")):
    if weights.suffix in {".pt", ".onnx"}:
        print(weights.relative_to(run_root), f"{weights.stat().st_size / 1024**2:.1f} MiB")
print(run_root)
```

---

## Лаборатория семи задач (после Seraphim)

Detect и instance-seg заново не учим. Сначала `git pull`, потом готовые веса, потом пять отдельных smoke. Каждый smoke — отдельный процесс, затем очистка GPU. На борт остаются предобученные `yolo26n-*.pt`, пока val не покажет выигрыш.

```python
%cd /content/NULLXES-CERBER-ULTRA
!git pull
```

Капни в `data/probe/{street,indoor,air,people}/` по несколько jpg. Если папка пустая, probe сам тянет `bus.jpg` / `boats.jpg`.

```python
import gc
import torch

probe_root = run_root / "probe"
execute(
    "cerber.experiments.probe",
    "--catalog", "configs/probe/catalog.yaml",
    "--source", str(repo / "data/probe"),
    "--output", str(probe_root),
    "--device", "0",
)
print((probe_root / "probe-index.json").read_text(encoding="utf-8"))
```

Численный val готовых голов (размеченные tiny-сеты, не свои кадры):

```python
from ultralytics import YOLO
import json, gc, torch

pretrained = [
    ("detect", "yolo26n.pt", "coco8.yaml", 640),
    ("segment", "yolo26n-seg.pt", "coco8-seg.yaml", 640),
    ("pose", "yolo26n-pose.pt", "coco8-pose.yaml", 640),
    ("obb", "yolo26n-obb.pt", "dota8.yaml", 1024),
    ("semantic", "yolo26n-sem.pt", "cityscapes8.yaml", 640),
    ("depth", "yolo26n-depth.pt", "depth8.yaml", 768),
    ("classify", "yolo26n-cls.pt", "imagenet10", 224),
]
out = run_root / "pretrained-val"
out.mkdir(parents=True, exist_ok=True)
for name, weights, data, imgsz in pretrained:
    metrics = YOLO(weights).val(data=data, imgsz=imgsz, device=0)
    from cerber.experiments.metrics import extract_metrics
    payload = {"name": name, "weights": weights, "data": data, "imgsz": imgsz, **extract_metrics(metrics)}
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(name, payload)
    del metrics
    gc.collect()
    torch.cuda.empty_cache()
```

Пять smoke (отдельные конфиги, не общий цикл гиперпараметров):

```python
import gc, torch, yaml
from pathlib import Path

smoke_dir = run_root / "configs" / "smoke"
smoke_dir.mkdir(parents=True, exist_ok=True)
smokes = [
    "configs/experiments/dota8-obb-smoke.yaml",
    "configs/experiments/coco8-pose-smoke.yaml",
    "configs/experiments/cityscapes8-sem-smoke.yaml",
    "configs/experiments/depth8-smoke.yaml",
    "configs/experiments/imagenet10-cls-smoke.yaml",
]
for source in smokes:
    cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    cfg["project"] = str(run_root)
    dest = smoke_dir / Path(source).name
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    execute("cerber.experiments.train", "--config", dest)
    execute("cerber.experiments.val", "--config", dest)
    gc.collect()
    torch.cuda.empty_cache()
```

VisDrone: тот же `best.pt`, два `imgsz` — не новый train:

```python
execute("cerber.experiments.val", "--config", configs["visdrone"], "--imgsz", 640)
execute("cerber.experiments.val", "--config", configs["visdrone"], "--imgsz", 960)
```

Профили борта после probe: `configs/runtime-ground.yaml`, `runtime-indoor.yaml`, `runtime-air.yaml`, `runtime-drone.yaml`. Aux в YAML закомментирован, пока нет замера VRAM на 2080 Super.
