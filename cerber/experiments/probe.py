from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from cerber.config import RuntimeConfig, project_root
from cerber.core.detector import Detector
from cerber.core.pipeline import annotate, percentile
from cerber.core.result import FrameResult
from cerber.tasks import infer_task

SAMPLE_URLS = {
    "street": "https://ultralytics.com/images/bus.jpg",
    "people": "https://ultralytics.com/images/bus.jpg",
    "air": "https://ultralytics.com/images/boats.jpg",
    "indoor": "https://ultralytics.com/images/bus.jpg",
}


def _images(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if root.is_file():
        return [root]
    found = [path for path in sorted(root.rglob("*")) if path.suffix.lower() in suffixes]
    return found


def _ensure_samples(root: Path) -> list[Path]:
    images = _images(root)
    if images:
        return images
    root.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import safe_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"no probe images in {root} and ultralytics download is unavailable") from exc
    for name, url in SAMPLE_URLS.items():
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        safe_download(url=url, dir=dest, unzip=False)
    images = _images(root)
    if not images:
        raise RuntimeError(f"failed to collect probe images under {root}")
    return images


def _vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / 1024**2)
    except Exception:
        return None


def _reset_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _summarize(result: FrameResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": result.task,
        "n_instances": len(result.instances),
        "classes": sorted({item.name for item in result.instances}),
        "latency_ms": (result.finished_at - result.captured_at) * 1000.0,
    }
    if result.classification is not None:
        payload["classification"] = {
            "name": result.classification.name,
            "conf": result.classification.conf,
            "top5": [
                {"cls": cls, "name": name, "conf": conf} for cls, name, conf in result.classification.top5
            ],
        }
    if result.semantic is not None:
        payload["semantic_unique"] = [int(v) for v in np.unique(result.semantic)[:32]]
        payload["semantic_shape"] = list(result.semantic.shape)
    if result.depth is not None:
        finite = result.depth[np.isfinite(result.depth) & (result.depth > 0)]
        payload["depth_shape"] = list(result.depth.shape)
        if finite.size:
            payload["depth_m"] = {
                "min": float(finite.min()),
                "p50": float(np.median(finite)),
                "max": float(finite.max()),
            }
    return payload


def probe_module(
    module: dict[str, Any],
    images: list[Path],
    output: Path,
    device: int | str,
    warmup: int,
) -> dict[str, Any]:
    name = str(module.get("name") or infer_task(str(module["weights"])))
    weights = str(module["weights"])
    task = infer_task(weights, str(module["task"]) if module.get("task") else None)
    imgsz = int(module.get("imgsz", 640))
    run_dir = output / name
    vis_dir = run_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig(
        weights=weights,
        source=str(images[0]),
        imgsz=imgsz,
        device=device,
        task=task,
        persist=False,
        verbose=False,
    )
    _reset_vram()
    detector = Detector(config)
    latencies: list[float] = []
    samples: list[dict[str, Any]] = []
    for index, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        started = time.perf_counter()
        result = detector.infer(frame, persist=False, frame_id=index)
        elapsed = time.perf_counter() - started
        if index >= warmup:
            latencies.append(elapsed)
        vis = annotate(frame, result.instances, result)
        vis_path = vis_dir / f"{path.stem}.jpg"
        cv2.imwrite(str(vis_path), vis)
        if result.semantic is not None:
            np.save(run_dir / f"{path.stem}-semantic.npy", result.semantic)
        if result.depth is not None:
            np.save(run_dir / f"{path.stem}-depth.npy", result.depth)
        sample = {"image": str(path), "vis": str(vis_path), **_summarize(result)}
        samples.append(sample)
    timed = latencies or [s["latency_ms"] / 1000.0 for s in samples]
    summary = {
        "name": name,
        "weights": weights,
        "task": task,
        "imgsz": imgsz,
        "device": device,
        "n_images": len(samples),
        "p50_ms": percentile(timed, 50) * 1000.0,
        "p95_ms": percentile(timed, 95) * 1000.0,
        "vram_mb": _vram_mb(),
        "samples": samples,
    }
    (run_dir / "probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del detector
    _reset_vram()
    return {key: value for key, value in summary.items() if key != "samples"}


def load_catalog(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"probe catalog must be a mapping: {path}")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe pretrained YOLO26 heads without training")
    parser.add_argument("--catalog", default="configs/probe/catalog.yaml")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int)
    args = parser.parse_args(argv)
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = project_root() / catalog_path
    catalog = load_catalog(catalog_path)
    source = Path(args.source or catalog.get("source") or "data/probe")
    if not source.is_absolute():
        source = project_root() / source
    output = Path(args.output or catalog.get("output") or "outputs/probe")
    if not output.is_absolute():
        output = project_root() / output
    output.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device is not None else catalog.get("device", 0)
    warmup = int(args.warmup if args.warmup is not None else catalog.get("warmup", 3))
    images = _ensure_samples(source)
    reports = []
    for module in catalog.get("modules") or []:
        reports.append(probe_module(module, images, output, device, warmup))
        print(json.dumps(reports[-1], indent=2))
    index = {"source": str(source), "output": str(output), "modules": reports}
    (output / "probe-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
