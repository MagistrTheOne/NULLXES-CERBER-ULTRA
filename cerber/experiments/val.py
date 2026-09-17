from __future__ import annotations

import argparse
import json
import sys

from cerber.experiments.config import load_experiment_config
from cerber.experiments.metrics import extract_metrics


def named_per_class(names: object, payload: dict) -> dict[str, dict[str, float]] | None:
    if not isinstance(names, dict):
        return None
    precision = payload.get("precision_per_class") or []
    recall = payload.get("recall_per_class") or []
    map50 = payload.get("map50_per_class") or []
    maps = payload.get("map50-95_per_class") or []
    result: dict[str, dict[str, float]] = {}
    for key, name in names.items():
        index = int(key)
        item: dict[str, float] = {}
        if index < len(precision):
            item["precision"] = float(precision[index])
        if index < len(recall):
            item["recall"] = float(recall[index])
        if index < len(map50):
            item["map50"] = float(map50[index])
        if index < len(maps):
            item["map50-95"] = float(maps[index])
        if item:
            result[str(name)] = item
    return result or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO26 experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--data")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--split", default="val")
    args = parser.parse_args(argv)
    experiment = load_experiment_config(args.config)
    weights = args.weights or str(experiment.best_weights())
    data = args.data or experiment.data
    imgsz = args.imgsz if args.imgsz is not None else experiment.imgsz
    split = str(args.split)
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=experiment.device, split=split)
    payload = {
        "weights": weights,
        "data": data,
        "imgsz": imgsz,
        "split": split,
        "task": experiment.task,
        **extract_metrics(metrics),
    }
    per_class = named_per_class(getattr(model, "names", {}), payload)
    if per_class:
        payload["per_class"] = per_class
    text = json.dumps(payload, indent=2)
    print(text)
    run_dir = experiment.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    tagged = run_dir / f"cerber-val-imgsz{imgsz}-{split}.json"
    tagged.write_text(text, encoding="utf-8")
    if split == "val" and imgsz == experiment.imgsz:
        (run_dir / "cerber-val.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
