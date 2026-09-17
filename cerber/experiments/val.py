from __future__ import annotations

import argparse
import json
import sys

from cerber.experiments.config import load_experiment_config
from cerber.experiments.metrics import extract_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO26 experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--data")
    parser.add_argument("--imgsz", type=int)
    args = parser.parse_args(argv)
    experiment = load_experiment_config(args.config)
    weights = args.weights or str(experiment.best_weights())
    data = args.data or experiment.data
    imgsz = args.imgsz if args.imgsz is not None else experiment.imgsz
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=experiment.device)
    payload = {
        "weights": weights,
        "data": data,
        "imgsz": imgsz,
        "task": experiment.task,
        **extract_metrics(metrics),
    }
    print(json.dumps(payload, indent=2))
    run_dir = experiment.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cerber-val.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
