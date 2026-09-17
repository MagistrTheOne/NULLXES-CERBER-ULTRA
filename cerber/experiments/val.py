from __future__ import annotations

import argparse
import json
import sys

from cerber.experiments.config import load_experiment_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO26 experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    args = parser.parse_args(argv)
    experiment = load_experiment_config(args.config)
    weights = args.weights or str(experiment.best_weights())
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=experiment.data, imgsz=experiment.imgsz, device=experiment.device)
    box = getattr(metrics, "box", None)
    payload = {
        "weights": weights,
        "data": experiment.data,
        "map50-95": float(getattr(box, "map", 0.0) or 0.0) if box is not None else None,
        "map50": float(getattr(box, "map50", 0.0) or 0.0) if box is not None else None,
    }
    print(json.dumps(payload, indent=2))
    run_dir = experiment.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cerber-val.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
