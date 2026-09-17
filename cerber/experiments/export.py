from __future__ import annotations

import argparse
import json
import sys

from cerber.experiments.config import load_experiment_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a trained YOLO26 checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--format", default="onnx")
    args = parser.parse_args(argv)
    experiment = load_experiment_config(args.config)
    weights = args.weights or str(experiment.best_weights())
    from ultralytics import YOLO

    model = YOLO(weights)
    exported = model.export(format=args.format, imgsz=experiment.imgsz, device=experiment.device)
    payload = {"weights": weights, "format": args.format, "exported": str(exported)}
    print(json.dumps(payload, indent=2))
    run_dir = experiment.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cerber-export.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
