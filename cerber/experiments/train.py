from __future__ import annotations

import argparse
import json
import sys

from cerber.experiments.config import load_experiment_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO26 experiment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    experiment = load_experiment_config(args.config)
    from ultralytics import YOLO

    model = YOLO(experiment.model)
    results = model.train(**experiment.train_kwargs())
    run_dir = experiment.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "model": experiment.model,
        "data": experiment.data,
        "save_dir": str(getattr(results, "save_dir", run_dir)),
        "best": str(experiment.best_weights()),
        "last": str(experiment.last_weights()),
    }
    (run_dir / "cerber-experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
