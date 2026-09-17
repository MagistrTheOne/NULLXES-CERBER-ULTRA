from __future__ import annotations

import argparse
import logging
import sys

from cerber.config import load_runtime_config
from cerber.core.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cerber", description="CERBER-ULTRA runtime")
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--source")
    parser.add_argument("--weights")
    parser.add_argument("--device")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = {
        "source": args.source,
        "weights": args.weights,
        "device": args.device,
    }
    if args.show:
        overrides["show"] = True
    config = load_runtime_config(args.config, **overrides)
    Pipeline(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
