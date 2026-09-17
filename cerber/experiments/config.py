from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cerber.config import project_root


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    data: str
    epochs: int = 100
    imgsz: int = 640
    device: int | str = 0
    batch: int | float = 16
    optimizer: str = "auto"
    project: str = "outputs"
    name: str = "exp"
    patience: int = 50
    workers: int = 8
    exist_ok: bool = True
    extra: dict[str, Any] | None = None

    def train_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "data": self.data,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "device": self.device,
            "batch": self.batch,
            "optimizer": self.optimizer,
            "project": str(self.resolved_project()),
            "name": self.name,
            "patience": self.patience,
            "workers": self.workers,
            "exist_ok": self.exist_ok,
        }
        if self.extra:
            kwargs.update(self.extra)
        return kwargs

    def resolved_project(self) -> Path:
        path = Path(self.project)
        if path.is_absolute():
            return path
        return project_root() / path

    def run_dir(self) -> Path:
        return self.resolved_project() / self.name

    def best_weights(self) -> Path:
        return self.run_dir() / "weights" / "best.pt"

    def last_weights(self) -> Path:
        return self.run_dir() / "weights" / "last.pt"


KNOWN = {
    "model",
    "data",
    "epochs",
    "imgsz",
    "device",
    "batch",
    "optimizer",
    "project",
    "name",
    "patience",
    "workers",
    "exist_ok",
}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"experiment config must be a mapping: {config_path}")
    extra = {key: value for key, value in raw.items() if key not in KNOWN}
    device = raw.get("device", 0)
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    return ExperimentConfig(
        model=str(raw["model"]),
        data=str(raw["data"]),
        epochs=int(raw.get("epochs", 100)),
        imgsz=int(raw.get("imgsz", 640)),
        device=device,
        batch=raw.get("batch", 16),
        optimizer=str(raw.get("optimizer", "auto")),
        project=str(raw.get("project", "outputs")),
        name=str(raw.get("name", "exp")),
        patience=int(raw.get("patience", 50)),
        workers=int(raw.get("workers", 8)),
        exist_ok=bool(raw.get("exist_ok", True)),
        extra=extra or None,
    )
