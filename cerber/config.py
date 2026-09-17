from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _as_source(value: Any) -> int | str:
    if isinstance(value, bool):
        raise ValueError("source cannot be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def _as_device(value: Any) -> int | str:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    return text


@dataclass(frozen=True)
class RuntimeConfig:
    weights: str
    source: int | str = 0
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: int | str = 0
    tracker: str = "bytetrack.yaml"
    task: str = "segment"
    persist: bool = True
    show: bool = False
    verbose: bool = False
    max_lost: int = 30
    events_dir: str = "outputs/events"
    log_every: int = 30

    def resolved_weights(self) -> str:
        path = Path(self.weights)
        if path.is_file():
            return str(path)
        candidate = project_root() / self.weights
        if candidate.is_file():
            return str(candidate)
        return self.weights

    def resolved_events_dir(self) -> Path:
        path = Path(self.events_dir)
        if path.is_absolute():
            return path
        return project_root() / path


def load_runtime_config(path: str | Path, **overrides: Any) -> RuntimeConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {config_path}")
    raw.update({key: value for key, value in overrides.items() if value is not None})
    task = str(raw.get("task", "segment")).lower()
    if task not in {"detect", "segment"}:
        raise ValueError(f"unsupported task: {task}")
    return RuntimeConfig(
        weights=str(raw["weights"]),
        source=_as_source(raw.get("source", 0)),
        imgsz=int(raw.get("imgsz", 640)),
        conf=float(raw.get("conf", 0.25)),
        iou=float(raw.get("iou", 0.7)),
        device=_as_device(raw.get("device", 0)),
        tracker=str(raw.get("tracker", "bytetrack.yaml")),
        task=task,
        persist=bool(raw.get("persist", True)),
        show=bool(raw.get("show", False)),
        verbose=bool(raw.get("verbose", False)),
        max_lost=int(raw.get("max_lost", 30)),
        events_dir=str(raw.get("events_dir", "outputs/events")),
        log_every=int(raw.get("log_every", 30)),
    )
