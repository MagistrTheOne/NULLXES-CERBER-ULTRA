from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from cerber.tasks import TASKS, infer_task


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
        return int(value)
    text = str(value).strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    return text


@dataclass(frozen=True)
class AuxModule:
    weights: str
    task: str
    every: int = 5
    imgsz: int | None = None
    persist: bool = False


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
    aux: tuple[AuxModule, ...] = ()

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

    def for_aux(self, module: AuxModule) -> RuntimeConfig:
        return replace(
            self,
            weights=module.weights,
            task=module.task,
            persist=False,
            imgsz=module.imgsz if module.imgsz is not None else self.imgsz,
        )


def _parse_aux(raw: Any, parent_weights: str) -> tuple[AuxModule, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ValueError("aux must be a list of modules")
    modules: list[AuxModule] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("aux item must be a mapping")
        weights = str(item.get("weights") or parent_weights)
        task = infer_task(weights, str(item["task"]) if item.get("task") else None)
        modules.append(
            AuxModule(
                weights=weights,
                task=task,
                every=max(1, int(item.get("every", 5))),
                imgsz=int(item["imgsz"]) if item.get("imgsz") is not None else None,
                persist=False,
            )
        )
    return tuple(modules)


def load_runtime_config(path: str | Path, **overrides: Any) -> RuntimeConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {config_path}")
    raw.update({key: value for key, value in overrides.items() if value is not None})
    weights = str(raw["weights"])
    explicit = raw.get("task")
    task = infer_task(weights, str(explicit).lower() if explicit is not None else None)
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    return RuntimeConfig(
        weights=weights,
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
        aux=_parse_aux(raw.get("aux"), weights),
    )
