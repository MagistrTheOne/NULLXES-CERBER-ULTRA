from __future__ import annotations

from pathlib import Path

TASKS = frozenset({"detect", "segment", "semantic", "depth", "pose", "obb", "classify"})
TRACKABLE_TASKS = frozenset({"detect", "segment", "pose", "obb"})
DENSE_TASKS = frozenset({"semantic", "depth"})

# Longer suffixes first so `-sem` is not confused with detect defaults.
_WEIGHT_HINTS = (
    ("-sem", "semantic"),
    ("-depth", "depth"),
    ("-pose", "pose"),
    ("-obb", "obb"),
    ("-cls", "classify"),
    ("-seg", "segment"),
)


def infer_task(weights: str, explicit: str | None = None) -> str:
    if explicit:
        task = explicit.strip().lower()
        if task not in TASKS:
            raise ValueError(f"unsupported task: {task}")
        return task
    name = Path(weights).name.lower()
    for hint, task in _WEIGHT_HINTS:
        if hint in name:
            return task
    return "detect"
