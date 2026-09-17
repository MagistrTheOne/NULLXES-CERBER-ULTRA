from __future__ import annotations

from typing import Any

import numpy as np


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        array = np.asarray(value)
        if array.size == 0:
            return None
        number = float(array.reshape(-1)[0])
    if number != number:  # NaN
        return None
    return number


def _list(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    return [float(item) for item in array]


def extract_metrics(metrics: object) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    box = getattr(metrics, "box", None)
    if box is not None:
        payload["map50-95"] = _f(getattr(box, "map", None))
        payload["map50"] = _f(getattr(box, "map50", None))
        payload["precision"] = _f(getattr(box, "mp", None))
        payload["recall"] = _f(getattr(box, "mr", None))
        payload["map50-95_per_class"] = _list(getattr(box, "maps", None))
    seg = getattr(metrics, "seg", None)
    if seg is not None:
        payload["mask_map50-95"] = _f(getattr(seg, "map", None))
        payload["mask_map50"] = _f(getattr(seg, "map50", None))
        payload["mask_map50-95_per_class"] = _list(getattr(seg, "maps", None))
    pose = getattr(metrics, "pose", None)
    if pose is not None:
        payload["pose_map50-95"] = _f(getattr(pose, "map", None))
        payload["pose_map50"] = _f(getattr(pose, "map50", None))
        payload["pose_map50-95_per_class"] = _list(getattr(pose, "maps", None))
    payload["miou"] = _f(getattr(metrics, "miou", None))
    payload["pixel_accuracy"] = _f(getattr(metrics, "pixel_accuracy", None))
    payload["abs_rel"] = _f(getattr(metrics, "abs_rel", None))
    payload["rmse"] = _f(getattr(metrics, "rmse", None))
    payload["delta1"] = _f(getattr(metrics, "delta1", None))
    payload["silog"] = _f(getattr(metrics, "silog", None))
    payload["top1"] = _f(getattr(metrics, "top1", None))
    payload["top5"] = _f(getattr(metrics, "top5", None))
    return {key: value for key, value in payload.items() if value is not None}
