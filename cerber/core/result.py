from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    cls: int
    conf: float
    name: str
    track_id: int | None = None
    mask: np.ndarray | None = None
    keypoints: np.ndarray | None = None
    obb: tuple[tuple[float, float], ...] | None = None
    obb_xywhr: tuple[float, float, float, float, float] | None = None


@dataclass(frozen=True)
class Classification:
    cls: int
    name: str
    conf: float
    top5: tuple[tuple[int, str, float], ...] = ()


@dataclass
class FrameResult:
    frame_id: int
    captured_at: float
    finished_at: float
    task: str
    model: str
    names: dict[int, str]
    instances: list[Detection] = field(default_factory=list)
    semantic: np.ndarray | None = None
    semantic_frame_id: int | None = None
    depth: np.ndarray | None = None
    depth_frame_id: int | None = None
    classification: Classification | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def stale_maps(self) -> dict[str, bool]:
        return {
            "semantic": self.semantic is not None and self.semantic_frame_id != self.frame_id,
            "depth": self.depth is not None and self.depth_frame_id != self.frame_id,
        }


def merge_aux(primary: FrameResult, aux: FrameResult | None) -> FrameResult:
    if aux is None:
        return primary
    if aux.semantic is not None:
        primary.semantic = aux.semantic
        primary.semantic_frame_id = aux.frame_id
    if aux.depth is not None:
        primary.depth = aux.depth
        primary.depth_frame_id = aux.frame_id
    if aux.classification is not None:
        primary.classification = aux.classification
    return primary
