from __future__ import annotations

import time

import numpy as np

from cerber.config import RuntimeConfig
from cerber.core.adapters import adapt_result
from cerber.core.result import Detection, FrameResult
from cerber.tasks import TRACKABLE_TASKS

__all__ = ["Detection", "Detector"]


class Detector:
    def __init__(self, config: RuntimeConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.resolved_weights())
        names = getattr(self.model, "names", {}) or {}
        self.names = {int(key): str(value) for key, value in names.items()}

    def predict(self, frame: np.ndarray) -> list[Detection]:
        return self.infer(frame, persist=False).instances

    def track(self, frame: np.ndarray, tracker: str) -> list[Detection]:
        return self.infer(frame, persist=True, tracker=tracker).instances

    def infer(
        self,
        frame: np.ndarray,
        persist: bool | None = None,
        tracker: str | None = None,
        frame_id: int = 0,
        captured_at: float | None = None,
    ) -> FrameResult:
        captured = time.time() if captured_at is None else captured_at
        use_track = (self.config.persist if persist is None else persist) and self.config.task in TRACKABLE_TASKS
        kwargs = {
            "imgsz": self.config.imgsz,
            "conf": self.config.conf,
            "iou": self.config.iou,
            "device": self.config.device,
            "verbose": self.config.verbose,
        }
        if use_track:
            results = self.model.track(frame, persist=True, tracker=tracker or self.config.tracker, **kwargs)
        else:
            results = self.model.predict(frame, **kwargs)
        finished = time.time()
        raw = results[0] if results else object()
        return adapt_result(
            raw,
            task=self.config.task,
            model=self.config.weights,
            frame_id=frame_id,
            captured_at=captured,
            finished_at=finished,
            names=self.names,
        )
