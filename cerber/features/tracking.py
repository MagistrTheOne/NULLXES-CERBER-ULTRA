from __future__ import annotations

import numpy as np

from cerber.core.detector import Detection, Detector


class Tracker:
    def __init__(self, detector: Detector, tracker: str = "bytetrack.yaml") -> None:
        self.detector = detector
        self.tracker = tracker

    def update(self, frame: np.ndarray) -> list[Detection]:
        return self.detector.track(frame, tracker=self.tracker)
