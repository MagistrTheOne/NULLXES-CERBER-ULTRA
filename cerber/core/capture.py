from __future__ import annotations

import time
from typing import Iterator

import cv2
import numpy as np


class Capture:
    def __init__(self, source: int | str) -> None:
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video source: {source}")

    def read(self) -> tuple[np.ndarray | None, float]:
        ok, frame = self._cap.read()
        timestamp = time.time()
        if not ok or frame is None:
            return None, timestamp
        return frame, timestamp

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        while True:
            frame, timestamp = self.read()
            if frame is None:
                break
            yield frame, timestamp

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> Capture:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
