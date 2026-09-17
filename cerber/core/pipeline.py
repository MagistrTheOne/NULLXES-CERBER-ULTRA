from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import replace

import cv2
import numpy as np

from cerber.config import RuntimeConfig
from cerber.core.capture import Capture
from cerber.core.detector import Detection, Detector
from cerber.features.events import EventLog
from cerber.features.scene import Scene
from cerber.features.tracking import Tracker

LOGGER = logging.getLogger("cerber")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection.xyxy)
        label = detection.name
        if detection.track_id is not None:
            label = f"{detection.track_id}:{label}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{label} {detection.conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        if detection.mask is not None:
            mask = detection.mask
            if mask.shape[:2] != canvas.shape[:2]:
                mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = canvas.copy()
            overlay[mask.astype(bool)] = (0, 128, 255)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    return canvas


class Pipeline:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.detector = Detector(config)
        self.tracker = Tracker(self.detector, tracker=config.tracker) if config.persist else None
        self.scene = Scene(max_lost=config.max_lost)
        self.events = EventLog(config.resolved_events_dir())
        self.latencies: deque[float] = deque(maxlen=300)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        if self.tracker is not None:
            return self.tracker.update(frame)
        return self.detector.predict(frame)

    def step(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        started = time.perf_counter()
        detections = self.infer(frame)
        tracks = self.scene.update(detections, timestamp)
        self.events.emit(tracks, timestamp)
        self.latencies.append(time.perf_counter() - started)
        return detections

    def metrics(self) -> dict[str, float]:
        samples = list(self.latencies)
        if not samples:
            return {"fps": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        mean = sum(samples) / len(samples)
        return {
            "fps": (1.0 / mean) if mean > 0 else 0.0,
            "p50_ms": percentile(samples, 50) * 1000.0,
            "p95_ms": percentile(samples, 95) * 1000.0,
        }

    def run(self) -> dict[str, float]:
        frame_index = 0
        with Capture(self.config.source) as capture:
            for frame, timestamp in capture.frames():
                detections = self.step(frame, timestamp)
                frame_index += 1
                if frame_index % self.config.log_every == 0:
                    stats = self.metrics()
                    LOGGER.info(
                        "frame=%s dets=%s fps=%.1f p50=%.1fms p95=%.1fms",
                        frame_index,
                        len(detections),
                        stats["fps"],
                        stats["p50_ms"],
                        stats["p95_ms"],
                    )
                if self.config.show:
                    vis = annotate(frame, detections)
                    try:
                        cv2.imshow("CERBER", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error as exc:
                        LOGGER.warning("show disabled: %s", exc)
                        self.config = replace(self.config, show=False)
        if self.config.show:
            cv2.destroyAllWindows()
        stats = self.metrics()
        LOGGER.info(
            "done frames=%s fps=%.1f p50=%.1fms p95=%.1fms events=%s",
            frame_index,
            stats["fps"],
            stats["p50_ms"],
            stats["p95_ms"],
            self.events.path,
        )
        return stats
