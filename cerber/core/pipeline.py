from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import replace

import cv2
import numpy as np

from cerber.config import AuxModule, RuntimeConfig
from cerber.core.capture import Capture
from cerber.core.detector import Detection, Detector
from cerber.core.result import FrameResult, merge_aux
from cerber.features.events import EventLog
from cerber.features.scene import Scene
from cerber.tasks import TRACKABLE_TASKS

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


def _draw_semantic(canvas: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    if semantic.shape[:2] != canvas.shape[:2]:
        semantic = cv2.resize(semantic.astype(np.float32), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    hue = (semantic.astype(np.uint8) * 13) % 180
    hsv = np.zeros((*semantic.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 180
    hsv[..., 2] = np.where(semantic > 0, 180, 0).astype(np.uint8)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(canvas, 0.65, color, 0.35, 0)


def _draw_depth(canvas: np.ndarray, depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return canvas
    scaled = np.clip((depth - float(finite.min())) / max(float(np.ptp(finite)), 1e-6), 0, 1)
    color = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if color.shape[:2] != canvas.shape[:2]:
        color = cv2.resize(color, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR)
    inset = cv2.resize(color, (max(80, canvas.shape[1] // 5), max(60, canvas.shape[0] // 5)))
    h, w = inset.shape[:2]
    canvas[8 : 8 + h, 8 : 8 + w] = inset
    return canvas


def annotate(frame: np.ndarray, detections: list[Detection], result: FrameResult | None = None) -> np.ndarray:
    canvas = frame.copy()
    if result is not None and result.semantic is not None:
        canvas = _draw_semantic(canvas, result.semantic)
    for detection in detections:
        if detection.obb:
            pts = np.array(detection.obb, dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 2)
        else:
            x1, y1, x2, y2 = (int(v) for v in detection.xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, _, _ = (int(v) for v in detection.xyxy)
        label = detection.name
        if detection.track_id is not None:
            label = f"{detection.track_id}:{label}"
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
        if detection.keypoints is not None:
            for point in np.asarray(detection.keypoints):
                if len(point) >= 3 and float(point[2]) < 0.25:
                    continue
                cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (255, 0, 255), -1)
    if result is not None and result.depth is not None:
        canvas = _draw_depth(canvas, result.depth)
    if result is not None and result.classification is not None:
        item = result.classification
        cv2.putText(
            canvas,
            f"{item.name} {item.conf:.2f}",
            (12, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


class Pipeline:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.detector = Detector(config)
        self.aux: list[tuple[AuxModule, Detector]] = [
            (module, Detector(config.for_aux(module))) for module in config.aux
        ]
        self._aux_last: dict[str, FrameResult] = {}
        self.scene = Scene(max_lost=config.max_lost)
        self.events = EventLog(config.resolved_events_dir())
        self.latencies: deque[float] = deque(maxlen=300)

    def infer(self, frame: np.ndarray, frame_id: int = 0, captured_at: float | None = None) -> FrameResult:
        persist = self.config.persist and self.config.task in TRACKABLE_TASKS
        result = self.detector.infer(frame, persist=persist, frame_id=frame_id, captured_at=captured_at)
        for module, detector in self.aux:
            if frame_id % module.every == 0:
                self._aux_last[module.task] = detector.infer(
                    frame,
                    persist=False,
                    frame_id=frame_id,
                    captured_at=captured_at,
                )
            result = merge_aux(result, self._aux_last.get(module.task))
        return result

    def step(self, frame: np.ndarray, timestamp: float, frame_id: int = 0) -> FrameResult:
        started = time.perf_counter()
        result = self.infer(frame, frame_id=frame_id, captured_at=timestamp)
        tracks = self.scene.update(result.instances, timestamp)
        self.events.emit(tracks, timestamp)
        self.latencies.append(time.perf_counter() - started)
        return result

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
                result = self.step(frame, timestamp, frame_id=frame_index)
                frame_index += 1
                if frame_index % self.config.log_every == 0:
                    stats = self.metrics()
                    stale = result.stale_maps()
                    LOGGER.info(
                        "frame=%s dets=%s fps=%.1f p50=%.1fms p95=%.1fms stale=%s",
                        frame_index,
                        len(result.instances),
                        stats["fps"],
                        stats["p50_ms"],
                        stats["p95_ms"],
                        stale,
                    )
                if self.config.show:
                    vis = annotate(frame, result.instances, result)
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
