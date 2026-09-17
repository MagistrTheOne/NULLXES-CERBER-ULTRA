from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cerber.config import RuntimeConfig


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    cls: int
    conf: float
    name: str
    track_id: int | None = None
    mask: np.ndarray | None = None


class Detector:
    def __init__(self, config: RuntimeConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.resolved_weights())
        names = getattr(self.model, "names", {}) or {}
        self.names = {int(key): str(value) for key, value in names.items()}

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            verbose=self.config.verbose,
        )
        return self._parse(results[0])

    def track(self, frame: np.ndarray, tracker: str) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracker,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            verbose=self.config.verbose,
        )
        return self._parse(results[0])

    def _parse(self, result: object) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        track_ids = None
        raw_ids = getattr(boxes, "id", None)
        if raw_ids is not None:
            track_ids = raw_ids.cpu().numpy().astype(int)
        masks = None
        mask_data = getattr(getattr(result, "masks", None), "data", None)
        if mask_data is not None:
            masks = mask_data.cpu().numpy()
        detections: list[Detection] = []
        for index, box in enumerate(xyxy):
            cls_id = int(cls_ids[index])
            mask = None
            if masks is not None and index < len(masks):
                mask = (masks[index] > 0.5).astype(np.uint8)
            detections.append(
                Detection(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    cls=cls_id,
                    conf=float(confs[index]),
                    name=self.names.get(cls_id, str(cls_id)),
                    track_id=int(track_ids[index]) if track_ids is not None else None,
                    mask=mask,
                )
            )
        return detections
