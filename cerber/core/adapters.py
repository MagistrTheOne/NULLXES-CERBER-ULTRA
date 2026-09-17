from __future__ import annotations

from typing import Any

import numpy as np

from cerber.core.result import Classification, Detection, FrameResult


def as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    array = np.asarray(value)
    if array.dtype == object and array.shape == ():
        return None
    return array


def _names(result: object, fallback: dict[int, str] | None = None) -> dict[int, str]:
    raw = getattr(result, "names", None) or fallback or {}
    return {int(key): str(value) for key, value in dict(raw).items()}


def _aabb(corners: np.ndarray) -> tuple[float, float, float, float]:
    xs = corners[:, 0]
    ys = corners[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def parse_instances(result: object, names: dict[int, str] | None = None) -> list[Detection]:
    labels = _names(result, names)
    boxes = getattr(result, "boxes", None)
    obb = getattr(result, "obb", None)
    keypoints = getattr(result, "keypoints", None)
    masks = getattr(getattr(result, "masks", None), "data", None)
    mask_array = as_numpy(masks)

    detections: list[Detection] = []
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        xyxy = as_numpy(boxes.xyxy)
        if xyxy is not None and len(xyxy):
            cls_ids = as_numpy(getattr(boxes, "cls", None))
            confs = as_numpy(getattr(boxes, "conf", None))
            track_ids = as_numpy(getattr(boxes, "id", None))
            kpts = as_numpy(getattr(keypoints, "data", None)) if keypoints is not None else None
            for index, box in enumerate(xyxy):
                cls_id = int(cls_ids[index]) if cls_ids is not None else 0
                mask = None
                if mask_array is not None and index < len(mask_array):
                    mask = (mask_array[index] > 0.5).astype(np.uint8)
                pose = None
                if kpts is not None and index < len(kpts):
                    pose = np.asarray(kpts[index], dtype=np.float32)
                detections.append(
                    Detection(
                        xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        cls=cls_id,
                        conf=float(confs[index]) if confs is not None else 0.0,
                        name=labels.get(cls_id, str(cls_id)),
                        track_id=int(track_ids[index]) if track_ids is not None else None,
                        mask=mask,
                        keypoints=pose,
                    )
                )
            return detections

    if obb is not None and getattr(obb, "xyxyxyxy", None) is not None:
        corners = as_numpy(obb.xyxyxyxy)
        if corners is None or len(corners) == 0:
            return []
        cls_ids = as_numpy(getattr(obb, "cls", None))
        confs = as_numpy(getattr(obb, "conf", None))
        track_ids = as_numpy(getattr(obb, "id", None))
        xywhr = as_numpy(getattr(obb, "xywhr", None))
        for index, poly in enumerate(corners):
            pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
            cls_id = int(cls_ids[index]) if cls_ids is not None else 0
            rotation = None
            if xywhr is not None and index < len(xywhr):
                rotation = tuple(float(v) for v in xywhr[index][:5])
            detections.append(
                Detection(
                    xyxy=_aabb(pts),
                    cls=cls_id,
                    conf=float(confs[index]) if confs is not None else 0.0,
                    name=labels.get(cls_id, str(cls_id)),
                    track_id=int(track_ids[index]) if track_ids is not None else None,
                    obb=tuple((float(x), float(y)) for x, y in pts),
                    obb_xywhr=rotation,
                )
            )
    return detections


def parse_semantic(result: object) -> np.ndarray | None:
    mask = getattr(getattr(result, "semantic_mask", None), "data", None)
    array = as_numpy(mask)
    if array is None:
        return None
    return np.asarray(array).squeeze()


def parse_depth(result: object) -> np.ndarray | None:
    depth = getattr(getattr(result, "depth", None), "data", None)
    array = as_numpy(depth)
    if array is None:
        return None
    return np.asarray(array, dtype=np.float32).squeeze()


def parse_classification(result: object, names: dict[int, str] | None = None) -> Classification | None:
    probs = getattr(result, "probs", None)
    if probs is None:
        return None
    labels = _names(result, names)
    top1 = int(getattr(probs, "top1", 0))
    conf_raw = as_numpy(getattr(probs, "top1conf", 0.0))
    conf = float(conf_raw.reshape(-1)[0]) if conf_raw is not None else 0.0
    raw_top5 = getattr(probs, "top5", None) or [top1]
    data = as_numpy(getattr(probs, "data", None))
    top5: list[tuple[int, str, float]] = []
    for cls_id in raw_top5:
        idx = int(cls_id)
        score = float(data[idx]) if data is not None and idx < len(data) else conf
        top5.append((idx, labels.get(idx, str(idx)), score))
    return Classification(cls=top1, name=labels.get(top1, str(top1)), conf=conf, top5=tuple(top5))


def adapt_result(
    result: object,
    *,
    task: str,
    model: str,
    frame_id: int = 0,
    captured_at: float = 0.0,
    finished_at: float = 0.0,
    names: dict[int, str] | None = None,
) -> FrameResult:
    labels = names or _names(result)
    semantic = parse_semantic(result)
    depth = parse_depth(result)
    return FrameResult(
        frame_id=frame_id,
        captured_at=captured_at,
        finished_at=finished_at,
        task=task,
        model=model,
        names=labels,
        instances=parse_instances(result, labels),
        semantic=semantic,
        semantic_frame_id=frame_id if semantic is not None else None,
        depth=depth,
        depth_frame_id=frame_id if depth is not None else None,
        classification=parse_classification(result, labels),
    )
