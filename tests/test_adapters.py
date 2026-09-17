from types import SimpleNamespace

import numpy as np

from cerber.core.adapters import adapt_result, parse_classification, parse_instances, parse_semantic
from cerber.core.result import FrameResult, merge_aux
from cerber.tasks import infer_task


def test_infer_task_from_weights() -> None:
    assert infer_task("yolo26n.pt") == "detect"
    assert infer_task("yolo26n-seg.pt") == "segment"
    assert infer_task("yolo26n-sem.pt") == "semantic"
    assert infer_task("yolo26n-depth.pt") == "depth"
    assert infer_task("yolo26n-pose.pt") == "pose"
    assert infer_task("yolo26n-obb.pt") == "obb"
    assert infer_task("yolo26n-cls.pt") == "classify"


def test_parse_boxes_and_masks() -> None:
    result = SimpleNamespace(
        names={0: "person"},
        boxes=SimpleNamespace(
            xyxy=np.array([[1.0, 2.0, 3.0, 4.0]]),
            cls=np.array([0.0]),
            conf=np.array([0.9]),
            id=np.array([7.0]),
        ),
        masks=SimpleNamespace(data=np.ones((1, 4, 4), dtype=np.float32)),
        keypoints=None,
        obb=None,
        semantic_mask=None,
        depth=None,
        probs=None,
    )
    detections = parse_instances(result)
    assert len(detections) == 1
    assert detections[0].name == "person"
    assert detections[0].track_id == 7
    assert detections[0].mask is not None
    assert detections[0].mask.shape == (4, 4)


def test_parse_obb_not_as_instance_masks() -> None:
    corners = np.array([[[0, 0], [10, 0], [10, 4], [0, 4]]], dtype=np.float32)
    result = SimpleNamespace(
        names={1: "ship"},
        boxes=None,
        masks=None,
        keypoints=None,
        obb=SimpleNamespace(
            xyxyxyxy=corners,
            cls=np.array([1.0]),
            conf=np.array([0.8]),
            id=None,
            xywhr=np.array([[5.0, 2.0, 10.0, 4.0, 0.1]]),
        ),
        semantic_mask=None,
        depth=None,
        probs=None,
    )
    detections = parse_instances(result)
    assert len(detections) == 1
    assert detections[0].name == "ship"
    assert detections[0].obb is not None
    assert detections[0].xyxy == (0.0, 0.0, 10.0, 4.0)


def test_semantic_is_class_map() -> None:
    class_map = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    result = SimpleNamespace(semantic_mask=SimpleNamespace(data=class_map), masks=None, boxes=None)
    parsed = parse_semantic(result)
    assert parsed is not None
    assert parsed.shape == (2, 2)
    assert parsed[0, 1] == 1
    adapted = adapt_result(result, task="semantic", model="yolo26n-sem.pt", names={0: "road", 1: "car", 2: "sky"})
    assert adapted.instances == []
    assert adapted.semantic is not None


def test_classification_top1() -> None:
    result = SimpleNamespace(
        names={0: "cat", 1: "dog", 2: "bus"},
        probs=SimpleNamespace(top1=2, top1conf=0.91, top5=[2, 1, 0], data=np.array([0.05, 0.04, 0.91])),
    )
    item = parse_classification(result)
    assert item is not None
    assert item.name == "bus"
    assert item.top5[0][1] == "bus"


def test_merge_aux_keeps_stale_frame_id() -> None:
    primary = FrameResult(
        frame_id=9,
        captured_at=0.0,
        finished_at=0.1,
        task="segment",
        model="yolo26n-seg.pt",
        names={},
    )
    aux = FrameResult(
        frame_id=5,
        captured_at=0.0,
        finished_at=0.1,
        task="depth",
        model="yolo26n-depth.pt",
        names={},
        depth=np.ones((2, 2), dtype=np.float32),
        depth_frame_id=5,
    )
    merged = merge_aux(primary, aux)
    assert merged.depth_frame_id == 5
    assert merged.stale_maps()["depth"] is True
