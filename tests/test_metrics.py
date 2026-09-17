from types import SimpleNamespace

from cerber.experiments.metrics import extract_metrics


def test_extract_detect_and_seg_metrics() -> None:
    metrics = SimpleNamespace(
        box=SimpleNamespace(map=0.16, map50=0.29, mp=0.41, mr=0.32, maps=[0.1, 0.2]),
        seg=SimpleNamespace(map=0.33, map50=0.51, maps=[0.3]),
    )
    payload = extract_metrics(metrics)
    assert payload["map50-95"] == 0.16
    assert payload["recall"] == 0.32
    assert payload["mask_map50"] == 0.51
    assert payload["map50-95_per_class"] == [0.1, 0.2]


def test_extract_semantic_depth_pose_cls() -> None:
    metrics = SimpleNamespace(
        miou=0.78,
        pixel_accuracy=0.9,
        abs_rel=0.11,
        rmse=0.4,
        delta1=0.88,
        silog=0.12,
        pose=SimpleNamespace(map=0.57, map50=0.83, maps=None),
        top1=0.71,
        top5=0.90,
    )
    payload = extract_metrics(metrics)
    assert payload["miou"] == 0.78
    assert payload["abs_rel"] == 0.11
    assert payload["pose_map50"] == 0.83
    assert payload["top1"] == 0.71
    assert "map50-95" not in payload
