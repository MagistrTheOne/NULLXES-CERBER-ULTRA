import numpy as np

from cerber.core.capture import Capture
from cerber.core.pipeline import percentile


def test_percentile_empty() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_known() -> None:
    values = [0.01, 0.02, 0.03, 0.04, 0.10]
    assert abs(percentile(values, 50) - 0.03) < 1e-9
    assert percentile(values, 95) >= percentile(values, 50)


def test_capture_video(tmp_path) -> None:
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    with Capture(str(path)) as capture:
        frames = list(capture.frames())
    assert len(frames) == 3
    assert frames[0][0].shape == (64, 64, 3)
