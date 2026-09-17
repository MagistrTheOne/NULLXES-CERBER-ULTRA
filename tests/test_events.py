from pathlib import Path

from cerber.core.detector import Detection
from cerber.features.events import EventLog
from cerber.features.scene import Scene


def _det(track_id: int) -> Detection:
    return Detection(xyxy=(0.0, 0.0, 10.0, 10.0), cls=2, name="car", conf=0.8, track_id=track_id)


def test_enter_lost_leave(tmp_path: Path) -> None:
    scene = Scene(max_lost=1)
    log = EventLog(tmp_path)
    events = log.emit(scene.update([_det(5)], 1.0), 1.0)
    assert [event.type for event in events] == ["enter"]
    events = log.emit(scene.update([], 2.0), 2.0)
    assert [event.type for event in events] == ["lost"]
    events = log.emit(scene.update([], 3.0), 3.0)
    assert [event.type for event in events] == ["leave"]
    assert events[0].track_id == 5
    assert events[0].name == "car"
    lines = log.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
