from cerber.core.detector import Detection
from cerber.features.scene import Scene


def _det(track_id: int, cls: int = 0, name: str = "person") -> Detection:
    return Detection(
        xyxy=(1.0, 2.0, 3.0, 4.0),
        cls=cls,
        name=name,
        conf=0.9,
        track_id=track_id,
    )


def test_scene_enter_and_age() -> None:
    scene = Scene(max_lost=2)
    scene.update([_det(7)], timestamp=1.0)
    assert 7 in scene.tracks
    assert scene.tracks[7].age == 1
    scene.update([_det(7)], timestamp=2.0)
    assert scene.tracks[7].age == 2
    assert scene.tracks[7].lost == 0


def test_scene_lost_then_drop() -> None:
    scene = Scene(max_lost=2)
    scene.update([_det(3)], timestamp=1.0)
    scene.update([], timestamp=2.0)
    assert scene.tracks[3].lost == 1
    scene.update([], timestamp=3.0)
    assert scene.tracks[3].lost == 2
    scene.update([], timestamp=4.0)
    assert 3 not in scene.tracks


def test_scene_ignores_untracked() -> None:
    scene = Scene()
    scene.update(
        [Detection(xyxy=(0, 0, 1, 1), cls=0, name="car", conf=0.5, track_id=None)],
        timestamp=1.0,
    )
    assert scene.tracks == {}
