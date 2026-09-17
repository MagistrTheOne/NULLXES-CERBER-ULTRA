from __future__ import annotations

from dataclasses import dataclass, replace

from cerber.core.detector import Detection


@dataclass(frozen=True)
class TrackState:
    track_id: int
    cls: int
    name: str
    xyxy: tuple[float, float, float, float]
    conf: float
    age: int
    lost: int
    last_seen: float


class Scene:
    def __init__(self, max_lost: int = 30) -> None:
        self.max_lost = max_lost
        self.tracks: dict[int, TrackState] = {}

    def update(self, detections: list[Detection], timestamp: float) -> dict[int, TrackState]:
        seen: set[int] = set()
        for detection in detections:
            if detection.track_id is None:
                continue
            seen.add(detection.track_id)
            previous = self.tracks.get(detection.track_id)
            self.tracks[detection.track_id] = TrackState(
                track_id=detection.track_id,
                cls=detection.cls,
                name=detection.name,
                xyxy=detection.xyxy,
                conf=detection.conf,
                age=(previous.age + 1) if previous is not None else 1,
                lost=0,
                last_seen=timestamp,
            )
        stale: list[int] = []
        for track_id, state in self.tracks.items():
            if track_id in seen:
                continue
            lost = state.lost + 1
            if lost > self.max_lost:
                stale.append(track_id)
            else:
                self.tracks[track_id] = replace(state, lost=lost)
        for track_id in stale:
            del self.tracks[track_id]
        return dict(self.tracks)
