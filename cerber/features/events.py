from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cerber.features.scene import TrackState


@dataclass(frozen=True)
class Event:
    ts: float
    type: str
    track_id: int
    cls: int
    name: str
    xyxy: tuple[float, float, float, float]
    conf: float
    lost: int


class EventLog:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.output_dir / f"events-{stamp}.jsonl"
        self._previous: dict[int, TrackState] = {}

    def emit(self, tracks: dict[int, TrackState], timestamp: float) -> list[Event]:
        current_ids = set(tracks)
        previous_ids = set(self._previous)
        events: list[Event] = []
        for track_id in current_ids - previous_ids:
            events.append(self._from_state("enter", tracks[track_id], timestamp))
        for track_id in previous_ids - current_ids:
            events.append(self._from_state("leave", self._previous[track_id], timestamp))
        for track_id in current_ids & previous_ids:
            state = tracks[track_id]
            if state.lost == 1:
                events.append(self._from_state("lost", state, timestamp))
        if events:
            with self.path.open("a", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(asdict(event), ensure_ascii=True) + "\n")
        self._previous = dict(tracks)
        return events

    @staticmethod
    def _from_state(event_type: str, state: TrackState, timestamp: float) -> Event:
        return Event(
            ts=timestamp,
            type=event_type,
            track_id=state.track_id,
            cls=state.cls,
            name=state.name,
            xyxy=state.xyxy,
            conf=state.conf,
            lost=state.lost,
        )
