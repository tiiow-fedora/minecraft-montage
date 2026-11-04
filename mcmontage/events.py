"""Shared data types for the overhaul modules.

Everything that crosses a module boundary lives here so the detectors, the
camera planner, the editor and the batch queue agree on one vocabulary:

    Event      a timestamped thing that happened in a replay (input to highlights)
    Highlight  a span of replay time worth putting in the montage (output of
               highlights, input to campath / beatsync / queue)

Times are always milliseconds of *replay* time, matching Track.points[i][0] and
the "timestamp" keyframe property. Positions are Minecraft world coordinates.

Event feed (optional, JSON): tracks alone show where entities were, not what
happened. A richer extractor can write a sidecar next to the tracks cache:

    {"version": 1,
     "events": [ {"t": 123456, "kind": "death", "entity": 7,
                  "pos": [x, y, z], "target": null, "meta": {}} ] }

Known kinds: death, kill, damage, block_place, block_break, chat, respawn.
Unknown kinds are preserved and ignored by the built-in detectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

EVENT_KINDS = ("death", "kill", "damage", "block_place", "block_break", "chat", "respawn")
HIGHLIGHT_KINDS = ("death", "kill", "speed", "build", "fall", "custom")


class EventError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    t: int                       # replay ms
    kind: str
    entity: int | None = None    # network entity id the event is about
    pos: tuple | None = None     # (x, y, z) or None
    target: int | None = None    # e.g. the victim of a kill
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"t": int(self.t), "kind": self.kind, "entity": self.entity,
                "pos": list(self.pos) if self.pos else None,
                "target": self.target, "meta": dict(self.meta)}

    @staticmethod
    def from_json(d: dict) -> "Event":
        try:
            pos = d.get("pos")
            return Event(
                t=int(d["t"]), kind=str(d["kind"]), entity=d.get("entity"),
                pos=tuple(float(v) for v in pos) if pos else None,
                target=d.get("target"), meta=dict(d.get("meta") or {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise EventError(f"bad event {d!r}: {e}") from e


@dataclass(frozen=True)
class Highlight:
    kind: str                    # one of HIGHLIGHT_KINDS
    start_ms: int                # replay ms, inclusive
    end_ms: int                  # replay ms, exclusive; end > start
    score: float                 # 0..1, higher = better; comparable across kinds
    entity: int | None = None    # who it is about
    pos: tuple | None = None     # world position of the focal point
    label: str = ""              # short human text, e.g. "Death at 1:23"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.end_ms <= self.start_ms:
            raise EventError(f"highlight end {self.end_ms} <= start {self.start_ms}")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_json(self) -> dict:
        return {"kind": self.kind, "start_ms": int(self.start_ms),
                "end_ms": int(self.end_ms), "score": round(float(self.score), 4),
                "entity": self.entity,
                "pos": list(self.pos) if self.pos else None,
                "label": self.label, "meta": dict(self.meta)}

    @staticmethod
    def from_json(d: dict) -> "Highlight":
        try:
            pos = d.get("pos")
            return Highlight(
                kind=str(d["kind"]), start_ms=int(d["start_ms"]),
                end_ms=int(d["end_ms"]), score=float(d["score"]),
                entity=d.get("entity"),
                pos=tuple(float(v) for v in pos) if pos else None,
                label=str(d.get("label", "")), meta=dict(d.get("meta") or {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise EventError(f"bad highlight {d!r}: {e}") from e


def load_events(path) -> list[Event]:
    """Read an event sidecar; a missing file means no events, not an error."""
    p = Path(path)
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("events", []) if isinstance(data, dict) else data
    return sorted((Event.from_json(e) for e in raw), key=lambda e: e.t)


def save_events(path, events) -> None:
    Path(path).write_text(
        json.dumps({"version": 1, "events": [e.to_json() for e in events]}),
        encoding="utf-8")
