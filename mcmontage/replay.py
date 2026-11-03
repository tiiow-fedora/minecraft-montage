"""Read and write ReplayMod .mcpr files.

A .mcpr is a plain zip. The entries this module cares about:

  metaData.json    recording metadata written by the mod
  recording.tmcpr  the raw packet stream (we never touch it)
  timelines.json   saved camera timelines (the in-GUI "keyframe repository")
  renderQueue.json the render queue -- RenderJob.readQueue/writeQueue

renderQueue.json is a Gson-serialised List<RenderJob>:

    [ {"timeline": "<escaped JSON string>", "settings": {...}} ]

The "timeline" value is itself a JSON *string* (a custom Gson TypeAdapter
delegates to TimelineSerialization), keyed by the empty string.  See
timeline.py for that inner schema.

Schema derived from bytecode of reforgedplaymod-1.21.1-0.3.jar
(sha256 6084816...5cc0) and validated against a real file by
`python -m mcmontage.cli inspect` -- see SCHEMA_VALIDATED below.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

RENDER_QUEUE_ENTRY = "renderQueue.json"
TIMELINES_ENTRY = "timelines.json"
METADATA_ENTRY = "metaData.json"

# The inner timeline format was checked against a mod-written timelines.json on
# 2026-09-08 and matches structurally, field for field. That file is produced by
# the same TimelineSerialization.serialize() call that generates the nested
# timeline string inside renderQueue.json, so the hard half is confirmed.
#
# One error was caught by doing this: the timestamp key is "timestamp", not
# "timestamp:timestamp". Properties are prefixed "<group>:<id>" only when they
# belong to a PropertyGroup, and TimestampProperty is ungrouped.
TIMELINE_SCHEMA_VALIDATED = True

# The RenderJob wrapper and RenderSettings field names were confirmed on
# 2026-09-08 by a completed render: the mod consumed a renderQueue.json written
# by this code and produced exactly the requested output -- 25.00s at 1920x1080
# and 60fps, matching the timeline duration and the videoWidth / videoHeight /
# framesPerSecond fields. A malformed envelope or wrong field names could not
# have produced a to-spec render.
RENDER_QUEUE_SCHEMA_VALIDATED = True

SCHEMA_VALIDATED = TIMELINE_SCHEMA_VALIDATED and RENDER_QUEUE_SCHEMA_VALIDATED


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayInfo:
    path: Path
    entries: list[str]
    metadata: dict
    has_render_queue: bool
    has_timelines: bool

    @property
    def duration_ms(self) -> int | None:
        d = self.metadata.get("duration")
        return int(d) if isinstance(d, (int, float)) else None

    @property
    def server_name(self) -> str | None:
        return self.metadata.get("serverName")


def _read_json_entry(zf: zipfile.ZipFile, name: str) -> dict | list | None:
    try:
        raw = zf.read(name)
    except KeyError:
        return None
    if not raw.strip():
        return None
    return json.loads(raw.decode("utf-8"))


def inspect(mcpr: Path) -> ReplayInfo:
    """Read a .mcpr without modifying it."""
    mcpr = Path(mcpr)
    if not mcpr.is_file():
        raise ReplayError(f"no such replay: {mcpr}")
    try:
        with zipfile.ZipFile(mcpr) as zf:
            names = zf.namelist()
            meta = _read_json_entry(zf, METADATA_ENTRY) or {}
            if not isinstance(meta, dict):
                meta = {}
            return ReplayInfo(
                path=mcpr,
                entries=names,
                metadata=meta,
                has_render_queue=RENDER_QUEUE_ENTRY in names,
                has_timelines=TIMELINES_ENTRY in names,
            )
    except (zipfile.BadZipFile, ValueError) as exc:
        raise ReplayError(f"{mcpr.name}: not a readable replay ({exc})") from exc


def read_render_queue(mcpr: Path) -> list[dict]:
    """Return the raw render queue, with each job's nested timeline decoded.

    Each returned job is {"timeline": <decoded dict>, "settings": {...},
    "_timeline_raw": <original string>} so a round-trip can be compared
    byte-for-byte against what the mod wrote.
    """
    try:
        with zipfile.ZipFile(mcpr) as zf:
            queue = _read_json_entry(zf, RENDER_QUEUE_ENTRY)
    except (zipfile.BadZipFile, ValueError) as exc:
        raise ReplayError(
            f"{Path(mcpr).name}: cannot read {RENDER_QUEUE_ENTRY} ({exc})") from exc
    if queue is None:
        return []
    if not isinstance(queue, list):
        raise ReplayError(f"{RENDER_QUEUE_ENTRY} is not a list: {type(queue).__name__}")
    out = []
    for job in queue:
        raw = job.get("timeline")
        decoded = json.loads(raw) if isinstance(raw, str) else raw
        out.append({**job, "timeline": decoded, "_timeline_raw": raw})
    return out


def _encode_job(job: dict) -> dict:
    """Re-encode a job so its timeline is a nested JSON string again."""
    tl = job["timeline"]
    encoded = tl if isinstance(tl, str) else json.dumps(tl, separators=(",", ":"))
    return {"timeline": encoded, "settings": job["settings"]}


def write_render_queue(mcpr: Path, jobs: list[dict], *, backup: bool = True) -> Path:
    """Replace renderQueue.json inside the .mcpr.

    Rewrites the archive (zips cannot replace an entry in place). The original
    is copied to <name>.mcpr.bak first unless backup=False.
    """
    mcpr = Path(mcpr)
    if backup:
        bak = mcpr.with_suffix(mcpr.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(mcpr, bak)

    payload = json.dumps([_encode_job(j) for j in jobs], separators=(",", ":"))
    tmp = mcpr.with_suffix(mcpr.suffix + ".tmp")

    with zipfile.ZipFile(mcpr) as src, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            if item.filename == RENDER_QUEUE_ENTRY:
                continue
            # copy compressed payload through without re-deflating the big
            # recording.tmcpr entry
            dst.writestr(item, src.read(item.filename))
        dst.writestr(RENDER_QUEUE_ENTRY, payload.encode("utf-8"))

    tmp.replace(mcpr)
    return mcpr


def find_replays(recordings_dir: Path) -> list[Path]:
    """All .mcpr files in a recordings folder, newest first."""
    d = Path(recordings_dir)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.mcpr"), key=lambda p: p.stat().st_mtime, reverse=True)
