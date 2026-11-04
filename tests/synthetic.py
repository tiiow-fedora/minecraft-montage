"""Synthetic fixtures: fake ReplayMod recordings, entity tracks, clips, music.

Nothing here needs Minecraft, ReplayMod or Java.  A real .mcpr is a zip whose
interesting entries are small JSON files; ``recording.tmcpr`` (the packet
stream) is opaque to this project, so it is filled with deterministic filler.

Everything is deterministic: same arguments -> same bytes (fixed zip
timestamps, seeded RNG), so generated archives can be compared and golden
files never churn.
"""

from __future__ import annotations

import datetime
import json
import math
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def ms(dt: datetime.datetime) -> int:
    """Local naive datetime -> epoch ms, the way ReplayMod writes metaData.date.

    sessions.read_clip reads it back with ``fromtimestamp`` (local time), so
    the round trip is timezone independent.
    """
    return int(dt.timestamp() * 1000)


def _zip_write(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, data)


def make_mcpr(
    path,
    *,
    duration_ms: int = 600_000,
    date: datetime.datetime | int | None = datetime.datetime(2026, 9, 1, 20, 0),
    server_name: str | None = "synthetic.example",
    render_queue: list | str | bytes | None = None,
    timelines: dict | None = None,
    metadata: dict | None = None,
    recording_kib: int = 8,
    extra_entries: dict | None = None,
    omit_metadata: bool = False,
) -> Path:
    """Write a synthetic .mcpr and return its path.

    ``render_queue`` is written verbatim if it is str/bytes, JSON-encoded
    otherwise.  ``metadata`` is merged over the defaults (a value of None drops
    the key, to model a recording missing a field).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "singleplayer": False,
        "serverName": server_name,
        "duration": duration_ms,
        "date": ms(date) if isinstance(date, datetime.datetime) else date,
        "mcversion": "1.21.1",
        "fileFormat": "MCPR",
        "fileFormatVersion": 14,
        "protocol": 767,
        "generator": "synthetic-test-harness",
        "selfId": 0,
    }
    for k, v in (metadata or {}).items():
        if v is None:
            meta.pop(k, None)
        else:
            meta[k] = v
    meta = {k: v for k, v in meta.items() if v is not None}

    rng = random.Random(duration_ms)
    filler = bytes(rng.randrange(256) for _ in range(256)) * (recording_kib * 4)

    with zipfile.ZipFile(path, "w") as zf:
        if not omit_metadata:
            _zip_write(zf, "metaData.json", json.dumps(meta).encode())
        _zip_write(zf, "recording.tmcpr", filler)
        if timelines is not None:
            _zip_write(zf, "timelines.json", json.dumps(timelines).encode())
        if render_queue is not None:
            if isinstance(render_queue, bytes):
                raw = render_queue
            elif isinstance(render_queue, str):
                raw = render_queue.encode()
            else:
                raw = json.dumps(render_queue, separators=(",", ":")).encode()
            _zip_write(zf, "renderQueue.json", raw)
        for name, data in (extra_entries or {}).items():
            _zip_write(zf, name, data if isinstance(data, bytes) else data.encode())
    return path


# --- entity tracks ----------------------------------------------------------

def walk_points(t0, t1, step_ms, start=(0.0, 64.0, 0.0), velocity=(0.0, 0.0, 0.0),
                jitter=0.0, seed=0):
    """Straight-line walk sampled every step_ms: [(t,x,y,z,yaw,pitch), ...]."""
    rng = random.Random(seed)
    pts = []
    t = t0
    while t <= t1:
        s = (t - t0) / 1000.0
        pts.append((
            t,
            start[0] + velocity[0] * s + rng.uniform(-jitter, jitter),
            start[1] + velocity[1] * s,
            start[2] + velocity[2] * s + rng.uniform(-jitter, jitter),
            0.0, 0.0,
        ))
        t += step_ms
    return pts


def circle_points(t0, t1, step_ms, centre=(0.0, 64.0, 0.0), radius=10.0,
                  period_ms=20_000):
    pts = []
    t = t0
    while t <= t1:
        a = 2 * math.pi * (t - t0) / period_ms
        pts.append((t, centre[0] + radius * math.cos(a), centre[1],
                    centre[2] + radius * math.sin(a), 0.0, 0.0))
        t += step_ms
    return pts


def write_tracks(path, tracks: dict) -> Path:
    """Write the JSON DumpTracks.java would: {"<entity id>": [[t,x,y,z,yaw,pitch],...]}"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): [list(p) for p in v]
                                for k, v in tracks.items()}), encoding="utf-8")
    return path


def standard_tracks(duration_ms=60_000):
    """A player (entity 100, sampled often, walking) among quieter mobs."""
    return {
        100: walk_points(0, duration_ms, 50, velocity=(2.0, 0.0, 1.0), jitter=0.2, seed=1),
        200: walk_points(1000, duration_ms // 2, 500, start=(5, 64, 5), velocity=(0.5, 0, 0)),
        300: walk_points(0, duration_ms, 2000, start=(-30, 70, 12)),  # stationary mob
    }


# --- media (needs ffmpeg) ---------------------------------------------------

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ff(args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                   check=True)


def make_clip(path, seconds=3.0, size=(320, 180), fps=30, audio=False,
              pattern="testsrc2") -> Path:
    """A tiny ReplayMod-like clip.  Real renders have no audio (-an)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    args = ["-f", "lavfi", "-i", f"{pattern}=s={w}x{h}:r={fps}:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=f=440:d={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac"] if audio else ["-an"]
    _ff([*args, str(path)])
    return path


def make_tone(path, seconds=2.0, freq=440, with_cover_art=False) -> Path:
    """A short sine-wave music file (wav/mp3/m4a chosen by suffix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-f", "lavfi", "-i", f"sine=f={freq}:d={seconds}"]
    if with_cover_art:
        args += ["-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
                 "-map", "0:a", "-map", "1:v", "-c:v", "mjpeg", "-id3v2_version", "3"]
    _ff([*args, str(path)])
    return path
