"""Entity position tracks extracted from a replay's packet stream.

The heavy lifting is done by ReplayStudio's own EntityPositionTracker, driven by
tools/DumpTracks.java against the mod jar, rather than by reimplementing
Minecraft packet parsing here. Whatever the mod believes about entity positions
is exactly what this module sees.

That extraction is also what makes a player-following camera possible: the
camera path itself ends up as ordinary position keyframes, so the render needs
no runtime entity id -- the id is only needed here, at plan time.
"""

from __future__ import annotations

import bisect
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CP_FILE = REPO / "build" / "cp.txt"
CLASSES = REPO / "build"


class TrackError(RuntimeError):
    pass


@dataclass
class Track:
    entity_id: int
    points: list  # [(t_ms, x, y, z, yaw, pitch), ...] sorted by t

    @property
    def times(self):
        return [p[0] for p in self.points]

    @property
    def start(self):
        return self.points[0][0]

    @property
    def end(self):
        return self.points[-1][0]

    def at(self, ts):
        """Linearly interpolated (x, y, z) at replay timestamp `ts`."""
        pts = self.points
        if ts <= pts[0][0]:
            return tuple(pts[0][1:4])
        if ts >= pts[-1][0]:
            return tuple(pts[-1][1:4])
        times = self.times
        i = bisect.bisect_right(times, ts)
        a, b = pts[i - 1], pts[i]
        span = b[0] - a[0]
        f = 0.0 if span == 0 else (ts - a[0]) / span
        return tuple(a[j] + (b[j] - a[j]) * f for j in (1, 2, 3))

    def distance_travelled(self):
        return sum(
            math.dist(a[1:4], b[1:4]) for a, b in zip(self.points, self.points[1:])
        )

    def smoothed(self, window_ms=4000, step_ms=250):
        """A low-passed copy, for use as a camera target.

        A player's raw position jitters -- strafing, jumping, stepping round a
        build. Pointing a camera straight at that judders badly, so resample at a
        fixed step and take a centred moving average. The result trails the real
        position slightly, which reads as a camera operator following someone
        rather than a camera welded to them.
        """
        if not self.points:
            raise TrackError("empty track")
        t0, t1 = self.start, self.end
        n = max(2, int((t1 - t0) / step_ms) + 1)
        grid = [t0 + i * step_ms for i in range(n)]
        raw = [self.at(t) for t in grid]

        half = max(1, int((window_ms / step_ms) / 2))
        out = []
        for i in range(len(raw)):
            lo, hi = max(0, i - half), min(len(raw), i + half + 1)
            chunk = raw[lo:hi]
            out.append(
                (
                    grid[i],
                    sum(c[0] for c in chunk) / len(chunk),
                    sum(c[1] for c in chunk) / len(chunk),
                    sum(c[2] for c in chunk) / len(chunk),
                    0.0,
                    0.0,
                )
            )
        return Track(self.entity_id, out)


def dump(mcpr, out_json, java="java"):
    """Run the Java extractor against a COPY of the replay.

    EntityPositionTracker writes a cache entry back into whatever replay file it
    is handed, so the original is never passed to it.
    """
    mcpr = Path(mcpr)
    out_json = Path(out_json)
    if not mcpr.is_file():
        raise TrackError(f"no such replay: {mcpr}")
    if not CP_FILE.is_file():
        raise TrackError(
            f"missing {CP_FILE} -- run: python tools/make_cp.py  (and compile "
            "tools/DumpTracks.java first)"
        )
    if not (CLASSES / "DumpTracks.class").is_file():
        raise TrackError(
            "DumpTracks.class not built -- run:\n"
            '  javac -cp "<mod jar>" -d build tools/DumpTracks.java'
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "copy.mcpr"
        shutil.copy2(mcpr, copy)
        proc = subprocess.run(
            [java, f"@{CP_FILE}", "DumpTracks", str(copy), str(out_json)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise TrackError(f"DumpTracks failed (exit {proc.returncode}):\n{tail}")
    return out_json


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for eid, pts in data.items():
        pts = sorted(pts, key=lambda p: p[0])
        if len(pts) >= 2:
            out[int(eid)] = Track(int(eid), [tuple(p) for p in pts])
    return out


def identify_player(tracks, min_distance=20.0):
    """Best guess at the recording player's entity.

    The player's position is updated far more often than anything else in the
    replay, so sample count is the dominant signal; movement and an early start
    break ties. Verify with `mcmontage tracks` before trusting it -- a busy
    server with other players would make this ambiguous.
    """
    if not tracks:
        raise TrackError("no tracks")
    movers = [t for t in tracks.values() if t.distance_travelled() >= min_distance]
    pool = movers or list(tracks.values())
    return max(pool, key=lambda t: (len(t.points), t.end - t.start, -t.start))


SMOOTH_CANDIDATES = (16000, 8000, 4000, 2000, 1000, 500)


def max_deviation(track, smoothed, start_ms, end_ms, keyframes):
    """Worst-case gap, in blocks, between a smoothed target and the real position.

    Measured at the keyframe sample points, because those are the only places
    the camera is actually pinned.
    """
    worst = 0.0
    for i in range(keyframes):
        f = i / max(1, keyframes - 1)
        ts = start_ms + (end_ms - start_ms) * f
        a = track.at(ts)
        b = smoothed.at(ts)
        d = math.dist(a, b)
        worst = max(worst, d)
    return worst


def choose_smoothing(track, start_ms, end_ms, keyframes, max_deviation_blocks):
    """Largest smoothing window that still keeps the subject in frame.

    Smoothing removes jitter but also cuts corners: the smoothed point lags
    behind wherever you actually were. How much lag a given window causes
    depends entirely on how fast you were moving, so no fixed window -- and no
    window scaled by the timelapse speed-up -- is right for every replay. On a
    slow build 8s is fine; on an 11-minute session covering 5000 blocks the same
    window puts the camera 70+ blocks off target and you leave the shot.

    So rather than guess, measure: try progressively smaller windows and take
    the first whose worst-case deviation stays inside the budget. Returns
    (window_ms, deviation_blocks); window_ms of 0 means no smoothing, which is
    already smooth because the keyframe sampling itself low-passes the path.
    """
    for window in SMOOTH_CANDIDATES:
        smoothed = track.smoothed(window_ms=window)
        dev = max_deviation(track, smoothed, start_ms, end_ms, keyframes)
        if dev <= max_deviation_blocks:
            return window, dev
    return 0, 0.0


def rank(tracks, limit=10):
    """Diagnostic ranking, most player-like first."""
    rows = [
        (t.entity_id, len(t.points), t.start, t.end, t.distance_travelled())
        for t in tracks.values()
    ]
    rows.sort(key=lambda r: (-r[1], -(r[3] - r[2])))
    return rows[:limit]
