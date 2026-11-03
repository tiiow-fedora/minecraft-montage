"""Group replays into play sessions, and name them accordingly.

ReplayMod names recordings by wall-clock timestamp, which says nothing about
which ones belong together. This groups them into "play days" so a night's
worth of clips can be prepared, rendered and stitched as one montage.

The day boundary is NOT midnight. A session that runs from 22:00 to 01:30 is
obviously one sitting, but a midnight boundary would split it across two days
and number the clips wrongly. So a play-day runs from CUTOFF_HOUR to
CUTOFF_HOUR (04:00 by default): anything recorded before 04:00 counts towards
the previous day.
"""

from __future__ import annotations

import datetime
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

# A play-day runs from this hour to this hour the next day. Playing past
# midnight still counts as the day you started.
CUTOFF_HOUR = 4


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Clip:
    path: Path
    recorded: datetime.datetime
    duration_ms: int

    @property
    def ends(self):
        return self.recorded + datetime.timedelta(milliseconds=self.duration_ms)

    @property
    def minutes(self):
        return self.duration_ms / 60000.0


@dataclass
class Session:
    day: datetime.date       # the play-day these belong to
    index: int               # 1-based, chronological among all sessions
    clips: list              # Clips, in recording order

    @property
    def total_minutes(self):
        return sum(c.minutes for c in self.clips)

    @property
    def started(self):
        return self.clips[0].recorded

    @property
    def ended(self):
        return self.clips[-1].ends

    def name_for(self, clip, scheme="day{session}_{n}"):
        n = self.clips.index(clip) + 1
        return scheme.format(session=self.index, n=n,
                             date=self.day.isoformat())


def read_clip(mcpr):
    """Recording time and duration, straight from the replay's own metadata.

    Deliberately not file mtime: writing a render queue rewrites the archive and
    bumps mtime, so mtime stops reflecting when the session actually happened.
    """
    mcpr = Path(mcpr)
    try:
        with zipfile.ZipFile(mcpr) as z:
            meta = json.loads(z.read("metaData.json").decode("utf-8"))
    except (KeyError, OSError, zipfile.BadZipFile, ValueError) as exc:
        raise SessionError(f"{mcpr.name}: cannot read metaData.json ({exc})") from exc

    ts = meta.get("date")
    if not ts:
        raise SessionError(f"{mcpr.name}: no recording date in metadata")
    return Clip(
        path=mcpr,
        recorded=datetime.datetime.fromtimestamp(ts / 1000),
        duration_ms=int(meta.get("duration") or 0),
    )


def play_day(when, cutoff_hour=CUTOFF_HOUR):
    """The play-day a moment belongs to (before the cutoff = previous day)."""
    d = when.date()
    if when.hour < cutoff_hour:
        d -= datetime.timedelta(days=1)
    return d


def group(paths, cutoff_hour=CUTOFF_HOUR, max_gap_minutes=None):
    """Group replays into Sessions, oldest first.

    By default a session is a whole play-day. Pass `max_gap_minutes` to split
    further whenever there is a long quiet stretch between the end of one clip
    and the start of the next -- useful if you play twice in a day and want two
    separate montages.
    """
    clips = sorted((read_clip(p) for p in paths), key=lambda c: c.recorded)
    if not clips:
        return []

    groups = []
    current = [clips[0]]
    for prev, clip in zip(clips, clips[1:]):
        same_day = (play_day(prev.recorded, cutoff_hour)
                    == play_day(clip.recorded, cutoff_hour))
        gap_ok = True
        if max_gap_minutes is not None:
            gap = (clip.recorded - prev.ends).total_seconds() / 60.0
            gap_ok = gap <= max_gap_minutes
        if same_day and gap_ok:
            current.append(clip)
        else:
            groups.append(current)
            current = [clip]
    groups.append(current)

    return [
        Session(day=play_day(g[0].recorded, cutoff_hour), index=i, clips=g)
        for i, g in enumerate(groups, 1)
    ]


def rename_plan(sessions, scheme="day{session}_{n}"):
    """[(current_path, new_path)] for clips whose name is already wrong."""
    plan = []
    for s in sessions:
        for c in s.clips:
            new = c.path.with_name(s.name_for(c, scheme) + c.path.suffix)
            if new != c.path:
                plan.append((c.path, new))
    return plan


def apply_renames(plan):
    """Rename via temporary names, so a target never blocks a source.

    Renumbering a set of files inevitably means some new name is another file's
    current name, so a straight rename can clobber. Parking everything under
    temporary names first makes the order irrelevant.

    Everything is validated before the first rename, and a failure part-way
    puts parked files back, so an error never strands recordings under
    __mcmontage_rename_N.tmp names.
    """
    if not plan:
        return []
    sources = {src for src, _ in plan}
    targets = [dst for _, dst in plan]
    if len(set(targets)) != len(targets):
        raise SessionError("two files would be renamed to the same name")
    for src, dst in plan:
        if not src.is_file():
            raise SessionError(f"gone before rename: {src}")
        if dst.exists() and dst not in sources:
            raise SessionError(
                f"will not overwrite {dst.name}: it exists and is not part of "
                "this rename")

    parked = []          # (tmp, src, dst)
    placed = []          # (dst, tmp)
    try:
        for i, (src, dst) in enumerate(plan):
            tmp = src.with_name(f"__mcmontage_rename_{i}.tmp")
            src.rename(tmp)
            parked.append((tmp, src, dst))
        for tmp, src, dst in parked:
            tmp.rename(dst)
            placed.append((dst, tmp))
    except OSError:
        for dst, tmp in reversed(placed):
            dst.rename(tmp)
        for tmp, src, _ in parked:
            tmp.rename(src)
        raise
    return [(src, dst) for _, src, dst in parked]
