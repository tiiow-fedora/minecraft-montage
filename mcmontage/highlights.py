"""Automatic highlight detection from replay tracks and (optional) events.

Why this exists: a montage is made of the few seconds of a long replay worth
watching. Finding them by hand means scrubbing hours of footage, so this module
proposes candidates -- deaths, kills, fast travel, long falls, building
sessions -- each with a 0..1 score that is comparable across kinds so a later
stage (:func:`select`) can pick the best ones to fit a target running time.

Observed vs inferred (read this before trusting a result)
---------------------------------------------------------
* Anything derived from an :class:`~mcmontage.events.Event` (death, kill,
  block_place) is *observed*: the replay said it happened.
* Everything derived from position tracks alone is *inferred*. Speed and
  falls are measured directly from positions, so they are reliable, but the
  *meaning* is a guess (fast could be a boat, a horse or elytra). Death
  inference and the track-only build detector are heuristics: they can miss
  real events and can fire on look-alikes (a plain ``/tp`` after a lag spike
  looks like a respawn; an AFK-farm player spinning in place looks like
  building). Every highlight records ``meta["source"]`` so downstream code and
  the human editing the list can tell the difference.

Conventions: times are replay milliseconds; detectors return *core* spans
(just the moment itself, no padding); :func:`detect_all` adds pre/post roll,
clamps to the replay, merges overlaps and normalises scores. Pure standard
library, deterministic, and never emits NaN/inf (non-finite track samples are
dropped up front).
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .events import Event, Highlight
from .tracks import Track, TrackError, identify_player


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class DetectConfig:
    """Thresholds for every detector. All times ms, distances blocks."""

    # --- deaths -----------------------------------------------------------
    death_dedupe_ms: int = 2000          # death events closer than this are one death
    respawn_teleport_blocks: float = 30.0  # jump this big between samples = teleport
    death_gap_ms: int = 1000             # sample gap that, with a teleport, implies a respawn
    death_fall_blocks: float = 20.0      # drop just before a teleport that implies a fall death
    death_fall_lookback_ms: int = 3000
    death_inferred_confidence: float = 0.7  # score of an inferred (not observed) death

    # --- kills ------------------------------------------------------------
    kill_streak_gap_ms: int = 8000       # kills this close chain into one streak
    kill_base_score: float = 0.4
    kill_streak_bonus: float = 0.2       # added per extra kill in the streak

    # --- rapid movement ---------------------------------------------------
    speed_step_ms: int = 250             # resampling step (Track.at)
    speed_window_ms: int = 2000          # sliding window for sustained speed
    fast_bps: float = 10.0               # horizontal blocks/s; sprint-jump is ~7
    elytra_bps: float = 25.0             # at/above this the label says elytra-like
    teleport_blocks: float = 40.0        # one raw sample step this big is a teleport, not motion
    fall_min_vspeed: float = 6.0         # blocks/s downward for a step to count as falling
    fall_min_drop: float = 12.0          # total drop for a "fall" highlight
    fall_full_drop: float = 60.0         # drop that earns the max fall score

    # --- builds -----------------------------------------------------------
    build_gap_ms: int = 5000             # placements further apart end a cluster
    build_radius: float = 24.0           # placements must stay this near the cluster centre
    build_min_blocks: int = 12           # fewest placements for a build
    build_full_blocks: int = 100         # placements that earn the max build score
    # track-only fallback (heuristic):
    build_window_ms: int = 20000
    build_fallback_radius: float = 8.0   # player stays within this of its window centre
    build_min_activity_per_s: float = 1.0  # samples/s with any small change
    build_min_look_deg_per_s: float = 10.0  # summed |dyaw|+|dpitch| per second

    # --- assembly (detect_all) ---------------------------------------------
    pad_ms: dict = field(default_factory=lambda: {
        # kind: (pre-roll, post-roll)
        "death": (4000, 3000),
        "kill": (3000, 2000),
        "speed": (1000, 1000),
        "fall": (1500, 2000),
        "build": (2000, 2000),
    })
    # Multiplier per kind so a mediocre build does not outrank a death.
    kind_weight: dict = field(default_factory=lambda: {
        "death": 1.0, "kill": 1.0, "fall": 0.8, "speed": 0.7, "build": 0.6,
    })


DEFAULT_CONFIG = DetectConfig()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return 0.0 if v <= 0 or v != v else (1.0 if v >= 1 else float(v))


def _fmt(ms: float) -> str:
    s = max(0, int(ms // 1000))
    return f"{s // 60}:{s % 60:02d}"


def _clean(track: Track | None) -> Track | None:
    """Copy of `track` with non-finite samples dropped, 6-tuples, time-sorted.

    Garbage in a track (NaN from a bad packet) would otherwise leak into
    scores and JSON. Returns None if fewer than two usable samples remain,
    since nothing can be measured from one point.
    """
    if track is None:
        return None
    pts = []
    for p in track.points:
        try:
            vals = [float(v) for v in tuple(p)[:6]]
        except (TypeError, ValueError):
            continue
        if len(vals) < 4 or not all(math.isfinite(v) for v in vals):
            continue
        vals += [0.0] * (6 - len(vals))
        pts.append(tuple(vals))
    pts.sort(key=lambda p: p[0])
    if len(pts) < 2:
        return None
    return Track(track.entity_id, pts)


def _dist(a, b) -> float:
    return math.dist(a[1:4], b[1:4])


def _dist_xz(a, b) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])  # a, b are (x, y, z) positions


def _angle_delta(a: float, b: float) -> float:
    return abs((b - a + 180.0) % 360.0 - 180.0)


def _as_events(events: Iterable[Event] | None) -> list[Event]:
    return sorted(events or (), key=lambda e: e.t)


def _teleport_intervals(tr: Track, cfg: DetectConfig) -> list[tuple[float, float]]:
    """Time spans between consecutive raw samples that jump implausibly far.

    Interpolating across such a jump with Track.at would fake a burst of
    speed, so movement detectors exclude these spans. Relies on tracks being
    dense (a sample every tick or so), as DumpTracks produces.
    """
    out = []
    for a, b in zip(tr.points, tr.points[1:]):
        if _dist(a, b) >= cfg.teleport_blocks:
            out.append((a[0], b[0]))
    return out


def _grid(tr: Track, cfg: DetectConfig):
    """Resample onto a fixed grid; returns (times, positions, bad_step flags)."""
    step = max(1, int(cfg.speed_step_ms))
    t0, t1 = tr.start, tr.end
    n = int((t1 - t0) // step) + 1
    times = [t0 + i * step for i in range(n)]
    pos = [tr.at(t) for t in times]
    bad = [False] * max(0, n - 1)
    for ta, tb in _teleport_intervals(tr, cfg):
        lo = int((ta - t0) // step)
        hi = max(lo, int(math.ceil((tb - t0) / step)) - 1)
        for k in range(max(0, lo), min(len(bad) - 1, hi) + 1):
            bad[k] = True
    return times, pos, bad, step


def _runs(indices: Sequence[int], max_gap: int = 1) -> list[tuple[int, int]]:
    """Group sorted ints into (first, last) runs where neighbours differ by <= max_gap."""
    runs: list[tuple[int, int]] = []
    for i in indices:
        if runs and i - runs[-1][1] <= max_gap:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


# --------------------------------------------------------------------------
# deaths
# --------------------------------------------------------------------------

def detect_deaths(tracks: Mapping[int, Track], player_id: int | None,
                  events: Iterable[Event] = (), cfg: DetectConfig | None = None
                  ) -> list[Highlight]:
    """Deaths of the recording player.

    OBSERVED: ``death`` events whose entity is the player (near-duplicates
    within ``death_dedupe_ms`` collapse). If at least one exists, only those
    are used -- an event feed is authoritative, and mixing in heuristics would
    double-count.

    INFERRED (only when the feed has no player death): in the player's track,
    a sample-to-sample jump of >= ``respawn_teleport_blocks`` is a respawn
    candidate when either (a) it follows a time gap >= ``death_gap_ms``
    (loading/respawn screen) or (b) the player had just dropped >=
    ``death_fall_blocks`` (fall death, no gap). Plain teleports with neither
    are ignored. A lag spike plus ``/tp`` is indistinguishable from (a).
    The death is placed at the last sample before the jump.
    """
    cfg = cfg or DEFAULT_CONFIG
    if player_id is None:
        return []
    evs = _as_events(events)
    observed = [e for e in evs if e.kind == "death" and e.entity == player_id]
    tr = _clean(tracks.get(player_id)) if tracks else None

    out: list[Highlight] = []
    if observed:
        last_t = None
        for e in observed:
            if last_t is not None and e.t - last_t < cfg.death_dedupe_ms:
                continue
            last_t = e.t
            pos = tuple(e.pos) if e.pos else (tuple(tr.at(e.t)) if tr else None)
            out.append(Highlight("death", int(e.t), int(e.t) + 1, 1.0,
                                 entity=player_id, pos=pos,
                                 label=f"Death at {_fmt(e.t)}",
                                 meta={"source": "event"}))
        return out

    if tr is None:
        return []
    pts = tr.points
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if _dist(a, b) < cfg.respawn_teleport_blocks:
            continue
        gap = (b[0] - a[0]) >= cfg.death_gap_ms
        peak = a[2]
        j = i - 1
        while j >= 0 and pts[j][0] >= a[0] - cfg.death_fall_lookback_ms:
            peak = max(peak, pts[j][2])
            j -= 1
        fall = (peak - a[2]) >= cfg.death_fall_blocks
        if not (gap or fall):
            continue
        score = cfg.death_inferred_confidence + (0.1 if gap and fall else 0.0)
        t = int(a[0])
        out.append(Highlight("death", t, t + 1, _clamp01(score),
                             entity=player_id, pos=tuple(a[1:4]),
                             label=f"Death at {_fmt(t)} (inferred)",
                             meta={"source": "inferred", "gap": gap, "fall": fall}))
    return out


# --------------------------------------------------------------------------
# kills
# --------------------------------------------------------------------------

_STREAK_NAMES = {1: "Kill", 2: "Double kill", 3: "Triple kill", 4: "Quad kill"}


def detect_kills(events: Iterable[Event], player_id: int | None,
                 cfg: DetectConfig | None = None) -> list[Highlight]:
    """Kills by the player, chained into streaks (all OBSERVED).

    Kills closer than ``kill_streak_gap_ms`` to the previous one belong to the
    same streak; one highlight per streak, scored
    ``base + bonus * (n - 1)`` so a triple outranks three separate kills.
    """
    cfg = cfg or DEFAULT_CONFIG
    if player_id is None:
        return []
    kills = [e for e in _as_events(events) if e.kind == "kill" and e.entity == player_id]
    groups: list[list[Event]] = []
    for e in kills:
        if groups and e.t - groups[-1][-1].t <= cfg.kill_streak_gap_ms:
            groups[-1].append(e)
        else:
            groups.append([e])
    out = []
    for g in groups:
        n = len(g)
        pos = next((tuple(e.pos) for e in reversed(g) if e.pos), None)
        out.append(Highlight(
            "kill", int(g[0].t), int(g[-1].t) + 1,
            _clamp01(cfg.kill_base_score + cfg.kill_streak_bonus * (n - 1)),
            entity=player_id, pos=pos,
            label=_STREAK_NAMES.get(n, f"{n}-kill streak"),
            meta={"source": "event", "kills": n,
                  "targets": [e.target for e in g if e.target is not None]}))
    return out


# --------------------------------------------------------------------------
# rapid movement and falls
# --------------------------------------------------------------------------

def detect_rapid_movement(track: Track, cfg: DetectConfig | None = None
                          ) -> list[Highlight]:
    """Sustained fast travel (kind "speed") and long falls (kind "fall").

    MEASURED from positions. The track is resampled with ``Track.at`` onto a
    fixed grid; horizontal speed is net displacement over a sliding
    ``speed_window_ms`` window, so a single-tick spike or a teleport cannot
    qualify -- only travel sustained for the whole window. Overlapping active
    windows merge into one span. Peaks at/above ``elytra_bps`` are labelled
    elytra-like (``meta["elytra_like"]``); we cannot actually tell elytra from
    an ice boat from tracks alone.

    Falls: runs of grid steps descending faster than ``fall_min_vspeed``
    (one non-falling step is bridged so a bounce does not split a fall)
    whose total drop is >= ``fall_min_drop``.
    """
    cfg = cfg or DEFAULT_CONFIG
    tr = _clean(track)
    if tr is None:
        return []
    times, pos, bad, step = _grid(tr, cfg)
    n = len(times)
    out: list[Highlight] = []
    eid = tr.entity_id

    # --- speed -----------------------------------------------------------
    w = max(1, int(round(cfg.speed_window_ms / step)))
    if n > w:
        badpre = [0]
        for b in bad:
            badpre.append(badpre[-1] + (1 if b else 0))
        active, speeds = [], {}
        secs = w * step / 1000.0
        for i in range(n - w):
            if badpre[i + w] - badpre[i]:
                continue
            v = _dist_xz(pos[i], pos[i + w]) / secs
            if v >= cfg.fast_bps:
                active.append(i)
                speeds[i] = v
        for first, last in _runs(active, 1):
            vs = [speeds[i] for i in range(first, last + 1) if i in speeds]
            mean, peak = sum(vs) / len(vs), max(vs)
            elytra = peak >= cfg.elytra_bps
            score = 0.4 + 0.6 * _clamp01((mean - cfg.fast_bps) / (3 * cfg.fast_bps))
            s, e = times[first], times[last + w]
            out.append(Highlight(
                "speed", int(s), int(e), _clamp01(score), entity=eid,
                pos=tuple(pos[last + w]),
                label=(f"Elytra-speed flight ({peak:.0f} b/s)" if elytra
                       else f"Fast travel ({peak:.0f} b/s)"),
                meta={"source": "measured", "mean_bps": round(mean, 2),
                      "peak_bps": round(peak, 2), "elytra_like": elytra}))

    # --- falls -----------------------------------------------------------
    dt_s = step / 1000.0
    falling = [k for k in range(n - 1)
               if not bad[k] and (pos[k][1] - pos[k + 1][1]) / dt_s >= cfg.fall_min_vspeed]
    for first, last in _runs(falling, 2):
        drop = pos[first][1] - pos[last + 1][1]
        if drop < cfg.fall_min_drop:
            continue
        score = 0.4 + 0.6 * _clamp01(drop / cfg.fall_full_drop)
        out.append(Highlight(
            "fall", int(times[first]), int(times[last + 1]), _clamp01(score),
            entity=eid, pos=tuple(pos[last + 1]),
            label=f"Fall of {drop:.0f} blocks",
            meta={"source": "measured", "drop": round(drop, 2)}))
    out.sort(key=lambda h: (h.start_ms, h.end_ms, h.kind))
    return out


# --------------------------------------------------------------------------
# builds
# --------------------------------------------------------------------------

def detect_builds(track: Track, events: Iterable[Event] = (),
                  cfg: DetectConfig | None = None) -> list[Highlight]:
    """Building sessions (kind "build").

    OBSERVED path: ``block_place`` events by this entity are clustered -- a
    placement joins the current cluster if it is within ``build_gap_ms`` of the
    previous one and within ``build_radius`` of the cluster centre (events with
    no position are time-only). Clusters with >= ``build_min_blocks``
    placements become highlights. If the player has any block_place events,
    only this path runs.

    HEURISTIC fallback (no block_place events at all): over sliding
    ``build_window_ms`` windows, flag stretches where the player stays within
    ``build_fallback_radius`` of the window centre while producing lots of
    small position/yaw/pitch changes. This is a guess at "standing still and
    working"; it also matches fishing, farming or fidgeting at a menu, and it
    misses builds done while walking around a large structure. Treat it as a
    suggestion list, not ground truth.
    """
    cfg = cfg or DEFAULT_CONFIG
    tr = _clean(track)
    eid = track.entity_id if track is not None else None
    places = [e for e in _as_events(events)
              if e.kind == "block_place" and (e.entity is None or e.entity == eid)]
    if places:
        return _builds_from_events(places, eid, cfg)
    if tr is None:
        return []
    return _builds_from_track(tr, cfg)


def _builds_from_events(places: list[Event], eid, cfg: DetectConfig) -> list[Highlight]:
    clusters: list[dict] = []
    cur: dict | None = None
    for e in places:
        joins = False
        if cur is not None and e.t - cur["evs"][-1].t <= cfg.build_gap_ms:
            if e.pos is None or cur["n"] == 0:
                joins = True
            else:
                c = [s / cur["n"] for s in cur["sum"]]
                joins = math.dist(c, e.pos[:3]) <= cfg.build_radius
        if not joins:
            cur = {"evs": [], "sum": [0.0, 0.0, 0.0], "n": 0}
            clusters.append(cur)
        cur["evs"].append(e)
        if e.pos is not None:
            for k in range(3):
                cur["sum"][k] += float(e.pos[k])
            cur["n"] += 1
    out = []
    for c in clusters:
        evs = c["evs"]
        if len(evs) < cfg.build_min_blocks:
            continue
        centre = tuple(s / c["n"] for s in c["sum"]) if c["n"] else None
        score = 0.4 + 0.6 * _clamp01(len(evs) / cfg.build_full_blocks)
        out.append(Highlight(
            "build", int(evs[0].t), int(evs[-1].t) + 1, _clamp01(score),
            entity=eid, pos=centre, label=f"Build ({len(evs)} blocks)",
            meta={"source": "event", "blocks": len(evs)}))
    return out


def _builds_from_track(tr: Track, cfg: DetectConfig) -> list[Highlight]:
    pts = tr.points
    times = [p[0] for p in pts]
    # Per-segment activity and look-change, as prefix sums for O(1) windows.
    act = [0]
    look = [0.0]
    for a, b in zip(pts, pts[1:]):
        dl = _angle_delta(a[4], b[4]) + _angle_delta(a[5], b[5])
        moved = _dist(a, b) > 0.02 or dl > 0.5
        act.append(act[-1] + (1 if moved else 0))
        look.append(look[-1] + dl)
    win = max(1, int(cfg.build_window_ms))
    hop = max(1, win // 4)
    secs = win / 1000.0
    starts, rates = [], {}
    ws = tr.start
    while ws + win <= tr.end:
        lo = bisect.bisect_left(times, ws)
        hi = bisect.bisect_right(times, ws + win) - 1
        if hi - lo >= 2:
            seg_n = act[hi] - act[lo]
            seg_look = look[hi] - look[lo]
            # Activity must be spread across the whole window (every quarter),
            # not one burst that happens to average out above the threshold.
            bins_ok = True
            for k in range(4):
                blo = bisect.bisect_left(times, ws + k * hop)
                bhi = bisect.bisect_right(times, ws + (k + 1) * hop) - 1
                if (act[max(blo, bhi)] - act[blo]) / (hop / 1000.0) < cfg.build_min_activity_per_s:
                    bins_ok = False
                    break
            if (bins_ok and seg_n / secs >= cfg.build_min_activity_per_s
                    and seg_look / secs >= cfg.build_min_look_deg_per_s):
                chunk = pts[lo:hi + 1]
                cx = sum(p[1] for p in chunk) / len(chunk)
                cy = sum(p[2] for p in chunk) / len(chunk)
                cz = sum(p[3] for p in chunk) / len(chunk)
                if max(math.dist((p[1], p[2], p[3]), (cx, cy, cz)) for p in chunk) \
                        <= cfg.build_fallback_radius:
                    idx = int((ws - tr.start) // hop)
                    starts.append(idx)
                    rates[idx] = (seg_n / secs, (cx, cy, cz))
        ws += hop
    out = []
    for first, last in _runs(starts, 1):
        s = tr.start + first * hop
        e = tr.start + last * hop + win
        rs = [rates[i][0] for i in range(first, last + 1) if i in rates]
        rate = sum(rs) / len(rs)
        mid = first + (last - first) // 2
        centre = rates[mid][1] if mid in rates else rates[first][1]
        score = (0.3 + 0.35 * _clamp01((e - s) / 60000.0)
                 + 0.35 * _clamp01(rate / (4 * cfg.build_min_activity_per_s)))
        out.append(Highlight(
            "build", int(s), int(e), _clamp01(score), entity=tr.entity_id,
            pos=centre, label=f"Possible build ({(e - s) / 1000:.0f}s)",
            meta={"source": "heuristic", "activity_per_s": round(rate, 2)}))
    return out


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def _merge_same_kind(hs: list[Highlight]) -> list[Highlight]:
    """Union overlapping/touching highlights of the same kind and entity.

    The merged score is the max (a long stretch is not better than its best
    moment), and label/pos/meta come from that best member.
    """
    groups: dict[tuple, list[Highlight]] = {}
    for h in hs:
        groups.setdefault((h.kind, h.entity), []).append(h)
    out: list[Highlight] = []
    for key in sorted(groups, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
        cur: list[Highlight] = []
        cur_end = None
        batches: list[list[Highlight]] = []
        for h in sorted(groups[key], key=lambda h: (h.start_ms, h.end_ms)):
            if cur and h.start_ms <= cur_end:
                cur.append(h)
                cur_end = max(cur_end, h.end_ms)
            else:
                cur = [h]
                cur_end = h.end_ms
                batches.append(cur)
        for b in batches:
            if len(b) == 1:
                out.append(b[0])
                continue
            best = max(b, key=lambda h: (h.score, -h.start_ms))
            meta = dict(best.meta)
            meta["merged"] = len(b)
            out.append(Highlight(best.kind, min(h.start_ms for h in b),
                                 max(h.end_ms for h in b), best.score,
                                 entity=best.entity, pos=best.pos,
                                 label=best.label, meta=meta))
    return out


def detect_all(tracks: Mapping[int, Track] | Iterable[Track],
               events: Iterable[Event] = (), player_id: int | None = None,
               cfg: DetectConfig | None = None) -> list[Highlight]:
    """Run every detector and assemble one chronological highlight list.

    Steps: auto-pick the player (``identify_player``) if not given; run the
    detectors; pad each span with its kind's pre/post roll (so the viewer sees
    the lead-up and aftermath, not just the instant); clamp to the replay's
    time range; merge overlaps within a kind; apply the per-kind weight and
    clamp scores to 0..1 (rounded to 4 places, matching JSON storage). Output
    is sorted by (start, end, kind) and is deterministic.
    """
    cfg = cfg or DEFAULT_CONFIG
    if not isinstance(tracks, Mapping):
        tracks = {t.entity_id: t for t in tracks}
    tracks = {k: t for k, t in tracks.items() if _clean(t) is not None}
    evs = _as_events(events)

    if player_id is None and tracks:
        try:
            player_id = identify_player(tracks).entity_id
        except TrackError:
            player_id = None

    raw: list[Highlight] = []
    raw += detect_deaths(tracks, player_id, evs, cfg)
    raw += detect_kills(evs, player_id, cfg)
    ptrack = tracks.get(player_id) if player_id is not None else None
    if ptrack is not None:
        raw += detect_rapid_movement(ptrack, cfg)
        raw += detect_builds(ptrack, evs, cfg)

    if tracks:
        lo = min(_clean(t).start for t in tracks.values())
        hi = max(_clean(t).end for t in tracks.values())
    else:
        lo, hi = 0, math.inf

    padded: list[Highlight] = []
    for h in raw:
        pre, post = cfg.pad_ms.get(h.kind, (0, 0))
        s = max(lo, h.start_ms - pre)
        e = min(hi, h.end_ms + post)
        if e <= s:
            continue
        padded.append(Highlight(h.kind, int(s), int(e), h.score, h.entity,
                                h.pos, h.label, h.meta))

    merged = _merge_same_kind(padded)
    final = []
    for h in merged:
        w = cfg.kind_weight.get(h.kind, 1.0)
        score = round(_clamp01(h.score * w), 4)
        final.append(Highlight(h.kind, h.start_ms, h.end_ms, score, h.entity,
                               h.pos, h.label, h.meta))
    final.sort(key=lambda h: (h.start_ms, h.end_ms, h.kind))
    return final


def select(highlights: Iterable[Highlight], target_ms: int,
           min_gap_ms: int = 2000, max_count: int | None = None) -> list[Highlight]:
    """Pick the best-scoring non-overlapping subset within a duration budget.

    Greedy by score (ties: earlier start, then shorter): a highlight is taken
    if it fits the remaining budget, keeps ``min_gap_ms`` clear of every
    already-chosen one (gap 0 still forbids overlap), and ``max_count`` is not
    reached. Greedy is not guaranteed optimal for the budgeted-interval
    problem, but it is predictable and always keeps the top-scoring moment
    that fits, which is what an editor expects. A highlight longer than the
    whole budget is skipped, never truncated. Returned chronologically.
    """
    if target_ms <= 0 or (max_count is not None and max_count <= 0):
        return []
    gap = max(0, int(min_gap_ms))
    order = sorted(highlights, key=lambda h: (-h.score, h.start_ms, h.duration_ms, h.kind))
    chosen: list[Highlight] = []
    used = 0
    for h in order:
        if max_count is not None and len(chosen) >= max_count:
            break
        if used + h.duration_ms > target_ms:
            continue
        if any(h.start_ms < c.end_ms + gap and c.start_ms < h.end_ms + gap
               for c in chosen):
            continue
        chosen.append(h)
        used += h.duration_ms
    chosen.sort(key=lambda h: (h.start_ms, h.end_ms, h.kind))
    return chosen


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save_highlights(path, hs: Iterable[Highlight]) -> None:
    """Write highlights as ``{"version": 1, "highlights": [...]}`` JSON."""
    Path(path).write_text(
        json.dumps({"version": 1, "highlights": [h.to_json() for h in hs]}, indent=1),
        encoding="utf-8")


def load_highlights(path) -> list[Highlight]:
    """Read highlights saved by :func:`save_highlights` (a bare list also works).

    Raises FileNotFoundError for a missing file and EventError for malformed
    entries -- unlike events, a missing highlight file is a real mistake.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("highlights", []) if isinstance(data, dict) else data
    return [Highlight.from_json(d) for d in raw]
