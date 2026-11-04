"""Camera-path planner: richer renderQueue keyframes from a Highlight + Track.

ReplayMod only interpolates linearly, with a cubic spline, or with a
catmull-rom spline; it has no easing curves. Easing therefore has to be *baked*
into the keyframes: we sample the shot at evenly spaced video times but move the
camera along its route by ``easing(f)`` rather than ``f``, and let the
interpolator join the dots. More keyframes = a more faithful easing curve.

Layers, bottom to top:

    easing functions (EASINGS / get_easing)
    shot builders    orbit_shot, dolly_shot, crane_shot, flyby_shot,
                     static_shot / hold, follow_shot      -> timeline.Path
    speed_ramp       baked TIME keyframes for slow-motion  -> timeline.Path
    choose_shot      Highlight kind -> deterministic ShotRecipe
    plan_highlight   Highlight + Track -> (time Path, position Path)
    plan_montage     many Highlights -> render jobs for replay.write_render_queue

All shot builders funnel through ``_finalise`` which enforces the safety rules
once, in one place: camera never below the world floor, never closer than
``min_distance`` to its target, pitch clamped to +/-90, strictly increasing
integer keyframe times, and finite plain-float output (JSON-serialisable).
The final yaw/pitch always come from ``timeline.look_at`` on the *adjusted*
eye, so rotation stays consistent with what the camera actually sees.
"""

from __future__ import annotations

import bisect
import math
import numbers
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path as FsPath
from typing import Callable, Mapping, Sequence

from . import timeline, tracks
from .events import Highlight
from .timeline import PROP_POSITION, PROP_ROTATION, Path

Vec3 = Sequence[float]


class CamPathError(ValueError):
    """Bad input to the camera planner."""


# --------------------------------------------------------------------------- easing

def _clamp01(t: float) -> float:
    return 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else t


def _endpoints_exact(fn: Callable[[float], float]) -> Callable[[float], float]:
    """Clamp input and pin f(0)=0, f(1)=1 exactly (float error would otherwise
    leave a camera a hair off its start/end mark)."""

    def wrapped(t: float) -> float:
        t = _clamp01(float(t))
        if t == 0.0:
            return 0.0
        if t == 1.0:
            return 1.0
        return fn(t)

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def _in_out(ease_in: Callable[[float], float]) -> Callable[[float], float]:
    """Build a symmetric in-out from an ease-in (power-law style)."""

    def f(t: float) -> float:
        return 0.5 * ease_in(2 * t) if t < 0.5 else 1 - 0.5 * ease_in(2 - 2 * t)

    return f


_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1


def _out_back(t: float) -> float:
    return 1 + _C3 * (t - 1) ** 3 + _C1 * (t - 1) ** 2


def _in_back(t: float) -> float:
    return _C3 * t ** 3 - _C1 * t ** 2


def _in_out_back(t: float) -> float:
    if t < 0.5:
        return ((2 * t) ** 2 * ((_C2 + 1) * 2 * t - _C2)) / 2
    return ((2 * t - 2) ** 2 * ((_C2 + 1) * (2 * t - 2) + _C2) + 2) / 2


def _linear(t): return t
def _in_quad(t): return t * t
def _in_cubic(t): return t ** 3
def _in_quart(t): return t ** 4
def _out_of(ein): return lambda t: 1 - ein(1 - t)
def _in_sine(t): return 1 - math.cos(t * math.pi / 2)
def _out_sine(t): return math.sin(t * math.pi / 2)
def _in_out_sine(t): return -(math.cos(math.pi * t) - 1) / 2
def _smoothstep(t): return t * t * (3 - 2 * t)
def _smootherstep(t): return t * t * t * (t * (6 * t - 15) + 10)


_RAW: dict[str, Callable[[float], float]] = {
    "linear": _linear,
    "ease_in_quad": _in_quad,
    "ease_out_quad": _out_of(_in_quad),
    "ease_in_out_quad": _in_out(_in_quad),
    "ease_in_cubic": _in_cubic,
    "ease_out_cubic": _out_of(_in_cubic),
    "ease_in_out_cubic": _in_out(_in_cubic),
    "ease_in_quart": _in_quart,
    "ease_out_quart": _out_of(_in_quart),
    "ease_in_out_quart": _in_out(_in_quart),
    "ease_in_sine": _in_sine,
    "ease_out_sine": _out_sine,
    "ease_in_out_sine": _in_out_sine,
    "smoothstep": _smoothstep,
    "smootherstep": _smootherstep,
    "ease_in_back": _in_back,
    "ease_out_back": _out_back,
    "ease_in_out_back": _in_out_back,
}
# Unqualified names default to the quad family (the gentlest), cubic for in_out
# is intentionally NOT used so that "ease_in/out/in_out" are one coherent family.
_ALIASES = {
    "ease_in": "ease_in_quad",
    "ease_out": "ease_out_quad",
    "ease_in_out": "ease_in_out_quad",
}

EASINGS: dict[str, Callable[[float], float]] = {
    **{k: _endpoints_exact(v) for k, v in _RAW.items()},
}
for _alias, _target in _ALIASES.items():
    EASINGS[_alias] = EASINGS[_target]

#: Easings that overshoot [0, 1] (not monotonic); callers may want to avoid them
#: for time mapping, where overshoot would play the replay backwards.
OVERSHOOTING = frozenset({"ease_in_back", "ease_out_back", "ease_in_out_back"})


def get_easing(name: str) -> Callable[[float], float]:
    """Look up an easing by name; ValueError (listing choices) if unknown."""
    try:
        return EASINGS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown easing {name!r}; choose from {sorted(EASINGS)}"
        ) from None


def _resolve_easing(e) -> Callable[[float], float]:
    return e if callable(e) else get_easing(e)


# --------------------------------------------------------------------------- safety

@dataclass(frozen=True)
class Safety:
    """Camera sanity limits applied to every shot.

    min_y: lowest allowed eye y (1.18+ worlds floor at -64; going lower shows
           the void / clips into bedrock-less nothing).
    min_distance: eyes closer than this to the target are pushed outward, since
           the near-plane clips the subject and pitch swings wildly.
    max_y: optional ceiling (build limit is 320).
    """

    min_y: float = -64.0
    min_distance: float = 2.0
    max_y: float | None = None

    def __post_init__(self):
        if not (math.isfinite(self.min_y) and math.isfinite(self.min_distance)
                and self.min_distance >= 0):
            raise CamPathError("Safety needs finite min_y and min_distance >= 0")
        if self.max_y is not None and not (math.isfinite(self.max_y)
                                           and self.max_y >= self.min_y):
            raise CamPathError("Safety.max_y must be finite and >= min_y")


def clamp_pitch(pitch: float) -> float:
    """Pitch limited to what a Minecraft camera can show."""
    return max(-90.0, min(90.0, pitch))


def _finite_vec(v, what: str) -> tuple[float, float, float]:
    try:
        out = tuple(float(x) for x in v)
    except (TypeError, ValueError):
        raise CamPathError(f"{what} must be 3 numbers, got {v!r}") from None
    if len(out) != 3 or not all(math.isfinite(x) for x in out):
        raise CamPathError(f"{what} must be 3 finite numbers, got {v!r}")
    return out  # type: ignore[return-value]


def _finite(x, what: str) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise CamPathError(f"{what} must be a number, got {x!r}") from None
    if not math.isfinite(f):
        raise CamPathError(f"{what} must be finite, got {x!r}")
    return f


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _check_common(duration_ms, keyframes) -> int:
    d = _finite(duration_ms, "duration_ms")
    if d < 1:
        raise CamPathError("duration_ms must be >= 1")
    if not _is_int(keyframes) or keyframes < 2:
        raise CamPathError("a shot needs at least 2 integer keyframes")
    return int(round(d))


def _safe_eye(eye, target, safety: Safety):
    """Clamp the eye into [min_y, max_y], then push it out to min_distance.

    The push is radial (keeps the framing). If that would leave the allowed
    height band, the eye is pinned to the band edge and slid horizontally
    instead, which always reaches min_distance (a purely horizontal move of
    ``sqrt(d^2 - dy^2)`` suffices, and is 0 when |dy| alone is enough).
    """
    x, y, z = eye
    md = safety.min_distance

    def clamp_y(v):
        v = max(safety.min_y, v)
        return v if safety.max_y is None else min(safety.max_y, v)

    y = clamp_y(y)
    dx, dy, dz = x - target[0], y - target[1], z - target[2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d >= md - 1e-9:
        return (x, y, z)
    if d < 1e-9:
        dx, dy, dz, d = 1.0, 0.0, 0.0, 1.0  # arbitrary but deterministic
    s = md / d
    ny = target[1] + dy * s
    if clamp_y(ny) == ny:
        return (target[0] + dx * s, ny, target[2] + dz * s)
    y = clamp_y(ny)
    dy = y - target[1]
    h = math.sqrt(max(0.0, md * md - dy * dy))
    hd = math.hypot(dx, dz)
    ux, uz = (dx / hd, dz / hd) if hd > 1e-9 else (1.0, 0.0)
    return (target[0] + ux * h, y, target[2] + uz * h)


def _finalise(eyes, times, targets, interpolator, safety: Safety | None) -> Path:
    """Shared tail of every shot builder: dedupe times, apply safety, emit Path."""
    safety = safety or Safety()
    n = len(eyes)
    if not (len(times) == n == len(targets)):
        raise CamPathError("eyes/times/targets length mismatch")
    kept: list[int] = []
    last_t: int | None = None
    for i, t in enumerate(times):
        t = int(round(t))
        if last_t is not None and t <= last_t:
            if i != n - 1:
                continue
            # The final keyframe defines the shot length; displace earlier ones.
            while kept and int(round(times[kept[-1]])) >= t:
                kept.pop()
        kept.append(i)
        last_t = t
    if len(kept) < 2:
        raise CamPathError("shot collapsed to fewer than 2 distinct keyframe times")

    k_times = [int(round(times[i])) for i in kept]
    k_targets = [_finite_vec(targets[i], "target") for i in kept]
    k_eyes = [
        _safe_eye(_finite_vec(eyes[i], "eye"), k_targets[j], safety)
        for j, i in enumerate(kept)
    ]
    interp = timeline.catmull_rom() if interpolator is None else interpolator
    path = timeline._camera_path(k_eyes, k_times, k_targets, interp)
    _fix_degenerate_yaw(path, k_eyes, k_targets)
    for kf in path.keyframes:
        props = kf["properties"]
        rot = props[PROP_ROTATION]
        rot[1] = clamp_pitch(rot[1])
        props[PROP_ROTATION] = [v + 0.0 for v in rot]       # -0.0 -> 0.0
        props[PROP_POSITION] = [v + 0.0 for v in props[PROP_POSITION]]
    return path


def _fix_degenerate_yaw(path: Path, eyes, targets) -> None:
    """Hold yaw steady where the eye is (almost) straight above/below the target.

    There ``atan2`` has no meaningful answer (it returns 0/-0/180 arbitrarily),
    so neighbouring keyframes would spin the camera a half turn for nothing.
    Degenerate keyframes borrow the nearest well-defined yaw, then the whole
    sequence is re-unwrapped.
    """
    deg = [math.hypot(e[0] - t[0], e[2] - t[2]) < 1e-6 for e, t in zip(eyes, targets)]
    if not any(deg) or all(deg):
        return
    yaws = [kf["properties"][PROP_ROTATION][0] for kf in path.keyframes]
    n = len(yaws)
    filled = list(yaws)
    last = None
    for i in range(n):            # forward fill
        if deg[i]:
            filled[i] = last
        else:
            last = yaws[i]
    nxt = None
    for i in reversed(range(n)):  # leading degenerates take the next good yaw
        if not deg[i]:
            nxt = yaws[i]
        elif filled[i] is None:
            filled[i] = nxt
    wrapped = [((y + 180.0) % 360.0) - 180.0 for y in filled]
    for kf, y in zip(path.keyframes, timeline._unwrap(wrapped)):
        kf["properties"][PROP_ROTATION][0] = round(y, 3)


def _sample_fracs(n: int) -> list[float]:
    return [i / (n - 1) for i in range(n)]


# --------------------------------------------------------------------------- shots

def orbit_shot(
    centre: Vec3,
    duration_ms: int,
    *,
    radius: float = 10.0,
    end_radius: float | None = None,
    height: float = 4.0,
    end_height: float | None = None,
    revolutions: float = 0.5,
    start_angle_deg: float = 0.0,
    easing="ease_in_out_sine",
    radius_easing=None,
    keyframes: int = 24,
    interpolator=None,
    safety: Safety | None = None,
    target_fn: Callable[[float], Vec3] | None = None,
) -> Path:
    """Orbit with eased angular velocity and optionally eased radius/height.

    The angle advances by ``easing(f)`` so the camera accelerates into and out
    of the swing; ``end_radius`` / ``end_height`` give spirals (e.g. a spiral-in
    that tightens onto the subject). ``target_fn(f)`` (f in 0..1 of the shot)
    replaces the fixed ``centre`` for orbits around a moving subject; ``height``
    is relative to the target's y. Negative ``revolutions`` orbit the other way.
    """
    dur = _check_common(duration_ms, keyframes)
    centre = _finite_vec(centre, "centre")
    ease = _resolve_easing(easing)
    r_ease = _resolve_easing(radius_easing) if radius_easing is not None else ease
    r0 = _finite(radius, "radius")
    r1 = r0 if end_radius is None else _finite(end_radius, "end_radius")
    h0 = _finite(height, "height")
    h1 = h0 if end_height is None else _finite(end_height, "end_height")
    rev = _finite(revolutions, "revolutions")
    a0 = _finite(start_angle_deg, "start_angle_deg")

    eyes, times, targets = [], [], []
    for f in _sample_fracs(keyframes):
        p = ease(f)
        q = r_ease(f)
        tgt = centre if target_fn is None else _finite_vec(target_fn(f), "target")
        ang = math.radians(a0 + 360.0 * rev * p)
        r = r0 + (r1 - r0) * q
        h = h0 + (h1 - h0) * q
        eyes.append((tgt[0] + r * math.cos(ang), tgt[1] + h, tgt[2] + r * math.sin(ang)))
        targets.append(tgt)
        times.append(dur * f)
    return _finalise(eyes, times, targets, interpolator, safety)


def dolly_shot(
    start: Vec3,
    end: Vec3,
    look: Vec3,
    duration_ms: int,
    *,
    easing="ease_in_out_cubic",
    arc: float = 0.0,
    keyframes: int = 12,
    interpolator=None,
    safety: Safety | None = None,
) -> Path:
    """Straight eased move start -> end facing ``look``.

    ``arc`` bows the path sideways (horizontal, perpendicular to the move) by up
    to that many blocks mid-move, turning a flat push-in into a gentle curve.
    """
    dur = _check_common(duration_ms, keyframes)
    s, e, lk = _finite_vec(start, "start"), _finite_vec(end, "end"), _finite_vec(look, "look")
    ease = _resolve_easing(easing)
    arc = _finite(arc, "arc")
    dx, dz = e[0] - s[0], e[2] - s[2]
    hl = math.hypot(dx, dz)
    nx, nz = (-dz / hl, dx / hl) if hl > 1e-9 else (1.0, 0.0)

    eyes, times = [], []
    for f in _sample_fracs(keyframes):
        p = ease(f)
        bow = arc * math.sin(math.pi * _clamp01(p))
        eyes.append(
            (s[0] + (e[0] - s[0]) * p + nx * bow,
             s[1] + (e[1] - s[1]) * p,
             s[2] + (e[2] - s[2]) * p + nz * bow)
        )
        times.append(dur * f)
    return _finalise(eyes, times, [lk] * keyframes, interpolator, safety)


def crane_shot(
    target: Vec3,
    duration_ms: int,
    *,
    radius: float = 12.0,
    start_height: float = 1.0,
    end_height: float = 18.0,
    angle_deg: float = 0.0,
    sweep_deg: float = 25.0,
    easing="ease_in_out_sine",
    keyframes: int = 16,
    interpolator=None,
    safety: Safety | None = None,
) -> Path:
    """Camera rises (or sinks) while staying locked onto ``target``.

    A small ``sweep_deg`` of horizontal drift alongside the rise avoids the flat
    "elevator" look of a purely vertical move.
    """
    dur = _check_common(duration_ms, keyframes)
    t = _finite_vec(target, "target")
    ease = _resolve_easing(easing)
    r = _finite(radius, "radius")
    h0, h1 = _finite(start_height, "start_height"), _finite(end_height, "end_height")
    a0, sw = _finite(angle_deg, "angle_deg"), _finite(sweep_deg, "sweep_deg")
    eyes, times = [], []
    for f in _sample_fracs(keyframes):
        p = ease(f)
        ang = math.radians(a0 + sw * p)
        eyes.append((t[0] + r * math.cos(ang), t[1] + h0 + (h1 - h0) * p,
                     t[2] + r * math.sin(ang)))
        times.append(dur * f)
    return _finalise(eyes, times, [t] * keyframes, interpolator, safety)


def flyby_shot(
    target: Vec3,
    duration_ms: int,
    *,
    pass_distance: float = 8.0,
    height: float = 3.0,
    approach: float = 30.0,
    direction_deg: float = 0.0,
    easing="linear",
    keyframes: int = 20,
    interpolator=None,
    safety: Safety | None = None,
) -> Path:
    """Camera sweeps along a line that passes the target at ``pass_distance``.

    The line runs ``approach`` blocks before the closest point to the same
    distance *past* it, with the lens tracking the target throughout, so the
    subject whips from ahead to behind. Linear easing by default: a flyby reads
    as speed, and the pass itself is the peak.
    """
    dur = _check_common(duration_ms, keyframes)
    t = _finite_vec(target, "target")
    ease = _resolve_easing(easing)
    pd = _finite(pass_distance, "pass_distance")
    h = _finite(height, "height")
    ap = _finite(approach, "approach")
    d = math.radians(_finite(direction_deg, "direction_deg"))
    ux, uz = math.cos(d), math.sin(d)       # travel direction
    nx, nz = -uz, ux                        # side the camera passes on
    eyes, times = [], []
    for f in _sample_fracs(keyframes):
        s = -ap + 2 * ap * ease(f)
        eyes.append((t[0] + ux * s + nx * pd, t[1] + h, t[2] + uz * s + nz * pd))
        times.append(dur * f)
    return _finalise(eyes, times, [t] * keyframes, interpolator, safety)


def static_shot(
    eye: Vec3,
    look: Vec3,
    duration_ms: int,
    *,
    keyframes: int = 2,
    interpolator=None,
    safety: Safety | None = None,
) -> Path:
    """Locked-off camera. Interpolator defaults to linear: nothing moves."""
    dur = _check_common(duration_ms, keyframes)
    e, lk = _finite_vec(eye, "eye"), _finite_vec(look, "look")
    interp = timeline.LINEAR if interpolator is None else interpolator
    return _finalise([e] * keyframes, [dur * f for f in _sample_fracs(keyframes)],
                     [lk] * keyframes, interp, safety)


hold = static_shot  # the name editors use for "leave the camera where it is"


def _window_track(track: tracks.Track, lo: float, hi: float) -> tracks.Track:
    """Finite, time-sorted copy of ``track`` trimmed to what shapes [lo, hi].

    Keeps ``max(SMOOTH_CANDIDATES)`` ms either side (the widest moving-average
    half-window is half of that) plus one extra sample each side so
    interpolation at the edges is unchanged.
    """
    pts = []
    for p in track.points:
        try:
            v = tuple(float(c) for c in p)
        except (TypeError, ValueError):
            continue
        if len(v) >= 4 and all(math.isfinite(c) for c in v):
            pts.append(v)
    if not pts:
        raise CamPathError("follow_shot track has no finite points")
    pts.sort(key=lambda p: p[0])
    margin = max(tracks.SMOOTH_CANDIDATES) + 1000
    ts = [p[0] for p in pts]
    a = max(0, bisect.bisect_left(ts, lo - margin) - 1)
    b = min(len(pts), bisect.bisect_right(ts, hi + margin) + 1)
    return tracks.Track(track.entity_id, pts[a:b])


def follow_shot(
    track: tracks.Track,
    replay_start_ms: float,
    replay_end_ms: float,
    duration_ms: int,
    *,
    radius: float = 12.0,
    height: float = 5.0,
    revolutions: float = 0.25,
    start_angle_deg: float = 0.0,
    eye_height: float = 1.6,
    easing="ease_in_out_sine",
    max_deviation_blocks: float = 6.0,
    keyframes: int = 48,
    interpolator=None,
    safety: Safety | None = None,
    time_map: Callable[[float], float] | None = None,
) -> Path:
    """Orbit a *moving* subject sampled from a Track.

    The smoothing window comes from ``tracks.choose_smoothing`` so the camera
    never loses the subject. ``time_map(video_ms) -> replay_ms`` must be passed
    when a speed ramp is in use, otherwise the camera would look where the
    player was at the *linear* time, not at the time the frame actually shows.
    """
    dur = _check_common(duration_ms, keyframes)
    if not getattr(track, "points", None):
        raise CamPathError("follow_shot needs a non-empty track")
    replay_start_ms = _finite(replay_start_ms, "replay_start_ms")
    replay_end_ms = _finite(replay_end_ms, "replay_end_ms")
    if replay_end_ms <= replay_start_ms:
        raise CamPathError("replay_end_ms must be after replay_start_ms")
    # Smoothing and Track.at are linear in the track length per call, so a
    # whole-replay track would make this quadratic. Only the shot's neighbourhood
    # (plus the widest smoothing half-window) can influence it.
    track = _window_track(track, replay_start_ms, replay_end_ms)
    window, _dev = tracks.choose_smoothing(
        track, replay_start_ms, replay_end_ms, keyframes, max_deviation_blocks)
    sm = track.smoothed(window_ms=window) if window else track
    eh = _finite(eye_height, "eye_height")
    span = replay_end_ms - replay_start_ms

    def target_fn(f: float) -> Vec3:
        ts = time_map(dur * f) if time_map else replay_start_ms + span * f
        x, y, z = sm.at(ts)
        return (x, y + eh, z)

    return orbit_shot(
        target_fn(0.0), dur, radius=radius, height=height, revolutions=revolutions,
        start_angle_deg=start_angle_deg, easing=easing, keyframes=keyframes,
        interpolator=interpolator, safety=safety, target_fn=target_fn)


# --------------------------------------------------------------------------- speed ramp

@dataclass(frozen=True)
class SpeedProfile:
    """Dense monotone video-time -> replay-time table behind a speed ramp."""

    duration_ms: float
    replay_start_ms: float
    replay_end_ms: float
    table: tuple  # cumulative replay fraction at i/(len-1) of video time

    def replay_at(self, video_ms: float) -> float:
        """Replay timestamp shown at ``video_ms`` (clamped to the clip)."""
        if video_ms != video_ms:
            raise CamPathError("video_ms must not be NaN")
        f = _clamp01(video_ms / self.duration_ms) * (len(self.table) - 1)
        i = min(int(f), len(self.table) - 2)
        frac = f - i
        c = self.table[i] + (self.table[i + 1] - self.table[i]) * frac
        if video_ms <= 0:
            return float(self.replay_start_ms)
        if video_ms >= self.duration_ms:
            return float(self.replay_end_ms)
        return self.replay_start_ms + (self.replay_end_ms - self.replay_start_ms) * c


def speed_profile(
    replay_start_ms: float,
    replay_end_ms: float,
    duration_ms: float,
    rates: Sequence[tuple[float, float]],
    *,
    blend: float = 0.08,
    resolution: int = 400,
) -> SpeedProfile:
    """Piecewise playback speed, smoothed at the joins.

    ``rates`` is a list of ``(share, speed)``: each piece occupies ``share`` of
    the *video* (shares are normalised) and plays at relative ``speed``
    (0.25 = quarter speed). Only ratios matter: the whole curve is rescaled so
    the clip begins exactly at replay_start and ends exactly at replay_end,
    i.e. slow-mo on the peak is paid for by running the rest slightly faster.
    ``blend`` (fraction of the video) box-filters the rate so speed changes
    ease rather than snap. All rates are > 0, so the mapping is strictly
    increasing.
    """
    if replay_end_ms <= replay_start_ms:
        raise CamPathError("replay_end_ms must be after replay_start_ms")
    if duration_ms <= 0:
        raise CamPathError("duration_ms must be positive")
    if not rates:
        raise CamPathError("rates must not be empty")
    shares, speeds = [], []
    for share, speed in rates:
        if not (math.isfinite(share) and share > 0 and math.isfinite(speed) and speed > 0):
            raise CamPathError(f"bad rate piece {(share, speed)!r}: need share>0, speed>0")
        shares.append(float(share))
        speeds.append(float(speed))
    # Only ratios matter: normalise so extreme magnitudes cannot overflow the
    # cumulative sums (1e308 speeds) or lose everything to underflow.
    top_speed = max(speeds)
    speeds = [sp / top_speed for sp in speeds]
    top_share = max(shares)
    shares = [sh / top_share for sh in shares]
    total = sum(shares)
    blend = _finite(blend, "blend")
    n = max(20, int(resolution))
    rate = []
    for i in range(n):
        x = (i + 0.5) / n * total
        acc = 0.0
        for sh, sp in zip(shares, speeds):
            acc += sh
            if x <= acc:
                rate.append(sp)
                break
        else:
            rate.append(speeds[-1])
    half = int(max(0.0, blend) * n / 2)
    if half:
        sm = []
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            sm.append(sum(rate[lo:hi]) / (hi - lo))
        rate = sm
    cum = [0.0]
    for r in rate:
        cum.append(cum[-1] + r)
    top = cum[-1]
    table = tuple(min(1.0, c / top) for c in cum)
    return SpeedProfile(float(duration_ms), float(replay_start_ms),
                        float(replay_end_ms), table)


def speed_ramp(
    replay_start_ms: float,
    replay_end_ms: float,
    duration_ms: int,
    rates: Sequence[tuple[float, float]],
    *,
    keyframes: int = 40,
    blend: float = 0.08,
) -> Path:
    """TIME path with baked timestamp keyframes realising a speed ramp.

    ReplayMod's time interpolation is linear between timestamp keyframes, so a
    varying playback rate is just many keyframes sampled off the profile. The
    first/last keyframes are exactly (0 -> replay_start) and
    (duration -> replay_end); timestamps and times are strictly increasing
    (colliding samples are dropped, never the endpoints).
    """
    if not _is_int(keyframes) or keyframes < 2:
        raise CamPathError("speed_ramp needs at least 2 integer keyframes")
    dur = int(round(_finite(duration_ms, "duration_ms")))
    prof = speed_profile(replay_start_ms, replay_end_ms, dur, rates, blend=blend)
    pts: list[tuple[int, int]] = []
    for f in _sample_fracs(keyframes):
        t = int(round(dur * f))
        if f == 0.0:
            ts = int(replay_start_ms)
        elif f == 1.0:
            ts = int(replay_end_ms)
        else:
            ts = int(round(prof.replay_at(t)))
        if pts and (t <= pts[-1][0] or ts <= pts[-1][1]):
            if f != 1.0:
                continue
            while pts and (pts[-1][0] >= t or pts[-1][1] >= ts):
                pts.pop()
        pts.append((t, ts))
    if len(pts) < 2:
        raise CamPathError("speed ramp collapsed; span too short for the duration")
    p = Path()
    p.keyframes = [{"time": t, "properties": {timeline.PROP_TIMESTAMP: ts}} for t, ts in pts]
    p.interpolators.append({"type": timeline.LINEAR, "properties": [timeline.PROP_TIMESTAMP]})
    return p


# --------------------------------------------------------------------------- shot choice

@dataclass(frozen=True)
class ShotRecipe:
    """What to shoot: a shot kind, its easing, params, and a slow-mo plan."""

    shot: str                              # orbit|dolly|crane|flyby|static|follow
    easing: str = "ease_in_out_sine"
    params: dict = field(default_factory=dict)
    slowmo: bool = False
    slowmo_speed: float = 0.25             # relative speed of the peak piece
    slowmo_share: float = 0.3              # share of video spent in slow-mo
    peak_frac: float = 0.65                # where in the video the peak sits


SHOT_KINDS = ("orbit", "dolly", "crane", "flyby", "static", "follow")


def _rng_for(h: Highlight) -> random.Random:
    # crc32, not hash(): str hashing is randomised per process.
    return random.Random(zlib.crc32(f"{h.kind}:{int(h.start_ms)}".encode()))


def choose_shot(highlight: Highlight, track: tracks.Track | None = None) -> ShotRecipe:
    """Deterministic shot recipe for a highlight.

    Variation (which candidate, spin direction, start angle) is seeded from the
    highlight's kind and start time, so re-planning the same replay yields the
    same montage while neighbouring highlights still differ. ``follow`` is only
    offered when a track exists; without one it degrades to a flyby.
    """
    rng = _rng_for(highlight)
    sign = rng.choice((-1, 1))
    angle = float(rng.randrange(0, 360, 15))
    kind = highlight.kind
    has_track = track is not None and bool(getattr(track, "points", None))

    if kind == "death":
        cands = [
            ("orbit", "ease_in_out_sine",
             dict(radius=12.0, end_radius=5.0, height=5.0, end_height=2.0,
                  revolutions=0.6 * sign, start_angle_deg=angle)),
            ("crane", "ease_out_cubic",
             dict(radius=9.0, start_height=14.0, end_height=2.5, angle_deg=angle,
                  sweep_deg=40.0 * sign)),
        ]
        slowmo = True
    elif kind == "kill":
        cands = [
            ("flyby", "ease_in_out_quad",
             dict(pass_distance=7.0, height=2.5, approach=24.0, direction_deg=angle)),
            ("orbit", "ease_out_cubic",
             dict(radius=9.0, height=3.0, revolutions=0.5 * sign, start_angle_deg=angle)),
        ]
        slowmo = True
    elif kind == "speed":
        cands = [
            ("follow", "ease_in_out_sine",
             dict(radius=14.0, height=5.0, revolutions=0.3 * sign, start_angle_deg=angle)),
            ("flyby", "linear",
             dict(pass_distance=10.0, height=4.0, approach=40.0, direction_deg=angle)),
        ]
        slowmo = False
    elif kind == "build":
        cands = [
            ("orbit", "smoothstep",
             dict(radius=22.0, end_radius=16.0, height=10.0, end_height=14.0,
                  revolutions=0.75 * sign, start_angle_deg=angle)),
            ("crane", "ease_in_out_sine",
             dict(radius=20.0, start_height=3.0, end_height=26.0, angle_deg=angle,
                  sweep_deg=60.0 * sign)),
        ]
        slowmo = False
    elif kind == "fall":
        cands = [
            ("crane", "ease_in_cubic",
             dict(radius=10.0, start_height=18.0, end_height=-6.0, angle_deg=angle,
                  sweep_deg=20.0 * sign)),
            ("dolly", "ease_in_out_cubic",
             dict(distance=18.0, end_distance=8.0, height=6.0, end_height=-2.0,
                  arc=3.0 * sign, angle_deg=angle)),
        ]
        slowmo = False
    else:  # custom and anything unknown
        cands = [
            ("orbit", "ease_in_out_sine",
             dict(radius=12.0, height=4.0, revolutions=0.5 * sign, start_angle_deg=angle)),
            ("dolly", "ease_in_out_cubic",
             dict(distance=16.0, end_distance=8.0, height=4.0, end_height=3.0,
                  arc=2.0 * sign, angle_deg=angle)),
        ]
        slowmo = False

    shot, easing, params = cands[rng.randrange(len(cands))]
    if shot == "follow" and not has_track:
        shot, easing = "flyby", "linear"
        params = dict(pass_distance=10.0, height=4.0, approach=40.0, direction_deg=angle)
    peak = 0.6 + 0.1 * rng.random() if slowmo else 0.65
    return ShotRecipe(shot=shot, easing=easing, params=dict(params), slowmo=slowmo,
                      peak_frac=round(peak, 3))


# --------------------------------------------------------------------------- planning

def _default_keyframes(seconds: float) -> int:
    return max(12, min(90, int(seconds * 5) + 8))


def _ramp_rates(recipe: ShotRecipe) -> list[tuple[float, float]]:
    share = min(0.5, max(0.1, recipe.slowmo_share))
    start = min(max(recipe.peak_frac - share / 2, 0.05), 0.95 - share)
    return [(start, 1.0), (share, recipe.slowmo_speed), (1.0 - start - share, 1.0)]


def _focus_point(h: Highlight, track, target_offset: float):
    if h.pos is not None:
        p = _finite_vec(h.pos, "highlight.pos")
    elif track is not None and getattr(track, "points", None):
        mid = (h.start_ms + h.end_ms) / 2.0
        p = tuple(track.at(mid))
    else:
        raise CamPathError(
            f"highlight at {h.start_ms}ms has no pos and no track to derive one from")
    return (p[0], p[1] + target_offset, p[2])


def plan_highlight(
    highlight: Highlight,
    track: tracks.Track | None,
    seconds: float,
    *,
    recipe: ShotRecipe | None = None,
    keyframes: int | None = None,
    interpolator=None,
    slowmo: bool | None = None,
    safety: Safety | None = None,
    target_offset: float = 1.0,
) -> tuple[Path, Path]:
    """Plan one highlight as ``(time Path, position Path)``.

    The time path maps video 0..seconds onto the highlight's replay span (with
    a baked slow-motion ramp when the recipe asks for it and ``slowmo`` is not
    False). The position path is built by the recipe's shot, aimed at the
    highlight's ``pos`` (or the track at mid-span) raised by ``target_offset``
    so the camera frames the torso, not the feet.
    """
    secs = _finite(seconds, "seconds")
    if secs <= 0:
        raise CamPathError("seconds must be positive")
    dur = int(round(secs * 1000))
    if dur < 1:
        raise CamPathError("seconds too small")
    recipe = recipe or choose_shot(highlight, track)
    if recipe.shot not in SHOT_KINDS:
        raise CamPathError(f"unknown shot {recipe.shot!r}; choose from {SHOT_KINDS}")
    use_ramp = recipe.slowmo if slowmo is None else bool(slowmo)
    n = keyframes if keyframes is not None else _default_keyframes(secs)
    t0, t1 = int(highlight.start_ms), int(highlight.end_ms)

    time_map = None
    if use_ramp:
        rates = _ramp_rates(recipe)
        time_path = speed_ramp(t0, t1, dur, rates)
        prof = speed_profile(t0, t1, dur, rates)
        time_map = prof.replay_at
    else:
        time_path = timeline.time_path(t0, t1, dur)

    common = dict(keyframes=n, interpolator=interpolator, safety=safety)
    pr = dict(recipe.params)
    easing = recipe.easing

    if recipe.shot == "follow":
        if track is None or not getattr(track, "points", None):
            raise CamPathError("follow shot requires a track")
        pos = follow_shot(track, t0, t1, dur, easing=easing, time_map=time_map,
                          **pr, **common)
        return time_path, pos

    focus = _focus_point(highlight, track, target_offset)
    if recipe.shot == "orbit":
        pos = orbit_shot(focus, dur, easing=easing, **pr, **common)
    elif recipe.shot == "crane":
        pos = crane_shot(focus, dur, easing=easing, **pr, **common)
    elif recipe.shot == "flyby":
        pos = flyby_shot(focus, dur, easing=easing, **pr, **common)
    elif recipe.shot == "dolly":
        a = math.radians(pr.pop("angle_deg", 0.0))
        d0 = pr.pop("distance", 16.0)
        d1 = pr.pop("end_distance", d0)
        h0 = pr.pop("height", 4.0)
        h1 = pr.pop("end_height", h0)
        start = (focus[0] + d0 * math.cos(a), focus[1] + h0, focus[2] + d0 * math.sin(a))
        end = (focus[0] + d1 * math.cos(a), focus[1] + h1, focus[2] + d1 * math.sin(a))
        pos = dolly_shot(start, end, focus, dur, easing=easing, **pr, **common)
    else:  # static
        a = math.radians(pr.get("angle_deg", 0.0))
        d = pr.get("distance", 10.0)
        eye = (focus[0] + d * math.cos(a), focus[1] + pr.get("height", 3.0),
               focus[2] + d * math.sin(a))
        pos = static_shot(eye, focus, dur, keyframes=2, interpolator=interpolator,
                          safety=safety)
    return time_path, pos


_EXT = {"MP4_CUSTOM": ".mp4", "MP4_POTATO": ".mp4", "WEBM_CUSTOM": ".webm",
        "MKV_LOSSLESS": ".mkv", "PNG": ".png", "EXR": ".exr", "BLEND": ".blend"}


def plan_montage(
    highlights: Sequence[Highlight],
    tracks_or_track,
    seconds_each: float,
    out_dir=None,
    *,
    stem: str = "montage",
    settings: Mapping | None = None,
    **plan_kwargs,
) -> list[dict]:
    """One render job per highlight, ready for ``replay.write_render_queue``.

    ``tracks_or_track`` may be a single Track (used for every highlight), a
    ``{entity_id: Track}`` dict as returned by ``tracks.load`` (the highlight's
    entity picks its track; no match means no track), or None. ``settings`` are
    keyword args for ``timeline.render_settings`` (width, fps, ...) except
    ``output_file``, which is generated: ``<out_dir>/<stem>_h01_death.mp4``
    (1-based index in the order given, so names are distinct and stable).
    Extra kwargs go to ``plan_highlight``.
    """
    opts = dict(settings or {})
    if "output_file" in opts or "outputFile" in opts:
        raise CamPathError("settings must not contain the output file; it is generated")
    if not stem or any(c in stem for c in '/\\'):
        raise CamPathError(f"bad stem {stem!r}")
    probe = timeline.render_settings("x", **opts)  # validate early (ValueError)
    ext = _EXT.get(probe["encodingPreset"], ".mp4")
    width = max(2, len(str(len(highlights))))

    jobs = []
    for i, h in enumerate(highlights, 1):
        if isinstance(tracks_or_track, Mapping):
            trk = tracks_or_track.get(h.entity) if h.entity is not None else None
        else:
            trk = tracks_or_track
        tpath, ppath = plan_highlight(h, trk, seconds_each, **plan_kwargs)
        kind = "".join(c if c.isalnum() else "_" for c in h.kind) or "x"
        name = f"{stem}_h{i:0{width}d}_{kind}{ext}"
        out = str(FsPath(out_dir) / name) if out_dir is not None else name
        jobs.append(timeline.render_job(
            timeline.build_timeline(tpath, ppath),
            timeline.render_settings(out, **opts)))
    return jobs
