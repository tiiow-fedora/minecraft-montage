"""Generate ReplayMod camera timelines and render jobs programmatically.

Everything here targets the schema reverse-engineered from
reforgedplaymod-1.21.1-0.3.jar. The serialised timeline looks like:

    { "": [ <TIME path>, <POSITION path> ] }

Path order is fixed by SPTimeline$SPPath: TIME=0, POSITION=1.

Each path is:

    { "keyframes":    [ {"time": <ms on the video timeline>,
                         "properties": { "<group>:<id>": <value>, ... }} ],
      "segments":     [ <index into interpolators> | null, ... ],
      "interpolators":[ {"type": <see below>, "properties": ["<group>:<id>"]} ] }

Property ids and value encodings. A property key is "<group>:<id>" ONLY when the
property belongs to a PropertyGroup; ungrouped properties serialise under their
bare id. CameraProperties is a group ("camera"); TimestampProperty and
SpectatorProperty are standalone, so they are NOT prefixed:

    "timestamp"        int   -- millisecond offset into the replay
    "camera:position"  [x, y, z]            (doubles)
    "camera:rotation"  [yaw, pitch, roll]   (floats, yaw NOT wrapped to +/-180)
    "spectate"         int   -- network entity id (runtime value, see NOTE)

Verified against a mod-written timelines.json (2026-09-08). "spectate" is
inferred by analogy -- SpectatorProperty is likewise ungrouped -- and is the one
key here not directly observed, since that replay had no spectator keyframe.

Interpolator "type" is POLYMORPHIC:
    "linear"                                          -- plain string
    "cubic-spline"                                    -- plain string
    {"type":"catmull-rom-spline","alpha":<double>}    -- object

NOTE on spectating: SpectatorProperty is AbstractProperty<Integer> keyed on the
*network entity id*, which is assigned at runtime and only recoverable by
parsing recording.tmcpr. Free camera paths (orbit/dolly) need no such lookup,
which is why they are what this module builds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

PROP_TIMESTAMP = "timestamp"
PROP_POSITION = "camera:position"
PROP_ROTATION = "camera:rotation"
PROP_SPECTATE = "spectate"

LINEAR = "linear"
CUBIC = "cubic-spline"


def catmull_rom(alpha: float = 0.5) -> dict:
    """Catmull-rom interpolator type object (smooth, the good default)."""
    return {"type": "catmull-rom-spline", "alpha": alpha}


@dataclass
class Path:
    keyframes: list = field(default_factory=list)
    interpolators: list = field(default_factory=list)

    def to_json(self) -> dict:
        # One segment between each consecutive pair of keyframes; every segment
        # uses interpolator 0 when one is defined.
        n_segments = max(0, len(self.keyframes) - 1)
        idx = 0 if self.interpolators else None
        return {
            "keyframes": self.keyframes,
            "segments": [idx] * n_segments,
            "interpolators": self.interpolators,
        }


def look_at(eye, target):
    """Minecraft (yaw, pitch) in degrees for a camera at `eye` facing `target`.

    Minecraft's look vector is
        x = -sin(yaw)cos(pitch),  y = -sin(pitch),  z = cos(yaw)cos(pitch)
    so yaw = atan2(-dx, dz) and pitch = -asin(dy), positive pitch looking down.
    """
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    dz = target[2] - eye[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist == 0:
        return 0.0, 0.0
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = -math.degrees(math.asin(dy / dist))
    return yaw, pitch


def _unwrap(yaws):
    """Remove +/-180 discontinuities so interpolation does not spin backwards."""
    if not yaws:
        return yaws
    out = [yaws[0]]
    for y in yaws[1:]:
        prev = out[-1]
        while y - prev > 180:
            y -= 360
        while y - prev < -180:
            y += 360
        out.append(y)
    return out


def _camera_path(eyes, times, targets, interpolator):
    raw = [look_at(e, t) for e, t in zip(eyes, targets)]
    yaws = _unwrap([y for y, _ in raw])
    pitches = [p for _, p in raw]

    p = Path()
    for t, eye, yaw, pitch in zip(times, eyes, yaws, pitches):
        p.keyframes.append(
            {
                "time": int(t),
                "properties": {
                    PROP_POSITION: [round(float(v), 4) for v in eye],
                    PROP_ROTATION: [round(yaw, 3), round(pitch, 3), 0.0],
                },
            }
        )
    p.interpolators.append(
        {"type": interpolator, "properties": [PROP_POSITION, PROP_ROTATION]}
    )
    return p


def orbit_path(
    centre,
    radius,
    height,
    duration_ms,
    revolutions=1.0,
    start_angle_deg=0.0,
    keyframes=24,
    interpolator=None,
):
    """Camera circling `centre` at `radius`, `height` above it, always looking at it.

    `height` is relative to centre[1]. `keyframes` controls path fidelity, not
    smoothness -- interpolation handles smoothness.
    """
    if keyframes < 2:
        raise ValueError("orbit needs at least 2 keyframes")
    interpolator = catmull_rom() if interpolator is None else interpolator

    eyes = []
    times = []
    for i in range(keyframes):
        f = i / (keyframes - 1)
        ang = math.radians(start_angle_deg + 360.0 * revolutions * f)
        eyes.append(
            (
                centre[0] + radius * math.cos(ang),
                centre[1] + height,
                centre[2] + radius * math.sin(ang),
            )
        )
        times.append(round(duration_ms * f))

    return _camera_path(eyes, times, [centre] * keyframes, interpolator)


def dolly_path(
    start, end, look, duration_ms, keyframes=8, interpolator=None
):
    """Straight camera move from `start` to `end`, always facing `look`."""
    if keyframes < 2:
        raise ValueError("dolly needs at least 2 keyframes")
    interpolator = catmull_rom() if interpolator is None else interpolator
    eyes = []
    times = []
    for i in range(keyframes):
        f = i / (keyframes - 1)
        eyes.append(tuple(start[j] + (end[j] - start[j]) * f for j in range(3)))
        times.append(round(duration_ms * f))
    return _camera_path(eyes, times, [look] * keyframes, interpolator)


def follow_orbit_path(
    track,
    replay_start_ms,
    replay_end_ms,
    duration_ms,
    radius=12.0,
    height=6.0,
    revolutions=0.5,
    start_angle_deg=0.0,
    keyframes=60,
    interpolator=None,
    eye_height=1.6,
):
    """Camera orbiting a MOVING target -- the player -- rather than a fixed point.

    At each keyframe the target's position is sampled at the replay timestamp
    that keyframe maps to, so the orbit centre travels with the player while the
    camera continues to swing around them.

    `track` should normally be a smoothed Track (see tracks.Track.smoothed): raw
    player position judders far too much to aim a camera at.

    `revolutions` is deliberately below 1 by default. A full spin on top of the
    player's own movement reads as restless; a half turn over the shot lets the
    subject move while the camera drifts around them.

    More keyframes matter here than for a static orbit -- each one is a fresh
    sample of where the player was, so too few will cut corners across their path.
    """
    if keyframes < 2:
        raise ValueError("follow orbit needs at least 2 keyframes")
    if replay_end_ms <= replay_start_ms:
        raise ValueError("replay_end_ms must be after replay_start_ms")
    interpolator = catmull_rom() if interpolator is None else interpolator

    eyes, times, targets = [], [], []
    for i in range(keyframes):
        f = i / (keyframes - 1)
        ts = replay_start_ms + (replay_end_ms - replay_start_ms) * f
        px, py, pz = track.at(ts)
        target = (px, py + eye_height, pz)
        ang = math.radians(start_angle_deg + 360.0 * revolutions * f)
        eyes.append(
            (target[0] + radius * math.cos(ang),
             target[1] + height,
             target[2] + radius * math.sin(ang))
        )
        targets.append(target)
        times.append(round(duration_ms * f))

    return _camera_path(eyes, times, targets, interpolator)


def time_path(replay_start_ms, replay_end_ms, duration_ms):
    """Map video time onto replay time -- this is what makes a timelapse.

    Two keyframes: video 0ms -> replay_start_ms, video duration_ms ->
    replay_end_ms. Compressing a long replay span into a short video makes the
    world advance fast, and it happens in-engine rather than by dropping frames,
    so every rendered frame is a real frame.
    """
    if replay_end_ms <= replay_start_ms:
        raise ValueError("replay_end_ms must be after replay_start_ms")
    p = Path()
    p.keyframes = [
        {"time": 0, "properties": {PROP_TIMESTAMP: int(replay_start_ms)}},
        {"time": int(duration_ms), "properties": {PROP_TIMESTAMP: int(replay_end_ms)}},
    ]
    p.interpolators.append({"type": LINEAR, "properties": [PROP_TIMESTAMP]})
    return p


def speedup_factor(replay_start_ms, replay_end_ms, duration_ms):
    return (replay_end_ms - replay_start_ms) / float(duration_ms)


def build_timeline(time, position):
    """Assemble the {"": [TIME, POSITION]} timeline map."""
    return {"": [time.to_json(), position.to_json()]}


ENCODING_PRESETS = (
    "MP4_CUSTOM",
    "MP4_POTATO",
    "WEBM_CUSTOM",
    "MKV_LOSSLESS",
    "PNG",
    "EXR",
    "BLEND",
)
RENDER_METHODS = ("DEFAULT", "STEREOSCOPIC", "CUBIC", "EQUIRECTANGULAR", "ODS", "BLEND")
ANTI_ALIASING = ("NONE", "X2", "X4", "X8")


def render_settings(
    output_file,
    width=1920,
    height=1080,
    fps=60,
    bitrate=20_000_000,
    encoding_preset="MP4_CUSTOM",
    render_method="DEFAULT",
    name_tags=False,
    anti_aliasing="NONE",
):
    """Partial RenderSettings.

    Only overridden fields are emitted: RenderSettings has a public no-arg
    constructor, so Gson leaves anything absent at its built-in default.
    outputFile is a plain path string (@JsonAdapter FileTypeAdapter).
    """
    if anti_aliasing not in ANTI_ALIASING:
        raise ValueError("bad antiAliasing: %s" % anti_aliasing)
    if encoding_preset not in ENCODING_PRESETS:
        raise ValueError("bad encodingPreset: %s" % encoding_preset)
    if render_method not in RENDER_METHODS:
        raise ValueError("bad renderMethod: %s" % render_method)
    return {
        "renderMethod": render_method,
        "encodingPreset": encoding_preset,
        "videoWidth": int(width),
        "videoHeight": int(height),
        "framesPerSecond": int(fps),
        "bitRate": int(bitrate),
        "outputFile": output_file,
        "renderNameTags": bool(name_tags),
        "antiAliasing": anti_aliasing,
    }


def render_job(timeline, settings):
    """One entry of renderQueue.json (replay.write_render_queue re-encodes it)."""
    return {"timeline": timeline, "settings": settings}
