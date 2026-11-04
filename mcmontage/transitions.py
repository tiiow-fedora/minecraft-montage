"""ffmpeg filter-graph builder for clip transitions (xfade / acrossfade).

Everything here is pure string generation: no subprocess is run and nothing
touches the disk, so the output is deterministic and unit-testable without
ffmpeg.  `edit.build()` hard-codes one `fade` crossfade between every pair of
clips; this module is the generalisation (per-boundary transition choice,
clip speed, audio crossfades, validation) that the pipeline can adopt later.

Timeline model (the thing that is easy to get wrong)
----------------------------------------------------
`xfade` overlaps the tail of the running stream with the head of the next
clip, so every transition *shortens* the result by its own duration::

    total_0 = len(clip0)
    offset_i = total_{i-1} - dur_i          # xfade `offset` for boundary i
    total_i  = offset_i + len(clip_i)

`acrossfade` shortens audio by exactly the same amount, so with matching
durations the audio and video streams stay the same length.

Labels are returned *without* brackets (``"x2"``); use `FilterGraph.video_map`
for the bracketed form ``-map`` wants.  A `prefix` keeps labels unique when
several graphs are merged into one ``-filter_complex``.
"""

from __future__ import annotations

import math
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

#: Every transition name known to ffmpeg's xfade filter, in ffmpeg's own order.
#: Verified against ffmpeg 6.1.1 (see tests/test_transitions.py, which renders
#: each one).  Older builds lack the later entries -- see MIN_FFMPEG.
TRANSITIONS: tuple[str, ...] = (
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite", "radial",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circleopen", "circleclose", "vertopen", "vertclose", "horzopen",
    "horzclose", "dissolve", "pixelize",
    "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "hblur", "fadegrays",
    "wipetl", "wipetr", "wipebl", "wipebr",
    "squeezeh", "squeezev", "zoomin",
    "fadefast", "fadeslow",
    "hlwind", "hrwind", "vuwind", "vdwind",
    "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
)

#: Minimum ffmpeg version per transition group.  Only the registry as a whole
#: is verified (on 6.1.1); the earliest versions below are from ffmpeg's
#: changelog as best recalled, so treat them as hints and use
#: `available_transitions()` to ask the installed binary instead.
MIN_FFMPEG: dict[str, str] = {
    "fade..dissolve/pixelize/diag*": "4.3 (xfade introduced)",
    "hlslice..fadegrays/wipetl..wipebr/squeezeh/squeezev/zoomin": "5.0",
    "fadefast/fadeslow/hlwind..vdwind/cover*/reveal*": "5.0-5.1",
    "custom (expr=)": "4.3",
    "xfade_opencl/xfade_vulkan": "not used here",
}

CUSTOM = "custom"

#: Smallest transition worth emitting; shorter than a frame at any fps we use.
MIN_DURATION = 0.04

_EPS = 1e-6

#: Pools for pick_transitions.  "subtle" never draws attention to itself;
#: "dynamic" is for fast-paced montage; "fade" is a single plain crossfade.
STYLE_POOLS: dict[str, tuple[str, ...]] = {
    "subtle": ("fade", "dissolve", "fadeblack", "smoothleft", "smoothright",
               "fadegrays"),
    "dynamic": ("slideleft", "slideright", "slideup", "slidedown", "wipeleft",
                "wiperight", "circleopen", "circleclose", "radial", "zoomin",
                "pixelize", "diagtl", "diagbr", "hlslice", "vuslice",
                "coverleft", "revealright", "squeezeh"),
    "fade": ("fade",),
}

#: Default duration (seconds) per style.
STYLE_DURATIONS: dict[str, float] = {"subtle": 0.6, "dynamic": 0.4, "fade": 0.8}

#: Easing curves as xfade custom expressions.  In `expr`, P runs 1 -> 0 over
#: the transition, A is the outgoing pixel and B the incoming one, so the
#: incoming weight is (1 - ease(P)).  Commas are avoided on purpose so the
#: expressions survive filtergraph parsing even unescaped.
EASINGS: dict[str, str] = {
    "linear": "A*P+B*(1-P)",
    "in": "A*(1-(1-P)*(1-P))+B*((1-P)*(1-P))",           # starts slow
    "out": "A*(P*P)+B*(1-P*P)",                          # ends slow
    "inout": "A*(P*P*(3-2*P))+B*(1-P*P*(3-2*P))",        # smoothstep
}


def _r(x: float) -> float:
    """Round to milliseconds so sums of durations stay exactly consistent."""
    return round(float(x), 3)


def _fmt(x: float) -> str:
    """Compact, deterministic number formatting for filter arguments."""
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s or "0"


def esc_expr(s: str) -> str:
    """Escape an expression for use as an unquoted filter option value.

    The graph parser splits on `,` `;` `[` `]` before the filter sees its
    arguments and `:` separates options, so each needs a backslash.  Quoting is
    avoided because nested quote levels are what make ffmpeg escaping brittle.
    """
    out = s.replace("\\", "\\\\")
    for ch in (":", "'", ",", ";", "[", "]"):
        out = out.replace(ch, "\\" + ch)
    return out


@dataclass(frozen=True)
class FilterGraph:
    """A set of filter chains plus the label(s) of what they produce.

    Shared between transitions.py and titlecards.py so results can be joined:
    concatenate `chains`, then refer to `label` from further chains.

    chains       -- each a complete chain string ("[in]f1,f2[out]" or a
                    source chain such as "color=...[bg]")
    label        -- primary output label, no brackets
    duration     -- predicted length in seconds of the primary output
    audio_label  -- label of a matching audio stream, if the graph builds one
    kind         -- "video" or "audio": what `label` carries
    """

    chains: tuple[str, ...]
    label: str
    duration: float
    audio_label: Optional[str] = None
    kind: str = "video"

    def __post_init__(self) -> None:
        object.__setattr__(self, "chains", tuple(self.chains))

    @property
    def filter_complex(self) -> str:
        return ";".join(self.chains)

    @property
    def video_map(self) -> str:
        return f"[{self.label}]"

    @property
    def audio_map(self) -> Optional[str]:
        if self.kind == "audio":
            return f"[{self.label}]"
        return f"[{self.audio_label}]" if self.audio_label else None

    def extend(self, chains: Iterable[str]) -> "FilterGraph":
        """Copy with extra chains appended (label/duration unchanged)."""
        return FilterGraph(self.chains + tuple(chains), self.label,
                           self.duration, self.audio_label, self.kind)


@dataclass(frozen=True)
class Transition:
    """One boundary between two clips.

    name     -- an xfade transition from TRANSITIONS, or "custom" with `expr`
    duration -- seconds of overlap
    expr     -- xfade custom expression (P, A, B, X, Y, W, H ...).  Supplying
                one forces name to "custom" -- xfade ignores `expr` otherwise.
    """

    name: str = "fade"
    duration: float = 0.5
    expr: Optional[str] = None

    def __post_init__(self) -> None:
        name = self.name
        if self.expr is not None:
            name = CUSTOM
        elif name == CUSTOM:
            raise ValueError("transition 'custom' needs an expr")
        if name != CUSTOM and name not in TRANSITIONS:
            raise ValueError(
                f"unknown transition {self.name!r}; expected one of "
                f"TRANSITIONS or 'custom' with expr"
            )
        if not (self.duration >= MIN_DURATION):
            raise ValueError(
                f"transition duration must be >= {MIN_DURATION}s, "
                f"got {self.duration!r}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "duration", _r(self.duration))

    @classmethod
    def eased(cls, easing: str = "inout", duration: float = 0.5) -> "Transition":
        """A crossfade with an easing curve (xfade has no built-in easing)."""
        if easing not in EASINGS:
            raise ValueError(
                f"unknown easing {easing!r}; expected one of {sorted(EASINGS)}")
        return cls(CUSTOM, duration, EASINGS[easing])

    def with_duration(self, duration: float) -> "Transition":
        return Transition(self.name, duration, self.expr)

    def filter_args(self, offset: float) -> str:
        """The `xfade=...` filter string for this boundary."""
        s = (f"xfade=transition={self.name}:duration={_fmt(self.duration)}"
             f":offset={_fmt(offset)}")
        if self.expr is not None:
            s += f":expr={esc_expr(self.expr)}"
        return s


@dataclass(frozen=True)
class ClipSpec:
    """One input to the montage.

    index     -- ffmpeg input index (the N in `-i` order, referenced as [N:v])
    duration  -- source seconds to use (the clip is trimmed to this)
    speed     -- playback speed multiplier (2.0 = twice as fast); the clip's
                 length on the timeline is duration / speed
    has_audio -- whether input N carries an audio stream
    """

    index: int
    duration: float
    speed: float = 1.0
    has_audio: bool = False

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("clip index must be >= 0")
        if not self.duration > 0:
            raise ValueError(f"clip duration must be > 0, got {self.duration!r}")
        if not self.speed > 0:
            raise ValueError(f"clip speed must be > 0, got {self.speed!r}")

    @property
    def length(self) -> float:
        """Length on the output timeline after the speed change."""
        return _r(self.duration / self.speed)


def pick_transitions(n_clips: int, style: str = "subtle", seed: int = 0,
                     duration: Optional[float] = None) -> list[Transition]:
    """Choose a transition for each of the n_clips - 1 boundaries.

    Deterministic in (n_clips, style, seed) so a re-run yields the same video.
    The pool is shuffled and cycled rather than sampled independently, so every
    transition gets used before any repeats and neighbours never match (which
    reads as a stutter).
    """
    if style not in STYLE_POOLS:
        raise ValueError(
            f"unknown style {style!r}; expected one of {sorted(STYLE_POOLS)}")
    if n_clips < 0:
        raise ValueError("n_clips must be >= 0")
    dur = STYLE_DURATIONS[style] if duration is None else duration
    pool = STYLE_POOLS[style]
    rng = random.Random(seed)
    names: list[str] = []
    order: list[str] = []
    for _ in range(max(0, n_clips - 1)):
        if not order:
            order = list(pool)
            rng.shuffle(order)
            # don't start a new cycle with the name that ended the last one
            if names and len(order) > 1 and order[0] == names[-1]:
                order.append(order.pop(0))
        names.append(order.pop(0))
    return [Transition(nm, dur) for nm in names]


def validate_transitions(clips: Sequence[ClipSpec],
                         transitions: Sequence[Transition],
                         clamp: bool = False) -> list[Transition]:
    """Check (or, with clamp=True, repair) transition durations.

    Two rules, both from xfade's model:
      * a transition cannot be longer than either clip it joins;
      * a clip between two transitions must be long enough to host both
        (len >= prev + next), or the second xfade would begin before the first
        has finished and ffmpeg emits garbage / stalls.

    Returns the (possibly clamped) list.  Raises ValueError on a violation when
    clamp is False, or when clamping would push a duration under MIN_DURATION.
    """
    n = len(clips)
    if n == 0:
        raise ValueError("no clips supplied")
    if len(transitions) != n - 1:
        raise ValueError(
            f"{n} clips need {n - 1} transitions, got {len(transitions)}")
    lens = [c.length for c in clips]
    out = list(transitions)
    for i, t in enumerate(out):          # boundary i joins clip i and i+1
        prev_t = out[i - 1].duration if i > 0 else 0.0
        limit = min(lens[i] - prev_t, lens[i + 1])
        if t.duration > limit + _EPS:
            if not clamp:
                raise ValueError(
                    f"transition {i} ({t.name}, {t.duration}s) is too long: "
                    f"clip {i} is {lens[i]}s"
                    + (f" (minus {prev_t}s used by the previous transition)"
                       if i > 0 else "")
                    + f" and clip {i + 1} is {lens[i + 1]}s; max {_r(limit)}s"
                )
            new = math.floor(limit * 1000 + 1e-6) / 1000   # floor to ms
            if new < MIN_DURATION:
                raise ValueError(
                    f"cannot clamp transition {i}: clips too short "
                    f"({lens[i]}s / {lens[i + 1]}s)")
            out[i] = t.with_duration(new)
    return out


def _normalise_chain(clip: ClipSpec, label: str, width: int, height: int,
                     fps: float) -> str:
    """Trim, retime and conform one clip so xfade will accept it.

    xfade needs identical size, pixel format, SAR, frame rate and timebase on
    both inputs; `fps` also pins the timebase to 1/fps.
    """
    parts = [f"trim=duration={_fmt(clip.duration)}"]
    if abs(clip.speed - 1.0) > 1e-9:
        parts.append(f"setpts=(PTS-STARTPTS)/{_fmt_speed(clip.speed)}")
    else:
        parts.append("setpts=PTS-STARTPTS")
    parts += [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        f"fps={_fmt_fps(fps)}",
        "format=yuv420p",
        "setsar=1",
    ]
    return f"[{clip.index}:v]" + ",".join(parts) + f"[{label}]"


def _fmt_speed(s: float) -> str:
    return f"{s:.6f}".rstrip("0").rstrip(".")


def _fmt_fps(f: float) -> str:
    return str(int(f)) if float(f).is_integer() else f"{f:.3f}".rstrip("0")


def _check_clips(clips: Sequence[ClipSpec]) -> None:
    if not clips:
        raise ValueError("no clips supplied")
    seen = set()
    for c in clips:
        if c.index in seen:
            raise ValueError(
                f"input {c.index} used twice; a filter-graph input pad can be "
                f"consumed once -- pass the file with a second -i instead")
        seen.add(c.index)


def timeline(clips: Sequence[ClipSpec],
             transitions: Sequence[Transition]) -> tuple[list[float], float]:
    """(xfade offsets per boundary, total duration) for already-valid input."""
    total = clips[0].length
    offsets: list[float] = []
    for clip, t in zip(clips[1:], transitions):
        off = _r(total - t.duration)
        offsets.append(off)
        total = _r(off + clip.length)
    return offsets, total


def atempo_chain(speed: float) -> list[str]:
    """atempo filters for an arbitrary speed (each instance spans 0.5..2.0)."""
    if not speed > 0:
        raise ValueError("speed must be > 0")
    out: list[str] = []
    s = speed
    while s > 2.0 + 1e-9:
        out.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        out.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-9 or not out:
        out.append(f"atempo={_fmt_speed(s)}")
    return out


def build_audio_graph(clips: Sequence[ClipSpec],
                      transitions: Sequence[Transition],
                      sample_rate: int = 48000, prefix: str = "",
                      clamp: bool = False) -> FilterGraph:
    """Crossfade clip audio with the same overlaps as the video.

    Clips without audio contribute silence of the right length, so one silent
    clip cannot break the chain (acrossfade needs two real inputs).  Every
    stream is conformed to stereo float at `sample_rate` first for the same
    reason xfade needs matching video: acrossfade rejects mismatches.
    """
    _check_clips(clips)
    trans = validate_transitions(clips, transitions, clamp)
    chains: list[str] = []
    fmt = f"aformat=sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=stereo"
    for i, c in enumerate(clips):
        if c.has_audio:
            parts = [f"atrim=duration={_fmt(c.duration)}", "asetpts=PTS-STARTPTS"]
            if abs(c.speed - 1.0) > 1e-9:
                parts += atempo_chain(c.speed)
                parts.append(f"atrim=duration={_fmt(c.length)}")
                parts.append("asetpts=PTS-STARTPTS")
            parts.append(fmt)
            chains.append(f"[{c.index}:a]" + ",".join(parts) + f"[{prefix}a{i}]")
        else:
            chains.append(
                f"anullsrc=channel_layout=stereo:sample_rate={sample_rate},"
                f"atrim=duration={_fmt(c.length)},asetpts=PTS-STARTPTS,{fmt}"
                f"[{prefix}a{i}]")
    last = f"{prefix}a0"
    for i, t in enumerate(trans, start=1):
        nxt = f"{prefix}ax{i}"
        chains.append(
            f"[{last}][{prefix}a{i}]acrossfade=d={_fmt(t.duration)}"
            f":c1=tri:c2=tri[{nxt}]")
        last = nxt
    _, total = timeline(clips, trans)
    return FilterGraph(tuple(chains), last, total, None, "audio")


def build_video_graph(clips: Sequence[ClipSpec],
                      transitions: Optional[Sequence[Transition]] = None,
                      width: int = 1920, height: int = 1080, fps: float = 30,
                      *, style: str = "subtle", seed: int = 0,
                      clamp: bool = False, with_audio: Optional[bool] = None,
                      sample_rate: int = 48000, prefix: str = "") -> FilterGraph:
    """Normalise clips and chain them with xfade.

    transitions -- one per boundary; None picks them with pick_transitions
                   (style/seed).  A single clip needs none and passes through
                   normalisation only.
    clamp       -- shorten over-long transitions instead of raising
    with_audio  -- build the acrossfade graph too.  None = automatic: yes when
                   any clip has audio (ReplayMod renders are silent, so the
                   default for those is video only).
    """
    _check_clips(clips)
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height and fps must be positive")
    if transitions is None:
        transitions = pick_transitions(len(clips), style, seed)
    trans = validate_transitions(clips, transitions, clamp)

    chains = [
        _normalise_chain(c, f"{prefix}v{i}", width, height, fps)
        for i, c in enumerate(clips)
    ]
    offsets, total = timeline(clips, trans)
    last = f"{prefix}v0"
    for i, (t, off) in enumerate(zip(trans, offsets), start=1):
        nxt = f"{prefix}x{i}"
        chains.append(f"[{last}][{prefix}v{i}]{t.filter_args(off)}[{nxt}]")
        last = nxt

    audio_label = None
    if with_audio is None:
        with_audio = any(c.has_audio for c in clips)
    if with_audio:
        ag = build_audio_graph(clips, trans, sample_rate, prefix)
        chains += list(ag.chains)
        audio_label = ag.label
    return FilterGraph(tuple(chains), last, total, audio_label, "video")


def render_graph_args(graph: FilterGraph, inputs: Sequence, output,
                      *, crf: int = 18, preset: str = "medium",
                      audio_bitrate: str = "192k", overwrite: bool = True,
                      loglevel: str = "error", ffmpeg: str = "ffmpeg",
                      trim_to_duration: bool = True) -> list[str]:
    """Full ffmpeg argv for rendering `graph` to `output`.

    inputs must be in the order the graph's input indices assume.  The `-t` cap
    guards against a stray extra frame from rounding; set trim_to_duration
    False when the graph is going to be extended downstream.
    """
    args = [ffmpeg, "-hide_banner", "-loglevel", loglevel]
    args.append("-y" if overwrite else "-n")
    for p in inputs:
        args += ["-i", str(p)]
    args += ["-filter_complex", graph.filter_complex,
             "-map", graph.video_map if graph.kind == "video" else graph.audio_map]
    if graph.kind == "video":
        amap = graph.audio_map
        if amap:
            args += ["-map", amap, "-c:a", "aac", "-b:a", audio_bitrate]
        else:
            args += ["-an"]
        args += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    else:
        args += ["-c:a", "aac", "-b:a", audio_bitrate]
    if trim_to_duration:
        args += ["-t", _fmt(graph.duration)]
    args.append(str(output))
    return args


_available_cache: dict[str, tuple[str, ...]] = {}


def ffmpeg_has_xfade(ffmpeg: str = "ffmpeg") -> bool:
    """True if the binary exists and has the xfade filter (ffmpeg >= 4.3)."""
    if shutil.which(ffmpeg) is None:
        return False
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(re.search(r"^\s*\S+\s+xfade\s", out, re.M))


def available_transitions(ffmpeg: str = "ffmpeg") -> tuple[str, ...]:
    """TRANSITIONS entries the installed ffmpeg actually accepts.

    Parsed from `ffmpeg -h filter=xfade`, so an older build without e.g. the
    `*wind` or `reveal*` names simply gets a shorter list.  Cached per binary.
    """
    if ffmpeg in _available_cache:
        return _available_cache[ffmpeg]
    names: tuple[str, ...] = ()
    if shutil.which(ffmpeg):
        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-h", "filter=xfade"],
                capture_output=True, text=True, timeout=20).stdout
            found = set(re.findall(r"^\s+(\w+)\s+-?\d+\s", out, re.M))
            names = tuple(n for n in TRANSITIONS if n in found)
        except (OSError, subprocess.SubprocessError):
            names = ()
    _available_cache[ffmpeg] = names
    return names
