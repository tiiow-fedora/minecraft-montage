"""Automatic title, description and chapter generation.

Pure functions, no network and no ffmpeg: give it the clips that went into a
montage (name + duration) and it works out where each one starts in the
finished video, which accounts for crossfades eating `xfade` seconds per join.

YouTube's chapter rules (support.google.com/youtube/answer/9884579):
  * the list must start at 0:00
  * at least 3 timestamps
  * each chapter at least 10 seconds long
If any of that fails YouTube silently ignores the list, so `make_chapters`
enforces it (merging too-short chapters) and `chapters_ok` reports it.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

MAX_TITLE = 100
MIN_CHAPTER_SECONDS = 10.0
MIN_CHAPTERS = 3


@dataclass(frozen=True)
class ClipInfo:
    name: str                # label, e.g. "Building the farm"; falls back to Part N
    duration: float          # seconds, as rendered
    path: str = ""


@dataclass(frozen=True)
class Chapter:
    start: float
    title: str


def fmt_timestamp(seconds: float) -> str:
    """m:ss, or h:mm:ss from one hour. Rounds down, as YouTube does."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def clip_starts(durations, xfade: float = 0.0):
    """Start time of each clip in the joined video."""
    starts, t = [], 0.0
    for i, d in enumerate(durations):
        starts.append(max(0.0, t))
        t += d - (xfade if i < len(durations) - 1 else 0.0)
    return starts


def total_duration(durations, xfade: float = 0.0) -> float:
    if not durations:
        return 0.0
    return sum(durations) - xfade * (len(durations) - 1)


_NAME_NOISE = re.compile(r"[_\-]+")


def label_from_filename(stem: str, index: int) -> str:
    """'day2_3' / 'base_tour-final' -> readable label; opaque names -> Part N."""
    s = _NAME_NOISE.sub(" ", stem).strip()
    s = re.sub(r"\b(final|render|rendered|out)\b", "", s, flags=re.I).strip()
    if not s or re.fullmatch(r"(day\s*)?\d+( \d+)*", s, flags=re.I):
        return f"Part {index}"
    return s[:1].upper() + s[1:]


def make_chapters(clips, xfade: float = 0.0, intro_label: str = "Intro",
                  min_seconds: float = MIN_CHAPTER_SECONDS):
    """Chapters for the joined video, one per clip, YouTube-rule compliant.

    Clips shorter than `min_seconds` are folded into the previous chapter. If a
    single clip is long enough, it is not split: returns [] when fewer than
    MIN_CHAPTERS survive, because YouTube would ignore the list anyway.
    """
    clips = list(clips)
    if not clips:
        return []
    durations = [c.duration for c in clips]
    starts = clip_starts(durations, xfade)
    end = total_duration(durations, xfade)

    raw = []
    for i, (c, s) in enumerate(zip(clips, starts), start=1):
        title = (c.name or "").strip() or f"Part {i}"
        raw.append(Chapter(s, title))
    raw[0] = Chapter(0.0, raw[0].title if raw[0].title != "Part 1" else intro_label)

    merged = []
    bounds = [c.start for c in raw[1:]] + [end]
    for ch, nxt in zip(raw, bounds):
        # a chapter too short to stand alone, or one that would leave its
        # predecessor under min_seconds, is folded into that predecessor
        if merged and (nxt - ch.start < min_seconds
                       or ch.start - merged[-1].start < min_seconds):
            continue
        merged.append(ch)
    # a too-short tail chapter is likewise folded back
    while len(merged) > 1 and (end - merged[-1].start) < min_seconds:
        merged.pop()
    if len(merged) < MIN_CHAPTERS:
        return []
    return merged


def chapters_ok(chapters, total: float) -> list:
    """List of rule violations ([] means YouTube will accept the chapters)."""
    problems = []
    if not chapters:
        return ["no chapters"]
    if len(chapters) < MIN_CHAPTERS:
        problems.append(f"need at least {MIN_CHAPTERS} chapters")
    if chapters[0].start != 0:
        problems.append("first chapter must start at 0:00")
    for a, b in zip(chapters, chapters[1:]):
        if b.start <= a.start:
            problems.append("chapters must be in increasing order")
        elif b.start - a.start < MIN_CHAPTER_SECONDS:
            problems.append(f"{a.title!r} is shorter than {MIN_CHAPTER_SECONDS:.0f}s")
    if total - chapters[-1].start < MIN_CHAPTER_SECONDS:
        problems.append(f"{chapters[-1].title!r} is shorter than "
                        f"{MIN_CHAPTER_SECONDS:.0f}s")
    return problems


def chapters_block(chapters) -> str:
    return "\n".join(f"{fmt_timestamp(c.start)} {c.title}" for c in chapters)


# --------------------------------------------------------------------------
# title
# --------------------------------------------------------------------------

@dataclass
class Context:
    """Everything known about a montage, all optional."""
    world: str = ""                  # "Longhaul World"
    theme: str = ""                  # "Building the nether hub"
    day: datetime.date = None
    session_index: int = None        # "Day 3"
    series: str = ""                 # "Survival Diaries"
    clips: list = field(default_factory=list)
    xfade: float = 0.0
    music: list = field(default_factory=list)   # music.Track objects
    extra_tags: list = field(default_factory=list)


def generate_title(ctx: Context, limit: int = MAX_TITLE) -> str:
    """'<theme> | <series> – Day N' style, degrading gracefully, <= limit."""
    head = ctx.theme.strip() or (
        f"Exploring {ctx.world.strip()}" if ctx.world.strip() else "Minecraft Montage")
    tail_bits = []
    if ctx.series.strip():
        tail_bits.append(ctx.series.strip())
    if ctx.session_index:
        tail_bits.append(f"Day {ctx.session_index}")
    candidates = []
    if tail_bits:
        candidates.append(f"{head} | {' – '.join(tail_bits)}")
        candidates.append(f"{head} | {tail_bits[-1]}")
    candidates.append(head)
    for c in candidates:
        if len(c) <= limit:
            return c
    return head[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# description + tags
# --------------------------------------------------------------------------

def generate_tags(ctx: Context, max_chars: int = 450):
    base = ["minecraft", "montage", "cinematic", "replay mod"]
    for s in (ctx.world, ctx.series):
        if s.strip():
            base.append(s.strip().lower())
    base += [t.strip().lower() for t in ctx.extra_tags if t.strip()]
    seen, out, total = set(), [], 0
    for t in base:
        if t in seen:
            continue
        if total + len(t) > max_chars:
            break
        seen.add(t)
        out.append(t)
        total += len(t)
    return out


def generate_description(ctx: Context, intro: str = "", hashtags: bool = True,
                         with_music: bool = True) -> str:
    """Intro, chapters (if valid), music credits, hashtags.

    Music credits are required by the tracks' licences, so they come from
    `ctx.music` through the same function the uploader uses. The result is
    safe to pass through youtube.build_description again only if you do NOT
    also pass music there -- see upload_meta, which uses this output verbatim.
    """
    parts = []
    intro = (intro or "").strip()
    if not intro:
        bits = []
        if ctx.theme.strip():
            bits.append(ctx.theme.strip().rstrip("."))
        elif ctx.world.strip():
            bits.append(f"A montage from {ctx.world.strip()}")
        if ctx.day:
            bits.append(ctx.day.strftime("recorded %d %B %Y").replace(" 0", " "))
        intro = (" — ".join(bits) + ".") if bits else ""
    if intro:
        parts.append(intro)

    chapters = make_chapters(ctx.clips, ctx.xfade)
    if chapters:
        parts.append("Chapters:\n" + chapters_block(chapters))

    if with_music and ctx.music:
        from . import music as music_mod
        block = music_mod.attribution_block(ctx.music)
        if block:
            parts.append(block)

    if hashtags:
        tags = ["#Minecraft", "#Montage"]
        parts.append(" ".join(tags))
    return "\n\n".join(parts)
