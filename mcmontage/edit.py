"""Post-processing: turn rendered clips + a music bed into a finished video.

Deterministic and re-runnable -- the same inputs and settings always produce
the same output file, and nothing here mutates its inputs.

Note on audio: every ReplayMod encoding preset renders with `-an`, so clips
arrive with NO audio track. The music bed is therefore the only audio and needs
no sidechain ducking -- there is nothing to duck against. Fades and level are
all that is required.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Plain names, referenced relative to the ffmpeg cwd (see build_title_filter).
TITLE_FILE = "_title.txt"
SUBTITLE_FILE = "_subtitle.txt"
FONT_COPY = "_font.ttf"

DEFAULT_FONT = r"C:\Windows\Fonts\arialbd.ttf"


class EditError(RuntimeError):
    pass


@dataclass(frozen=True)
class Probe:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _run(args, desc, cwd=None):
    proc = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise EditError(f"{desc} failed (exit {proc.returncode}):\n{tail}")
    return proc


def probe(path) -> Probe:
    path = Path(path)
    if not path.is_file():
        raise EditError(f"no such file: {path}")
    proc = _run(
        [
            FFPROBE, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        f"ffprobe {path.name}",
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise EditError(f"{path.name} has no video stream")
    audio = any(s.get("codec_type") == "audio" for s in streams)

    num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    duration = data.get("format", {}).get("duration") or video.get("duration") or 0
    return Probe(
        path=path,
        duration=float(duration),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        has_audio=audio,
    )


def probe_audio(path) -> float:
    """Duration in seconds of an audio file.

    Use this rather than probe() for music: probe() insists on a video stream,
    and an MP3's streams are a trap here -- one carrying embedded cover art
    exposes an mjpeg/png "video" stream while one without art does not, so
    probing music as video succeeds or fails depending on whether the file
    happens to have artwork.
    """
    path = Path(path)
    if not path.is_file():
        raise EditError(f"no such file: {path}")
    proc = _run(
        [
            FFPROBE, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "format=duration", "-of",
            "default=nw=1:nk=1", str(path),
        ],
        f"ffprobe {path.name}",
    )
    out = (proc.stdout or "").strip()
    try:
        return float(out)
    except ValueError as e:
        raise EditError(f"{path.name}: no audio duration reported") from e


def _esc_text(s: str) -> str:
    """Escape text for ffmpeg drawtext.

    Commas must be backslash-escaped: the filtergraph parser splits filters on
    commas before it parses a filter's own arguments, so quoting alone does not
    protect them.
    """
    return (
        s.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
        .replace(",", r"\,")
    )


def _esc_expr(s: str) -> str:
    """Escape an ffmpeg expression for embedding in a filtergraph argument."""
    return s.replace(",", r"\,")


def build_title_filter(title, work_dir, subtitle=None, font=DEFAULT_FONT,
                       height=1080, hold=2.5, fade=0.6):
    """drawtext filters overlaying a title (and optional subtitle) at the start.

    The strings are passed via `textfile=` rather than `text=`. ffmpeg's
    filtergraph escaping has several nested levels and an apostrophe inside a
    quoted text= value cannot be escaped reliably; a UTF-8 file sidesteps all of
    it, so titles may contain apostrophes, commas and non-ASCII freely.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # The filtergraph is parsed at two nested levels, and a path containing an
    # apostrophe, comma, colon or bracket (a profile or user folder like
    # "John's Videos") cannot be escaped reliably through both. So no real path
    # ever enters the graph: the font is copied next to the title text and
    # ffmpeg is run with that folder as its cwd, referencing both by plain name.
    # Callers must run ffmpeg with cwd=work_dir.
    font_src = Path(font)
    if not font_src.is_file():
        raise EditError(f"font not found: {font_src}")
    shutil.copyfile(font_src, work_dir / FONT_COPY)
    fp = FONT_COPY
    size_main = max(28, int(height * 0.075))
    size_sub = max(18, int(height * 0.035))
    # alpha ramps in, holds, then ramps out
    alpha = _esc_expr(
        f"if(lt(t,{fade}),t/{fade},"
        f"if(lt(t,{hold}),1,"
        f"if(lt(t,{hold + fade}),({hold + fade}-t)/{fade},0)))"
    )

    title_file = work_dir / TITLE_FILE
    title_file.write_text(title, encoding="utf-8")
    filters = [
        f"drawtext=fontfile='{fp}':textfile='{TITLE_FILE}'"
        f":fontcolor=white:fontsize={size_main}"
        f":x=(w-text_w)/2:y=(h-text_h)/2-{int(height * 0.03)}"
        f":shadowcolor=black@0.6:shadowx=3:shadowy=3"
        f":alpha='{alpha}'"
    ]
    if subtitle:
        sub_file = work_dir / SUBTITLE_FILE
        sub_file.write_text(subtitle, encoding="utf-8")
        filters.append(
            f"drawtext=fontfile='{fp}':textfile='{SUBTITLE_FILE}'"
            f":fontcolor=white@0.85:fontsize={size_sub}"
            f":x=(w-text_w)/2:y=(h-text_h)/2+{int(height * 0.05)}"
            f":shadowcolor=black@0.6:shadowx=2:shadowy=2"
            f":alpha='{alpha}'"
        )
    return filters


def build(
    clips,
    output,
    music=None,
    title=None,
    subtitle=None,
    font=DEFAULT_FONT,
    xfade=1.0,
    fade_in=1.0,
    fade_out=2.5,
    music_gain_db=-16.0,
    music_fade_out=4.0,
    music_xfade=3.0,
    music_normalise=True,
    music_lufs=-20.0,
    crf=18,
    preset="medium",
    fps=None,
    dry_run=False,
):
    """Assemble `clips` into `output`, optionally with a title and music bed.

    clips  -- list of paths to rendered video files, joined in order
    music  -- one path, or several; several share the runtime equally and are
              crossfaded into each other by `music_xfade`
    xfade  -- crossfade duration in seconds between consecutive clips

    Returns the ffmpeg argument list actually used (handy for dry runs).
    """
    clips = [Path(c) for c in clips]
    if not clips:
        raise EditError("no clips supplied")
    for c in clips:
        if not c.is_file():
            raise EditError(f"no such clip: {c}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A relative name starting with "-" would be parsed by ffmpeg as an option.
    output = output.resolve()

    probes = [probe(c) for c in clips]
    width, height = probes[0].width, probes[0].height
    out_fps = fps or (round(probes[0].fps) if probes[0].fps else 60)

    args = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    for c in clips:
        args += ["-i", str(c.resolve())]
    # music may be one path or several; several share the runtime equally
    tracks = []
    if music:
        tracks = [Path(music)] if isinstance(music, (str, Path)) else [
            Path(m) for m in music]
        for m in tracks:
            if not m.is_file():
                raise EditError(f"no such music file: {m}")
            # Every track is looped, then trimmed to its share. Looping means a
            # track shorter than its share still fills it instead of leaving
            # silence, and the trim is what actually sets the length.
            args += ["-stream_loop", "-1", "-i", str(m.resolve())]

    chains = []

    # Normalise every clip to a common size/fps/format so xfade can chain them.
    for i, p in enumerate(probes):
        chains.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={out_fps},format=yuv420p,setsar=1[v{i}]"
        )

    # Chain crossfades. Each xfade shortens the timeline by its own duration.
    if len(clips) == 1:
        last = "v0"
        total = probes[0].duration
    else:
        total = probes[0].duration
        last = "v0"
        for i in range(1, len(clips)):
            offset = max(0.0, total - xfade)
            nxt = f"x{i}"
            chains.append(
                f"[{last}][v{i}]xfade=transition=fade:duration={xfade}"
                f":offset={offset:.3f}[{nxt}]"
            )
            last = nxt
            total = offset + probes[i].duration

    # Title overlay, then fade from/to black across the whole piece.
    vf = []
    work_dir = None
    if title:
        work_dir = output.parent / ".mcmontage_tmp"
        vf += build_title_filter(
            title, work_dir=work_dir,
            subtitle=subtitle, font=font, height=height,
        )
    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        vf.append(f"fade=t=out:st={max(0.0, total - fade_out):.3f}:d={fade_out}")
    if vf:
        chains.append(f"[{last}]" + ",".join(vf) + "[vout]")
        vmap = "[vout]"
    else:
        vmap = f"[{last}]"

    amap = None
    if tracks:
        n = len(tracks)
        # Each track gets an equal share. Crossfades overlap neighbours, so
        # every share is padded by the overlap it gives away; without that the
        # bed would finish (n-1)*music_xfade short of the video.
        share = (total + (n - 1) * music_xfade) / n if n > 1 else total

        # Normalise each track to a common loudness before sequencing them.
        # Royalty-free tracks are mastered wildly differently -- of the four
        # bundled here, one sits 21 dB below another -- so without this a
        # multi-track bed lurches from inaudible to blaring at every join. One
        # track alone does not need it, since its level is just the gain.
        norm = (f",loudnorm=I={music_lufs}:TP=-2:LRA=11"
                if (music_normalise and n > 1) else "")
        for i in range(n):
            chains.append(
                f"[{len(clips) + i}:a]atrim=0:{share:.3f},asetpts=N/SR/TB"
                f"{norm}"
                f",aformat=sample_fmts=fltp:sample_rates=48000"
                f":channel_layouts=stereo[m{i}]"
            )

        last_a = "m0"
        for i in range(1, n):
            nxt = f"ax{i}"
            chains.append(
                f"[{last_a}][m{i}]acrossfade=d={music_xfade}:c1=tri:c2=tri[{nxt}]"
            )
            last_a = nxt

        a_fade_out_start = max(0.0, total - music_fade_out)
        chains.append(
            f"[{last_a}]atrim=0:{total:.3f},asetpts=N/SR/TB,"
            f"volume={music_gain_db}dB,"
            f"afade=t=in:st=0:d=1.5,"
            f"afade=t=out:st={a_fade_out_start:.3f}:d={music_fade_out},"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[aout]"
        )
        amap = "[aout]"

    args += ["-filter_complex", ";".join(chains)]
    args += ["-map", vmap]
    if amap:
        args += ["-map", amap, "-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-an"]
    args += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", f"{total:.3f}",
        str(output),
    ]

    if dry_run:
        return args

    _run(args, "ffmpeg assemble", cwd=work_dir)
    if not output.is_file() or output.stat().st_size == 0:
        raise EditError(f"ffmpeg produced no output at {output}")
    return args


def describe_command(args) -> str:
    return " ".join(shlex.quote(a) for a in args)
