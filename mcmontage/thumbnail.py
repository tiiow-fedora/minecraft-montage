"""Thumbnail generator: pick the best frame, overlay text, apply a template.

Pipeline:
    sample_frames()  -- pull N evenly spaced frames out of a video with ffmpeg
    score_frame()    -- sharpness (Laplacian variance) + contrast (luma stddev),
                        with penalties for near-black / blown-out frames
    pick_frames()    -- best K frames that are not near-duplicates in time
    render()         -- crop to 16:9, apply a Template (text, banner, vignette)
    save()           -- write a PNG/JPEG under YouTube's 2 MB limit

Only Pillow is needed (already a dependency of tools/make_icon.py); no numpy.
"""

from __future__ import annotations

import io
import math
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

THUMB_SIZE = (1280, 720)      # YouTube's recommended size, 16:9
MIN_WIDTH = 640               # YouTube's minimum thumbnail width
MAX_BYTES = 2 * 1024 * 1024   # YouTube's thumbnail upload limit
MAX_TITLE_CHARS = 60

_LAPLACIAN = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1,
                                offset=128)

FONT_CANDIDATES = (
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)


class ThumbnailError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameScore:
    sharpness: float   # variance of the Laplacian, raw
    contrast: float    # stddev of luma, 0..~128
    brightness: float  # mean luma 0..255
    score: float       # combined, higher is better


def _analysis_copy(img: Image.Image) -> Image.Image:
    """Greyscale copy at a fixed width so scores compare across resolutions."""
    g = img.convert("L")
    w = 640
    if g.width != w:
        g = g.resize((w, max(1, round(g.height * w / g.width))), Image.BILINEAR)
    return g


def score_frame(img: Image.Image) -> FrameScore:
    g = _analysis_copy(img)
    lap = g.filter(_LAPLACIAN)
    # kernel offset is 128; variance is shift-invariant so it is unaffected
    sharp = ImageStat.Stat(lap).var[0]
    stat = ImageStat.Stat(g)
    contrast = stat.stddev[0]
    brightness = stat.mean[0]

    # Sharpness has a huge dynamic range; log-compress then normalise.
    sharp_n = min(1.0, math.log1p(sharp) / math.log1p(2000.0))
    contrast_n = min(1.0, contrast / 70.0)

    # Penalise frames that are almost black or almost white (loading screens,
    # fades, sky-only shots): ramps to 0 at the extremes, 1 in the mid range.
    if brightness < 40:
        exposure = max(0.0, brightness / 40.0)
    elif brightness > 215:
        exposure = max(0.0, (255.0 - brightness) / 40.0)
    else:
        exposure = 1.0

    score = (0.55 * sharp_n + 0.45 * contrast_n) * exposure
    return FrameScore(sharp, contrast, brightness, score)


# --------------------------------------------------------------------------
# frame extraction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    time: float
    image: Image.Image = field(repr=False, compare=False)
    score: FrameScore = None


def video_duration(path) -> float:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError as e:
        raise ThumbnailError(f"cannot read duration of {path}") from e


def sample_times(duration: float, count: int, margin: float = 0.05):
    """`count` timestamps spread over the middle of the video.

    The first/last `margin` fraction is skipped: that is where title cards and
    fade-outs live and they make poor thumbnails.
    """
    if count < 1:
        raise ThumbnailError("count must be >= 1")
    if duration <= 0:
        raise ThumbnailError("video has no duration")
    lo, hi = duration * margin, duration * (1 - margin)
    if count == 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


def extract_frame(video, t: float) -> Image.Image:
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise ThumbnailError(
            f"ffmpeg could not read a frame at {t:.1f}s: "
            + proc.stderr.decode("utf-8", "replace")[-300:])
    img = Image.open(io.BytesIO(proc.stdout))
    img.load()
    return img.convert("RGB")


def sample_frames(video, count: int = 24):
    video = Path(video)
    if not video.is_file():
        raise ThumbnailError(f"no such video: {video}")
    out = []
    for t in sample_times(video_duration(video), count):
        img = extract_frame(video, t)
        out.append(Candidate(t, img, score_frame(img)))
    return out


def pick_frames(candidates, k: int = 3, min_gap: float = 0.0):
    """Best `k` candidates by score, at least `min_gap` seconds apart."""
    chosen = []
    for c in sorted(candidates, key=lambda c: c.score.score, reverse=True):
        if all(abs(c.time - o.time) >= min_gap for o in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    return chosen


def best_frames(video, k: int = 3, samples: int = 24):
    cands = sample_frames(video, samples)
    duration = max(c.time for c in cands) if cands else 0
    return pick_frames(cands, k=k, min_gap=duration / (samples * 2))


# --------------------------------------------------------------------------
# templates + rendering
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Template:
    name: str
    title_color: tuple = (255, 255, 255)
    subtitle_color: tuple = (255, 214, 64)
    outline_color: tuple = (0, 0, 0)
    outline_px: int = 6
    title_size: int = 118            # px at 1280x720, shrinks to fit
    subtitle_size: int = 56
    anchor: str = "bottom-left"      # bottom-left | bottom-center | top-left | center
    margin: int = 48
    banner: tuple = None             # (r,g,b,a) strip behind the text, or None
    vignette: float = 0.0            # 0..1 darkening of the edges
    badge: str = None                # small corner text, e.g. "DAY 3"
    badge_fill: tuple = (45, 212, 191)
    uppercase: bool = True


TEMPLATES = {
    "bold": Template("bold", vignette=0.35),
    "minimal": Template("minimal", title_size=84, subtitle_size=44,
                        outline_px=3, anchor="bottom-left", uppercase=False,
                        banner=(0, 0, 0, 140)),
    "banner": Template("banner", anchor="bottom-center", banner=(15, 18, 24, 215),
                       title_size=104, outline_px=0, vignette=0.15),
    "poster": Template("poster", anchor="center", title_size=150,
                       subtitle_size=60, vignette=0.55),
}


def get_template(name: str) -> Template:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise ThumbnailError(
            f"no template {name!r} (have: {', '.join(sorted(TEMPLATES))})"
        ) from None


def find_font(size: int, font=None):
    for cand in ([font] if font else []) + list(FONT_CANDIDATES):
        try:
            return ImageFont.truetype(str(cand), size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def crop_to_16x9(img: Image.Image, size=THUMB_SIZE) -> Image.Image:
    """Centre-crop to the target aspect ratio, then resize."""
    tw, th = size
    w, h = img.size
    target = tw / th
    if w / h > target:
        nw = round(h * target)
        x = (w - nw) // 2
        img = img.crop((x, 0, x + nw, h))
    else:
        nh = round(w / target)
        y = (h - nh) // 2
        img = img.crop((0, y, w, y + nh))
    return img.resize(size, Image.LANCZOS)


def wrap_text(draw, text, font, max_width, max_lines=3):
    """Greedy word wrap. Returns lines; the last is ellipsised if truncated."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    return lines


def _fit_font(draw, text, base_size, max_width, max_lines, font_path):
    size = base_size
    while size > 24:
        f = find_font(size, font_path)
        lines = wrap_text(draw, text, f, max_width, max_lines=99)
        widest = max((draw.textlength(l, font=f) for l in lines), default=0)
        if len(lines) <= max_lines and widest <= max_width:
            return f, lines
        size -= 6
    f = find_font(24, font_path)
    return f, wrap_text(draw, text, f, max_width, max_lines=max_lines)


def _vignette(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return img
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([-w * 0.25, -h * 0.25, w * 1.25, h * 1.25], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=w // 8))
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    # where the mask is bright (centre) keep the image; at the edges blend dark
    blended = Image.composite(img, dark, mask)
    return Image.blend(img, blended, min(1.0, strength))


def _draw_block(draw, lines, font, xy, color, outline_color, outline_px,
                align_w=None, line_gap=8):
    x, y = xy
    for line in lines:
        lw = draw.textlength(line, font=font)
        lx = x + ((align_w - lw) / 2 if align_w else 0)
        draw.text((lx, y), line, font=font, fill=color,
                  stroke_width=outline_px, stroke_fill=outline_color)
        bbox = font.getbbox(line)
        y += (bbox[3] - bbox[1]) + line_gap + outline_px
    return y


def _block_height(font, lines, outline_px, line_gap=8):
    if not lines:
        return 0
    h = 0
    for line in lines:
        b = font.getbbox(line)
        h += (b[3] - b[1]) + line_gap + outline_px
    return h - line_gap


def render(frame: Image.Image, title: str = "", subtitle: str = "",
           template="bold", badge: str = None, font=None,
           size=THUMB_SIZE) -> Image.Image:
    """Compose the final thumbnail image (RGB, `size`)."""
    tpl = get_template(template) if isinstance(template, str) else template
    if badge is not None:
        tpl = replace(tpl, badge=badge)

    title = (title or "").strip()[: MAX_TITLE_CHARS * 2]
    subtitle = (subtitle or "").strip()
    if tpl.uppercase:
        title, subtitle = title.upper(), subtitle.upper()

    img = crop_to_16x9(frame.convert("RGB"), size)
    img = _vignette(img, tpl.vignette)
    W, H = img.size
    scale = W / THUMB_SIZE[0]
    m = round(tpl.margin * scale)
    max_w = W - 2 * m

    probe = ImageDraw.Draw(img)
    t_font, t_lines = _fit_font(probe, title, round(tpl.title_size * scale),
                                max_w, 3, font) if title else (None, [])
    s_font, s_lines = _fit_font(probe, subtitle, round(tpl.subtitle_size * scale),
                                max_w, 2, font) if subtitle else (None, [])
    ow = round(tpl.outline_px * scale)

    t_h = _block_height(t_font, t_lines, ow) if t_lines else 0
    s_h = _block_height(s_font, s_lines, ow) if s_lines else 0
    gap = round(14 * scale) if (t_lines and s_lines) else 0
    total_h = t_h + gap + s_h

    if tpl.anchor.startswith("top"):
        y0 = m
    elif tpl.anchor == "center":
        y0 = (H - total_h) // 2
    else:
        y0 = H - m - total_h

    centered = tpl.anchor in ("bottom-center", "center")

    if tpl.banner and total_h:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        pad = round(22 * scale)
        od.rectangle([0, max(0, y0 - pad), W, min(H, y0 + total_h + pad)],
                     fill=tpl.banner)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    d = ImageDraw.Draw(img)
    y = y0
    if t_lines:
        y = _draw_block(d, t_lines, t_font, (m, y), tpl.title_color,
                        tpl.outline_color, ow, align_w=max_w if centered else None)
        y += gap
    if s_lines:
        _draw_block(d, s_lines, s_font, (m, y), tpl.subtitle_color,
                    tpl.outline_color, max(1, ow // 2),
                    align_w=max_w if centered else None)

    if tpl.badge:
        bf = find_font(round(46 * scale), font)
        text = tpl.badge.upper()
        bw = d.textlength(text, font=bf)
        pad = round(14 * scale)
        bh = bf.getbbox(text)[3] - bf.getbbox(text)[1]
        x1, y1 = W - m, m
        d.rounded_rectangle([x1 - bw - 2 * pad, y1, x1, y1 + bh + 2 * pad],
                            radius=round(10 * scale), fill=tpl.badge_fill)
        d.text((x1 - bw - pad, y1 + pad - bf.getbbox(text)[1]), text, font=bf,
               fill=(10, 12, 16))
    return img


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------

def save(img: Image.Image, path, max_bytes: int = MAX_BYTES) -> Path:
    """Write PNG (or JPEG, shrinking quality until under `max_bytes`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".png":
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        if buf.tell() <= max_bytes:
            path.write_bytes(buf.getvalue())
            return path
        # PNG too big (photographic frame): fall back to JPEG beside it
        path = path.with_suffix(".jpg")
    rgb = img.convert("RGB")
    while True:
        for q in (95, 90, 85, 80, 70, 60, 50, 40):
            buf = io.BytesIO()
            rgb.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() <= max_bytes:
                path.write_bytes(buf.getvalue())
                return path
        # still too big: shrink, but never below YouTube's 640 px minimum width
        nw = round(rgb.width * 0.85)
        if nw < MIN_WIDTH:
            raise ThumbnailError(f"cannot get thumbnail under {max_bytes} bytes")
        rgb = rgb.resize((nw, round(rgb.height * 0.85)), Image.LANCZOS)


def make_thumbnail(video, out, title="", subtitle="", template="bold",
                   badge=None, samples=24, font=None):
    """One call: pick the best frame of `video`, render, save. Returns the path."""
    best = best_frames(video, k=1, samples=samples)
    if not best:
        raise ThumbnailError("no usable frames")
    img = render(best[0].image, title, subtitle, template, badge=badge, font=font)
    return save(img, out)
