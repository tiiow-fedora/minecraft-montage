"""Tests for mcmontage.titlecards: strings/escaping first, then real ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mcmontage import titlecards as tc
from mcmontage.titlecards import (
    ANIMATIONS, CardStyle, FontNotFoundError, build_card_graph, credits_card,
    esc_path, fade_alpha, find_font, fit_font_size, lower_third,
)
from mcmontage.transitions import FilterGraph, render_graph_args

FONT = find_font()
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
CAN_RENDER = HAVE_FFMPEG and FONT is not None and tc.drawtext_available()
needs_render = pytest.mark.skipif(
    not CAN_RENDER, reason="needs ffmpeg with drawtext and a font file")
needs_font = pytest.mark.skipif(FONT is None, reason="no font on this machine")

TRICKY = "It's 100%, \"quoted\": {a} \\ ünï 日本"


@pytest.fixture
def style():
    return CardStyle(font=FONT) if FONT else CardStyle(on_missing_font="blank")


# ------------------------------------------------------------------- fonts

def test_find_font_returns_existing_file_or_none():
    f = find_font()
    assert f is None or Path(f).is_file()


def test_find_font_env_override(tmp_path, monkeypatch):
    f = tmp_path / "my.ttf"
    f.write_bytes(b"x")
    monkeypatch.setenv(tc.FONT_ENV, str(f))
    assert find_font() == str(f)


def test_find_font_extra_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv(tc.FONT_ENV, raising=False)
    f = tmp_path / "x.ttf"
    f.write_bytes(b"x")
    assert find_font([str(f)]) == str(f)


def test_find_font_none_when_nothing_exists(monkeypatch):
    monkeypatch.delenv(tc.FONT_ENV, raising=False)
    monkeypatch.setattr(tc, "FONT_CANDIDATES", ())
    monkeypatch.setattr(tc, "_FONT_DIRS", ())
    assert find_font() is None


def test_candidates_cover_all_platforms():
    joined = "\n".join(tc.FONT_CANDIDATES)
    assert "Windows" in joined and "System/Library" in joined and "dejavu" in joined


def test_missing_font_errors_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "find_font", lambda extra=(): None)
    with pytest.raises(FontNotFoundError, match="no usable font"):
        build_card_graph("T", None, 1, 160, 90, 15, CardStyle(), tmp_path)


def test_missing_font_blank_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "find_font", lambda extra=(): None)
    g = build_card_graph("T", None, 1, 160, 90, 15,
                         CardStyle(on_missing_font="blank"), tmp_path)
    assert "drawtext" not in g.filter_complex
    assert g.label == "cardout" and not list(tmp_path.iterdir())


def test_explicit_bad_font_path(tmp_path):
    with pytest.raises(FontNotFoundError, match="font file not found"):
        build_card_graph("T", None, 1, 160, 90, 15,
                         CardStyle(font=str(tmp_path / "nope.ttf")), tmp_path)


# ------------------------------------------------------------------- style

def test_style_validation():
    with pytest.raises(ValueError):
        CardStyle(animation="explode")
    with pytest.raises(ValueError):
        CardStyle(gradient="spiral")
    with pytest.raises(ValueError):
        CardStyle(on_missing_font="shrug")
    with pytest.raises(ValueError):
        CardStyle(fade_in=-1)


@pytest.mark.parametrize("bad", ["red;drawtext", "a b", "x,y", "[v]"])
def test_style_rejects_unsafe_colours(bad):
    with pytest.raises(ValueError, match="invalid"):
        CardStyle(bg_color=bad)


def test_style_accepts_ffmpeg_colour_forms():
    CardStyle(bg_color="0x101820", font_color="white@0.9", bg_color2="#ff00aa")


# ----------------------------------------------------------- helper strings

def test_esc_path_windows_style():
    assert esc_path(r"C:\Windows\Fonts\arial.ttf") == r"C\:/Windows/Fonts/arial.ttf"


def test_fade_alpha_shapes():
    assert fade_alpha(3, 0.5, 0.5) == \
        "max(0,min(1,(t-0)/0.5))*max(0,min(1,(3-t)/0.5))"
    assert fade_alpha(3, 0, 0) == "1"
    assert fade_alpha(3, 0, 0.5, delay=1).startswith("gte(t,1)")


def test_fit_font_size_shrinks_long_text_only():
    assert fit_font_size(["Hi"], 80, 1920) == 80
    s = fit_font_size(["x" * 100], 80, 1920)
    assert s < 80 and 100 * s * 0.62 <= 1920 * 0.9 + 1


def test_fit_font_size_floor():
    assert fit_font_size(["x" * 5000], 80, 100) == 12


def test_background_solid_and_gradient():
    s = CardStyle(bg_color="red")
    assert tc.background_chain("bg", 320, 180, 30, 2, s) == \
        "color=c=red:s=320x180:r=30:d=2[bg]"
    g = tc.background_chain("bg", 320, 180, 30, 2,
                            CardStyle(bg_color="red", bg_color2="blue"))
    assert g.startswith("gradients=") and "c0=red:c1=blue" in g
    assert "y1=180" in g and ":seed=" in g and g.endswith("[bg]")


def test_gradient_directions():
    h = tc.background_chain("b", 320, 180, 30, 1, CardStyle(
        bg_color2="blue", gradient="horizontal"))
    assert "x1=320:y1=0" in h


# ------------------------------------------------------ build_card_graph str

@needs_font
def test_card_graph_shape(tmp_path, style):
    g = build_card_graph("Title", "Sub", 3.0, 1280, 720, 30, style, tmp_path)
    assert isinstance(g, FilterGraph)
    assert g.label == "cardout" and g.duration == 3.0 and g.kind == "video"
    assert g.chains[0].startswith("color=") and g.chains[0].endswith("[cardbg]")
    last = g.chains[-1]
    assert last.startswith("[cardbg]") and last.endswith("format=yuv420p,setsar=1[cardout]")
    assert last.count("drawtext=") == 2


@needs_font
def test_text_goes_through_utf8_textfiles(tmp_path, style):
    g = build_card_graph(TRICKY, "sub, with: stuff", 2, 640, 360, 30, style, tmp_path)
    assert (tmp_path / "card_title.txt").read_bytes() == TRICKY.encode("utf-8")
    assert (tmp_path / "card_subtitle.txt").read_text(encoding="utf-8") == \
        "sub, with: stuff"
    fc = g.filter_complex
    # the dangerous text never appears in the graph itself
    for bad in ("It's", "100%", "ünï", "quoted"):
        assert bad not in fc
    assert "expansion=none" in fc and "textfile=" in fc and ":text=" not in fc


@needs_font
def test_crlf_normalised(tmp_path, style):
    build_card_graph("a\r\nb", None, 1, 320, 180, 15, style, tmp_path)
    assert (tmp_path / "card_title.txt").read_bytes() == b"a\nb"


@needs_font
def test_prefix_namespaces_labels_and_files(tmp_path, style):
    g = build_card_graph("T", "S", 1, 320, 180, 15, style, tmp_path, prefix="intro")
    assert g.label == "introout" and (tmp_path / "intro_title.txt").exists()
    assert "[introbg]" in g.chains[0]


@needs_font
def test_no_subtitle_means_one_drawtext(tmp_path, style):
    g = build_card_graph("T", None, 1, 320, 180, 15, style, tmp_path)
    assert g.chains[-1].count("drawtext=") == 1
    assert not (tmp_path / "card_subtitle.txt").exists()


@needs_font
@pytest.mark.parametrize("anim", ANIMATIONS)
def test_every_animation_builds(tmp_path, anim):
    g = build_card_graph("T", "S", 2, 320, 180, 15,
                         CardStyle(font=FONT, animation=anim), tmp_path)
    assert g.label == "cardout"


@needs_font
def test_animation_expressions(tmp_path):
    def fc(anim):
        return build_card_graph("T", "S", 2, 320, 180, 15,
                                CardStyle(font=FONT, animation=anim),
                                tmp_path).filter_complex
    assert "alpha=" not in fc("none")
    assert "alpha=" in fc("fade") and "pow(" not in fc("fade")
    assert "pow(" in fc("slide-up")
    tw = fc("typewriter")
    assert "alphamerge" in tw and "overlay=x=" in tw and "blend=all_mode=multiply" in tw


@needs_font
def test_commas_in_expressions_are_escaped(tmp_path, style):
    fc = build_card_graph("T", "S", 2, 320, 180, 15, style, tmp_path).filter_complex
    assert "min(1,(" not in fc and "min(1\\,(" in fc


@needs_font
def test_long_title_shrinks_font(tmp_path, style):
    short = build_card_graph("Hi", None, 1, 640, 360, 15, style, tmp_path).filter_complex
    long_ = build_card_graph("x" * 200, None, 1, 640, 360, 15, style, tmp_path).filter_complex

    def size(s):
        return int(s.split("fontsize=")[1].split(":")[0])
    assert size(long_) < size(short)


@needs_font
def test_fades_clamped_to_half_duration(tmp_path):
    g = build_card_graph("T", None, 1.0, 320, 180, 15,
                         CardStyle(font=FONT, fade_in=5, fade_out=5), tmp_path)
    assert "/0.5" in g.filter_complex and "/5)" not in g.filter_complex


@pytest.mark.parametrize("kw,msg", [
    (dict(title=""), "title"), (dict(title="  "), "title"),
    (dict(duration=0), "duration"), (dict(width=0), "positive"),
    (dict(fps=-1), "positive"),
])
def test_card_graph_validation(tmp_path, style, kw, msg):
    args = dict(title="T", subtitle=None, duration=1, width=320, height=180,
                fps=15, style=style, work_dir=tmp_path)
    args.update(kw)
    with pytest.raises(ValueError, match=msg):
        build_card_graph(**args)


def test_work_dir_required(style):
    with pytest.raises(ValueError, match="work_dir"):
        build_card_graph("T", None, 1, 320, 180, 15, style, None)


@needs_font
def test_work_dir_created(tmp_path, style):
    d = tmp_path / "a" / "b"
    build_card_graph("T", None, 1, 320, 180, 15, style, d)
    assert d.is_dir()


# ------------------------------------------------------------------ credits

@needs_font
def test_credits_stack_one_drawtext_per_nonblank_line(tmp_path, style):
    g = credits_card(["A - X", "", "B - Y"], 4, 640, 360, 15, style, tmp_path)
    # heading + A + B
    assert g.chains[-1].count("drawtext=") == 3
    assert g.label == "creditsout" and g.duration == 4.0
    assert (tmp_path / "credits_line0.txt").read_text() == "Music"


@needs_font
def test_credits_no_heading(tmp_path, style):
    g = credits_card(["A"], 3, 640, 360, 15, style, tmp_path, heading=None)
    assert g.chains[-1].count("drawtext=") == 1


@needs_font
def test_credits_scroll_moves_with_t(tmp_path, style):
    g = credits_card(["A", "B"], 5, 640, 360, 15, style, tmp_path, mode="scroll")
    assert "*t/5" in g.filter_complex and "alpha=" not in g.filter_complex


@needs_font
def test_credits_stack_is_static_and_fades(tmp_path, style):
    g = credits_card(["A"], 5, 640, 360, 15, style, tmp_path, mode="stack")
    assert "*t/" not in g.filter_complex and "alpha=" in g.filter_complex


@needs_font
def test_credits_text_safe(tmp_path, style):
    g = credits_card([TRICKY], 3, 640, 360, 15, style, tmp_path)
    assert "ünï" not in g.filter_complex
    assert (tmp_path / "credits_line2.txt").read_text(encoding="utf-8") == TRICKY


def test_credits_validation(tmp_path, style):
    with pytest.raises(ValueError, match="mode"):
        credits_card(["a"], 3, 640, 360, 15, style, tmp_path, mode="zoom")
    with pytest.raises(ValueError):
        credits_card(["", " "], 3, 640, 360, 15, style, tmp_path, heading=None)
    with pytest.raises(ValueError, match="work_dir"):
        credits_card(["a"], 3, 640, 360, 15, style, None)


# -------------------------------------------------------------- lower third

@needs_font
def test_lower_third_chain(tmp_path, style):
    g = lower_third("Kill: Zombie", 2.0, 4.5, width=1280, height=720,
                    work_dir=tmp_path, style=style, in_label="x3", out_label="lt1")
    assert len(g.chains) == 1 and g.label == "lt1" and g.duration == 4.5
    c = g.chains[0]
    assert c.startswith("[x3]") and c.endswith("[lt1]")
    assert "enable='between(t,2,4.5)'" in c and "box=1" in c


@needs_font
def test_lower_third_subtitle_adds_second_text(tmp_path, style):
    g = lower_third("Death", 1, 2, width=1280, height=720, work_dir=tmp_path,
                    style=style, subtitle="by Creeper")
    assert g.chains[0].count("drawtext=") == 2
    assert (tmp_path / "lt_subtext.txt").read_text() == "by Creeper"


@needs_font
def test_lower_third_center_position_has_no_slide(tmp_path, style):
    g = lower_third("D", 1, 2, width=1280, height=720, work_dir=tmp_path,
                    style=style, position="bottom-center")
    assert "pow(" not in g.chains[0] and "(w-text_w)/2" in g.chains[0]


@needs_font
def test_lower_third_text_via_file(tmp_path, style):
    g = lower_third(TRICKY, 1, 2, width=1280, height=720, work_dir=tmp_path, style=style)
    assert "ünï" not in g.filter_complex
    assert (tmp_path / "lt_text.txt").read_text(encoding="utf-8") == TRICKY


@pytest.mark.parametrize("kw", [dict(start=3, end=2), dict(start=-1, end=2),
                                dict(start=1, end=1), dict(text=""),
                                dict(position="top")])
def test_lower_third_validation(tmp_path, style, kw):
    args = dict(text="x", start=1, end=2, width=640, height=360,
                work_dir=tmp_path, style=style)
    args.update(kw)
    with pytest.raises(ValueError):
        lower_third(**args)


def test_lower_third_blank_font_passthrough(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "find_font", lambda extra=(): None)
    g = lower_third("x", 1, 2, width=640, height=360, work_dir=tmp_path,
                    style=CardStyle(on_missing_font="blank"), in_label="a")
    assert g.chains == ("[a]null[ltout]",)


# --------------------------------------------------------------- integration

def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    return float(d["format"]["duration"]), v


def _render(g, out, inputs=(), trim=True):
    a = render_graph_args(g, list(inputs), out, preset="ultrafast",
                          trim_to_duration=trim)
    r = subprocess.run(a, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:] + "\n" + g.filter_complex
    return _probe(out)


@needs_render
@pytest.mark.parametrize("anim", ANIMATIONS)
def test_card_renders_with_dimensions_and_duration(tmp_path, anim):
    g = build_card_graph(TRICKY, "sub, title: ok", 1.0, 320, 180, 15,
                         CardStyle(font=FONT, animation=anim), tmp_path / "w")
    dur, v = _render(g, tmp_path / "c.mp4")
    assert (v["width"], v["height"]) == (320, 180)
    assert dur == pytest.approx(1.0, abs=0.12)


@needs_render
def test_gradient_card_renders(tmp_path):
    g = build_card_graph("Grad", None, 1.0, 320, 180, 15,
                         CardStyle(font=FONT, bg_color2="0x305080"), tmp_path / "w")
    dur, v = _render(g, tmp_path / "g.mp4")
    assert (v["width"], v["height"]) == (320, 180)


def _pixels(path, t, x, y, w=1, h=1):
    """Gray bytes of a w*h region of the frame at t (one ffmpeg call)."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
         "-vf", f"crop={w}:{h}:{x}:{y},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout


@needs_render
def test_text_actually_drawn_and_fade_out_clears(tmp_path):
    # animation "none": text is fully visible, so the centre must differ from the
    # background corner.
    g = build_card_graph("MMMMMMMM", None, 1.0, 320, 180, 15,
                         CardStyle(font=FONT, animation="none", bg_color="black"),
                         tmp_path / "w")
    _render(g, tmp_path / "n.mp4")
    assert max(_pixels(tmp_path / "n.mp4", 0.5, 60, 60, 200, 60)) > 100
    assert max(_pixels(tmp_path / "n.mp4", 0.5, 0, 0, 20, 20)) < 40


@needs_render
def test_typewriter_reveals_progressively(tmp_path):
    g = build_card_graph("MMMMMMMMMM", None, 2.0, 320, 180, 15,
                         CardStyle(font=FONT, animation="typewriter",
                                   bg_color="black", fade_out=0), tmp_path / "w")
    _render(g, tmp_path / "t.mp4")

    def ink(t):
        return sum(b > 100 for b in _pixels(tmp_path / "t.mp4", t, 0, 70, 320, 40))
    assert ink(0.45) < ink(1.8)


@needs_render
def test_credits_render_both_modes(tmp_path):
    for mode in ("stack", "scroll"):
        g = credits_card(["Song - Artist", "Other - Band"], 1.0, 320, 180, 15,
                         CardStyle(font=FONT), tmp_path / mode, mode=mode)
        dur, v = _render(g, tmp_path / f"{mode}.mp4")
        assert (v["width"], v["height"]) == (320, 180)
        assert dur == pytest.approx(1.0, abs=0.12)


@needs_render
def test_lower_third_enable_window(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:s=320x180:r=15:d=1.6", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    g = lower_third("MMMMMMMM", 0.6, 1.2, width=320, height=180,
                    work_dir=tmp_path / "w", style=CardStyle(font=FONT),
                    in_label="0:v")
    dur, v = _render(g, tmp_path / "lt.mp4", [src], trim=False)
    assert dur == pytest.approx(1.6, abs=0.15)

    def ink(t):
        return max(_pixels(tmp_path / "lt.mp4", t, 0, 120, 320, 60))
    assert ink(0.3) < 30 and ink(0.9) > 100 and ink(1.45) < 30
