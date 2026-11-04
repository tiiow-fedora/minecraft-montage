"""assemble() with title / credits / lower-third extras (single ffmpeg run)."""
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from mcmontage import auto, beatsync, edit, titlecards

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                reason="ffmpeg missing")

HAVE_TEXT = (bool(shutil.which("ffmpeg")) and titlecards.drawtext_available()
             and titlecards.find_font() is not None)
needs_text = pytest.mark.skipif(not HAVE_TEXT, reason="drawtext or a font is missing")


def make_cfg(tmp_path):
    return SimpleNamespace(work=tmp_path / "work", videos=tmp_path / "videos",
                           output=tmp_path / "out", width=320, height=180, fps=30,
                           bitrate=1_000_000, xfade=0.5, crf=30,
                           music_gain_db=-16.0, music_fade_out=1.0)


def make_clip(path, color="0x303030", size="160x90", rate=15, dur=3):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c={color}:s={size}:r={rate}:d={dur}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def clips3(tmp_path):
    return [make_clip(tmp_path / "a.mp4", "red"),
            make_clip(tmp_path / "b.mp4", "green", "120x120", 24),     # odd one out
            make_clip(tmp_path / "c.mp4", "blue", "200x90", 10)]


@pytest.fixture
def music(tmp_path):
    p = tmp_path / "m.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=4", str(p)], check=True)
    return p


def frame(path, t, size="80x45"):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(path),
                          "-frames:v", "1", "-vf", f"scale={size},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    assert out, f"no frame at {t}s"
    return out


def mad(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def chains_of(args):
    fc = args[args.index("-filter_complex") + 1]
    return re.split(r"(?<!\\);", fc)


def check_graph(args, n_inputs):
    """Every chain input is a real input pad or an earlier chain's output, each
    label is produced exactly once and consumed at most once, and the mapped
    output exists."""
    produced, consumed = {}, []
    for ch in chains_of(args):
        m = re.match(r"((?:\[[^\]]+\])*)", ch)
        ins = re.findall(r"\[([^\]]+)\]", m.group(1))
        outs = re.findall(r"\[([^\]]+)\]$", ch)
        for i in ins:
            if re.fullmatch(r"\d+:[va]", i):
                assert int(i.split(":")[0]) < n_inputs
            else:
                assert i in produced, f"{i} used before it is produced"
                consumed.append(i)
        for o in outs:
            assert o not in produced, f"duplicate label {o}"
            produced[o] = ch
    assert len(consumed) == len(set(consumed)), "a label is consumed twice"
    for i, a in enumerate(args):
        if a == "-map":
            assert args[i + 1].strip("[]") in produced
    return produced


# ---------------------------------------------------------------- dry runs

def test_no_extras_graph_unchanged(tmp_path, clips3):
    args = auto.assemble(make_cfg(tmp_path), clips3, None, tmp_path / "o.mp4",
                         sync=False, dry_run=True)
    produced = check_graph(args, 3)
    assert set(produced) == {"v0", "v1", "v2", "x1", "x2"}
    assert "drawtext=" not in " ".join(args)
    assert not (tmp_path / "work").exists()          # nothing written either


def test_graph_with_all_extras_is_one_invocation(tmp_path, clips3, music):
    args = auto.assemble(make_cfg(tmp_path), clips3, music, tmp_path / "o.mp4", sync=False,
                         dry_run=True, title="My Montage", subtitle="by me",
                         credits=["Music:", "  Song by Artist"],
                         lower_thirds=[("Kill", 1.0, 2.0), ("Death", 3.0, 4.0)],
                         card_style=titlecards.CardStyle(font="/nonexistent.ttf"),
                         on_missing_font="blank")
    produced = check_graph(args, 4)           # 3 clips + music
    assert args.count("ffmpeg") == 1 and args.count("-filter_complex") == 1
    for label in ("tcout", "crout", "jin", "jout", "mx2"):
        assert label in produced
    # blank font: cards exist but nothing is drawn
    assert "drawtext=" not in " ".join(args)


@needs_text
def test_unique_prefixes_with_text(tmp_path, clips3):
    args = auto.assemble(make_cfg(tmp_path), clips3, None, tmp_path / "o.mp4", sync=False,
                         dry_run=True, title="T", subtitle="S", credits=["a", "b"],
                         lower_thirds=[("one", 1, 2), ("two", 1.5, 2.5)])
    produced = check_graph(args, 3)
    assert {"lt0out", "lt1out"} <= set(produced)
    assert args[args.index("-map") + 1] == "[lt1out]"          # overlay is last
    assert (tmp_path / "work" / "assemble" / "o").is_dir()


def test_total_duration_includes_cards(tmp_path, clips3, music):
    args = auto.assemble(make_cfg(tmp_path), clips3, music, tmp_path / "o.mp4", sync=False,
                         dry_run=True, title="T", credits=["x"], intro_s=2, outro_s=2.5,
                         on_missing_font="blank")
    # 2 + 3*3 + 2.5 - 4 transitions of 0.5
    assert float(args[args.index("-t") + 1]) == pytest.approx(11.5)
    fc = args[args.index("-filter_complex") + 1]
    assert "atrim=0:11.500" in fc          # music bed covers cards too


def test_beat_sync_applies_to_clips_not_cards(tmp_path, monkeypatch):
    clips = [make_clip(tmp_path / f"s{i}.mp4", dur=6) for i in range(3)]
    music = tmp_path / "m.wav"
    music.write_bytes(b"x")
    beats = [i * 0.5 for i in range(1, 200)]
    monkeypatch.setattr(beatsync, "load_or_detect",
                        lambda p: beatsync.BeatGrid(120.0, beats, 0.9))
    args = auto.assemble(make_cfg(tmp_path), clips, music, tmp_path / "o.mp4",
                         dry_run=True, title="T", credits=["x"], intro_s=2, outro_s=3,
                         on_missing_font="blank")
    chains = chains_of(args)
    offs = {}
    for ch in chains:
        o, lab = re.search(r"offset=([\d.]+)", ch), re.search(r"\[(\w+)\]$", ch)
        if o and lab:
            offs[lab.group(1)] = float(o.group(1))
    start = 2 - 0.5                      # montage starts when the title fades out
    for k in ("mx1", "mx2"):
        r = (offs[k] + start) % 0.5
        assert min(r, 0.5 - r) < 2e-3, (k, offs[k])
    assert offs["jin"] == pytest.approx(1.5)         # card keeps its own length
    # montage is not stretched to the music: clips only lose/keep footage
    assert float(args[args.index("-t") + 1]) <= 2 + 18 + 3 - 3 * 0.5 + 1e-6


def test_beat_sync_falls_back_when_music_has_too_few_beats(tmp_path, monkeypatch, clips3):
    music = tmp_path / "m.wav"
    music.write_bytes(b"x")
    monkeypatch.setattr(beatsync, "load_or_detect",
                        lambda p: beatsync.BeatGrid(120.0, [0.5, 1.0], 0.9))
    args = auto.assemble(make_cfg(tmp_path), clips3, music, tmp_path / "o.mp4", dry_run=True)
    assert float(args[args.index("-t") + 1]) == pytest.approx(9 - 1.0)   # natural lengths


# ----------------------------------------------------------- font handling

def test_missing_font_is_autoerror(tmp_path, clips3, monkeypatch):
    monkeypatch.setattr(titlecards, "find_font", lambda *a, **k: None)
    monkeypatch.setattr(titlecards, "drawtext_available", lambda *a, **k: True)
    with pytest.raises(auto.AutoError, match="font"):
        auto.assemble(make_cfg(tmp_path), clips3, None, tmp_path / "o.mp4", sync=False,
                      dry_run=True, title="Hi")
    with pytest.raises(auto.AutoError, match="font"):
        auto.assemble(make_cfg(tmp_path), clips3, None, tmp_path / "o.mp4", sync=False,
                      dry_run=True, lower_thirds=[("x", 1, 2)])
    args = auto.assemble(make_cfg(tmp_path), clips3, None, tmp_path / "o.mp4", sync=False,
                         dry_run=True, title="Hi", credits=["c"],
                         lower_thirds=[("x", 1, 2)], on_missing_font="blank")
    assert "drawtext=" not in " ".join(args)
    check_graph(args, 3)


def test_no_drawtext_filter_is_autoerror_or_blank(tmp_path, clips3, monkeypatch):
    monkeypatch.setattr(titlecards, "drawtext_available", lambda *a, **k: False)
    with pytest.raises(auto.AutoError, match="drawtext"):
        auto.assemble(make_cfg(tmp_path), clips3, None, tmp_path / "o.mp4", sync=False,
                      dry_run=True, title="Hi")
    args = auto.assemble(make_cfg(tmp_path), clips3, None, tmp_path / "o.mp4", sync=False,
                         dry_run=True, title="Hi", lower_thirds=[("x", 1, 2)],
                         on_missing_font="blank")
    assert "drawtext=" not in " ".join(args)


def test_argument_validation(tmp_path, clips3):
    cfg, out = make_cfg(tmp_path), tmp_path / "o.mp4"
    kw = dict(sync=False, dry_run=True, on_missing_font="blank")
    with pytest.raises(auto.AutoError):
        auto.assemble(cfg, clips3, None, out, subtitle="orphan", **kw)
    with pytest.raises(auto.AutoError):
        auto.assemble(cfg, clips3, None, out, lower_thirds=[("x", 2, 1)], **kw)
    with pytest.raises(auto.AutoError):
        auto.assemble(cfg, clips3, None, out, lower_thirds=[("x", 99, 100)], **kw)
    with pytest.raises(auto.AutoError):
        auto.assemble(cfg, [], None, out, **kw)
    with pytest.raises(auto.AutoError):
        auto.assemble(cfg, clips3, None, out, title="t", intro_s=0, **kw)


def test_single_clip_with_cards_dry(tmp_path):
    one = make_clip(tmp_path / "one.mp4")
    args = auto.assemble(make_cfg(tmp_path), [one], None, tmp_path / "o.mp4", sync=False,
                         dry_run=True, title="T", credits=["c"], intro_s=2, outro_s=2,
                         on_missing_font="blank")
    check_graph(args, 1)
    assert float(args[args.index("-t") + 1]) == pytest.approx(2 + 3 + 2 - 1.0)


# ------------------------------------------------------ real ffmpeg renders

def test_render_mixed_clips_with_cards_duration(tmp_path, clips3, music):
    out = tmp_path / "o.mp4"
    auto.assemble(make_cfg(tmp_path), clips3, music, out, sync=False, title="Title",
                  subtitle="sub", credits=["Music:", "  Song by Artist"],
                  intro_s=2, outro_s=2.5, on_missing_font="blank")
    p = edit.probe(out)
    assert p.has_audio
    assert p.duration == pytest.approx(11.5, abs=0.3)
    assert (p.width, p.height) == (160, 90)              # first clip's size
    assert edit.probe_audio(out) == pytest.approx(11.5, abs=0.4)   # looped music bed


def test_render_single_clip_with_title(tmp_path):
    one = make_clip(tmp_path / "one.mp4")
    out = tmp_path / "o.mp4"
    auto.assemble(make_cfg(tmp_path), [one], None, out, sync=False, title="Solo",
                  intro_s=2, on_missing_font="blank")
    assert edit.probe(out).duration == pytest.approx(2 + 3 - 0.5, abs=0.3)


@needs_text
def test_render_real_text_duration_and_title_frame(tmp_path):
    clips = [make_clip(tmp_path / f"c{i}.mp4", dur=3) for i in range(2)]
    cfg = make_cfg(tmp_path)
    shown, blank = tmp_path / "shown.mp4", tmp_path / "blank.mp4"
    kw = dict(sync=False, title="HELLO WORLD", subtitle="a subtitle",
              credits=["Music:", "  Song by Artist"], intro_s=2, outro_s=2)
    auto.assemble(cfg, clips, None, shown, **kw)
    blank_style = titlecards.CardStyle(font="/nonexistent.ttf", on_missing_font="blank")
    auto.assemble(cfg, clips, None, blank, card_style=blank_style, **kw)
    expect = 2 + 3 + 3 + 2 - 3 * 0.5      # 2 cards + 2 clips, 3 crossfades
    for f in (shown, blank):
        assert edit.probe(f).duration == pytest.approx(expect, abs=0.3)
    # title fully faded in at t=1.0: text pixels differ from the text-less card
    assert mad(frame(shown, 1.0), frame(blank, 1.0)) > 1.0
    # credits at the very end likewise
    assert mad(frame(shown, expect - 0.8), frame(blank, expect - 0.8)) > 1.0
    # mid-montage the two renders agree: the cards did not leak into the clips
    assert mad(frame(shown, 4.0), frame(blank, 4.0)) < 0.5


@needs_text
def test_render_lower_third_frame_differs_only_inside_window(tmp_path):
    clip = make_clip(tmp_path / "c.mp4", dur=6)
    cfg = make_cfg(tmp_path)
    with_lt, without = tmp_path / "lt.mp4", tmp_path / "plain.mp4"
    auto.assemble(cfg, [clip], None, with_lt, sync=False,
                  lower_thirds=[("PLAYER KILL: ZOMBIE", 1.5, 3.5)])
    auto.assemble(cfg, [clip], None, without, sync=False)
    assert edit.probe(with_lt).duration == pytest.approx(edit.probe(without).duration, abs=0.2)
    assert mad(frame(with_lt, 2.5), frame(without, 2.5)) > 1.0      # inside window
    assert mad(frame(with_lt, 5.0), frame(without, 5.0)) < 0.5      # after it
    assert mad(frame(with_lt, 0.5), frame(without, 0.5)) < 0.5      # before it


@needs_text
def test_render_lower_third_after_title_uses_final_timeline(tmp_path):
    clip = make_clip(tmp_path / "c.mp4", dur=6)
    cfg = make_cfg(tmp_path)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    kw = dict(sync=False, title="T", intro_s=2)
    auto.assemble(cfg, [clip], None, a, lower_thirds=[("LOWER", 4.0, 6.0)], **kw)
    auto.assemble(cfg, [clip], None, b, **kw)
    assert mad(frame(a, 5.0), frame(b, 5.0)) > 1.0
    assert mad(frame(a, 3.0), frame(b, 3.0)) < 0.5
