"""Real ffmpeg, tiny synthetic clips.  Skipped when ffmpeg is not installed."""

import pytest

import synthetic as syn
from mcmontage import edit

pytestmark = pytest.mark.ffmpeg

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def find_font():
    import os
    for f in FONT_CANDIDATES:
        if os.path.isfile(f):
            return f
    pytest.skip("no usable TTF font found for drawtext")


def has_drawtext():
    import subprocess
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                         capture_output=True, text=True).stdout
    return "drawtext" in out


def test_probe_reads_synthetic_clip(tmp_path):
    c = syn.make_clip(tmp_path / "a.mp4", seconds=2, size=(320, 180), fps=30)
    p = edit.probe(c)
    assert (p.width, p.height) == (320, 180)
    assert p.fps == pytest.approx(30)
    assert p.duration == pytest.approx(2, abs=0.2)
    assert p.has_audio is False


def test_probe_audio_with_and_without_cover_art(tmp_path):
    plain = syn.make_tone(tmp_path / "plain.mp3", 2)
    art = syn.make_tone(tmp_path / "art.mp3", 2, with_cover_art=True)
    assert edit.probe_audio(plain) == pytest.approx(2, abs=0.3)
    assert edit.probe_audio(art) == pytest.approx(2, abs=0.3)


def test_probe_errors(tmp_path):
    with pytest.raises(edit.EditError):
        edit.probe(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(edit.EditError):
        edit.probe(junk)
    tone = syn.make_tone(tmp_path / "t.wav", 1)
    with pytest.raises(edit.EditError, match="no video stream"):
        edit.probe(tone)


def test_single_clip_roundtrip(tmp_path):
    c = syn.make_clip(tmp_path / "a.mp4", seconds=3)
    out = tmp_path / "out" / "final.mp4"
    edit.build([c], out, crf=35, preset="ultrafast")
    p = edit.probe(out)
    assert p.duration == pytest.approx(3, abs=0.3)
    assert (p.width, p.height) == (320, 180)
    assert not p.has_audio


def test_two_clips_duration_minus_crossfade(tmp_path):
    a = syn.make_clip(tmp_path / "a.mp4", seconds=3)
    b = syn.make_clip(tmp_path / "b.mp4", seconds=3, pattern="testsrc")
    out = tmp_path / "final.mp4"
    edit.build([a, b], out, xfade=1.0, crf=35, preset="ultrafast")
    assert edit.probe(out).duration == pytest.approx(5, abs=0.3)


def test_mixed_resolution_clips_are_letterboxed_to_first(tmp_path):
    a = syn.make_clip(tmp_path / "a.mp4", seconds=2, size=(320, 180), fps=30)
    b = syn.make_clip(tmp_path / "b.mp4", seconds=2, size=(160, 160), fps=25)
    out = tmp_path / "final.mp4"
    edit.build([a, b], out, xfade=0.5, crf=35, preset="ultrafast")
    p = edit.probe(out)
    assert (p.width, p.height) == (320, 180)
    assert p.fps == pytest.approx(30)


@pytest.mark.parametrize("n_tracks", [1, 2, 3])
def test_music_bed_matches_video_length(tmp_path, n_tracks):
    a = syn.make_clip(tmp_path / "a.mp4", seconds=6)
    tracks = [syn.make_tone(tmp_path / f"m{i}.wav", 1.5, freq=300 + 100 * i)
              for i in range(n_tracks)]       # shorter than their share: must loop
    out = tmp_path / "final.mp4"
    edit.build([a], out, music=tracks, music_xfade=0.5, crf=35, preset="ultrafast")
    p = edit.probe(out)
    assert p.has_audio
    assert p.duration == pytest.approx(6, abs=0.4)
    assert edit.probe_audio(out) == pytest.approx(6, abs=0.4)


def test_music_as_single_path_not_list(tmp_path):
    a = syn.make_clip(tmp_path / "a.mp4", seconds=2)
    m = syn.make_tone(tmp_path / "m.wav", 1)
    out = tmp_path / "final.mp4"
    edit.build([a], out, music=m, crf=35, preset="ultrafast")
    assert edit.probe(out).has_audio


def test_title_with_awkward_characters_renders(tmp_path):
    if not has_drawtext():
        pytest.skip("ffmpeg lacks drawtext")
    a = syn.make_clip(tmp_path / "a.mp4", seconds=3)
    out = tmp_path / "out" / "final.mp4"
    edit.build([a], out, title="Day 1, it's 100%: \\ \u00e9\u00e8 \u2603",
               subtitle="a, b: c's", font=find_font(), crf=35, preset="ultrafast")
    assert edit.probe(out).duration == pytest.approx(3, abs=0.3)


@pytest.mark.parametrize("dirname", [
    "O'Brien", "my videos", "a[1]", "semi;colon", "eq=ual", "com,ma", "per%cent",
    "back'to'back",
])
def test_title_works_with_awkward_output_dir(tmp_path, dirname):
    """textfile='<output dir>/...' sits inside filtergraph quoting (e.g. a
    Windows user called O'Brien)."""
    if not has_drawtext():
        pytest.skip("ffmpeg lacks drawtext")
    a = syn.make_clip(tmp_path / "a.mp4", seconds=2)
    out = tmp_path / dirname / "final.mp4"
    edit.build([a], out, title="Hello", subtitle="there", font=find_font(),
               crf=35, preset="ultrafast")
    assert out.is_file()


def test_ffmpeg_failure_is_reported_as_edit_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    with pytest.raises(edit.EditError):
        edit.build([bad], tmp_path / "o.mp4")


def test_xfade_zero_is_a_hard_cut_not_a_dropped_clip(tmp_path):
    a = syn.make_clip(tmp_path / "a.mp4", seconds=2)
    b = syn.make_clip(tmp_path / "b.mp4", seconds=2, pattern="testsrc")
    out = tmp_path / "final.mp4"
    edit.build([a, b], out, xfade=0, crf=35, preset="ultrafast")
    assert edit.probe(out).duration == pytest.approx(4, abs=0.3)


def test_short_video_with_several_tracks_does_not_hang(tmp_path):
    """share per track < music_xfade used to make acrossfade spin forever."""
    import subprocess
    real_run = subprocess.run

    def run_with_timeout(*a, **kw):
        kw.setdefault("timeout", 60)
        return real_run(*a, **kw)

    a = syn.make_clip(tmp_path / "a.mp4", seconds=2)
    tracks = [syn.make_tone(tmp_path / f"m{i}.wav", 5) for i in range(3)]
    out = tmp_path / "final.mp4"
    import unittest.mock as mock
    with mock.patch.object(edit.subprocess, "run", run_with_timeout):
        edit.build([a], out, music=tracks, crf=35, preset="ultrafast")
    p = edit.probe(out)
    assert p.has_audio and p.duration == pytest.approx(2, abs=0.4)
