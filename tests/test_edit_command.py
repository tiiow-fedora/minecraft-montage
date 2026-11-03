"""Golden tests for the ffmpeg command line edit.build() constructs.

ffprobe is stubbed so the goldens depend only on our code, not on the ffmpeg
version installed; the clips just need to exist.
"""

import pytest

from golden import assert_golden
from mcmontage import edit


@pytest.fixture
def fake_probe(monkeypatch):
    table = {}

    def _probe(path):
        return table[path.name]

    monkeypatch.setattr(edit, "probe", _probe)

    def add(name, duration, width=1920, height=1080, fps=60.0, audio=False):
        table[name] = edit.Probe(path=name, duration=duration, width=width,
                                 height=height, fps=fps, has_audio=audio)
    return add


def mk(tmp_path, *names):
    out = []
    for n in names:
        p = tmp_path / "clips" / n
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x")
        out.append(p)
    return out


def norm(args, tmp_path):
    return [a.replace(str(tmp_path), "<TMP>") for a in args]


def build(tmp_path, clips, **kw):
    return norm(edit.build(clips, tmp_path / "out" / "final.mp4", dry_run=True, **kw),
                tmp_path)


def test_golden_single_clip_plain(tmp_path, fake_probe):
    fake_probe("a.mp4", 25.0)
    assert_golden("ffmpeg_single_plain.json", build(tmp_path, mk(tmp_path, "a.mp4")))


def test_golden_two_clips_crossfade(tmp_path, fake_probe):
    fake_probe("a.mp4", 25.0)
    fake_probe("b.mp4", 20.0)
    assert_golden("ffmpeg_two_clips.json",
                  build(tmp_path, mk(tmp_path, "a.mp4", "b.mp4"), xfade=1.0))


def test_golden_three_clips_mixed_sizes_one_music(tmp_path, fake_probe):
    fake_probe("a.mp4", 25.0)
    fake_probe("b.mp4", 20.0, width=1280, height=720, fps=30.0)
    fake_probe("c.mp4", 10.5)
    m = tmp_path / "m1.mp3"; m.write_bytes(b"x")
    assert_golden("ffmpeg_three_clips_music.json", build(
        tmp_path, mk(tmp_path, "a.mp4", "b.mp4", "c.mp4"), music=m, xfade=1.5))


def test_golden_multi_music_with_title(tmp_path, fake_probe):
    fake_probe("a.mp4", 40.0)
    ms = []
    for i in range(3):
        m = tmp_path / f"m{i}.mp3"; m.write_bytes(b"x"); ms.append(m)
    assert_golden("ffmpeg_multi_music_title.json", build(
        tmp_path, mk(tmp_path, "a.mp4"), music=ms, title="Day 1, Pt. 2: it's alive",
        subtitle="50% done", font="/fonts/test.ttf"))


def test_title_text_goes_via_files_not_filtergraph(tmp_path, fake_probe):
    fake_probe("a.mp4", 10.0)
    title = "It's 100%, really: \\ done"
    args = build(tmp_path, mk(tmp_path, "a.mp4"), title=title, font="/f.ttf")
    graph = args[args.index("-filter_complex") + 1]
    for ch_run in ("really", "100%", "It's"):
        assert ch_run not in graph
    assert (tmp_path / "out" / ".mcmontage_tmp" / "_title.txt").read_text(
        encoding="utf-8") == title


def test_single_music_track_is_trimmed_to_video_length(tmp_path, fake_probe):
    fake_probe("a.mp4", 12.0)
    m = tmp_path / "m.mp3"; m.write_bytes(b"x")
    args = build(tmp_path, mk(tmp_path, "a.mp4"), music=[m])
    g = args[args.index("-filter_complex") + 1]
    assert "atrim=0:12.000" in g
    assert "loudnorm" not in g          # one track needs no normalising
    assert args[args.index("-t") + 1] == "12.000"


def test_total_length_deducts_each_crossfade(tmp_path, fake_probe):
    for n, d in (("a.mp4", 10.0), ("b.mp4", 8.0), ("c.mp4", 6.0)):
        fake_probe(n, d)
    args = build(tmp_path, mk(tmp_path, "a.mp4", "b.mp4", "c.mp4"), xfade=2.0)
    assert args[args.index("-t") + 1] == "20.000"   # 24 - 2*2


def test_xfade_offsets_chain(tmp_path, fake_probe):
    for n, d in (("a.mp4", 10.0), ("b.mp4", 8.0), ("c.mp4", 6.0)):
        fake_probe(n, d)
    g = build(tmp_path, mk(tmp_path, "a.mp4", "b.mp4", "c.mp4"), xfade=2.0)
    g = g[g.index("-filter_complex") + 1]
    assert "xfade=transition=fade:duration=2.0:offset=8.000[x1]" in g
    assert "[x1][v2]xfade=transition=fade:duration=2.0:offset=14.000[x2]" in g


def test_no_music_means_no_audio_stream(tmp_path, fake_probe):
    fake_probe("a.mp4", 5.0)
    args = build(tmp_path, mk(tmp_path, "a.mp4"))
    assert "-an" in args and "aac" not in args


def test_every_music_input_is_looped(tmp_path, fake_probe):
    fake_probe("a.mp4", 5.0)
    ms = []
    for i in range(2):
        m = tmp_path / f"m{i}.mp3"; m.write_bytes(b"x"); ms.append(m)
    args = build(tmp_path, mk(tmp_path, "a.mp4"), music=ms)
    assert args.count("-stream_loop") == 2


def test_missing_inputs_raise(tmp_path, fake_probe):
    with pytest.raises(edit.EditError):
        edit.build([], tmp_path / "o.mp4", dry_run=True)
    with pytest.raises(edit.EditError):
        edit.build([tmp_path / "nope.mp4"], tmp_path / "o.mp4", dry_run=True)
    fake_probe("a.mp4", 5.0)
    with pytest.raises(edit.EditError):
        edit.build(mk(tmp_path, "a.mp4"), tmp_path / "o.mp4",
                   music=tmp_path / "nope.mp3", dry_run=True)


def test_dry_run_does_not_create_output(tmp_path, fake_probe):
    fake_probe("a.mp4", 5.0)
    edit.build(mk(tmp_path, "a.mp4"), tmp_path / "out" / "o.mp4", dry_run=True)
    assert not (tmp_path / "out" / "o.mp4").exists()


def test_describe_command_quotes():
    assert edit.describe_command(["ffmpeg", "-i", "a b.mp4"]) == "ffmpeg -i 'a b.mp4'"


def test_esc_path_windows():
    assert edit._esc_path(r"C:\Windows\Fonts\arialbd.ttf") == r"C\:/Windows/Fonts/arialbd.ttf"


def test_esc_path_apostrophe_closes_and_reopens_quote():
    assert edit._esc_path("/home/O'Brien/x") == "/home/O'\\\\\\''Brien/x"


def test_xfade_zero_uses_concat_not_xfade(tmp_path, fake_probe):
    fake_probe("a.mp4", 5.0); fake_probe("b.mp4", 5.0)
    args = build(tmp_path, mk(tmp_path, "a.mp4", "b.mp4"), xfade=0)
    g = args[args.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0" in g and "xfade=" not in g
    assert args[args.index("-t") + 1] == "10.000"


def test_music_crossfade_is_capped_for_short_videos(tmp_path, fake_probe):
    fake_probe("a.mp4", 2.0)
    ms = []
    for i in range(3):
        m = tmp_path / f"m{i}.mp3"; m.write_bytes(b"x"); ms.append(m)
    args = build(tmp_path, mk(tmp_path, "a.mp4"), music=ms, music_xfade=3.0)
    g = args[args.index("-filter_complex") + 1]
    assert "acrossfade=d=1.0" in g and "acrossfade=d=3.0" not in g
