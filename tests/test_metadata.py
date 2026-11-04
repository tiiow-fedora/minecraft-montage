import datetime

import pytest

from mcmontage import metadata as md


def clips(*durs, names=None):
    names = names or [f"Part {i}" for i in range(1, len(durs) + 1)]
    return [md.ClipInfo(n, d) for n, d in zip(names, durs)]


@pytest.mark.parametrize("s,out", [(0, "0:00"), (59.9, "0:59"), (60, "1:00"),
                                   (3599, "59:59"), (3600, "1:00:00"),
                                   (3725, "1:02:05"), (-5, "0:00")])
def test_fmt_timestamp(s, out):
    assert md.fmt_timestamp(s) == out


def test_clip_starts_account_for_crossfade():
    assert md.clip_starts([20, 20, 20], 1.0) == [0, 19, 38]
    assert md.total_duration([20, 20, 20], 1.0) == 58
    assert md.clip_starts([20, 20], 0) == [0, 20]
    assert md.total_duration([], 1) == 0


def test_chapters_basic():
    ch = md.make_chapters(clips(30, 30, 30, names=["Intro cut", "Farm", "Nether"]), 1.0)
    assert [c.start for c in ch] == [0, 29, 58]
    assert [c.title for c in ch] == ["Intro cut", "Farm", "Nether"]
    assert md.chapters_ok(ch, md.total_duration([30, 30, 30], 1.0)) == []


def test_first_generic_part_is_labelled_intro():
    ch = md.make_chapters(clips(30, 30, 30))
    assert ch[0].title == "Intro" and ch[1].title == "Part 2"


def test_too_few_chapters_returns_empty():
    assert md.make_chapters(clips(60, 60)) == []
    assert md.make_chapters([]) == []


def test_short_clips_fold_into_previous():
    ch = md.make_chapters(clips(30, 5, 30, 30), 0)
    assert [c.start for c in ch] == [0, 35, 65]
    assert md.chapters_ok(ch, 95) == []


def test_short_first_clip_folds_following():
    ch = md.make_chapters(clips(4, 30, 30, 30), 0)
    assert ch[0].start == 0 and ch[1].start >= 10
    assert md.chapters_ok(ch, 94) == []


def test_short_tail_dropped():
    ch = md.make_chapters(clips(30, 30, 30, 4), 0)
    assert len(ch) == 3
    assert md.chapters_ok(ch, 94) == []


def test_chapters_ok_reports_violations():
    bad = [md.Chapter(1, "a"), md.Chapter(5, "b")]
    probs = md.chapters_ok(bad, 8)
    assert any("0:00" in p for p in probs)
    assert any("at least" in p for p in probs)
    assert any("shorter" in p for p in probs)
    assert md.chapters_ok([], 10) == ["no chapters"]


def test_chapters_block_format():
    s = md.chapters_block([md.Chapter(0, "A"), md.Chapter(75, "B"), md.Chapter(3700, "C")])
    assert s.splitlines() == ["0:00 A", "1:15 B", "1:01:40 C"]


@pytest.mark.parametrize("stem,idx,out", [
    ("day2_3", 3, "Part 3"), ("12", 1, "Part 1"), ("base_tour-final", 1, "Base tour"),
    ("", 2, "Part 2"), ("farm_render", 1, "Farm")])
def test_label_from_filename(stem, idx, out):
    assert md.label_from_filename(stem, idx) == out


def test_title_variants_and_limit():
    ctx = md.Context(theme="Building the Nether Hub", series="Survival Diaries",
                     session_index=3)
    assert md.generate_title(ctx) == "Building the Nether Hub | Survival Diaries – Day 3"
    assert md.generate_title(md.Context(world="Longhaul")) == "Exploring Longhaul"
    assert md.generate_title(md.Context()) == "Minecraft Montage"
    long = md.Context(theme="x" * 95, series="Series", session_index=2)
    t = md.generate_title(long)
    assert len(t) <= md.MAX_TITLE and t.startswith("x")
    assert len(md.generate_title(md.Context(theme="y" * 300))) <= md.MAX_TITLE
    mid = md.Context(theme="x" * 80, series="S" * 30, session_index=2)
    assert md.generate_title(mid) == "x" * 80 + " | Day 2"


def test_tags_dedupe_and_cap():
    ctx = md.Context(world="Longhaul", series="longhaul", extra_tags=["Farm", "farm", " "])
    tags = md.generate_tags(ctx)
    assert tags.count("longhaul") == 1 and tags.count("farm") == 1
    assert sum(len(t) for t in md.generate_tags(
        md.Context(extra_tags=["a" * 100] * 20))) <= 450


def test_description_contains_chapters_music_and_date():
    from pathlib import Path
    from mcmontage import music
    track = music.Track(id="t", file=Path("x.mp3"), title="Song", artist="Artist",
                        source="test", license="CC-BY-4.0")
    ctx = md.Context(theme="Farm day", world="W", day=datetime.date(2026, 9, 8),
                     clips=clips(30, 30, 30), music=[track])
    d = md.generate_description(ctx)
    assert "Farm day — recorded 8 September 2026." in d
    assert "Chapters:\n0:00 Intro\n0:30 Part 2\n1:00 Part 3" in d
    assert d.rstrip().endswith("#Minecraft #Montage")
    assert "Music:" in d and '"Song" by Artist' in d


def test_description_without_anything():
    assert md.generate_description(md.Context(), hashtags=False) == ""
    assert md.generate_description(md.Context(), intro="Hi", hashtags=False) == "Hi"
