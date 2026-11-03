import datetime as dt

import pytest

import synthetic as syn
from mcmontage import sessions


def rec(tmp_path, name, when, minutes=10):
    return syn.make_mcpr(tmp_path / name, date=when, duration_ms=minutes * 60_000)


def D(day, hour, minute=0, month=9):
    return dt.datetime(2026, month, day, hour, minute)


def test_play_day_cutoff():
    assert sessions.play_day(D(2, 3, 59)) == dt.date(2026, 9, 1)
    assert sessions.play_day(D(2, 4, 0)) == dt.date(2026, 9, 2)
    assert sessions.play_day(D(2, 0, 0)) == dt.date(2026, 9, 1)
    # month boundary
    assert sessions.play_day(dt.datetime(2026, 10, 1, 1)) == dt.date(2026, 9, 30)
    assert sessions.play_day(D(2, 3), cutoff_hour=0) == dt.date(2026, 9, 2)


def test_read_clip_uses_metadata_not_mtime(tmp_path):
    import os
    p = rec(tmp_path, "a.mcpr", D(1, 20), minutes=7)
    os.utime(p, (0, 0))
    c = sessions.read_clip(p)
    assert c.recorded == D(1, 20)
    assert c.minutes == 7
    assert c.ends == D(1, 20, 7)


@pytest.mark.parametrize("kwargs", [
    dict(omit_metadata=True),
    dict(date=None),
    dict(metadata={"date": 0}),
])
def test_read_clip_errors(tmp_path, kwargs):
    p = syn.make_mcpr(tmp_path / "a.mcpr", **kwargs)
    with pytest.raises(sessions.SessionError):
        sessions.read_clip(p)


def test_read_clip_not_a_zip(tmp_path):
    p = tmp_path / "a.mcpr"
    p.write_bytes(b"nope")
    with pytest.raises(sessions.SessionError):
        sessions.read_clip(p)


def test_read_clip_missing_duration_is_zero(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr", metadata={"duration": None})
    assert sessions.read_clip(p).duration_ms == 0


def test_group_past_midnight_is_one_session(tmp_path):
    paths = [rec(tmp_path, "a.mcpr", D(1, 22)),
             rec(tmp_path, "b.mcpr", D(1, 23, 30)),
             rec(tmp_path, "c.mcpr", D(2, 1, 15))]
    g = sessions.group(paths)
    assert len(g) == 1 and [c.path.name for c in g[0].clips] == ["a.mcpr", "b.mcpr", "c.mcpr"]
    assert g[0].day == dt.date(2026, 9, 1)


def test_group_splits_days_and_orders_chronologically(tmp_path):
    paths = [rec(tmp_path, "z.mcpr", D(3, 20)),     # given out of order
             rec(tmp_path, "a.mcpr", D(1, 20)),
             rec(tmp_path, "m.mcpr", D(1, 21))]
    g = sessions.group(paths)
    assert [s.index for s in g] == [1, 2]
    assert [[c.path.name for c in s.clips] for s in g] == [["a.mcpr", "m.mcpr"], ["z.mcpr"]]


def test_group_empty():
    assert sessions.group([]) == []


def test_group_max_gap_splits_same_day(tmp_path):
    paths = [rec(tmp_path, "a.mcpr", D(1, 10), minutes=10),
             rec(tmp_path, "b.mcpr", D(1, 10, 15), minutes=10),   # 5 min gap
             rec(tmp_path, "c.mcpr", D(1, 20), minutes=10)]       # ~9.5 h gap
    assert len(sessions.group(paths)) == 1
    g = sessions.group(paths, max_gap_minutes=30)
    assert [len(s.clips) for s in g] == [2, 1]
    assert g[0].day == g[1].day


def test_naming_scheme(tmp_path):
    paths = [rec(tmp_path, "x.mcpr", D(1, 20)), rec(tmp_path, "y.mcpr", D(1, 21)),
             rec(tmp_path, "w.mcpr", D(2, 20))]
    g = sessions.group(paths)
    assert [g[0].name_for(c) for c in g[0].clips] == ["day1_1", "day1_2"]
    assert g[1].name_for(g[1].clips[0]) == "day2_1"
    assert g[1].name_for(g[1].clips[0], "{date}_{n}") == "2026-09-02_1"


def test_rename_plan_skips_already_correct(tmp_path):
    a = rec(tmp_path, "day1_1.mcpr", D(1, 20))
    b = rec(tmp_path, "weird.mcpr", D(1, 21))
    plan = sessions.rename_plan(sessions.group([a, b]))
    assert plan == [(b, tmp_path / "day1_2.mcpr")]


def test_apply_renames_handles_swaps(tmp_path):
    a = rec(tmp_path, "day1_2.mcpr", D(1, 20))   # earliest but named _2
    b = rec(tmp_path, "day1_1.mcpr", D(1, 21))   # later but named _1
    a_bytes, b_bytes = a.read_bytes(), b.read_bytes()
    plan = sessions.rename_plan(sessions.group([a, b]))
    assert len(plan) == 2
    sessions.apply_renames(plan)
    assert (tmp_path / "day1_1.mcpr").read_bytes() == a_bytes
    assert (tmp_path / "day1_2.mcpr").read_bytes() == b_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["day1_1.mcpr", "day1_2.mcpr"]


def test_apply_renames_missing_source_changes_nothing(tmp_path):
    a = rec(tmp_path, "a.mcpr", D(1, 20))
    ghost = tmp_path / "ghost.mcpr"
    with pytest.raises(sessions.SessionError):
        sessions.apply_renames([(a, tmp_path / "n1.mcpr"), (ghost, tmp_path / "n2.mcpr")])
    assert a.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mcpr"]


def test_rename_does_not_clobber_unrelated_file(tmp_path):
    """A file named like a target but NOT part of the batch must not be silently
    overwritten (os.rename overwrites on POSIX)."""
    a = rec(tmp_path, "weird.mcpr", D(1, 20))
    bystander = tmp_path / "day1_1.mcpr"
    bystander.write_bytes(b"irreplaceable")
    plan = sessions.rename_plan(sessions.group([a]))
    with pytest.raises(sessions.SessionError, match="will not overwrite"):
        sessions.apply_renames(plan)
    assert bystander.read_bytes() == b"irreplaceable"
    assert a.exists(), "refusal must happen before anything is moved"


def test_apply_renames_rolls_back_on_midway_oserror(tmp_path, monkeypatch):
    a = rec(tmp_path, "a.mcpr", D(1, 20))
    b = rec(tmp_path, "b.mcpr", D(1, 21))
    plan = sessions.rename_plan(sessions.group([a, b]))
    from pathlib import Path
    real = Path.rename
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 4:          # 2 parks + 1 placement succeed, then fail
            raise PermissionError("locked")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(PermissionError):
        sessions.apply_renames(plan)
    monkeypatch.setattr(Path, "rename", real)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mcpr", "b.mcpr"]


def test_apply_renames_rejects_duplicate_targets(tmp_path):
    a = rec(tmp_path, "a.mcpr", D(1, 20))
    b = rec(tmp_path, "b.mcpr", D(1, 21))
    with pytest.raises(sessions.SessionError):
        sessions.apply_renames([(a, tmp_path / "x.mcpr"), (b, tmp_path / "x.mcpr")])
    assert a.exists() and b.exists()


def test_session_totals(tmp_path):
    g = sessions.group([rec(tmp_path, "a.mcpr", D(1, 20), 10),
                        rec(tmp_path, "b.mcpr", D(1, 21), 5)])[0]
    assert g.total_minutes == 15
    assert g.started == D(1, 20) and g.ended == D(1, 21, 5)
