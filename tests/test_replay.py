import json
import zipfile

import pytest

import synthetic as syn
from mcmontage import replay


def test_inspect_reports_metadata(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr", duration_ms=123_456, server_name="srv")
    info = replay.inspect(p)
    assert info.duration_ms == 123_456
    assert info.server_name == "srv"
    assert not info.has_render_queue and not info.has_timelines
    assert set(info.entries) == {"metaData.json", "recording.tmcpr"}


def test_inspect_flags_present_entries(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr", render_queue=[], timelines={})
    info = replay.inspect(p)
    assert info.has_render_queue and info.has_timelines


def test_inspect_missing_file(tmp_path):
    with pytest.raises(replay.ReplayError):
        replay.inspect(tmp_path / "nope.mcpr")


def test_inspect_tolerates_missing_or_bad_metadata(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr", omit_metadata=True)
    info = replay.inspect(p)
    assert info.metadata == {} and info.duration_ms is None

    p2 = syn.make_mcpr(tmp_path / "b.mcpr", omit_metadata=True,
                       extra_entries={"metaData.json": "[1,2,3]"})
    assert replay.inspect(p2).metadata == {}

    p3 = syn.make_mcpr(tmp_path / "c.mcpr", omit_metadata=True,
                       extra_entries={"metaData.json": "   "})
    assert replay.inspect(p3).metadata == {}


def test_read_render_queue_absent_and_empty(tmp_path):
    assert replay.read_render_queue(syn.make_mcpr(tmp_path / "a.mcpr")) == []
    assert replay.read_render_queue(
        syn.make_mcpr(tmp_path / "b.mcpr", render_queue="")) == []


def test_read_render_queue_rejects_non_list(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr", render_queue={"oops": 1})
    with pytest.raises(replay.ReplayError):
        replay.read_render_queue(p)


def test_nested_timeline_is_a_json_string_on_disk(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr")
    job = {"timeline": {"": []}, "settings": {"videoWidth": 640}}
    replay.write_render_queue(p, [job], backup=False)
    with zipfile.ZipFile(p) as z:
        raw = json.loads(z.read("renderQueue.json"))
    assert isinstance(raw, list) and isinstance(raw[0]["timeline"], str)
    assert json.loads(raw[0]["timeline"]) == {"": []}
    assert raw[0]["settings"] == {"videoWidth": 640}
    assert set(raw[0]) == {"timeline", "settings"}


def test_roundtrip_preserves_raw_string_byte_for_byte(tmp_path):
    inner = '{"":[ {"keyframes":[],"segments":[],"interpolators":[]} ]}'  # odd spacing
    p = syn.make_mcpr(tmp_path / "a.mcpr",
                      render_queue=[{"timeline": inner, "settings": {"a": 1}}])
    jobs = replay.read_render_queue(p)
    assert jobs[0]["_timeline_raw"] == inner
    replay.write_render_queue(p, [{"timeline": jobs[0]["_timeline_raw"],
                                   "settings": jobs[0]["settings"]}], backup=False)
    assert replay.read_render_queue(p)[0]["_timeline_raw"] == inner


def test_write_preserves_other_entries_exactly(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr", timelines={"x": 1},
                      extra_entries={"mods.json": "[]", "thumb.bin": b"\x00\x01\xff"})
    with zipfile.ZipFile(p) as z:
        before = {n: z.read(n) for n in z.namelist()}
    replay.write_render_queue(p, [{"timeline": {"": []}, "settings": {}}])
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        after = {n: z.read(n) for n in z.namelist()}
    assert after.pop("renderQueue.json")
    assert after == before


def test_write_replaces_rather_than_duplicates_entry(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr", render_queue=[])
    for _ in range(3):
        replay.write_render_queue(p, [{"timeline": {"": []}, "settings": {}}],
                                  backup=False)
    with zipfile.ZipFile(p) as z:
        assert z.namelist().count("renderQueue.json") == 1


def test_backup_made_once_and_never_overwritten(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr")
    orig = p.read_bytes()
    replay.write_render_queue(p, [{"timeline": {"": []}, "settings": {"n": 1}}])
    bak = p.with_suffix(".mcpr.bak")
    assert bak.read_bytes() == orig
    replay.write_render_queue(p, [{"timeline": {"": []}, "settings": {"n": 2}}])
    assert bak.read_bytes() == orig, "second write must not clobber the first backup"


def test_no_backup_when_disabled_and_no_tmp_left(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr")
    replay.write_render_queue(p, [], backup=False)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["a.mcpr"]


def test_failed_write_leaves_original_intact(tmp_path):
    p = syn.make_mcpr(tmp_path / "a.mcpr", render_queue=[])
    before = p.read_bytes()
    with pytest.raises(KeyError):          # job missing "settings"
        replay.write_render_queue(p, [{"timeline": {"": []}}], backup=False)
    assert p.read_bytes() == before


def test_find_replays_newest_first_and_ignores_others(tmp_path):
    import os
    a = syn.make_mcpr(tmp_path / "a.mcpr")
    b = syn.make_mcpr(tmp_path / "b.mcpr")
    (tmp_path / "c.mcpr.bak").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    assert replay.find_replays(tmp_path) == [b, a]
    assert replay.find_replays(tmp_path / "missing") == []


def test_schema_validated_flags_are_set():
    assert replay.SCHEMA_VALIDATED is True


def test_corrupt_replays_raise_replay_error_not_zipfile_error(tmp_path):
    """Callers (batch_prepare, the GUI) only catch ReplayError."""
    junk = tmp_path / "junk.mcpr"
    junk.write_bytes(b"this is not a zip")
    with pytest.raises(replay.ReplayError):
        replay.inspect(junk)
    with pytest.raises(replay.ReplayError):
        replay.read_render_queue(junk)

    badjson = syn.make_mcpr(tmp_path / "bad.mcpr", omit_metadata=True,
                            extra_entries={"metaData.json": "{not json"})
    with pytest.raises(replay.ReplayError):
        replay.inspect(badjson)

    badq = syn.make_mcpr(tmp_path / "q.mcpr", render_queue="[{oops")
    with pytest.raises(replay.ReplayError):
        replay.read_render_queue(badq)
