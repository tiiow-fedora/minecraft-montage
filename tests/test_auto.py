import json
import math
import shutil
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from mcmontage import auto, cli, replay as replay_mod
from mcmontage.events import Event, save_events
from mcmontage.tracks import Track


def make_cfg(tmp_path):
    return SimpleNamespace(work=tmp_path / "work", videos=tmp_path / "videos",
                           output=tmp_path / "out", width=320, height=180, fps=30,
                           bitrate=1_000_000, xfade=0.5, crf=30,
                           music_gain_db=-16.0, music_fade_out=1.0)


def sprint_track(eid=1, seconds=60):
    pts = []
    for i in range(seconds * 20):
        t = i * 50
        # walk slowly, then sprint 20-30s at 12 b/s
        sec = t / 1000
        x = sec * 3 if sec < 20 else (60 + (min(sec, 30) - 20) * 12 + max(0, sec - 30) * 3)
        pts.append((t, x, 64.0, 0.0, 0.0, 0.0))
    return Track(eid, pts)


def write_mcpr(path, duration=60000):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metaData.json", json.dumps({"duration": duration}))
        z.writestr("recording.tmcpr", b"x")


def test_find_highlights_uses_cached_tracks_and_events(tmp_path):
    cfg = make_cfg(tmp_path)
    rp = tmp_path / "a.mcpr"
    write_mcpr(rp)
    cfg.work.mkdir()
    tr = sprint_track()
    (cfg.work / "a.tracks.json").write_text(json.dumps(
        {str(tr.entity_id): [list(p) for p in tr.points]}))
    save_events(auto.events_path(cfg, rp), [Event(40000, "death", 1, (1, 64, 0))])
    found, tracks, player = auto.find_highlights(cfg, rp)
    assert player == 1
    kinds = {h.kind for h in found}
    assert "death" in kinds and "speed" in kinds


def test_find_highlights_empty_tracks(tmp_path):
    cfg = make_cfg(tmp_path)
    rp = tmp_path / "a.mcpr"
    write_mcpr(rp)
    cfg.work.mkdir()
    (cfg.work / "a.tracks.json").write_text("{}")
    with pytest.raises(auto.AutoError):
        auto.find_highlights(cfg, rp)


def test_end_to_end_plan_writes_render_queue(tmp_path):
    cfg = make_cfg(tmp_path)
    rp = tmp_path / "a.mcpr"
    write_mcpr(rp)
    cfg.work.mkdir()
    tr = sprint_track()
    (cfg.work / "a.tracks.json").write_text(json.dumps(
        {"1": [list(p) for p in tr.points]}))
    save_events(auto.events_path(cfg, rp), [Event(40000, "death", 1, (1, 64, 0))])
    found, tracks, _ = auto.find_highlights(cfg, rp)
    chosen = auto.choose(found, 30, 6)
    assert chosen and all(a.end_ms <= b.start_ms for a, b in zip(chosen, chosen[1:]))
    jobs = auto.plan_highlight_jobs(cfg, rp, chosen, tracks, 6)
    assert len(jobs) == len(chosen)
    names = [j["settings"]["outputFile"] for j in jobs]
    assert len(set(names)) == len(names)
    assert auto.write_jobs(rp, jobs) == len(jobs)
    back = replay_mod.read_render_queue(rp)
    assert len(back) == len(jobs)
    assert auto.write_jobs(rp, jobs[:1], append=True) == len(jobs) + 1


def test_choose_respects_count():
    from mcmontage.events import Highlight
    hs = [Highlight("kill", i * 10000, i * 10000 + 3000, 0.5 + i / 100) for i in range(8)]
    assert len(auto.choose(hs, 100, 10, max_count=3)) == 3
    assert auto.choose([], 60, 8) == []


def _boundaries(slots, xfade, start=0.0):
    """Timeline times at which each xfade starts (+ the end of the last clip)."""
    out, t = [], start
    for s in slots[:-1]:
        t += s - xfade
        out.append(t)
    return out, start + sum(slots) - xfade * (len(slots) - 1)


def _on_beat(t, beats, tol=1e-6):
    return any(abs(t - b) < tol for b in beats)


BEATS = [i * 0.5 for i in range(1, 400)]       # 120 bpm, first beat at 0.5s


def test_beat_slots_cuts_land_on_beats():
    slots = auto.beat_slots(6, BEATS, 0.4, targets=[3.1, 2.4, 5.2, 3.0, 4.4, 2.2])
    assert len(slots) == 6
    cuts, end = _boundaries(slots, 0.4)
    assert all(_on_beat(c, BEATS) for c in cuts), cuts
    assert _on_beat(end, BEATS)                    # last clip has no extra overlap
    assert all(b > a for a, b in zip(cuts, cuts[1:]))
    # slots are positive and carry the xfade overlap on every non-final clip
    assert all(x > 0.4 for x in slots[:-1])


def test_beat_slots_one_per_clip_never_cycled():
    # more clips than the old cut plan would produce: still one slot each,
    # all distinct boundaries (the old code repeated slot lengths cyclically)
    slots = auto.beat_slots(40, BEATS, 0.5, min_clip_s=1.0, max_clip_s=3.0)
    assert len(slots) == 40
    cuts, end = _boundaries(slots, 0.5)
    assert all(_on_beat(c, BEATS) for c in cuts) and _on_beat(end, BEATS)
    assert len(set(round(c, 3) for c in cuts)) == 39


def test_beat_slots_targets_followed_and_total_s():
    slots = auto.beat_slots(4, BEATS, 0.0, targets=[2, 4, 2, 4])
    assert [round(x) for x in slots] == [2, 4, 2, 4]
    even = auto.beat_slots(4, BEATS, 0.0, total_s=16)
    assert sum(even) == pytest.approx(16, abs=1.0)


def test_beat_slots_start_offset_shifts_grid():
    # clip 0 starts at 3.2s (after a title card): cuts must be beats on that clock
    slots = auto.beat_slots(4, BEATS, 0.4, start_s=3.2, targets=[3] * 4)
    cuts, end = _boundaries(slots, 0.4, start=3.2)
    assert all(_on_beat(c, BEATS) for c in cuts) and _on_beat(end, BEATS)
    assert cuts[0] > 3.2


def test_beat_slots_trailing_xfade_for_outro():
    slots = auto.beat_slots(3, BEATS, 0.5, targets=[3] * 3, trailing_xfade=0.7)
    plain = auto.beat_slots(3, BEATS, 0.5, targets=[3] * 3)
    assert slots[:2] == pytest.approx(plain[:2])
    assert slots[2] == pytest.approx(plain[2] + 0.7)


def test_beat_slots_limits_never_exceeded_and_stay_on_beat():
    limits = [2.2, 2.2, 2.2, 2.2]
    slots = auto.beat_slots(4, BEATS, 0.5, targets=[3, 3, 3, 3], limits=limits)
    assert all(s <= l + 1e-6 for s, l in zip(slots, limits)), slots
    cuts, end = _boundaries(slots, 0.5)
    assert all(_on_beat(c, BEATS) for c in cuts) and _on_beat(end, BEATS)


def test_beat_slots_errors():
    with pytest.raises(auto.AutoError):
        auto.beat_slots(3, [0.0], 1.0)
    with pytest.raises(auto.AutoError):
        auto.beat_slots(0, BEATS, 1.0)
    with pytest.raises(auto.AutoError):
        auto.beat_slots(5, BEATS[:3], 1.0)          # fewer beats than clips
    with pytest.raises(auto.AutoError):
        auto.beat_slots(3, BEATS, 1.0, targets=[3, 3])


def test_cli_parses_new_commands():
    for argv in (["highlights"], ["autoplan", "--dry-run"], ["assemble", "a.mp4", "--no-sync"],
                 ["queue", "status"]):
        # parsing only: unknown commands would SystemExit(2)
        try:
            cli.main(["--config", "/nonexistent.toml"] + argv)
        except SystemExit as e:
            assert e.code != 2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_assemble_real_ffmpeg(tmp_path):
    cfg = make_cfg(tmp_path)
    clips = []
    for i, col in enumerate(("red", "green", "blue")):
        p = tmp_path / f"c{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"color=c={col}:s=160x90:r=15:d=3", "-pix_fmt", "yuv420p", str(p)],
                       check=True)
        clips.append(p)
    music = tmp_path / "m.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=10", str(music)], check=True)
    out = tmp_path / "o.mp4"
    args = auto.assemble(cfg, clips, music, out, sync=False)
    assert out.is_file() and out.stat().st_size > 0
    from mcmontage import edit
    p = edit.probe(out)
    assert p.has_audio and 5.0 < p.duration < 9.5
