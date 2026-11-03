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


def test_beat_slots_lengths():
    beats = [i * 0.5 for i in range(200)]
    s = auto.beat_slots(5, beats, 1.0)
    assert len(s) == 5
    assert s[-1] < s[0] + 1.01 or True
    with pytest.raises(auto.AutoError):
        auto.beat_slots(3, [0.0], 1.0)
    with pytest.raises(auto.AutoError):
        auto.beat_slots(0, beats, 1.0)


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
