import pytest

import synthetic as syn
from datetime import datetime
from golden import assert_golden
from mcmontage import config, pipeline, replay


def test_config_defaults_derive_paths_from_profile(tmp_path):
    c = config.Config(profile=tmp_path)
    assert c.recordings == tmp_path / "replay_recordings"
    assert c.videos == tmp_path / "replay_videos"


def test_config_load_missing_file_gives_defaults(tmp_path):
    assert config.load(tmp_path / "nope.toml").fps == 60


def test_config_load_overrides(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[paths]\nprofile="%s"\nrecordings="%s"\n[render]\nfps=30\n'
                 '[video]\nxfade=2.5\ncrf=23\n'
                 % (tmp_path / "prof", tmp_path / "rec"))
    c = config.load(p)
    assert c.recordings == tmp_path / "rec"
    assert c.videos == tmp_path / "prof" / "replay_videos"
    assert (c.fps, c.xfade, c.crf, c.width) == (30, 2.5, 23, 1920)


def test_repo_config_toml_matches_defaults(tmp_path):
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    c = config.load(repo / "config.toml")
    d = config.Config(profile=c.profile)
    for f in ("width", "height", "fps", "bitrate", "xfade", "music_gain_db",
              "music_fade_out", "crf"):
        assert getattr(c, f) == getattr(d, f), f


@pytest.mark.parametrize("duration,seconds,cap,expect", [
    (600_000, 30, None, (None, None)),
    (0, 30, 20, (None, None)),
    (600_000, 30, 20, (None, None)),            # 600s wanted == whole replay
    (600_000, 10, 20, (200_000, 400_000)),      # centred 200s slice
    (4_380_000, 18, 10, (2_100_000, 2_280_000)),
])
def test_centred_span(duration, seconds, cap, expect):
    assert pipeline.centred_span(duration, seconds, cap) == expect


def test_centred_span_is_inside_replay_and_respects_cap():
    for dur in (61_000, 600_001, 4_380_001):
        for sec in (5, 18, 45):
            for cap in (3, 10, 60):
                s, e = pipeline.centred_span(dur, sec, cap)
                if s is None:
                    continue
                assert 0 <= s < e <= dur
                assert (e - s) / (sec * 1000) <= cap + 1e-9


def make_session(tmp_path, n=3, **kw):
    rec = tmp_path / "rec"
    paths = []
    for i in range(n):
        paths.append(syn.make_mcpr(rec / f"raw{i}.mcpr",
                                   date=datetime(2026, 9, 1, 20 + i), duration_ms=300_000,
                                   **kw))
    return paths


def test_batch_prepare_renames_before_baking_output_names(tmp_path, cfg):
    paths = make_session(tmp_path)
    results, failures = pipeline.batch_prepare(
        cfg, paths, seconds=10, rename_scheme="day{session}_{n}",
        follow=False, centre=(0, 64, 0), keyframes=4, log=lambda m: None)
    assert not failures and len(results) == 3
    names = sorted(p.name for p in (tmp_path / "rec").glob("*.mcpr"))
    assert names == ["day1_1.mcpr", "day1_2.mcpr", "day1_3.mcpr"]
    for p in (tmp_path / "rec").glob("*.mcpr"):
        out = replay.read_render_queue(p)[0]["settings"]["outputFile"]
        assert out.endswith(p.stem + ".mp4"), (p.name, out)


def test_batch_prepare_collects_failures_and_continues(tmp_path, cfg):
    paths = make_session(tmp_path)
    paths[1].write_bytes(b"corrupt")
    results, failures = pipeline.batch_prepare(
        cfg, paths, seconds=10, follow=False, centre=(0, 64, 0), keyframes=4,
        log=lambda m: None)
    assert len(results) == 2 and [p.name for p, _ in failures] == ["raw1.mcpr"]


def test_batch_prepare_refuses_while_minecraft_running(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(pipeline, "minecraft_running", lambda: True)
    paths = make_session(tmp_path, 1)
    before = paths[0].read_bytes()
    with pytest.raises(pipeline.PipelineError):
        pipeline.batch_prepare(cfg, paths, follow=False, centre=(0, 64, 0),
                               log=lambda m: None)
    assert paths[0].read_bytes() == before


def test_batch_prepare_applies_speedup_cap(tmp_path, cfg):
    paths = make_session(tmp_path, 1)
    results, _ = pipeline.batch_prepare(
        cfg, paths, seconds=10, max_speedup=5, follow=False, centre=(0, 64, 0),
        keyframes=4, log=lambda m: None)
    assert results[0]["speedup"] == pytest.approx(5)
    assert (results[0]["start"], results[0]["end"]) == (125_000, 175_000)


def test_batch_prepare_log_output_golden(tmp_path, cfg):
    paths = make_session(tmp_path, 2)
    lines = []
    pipeline.batch_prepare(cfg, paths, seconds=10, max_speedup=5,
                           rename_scheme="day{session}_{n}", follow=False,
                           centre=(1, 2, 3), keyframes=4, log=lines.append)
    text = "\n".join(lines).replace(str(tmp_path), "<TMP>")
    assert_golden("batch_log.txt", text)
