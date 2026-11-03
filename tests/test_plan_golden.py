"""End-to-end: synthetic .mcpr -> plan_replay -> renderQueue.json, vs goldens."""

import json
import zipfile

import pytest

import synthetic as syn
from golden import assert_golden
from mcmontage import pipeline, replay


def queue_json(mcpr):
    """renderQueue.json with each nested timeline decoded, for readable goldens."""
    with zipfile.ZipFile(mcpr) as z:
        raw = json.loads(z.read("renderQueue.json"))
    return [{"timeline": json.loads(j["timeline"]), "settings": j["settings"]}
            for j in raw]


def _normalise(summary):
    return {k: (str(v) if k == "replay" else v) for k, v in summary.items()}


def test_golden_static_plan(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "rec" / "day1_1.mcpr", duration_ms=600_000)
    pipeline.plan_replay(
        cfg, mcpr, seconds=30, follow=False, centre=(100.5, 64, -20.25),
        radius=12, height=6, revolutions=0.5, keyframes=7,
        out_file="/videos/day1_1.mp4", log=lambda m: None)
    assert_golden("plan_static.json", queue_json(mcpr))


def test_golden_follow_plan(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "rec" / "day1_1.mcpr", duration_ms=60_000)
    syn.write_tracks(cfg.work / "day1_1.tracks.json", syn.standard_tracks(60_000))
    summary = pipeline.plan_replay(
        cfg, mcpr, seconds=10, follow=True, radius=12, height=6,
        revolutions=0.5, keyframes=9, out_file="/videos/day1_1.mp4",
        log=lambda m: None)
    assert summary["entity"] == 100, "player should be auto-detected"
    assert_golden("plan_follow.json", queue_json(mcpr))
    s = _normalise(summary)
    s.pop("replay"); s.pop("out_file")
    s["travelled"] = round(s["travelled"], 3)
    assert_golden("plan_follow_summary.json", s)


def test_plan_uses_config_resolution_fps_bitrate(tmp_path, cfg):
    cfg.width, cfg.height, cfg.fps, cfg.bitrate = 1280, 720, 30, 5_000_000
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr")
    pipeline.plan_replay(cfg, mcpr, follow=False, centre=(0, 64, 0),
                         log=lambda m: None)
    s = queue_json(mcpr)[0]["settings"]
    assert (s["videoWidth"], s["videoHeight"], s["framesPerSecond"], s["bitRate"]) \
        == (1280, 720, 30, 5_000_000)


def test_default_output_file_is_videos_dir_plus_stem(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "my replay.mcpr")
    pipeline.plan_replay(cfg, mcpr, follow=False, centre=(0, 64, 0),
                         log=lambda m: None)
    out = queue_json(mcpr)[0]["settings"]["outputFile"]
    assert out.endswith("my replay.mp4")
    assert out.startswith(str((cfg.videos).resolve()))


def test_timeline_maps_requested_window(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", duration_ms=600_000)
    pipeline.plan_replay(cfg, mcpr, seconds=20, start=100_000, end=300_000,
                         follow=False, centre=(0, 64, 0), keyframes=4,
                         log=lambda m: None)
    tp, pp = queue_json(mcpr)[0]["timeline"][""]
    assert [k["properties"]["timestamp"] for k in tp["keyframes"]] == [100_000, 300_000]
    assert [k["time"] for k in tp["keyframes"]] == [0, 20_000]
    assert pp["keyframes"][-1]["time"] == 20_000


def test_append_keeps_existing_jobs(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr")
    kw = dict(follow=False, centre=(0, 64, 0), log=lambda m: None)
    pipeline.plan_replay(cfg, mcpr, out_file="one.mp4", **kw)
    s = pipeline.plan_replay(cfg, mcpr, out_file="two.mp4", append=True, **kw)
    assert s["jobs"] == 2
    assert [j["settings"]["outputFile"] for j in queue_json(mcpr)] == ["one.mp4", "two.mp4"]


def test_no_append_replaces_queue(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr")
    kw = dict(follow=False, centre=(0, 64, 0), log=lambda m: None)
    pipeline.plan_replay(cfg, mcpr, out_file="one.mp4", **kw)
    pipeline.plan_replay(cfg, mcpr, out_file="two.mp4", **kw)
    assert [j["settings"]["outputFile"] for j in queue_json(mcpr)] == ["two.mp4"]


def test_append_to_mod_written_queue(tmp_path, cfg):
    """A queue the *mod* wrote (extra settings fields, odd timeline) survives append."""
    mod_job = {"timeline": '{"":[]}',
               "settings": {"renderMethod": "CUBIC", "videoWidth": 4096,
                            "someFutureField": [1, 2]}}
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", render_queue=[mod_job])
    pipeline.plan_replay(cfg, mcpr, append=True, follow=False, centre=(0, 64, 0),
                         log=lambda m: None)
    jobs = queue_json(mcpr)
    assert jobs[0]["settings"] == mod_job["settings"]
    assert jobs[0]["timeline"] == {"": []}
    assert len(jobs) == 2


def test_empty_span_is_an_error_and_writes_nothing(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", metadata={"duration": 0})
    before = mcpr.read_bytes()
    with pytest.raises(pipeline.PipelineError):
        pipeline.plan_replay(cfg, mcpr, follow=False, centre=(0, 64, 0),
                             log=lambda m: None)
    assert mcpr.read_bytes() == before


def test_static_orbit_requires_centre(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr")
    for centre in (None, (1, 2), (1, None, 3)):
        with pytest.raises(pipeline.PipelineError):
            pipeline.plan_replay(cfg, mcpr, follow=False, centre=centre,
                                 log=lambda m: None)


def test_refuses_while_minecraft_running(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(pipeline, "minecraft_running", lambda: True)
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr")
    before = mcpr.read_bytes()
    with pytest.raises(pipeline.PipelineError, match="Minecraft is running"):
        pipeline.plan_replay(cfg, mcpr, follow=False, centre=(0, 64, 0),
                             log=lambda m: None)
    assert mcpr.read_bytes() == before


def test_unknown_follow_entity(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", duration_ms=60_000)
    syn.write_tracks(cfg.work / "a.tracks.json", syn.standard_tracks())
    with pytest.raises(pipeline.PipelineError, match="999"):
        pipeline.plan_replay(cfg, mcpr, follow_entity=999, log=lambda m: None)


def test_follow_entity_override_follows_that_entity(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", duration_ms=60_000)
    syn.write_tracks(cfg.work / "a.tracks.json", syn.standard_tracks())
    s = pipeline.plan_replay(cfg, mcpr, follow_entity=200, keyframes=5,
                             log=lambda m: None)
    assert s["entity"] == 200 and s["entity_how"] == "specified"


def test_planned_replay_still_inspects_cleanly(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr")
    pipeline.plan_replay(cfg, mcpr, follow=False, centre=(0, 64, 0),
                         log=lambda m: None)
    info = replay.inspect(mcpr)
    assert info.has_render_queue and info.duration_ms == 600_000


def test_minecraft_running_false_without_tasklist():
    # real function (conftest stubs the module attr, so re-import original)
    import importlib
    mod = importlib.reload(pipeline)
    import shutil
    if shutil.which("tasklist") is None:
        assert mod.minecraft_running() is False
