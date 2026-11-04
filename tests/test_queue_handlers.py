"""Tests for the real queue handlers and the ``queue`` CLI actions.

Nothing here touches ffmpeg, Minecraft or YouTube: pipeline/auto functions are
monkeypatched and time is injected.
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcmontage import auto as auto_mod
from mcmontage import cli
from mcmontage import jobqueue as jq
from mcmontage import pipeline
from mcmontage import queue_handlers as qh
from mcmontage import replay as replay_mod
from mcmontage import youtube as yt_mod
from mcmontage.jobqueue import (Job, JobContext, JobQueue, NeedsManual, Registry,
                                StopRequested, enqueue_session_batch)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- helpers

class FakeTime:
    """Clock + sleep pair: sleeping advances the clock instantly."""

    def __init__(self):
        self.t = 100.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class FakeStat:
    def __init__(self, size, mtime_ns=1):
        self.st_size = size
        self.st_mtime_ns = mtime_ns


@pytest.fixture
def cfg(tmp_path):
    return SimpleNamespace(
        work=tmp_path / "work", videos=tmp_path / "videos", output=tmp_path / "out",
        width=1280, height=720, fps=30, bitrate=1_000_000, xfade=1.0,
        music_gain_db=-16.0, music_fade_out=4.0, crf=20)


@pytest.fixture
def ft():
    return FakeTime()


@pytest.fixture(autouse=True)
def mc_closed(monkeypatch):
    monkeypatch.setattr(pipeline, "minecraft_running", lambda: False)


def make_ctx(kind, params, deps=None, dep_results=None, stop=None, logs=None):
    job = Job(id="j0099", kind=kind, params=params, depends_on=list(deps or []))
    return JobContext(job, None, (logs.append if logs is not None else (lambda m: None)),
                      dict(dep_results or {}), stop)


def handlers(cfg, ft, **kw):
    kw.setdefault("clock", ft.clock)
    kw.setdefault("sleep", ft.sleep)
    return qh.Handlers(cfg, **kw)


# ---------------------------------------------------------------- plan

@pytest.fixture
def plan_env(monkeypatch):
    calls = {}
    monkeypatch.setattr(replay_mod, "inspect",
                        lambda p: SimpleNamespace(duration_ms=3_600_000))

    def fake_plan(cfg, path, **kw):
        calls["path"], calls["kw"] = path, kw
        return {"replay": Path(path), "start": kw["start"], "end": kw["end"],
                "speedup": 12.5, "centre": (1, 2, 3), "out_file": "/v/x.mp4"}

    monkeypatch.setattr(pipeline, "plan_replay", fake_plan)
    return calls


def test_plan_calls_plan_replay_with_capped_span(cfg, ft, plan_env):
    h = handlers(cfg, ft)
    out = h.plan(make_ctx("plan", {"clip": "a.mcpr", "seconds": 20.0, "max_speedup": 30.0}))
    kw = plan_env["kw"]
    assert plan_env["path"] == Path("a.mcpr")
    assert kw["seconds"] == 20.0
    wanted = 20 * 30 * 1000
    assert kw["end"] - kw["start"] == wanted
    assert out["speedup"] == 12.5


def test_plan_result_is_json_with_strings(cfg, ft, plan_env):
    out = handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr", "seconds": 5}))
    json.dumps(out)
    assert out["replay"] == "a.mcpr"
    assert out["centre"] == [1, 2, 3]
    assert out["out_file"] == "/v/x.mp4"


def test_plan_without_speedup_cap_uses_whole_replay(cfg, ft, plan_env):
    handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr", "seconds": 20, "max_speedup": None}))
    assert plan_env["kw"]["start"] is None and plan_env["kw"]["end"] is None


def test_plan_zero_speedup_means_no_cap(cfg, ft, plan_env):
    handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr", "seconds": 20, "max_speedup": 0}))
    assert plan_env["kw"]["start"] is None


def test_plan_forwards_camera_options(cfg, ft, plan_env):
    cam = {"radius": 9.0, "height": 3.0, "revolutions": 1.0, "keyframes": 40}
    handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr", "camera": cam}))
    for k, v in cam.items():
        assert plan_env["kw"][k] == v


def test_plan_rejects_unknown_camera_option(cfg, ft, plan_env):
    with pytest.raises(ValueError, match="bogus"):
        handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr", "camera": {"bogus": 1}}))
    assert "kw" not in plan_env


def test_plan_requires_clip(cfg, ft, plan_env):
    with pytest.raises(ValueError, match="clip"):
        handlers(cfg, ft).plan(make_ctx("plan", {}))


def test_plan_parks_when_minecraft_running(cfg, ft, plan_env, monkeypatch):
    monkeypatch.setattr(pipeline, "minecraft_running", lambda: True)
    with pytest.raises(NeedsManual, match="Minecraft"):
        handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr"}))
    assert "kw" not in plan_env


def test_plan_honours_check_stop(cfg, ft, plan_env):
    with pytest.raises(StopRequested):
        handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr"}, stop=lambda: True))
    assert "kw" not in plan_env


def test_plan_propagates_pipeline_errors(cfg, ft, monkeypatch):
    monkeypatch.setattr(replay_mod, "inspect",
                        lambda p: SimpleNamespace(duration_ms=1000))

    def boom(*a, **k):
        raise pipeline.PipelineError("replay span is empty")

    monkeypatch.setattr(pipeline, "plan_replay", boom)
    with pytest.raises(pipeline.PipelineError):
        handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr"}))


def test_plan_logs_go_to_ctx_log(cfg, ft, monkeypatch):
    monkeypatch.setattr(replay_mod, "inspect", lambda p: SimpleNamespace(duration_ms=1000))

    def fake(cfg, path, log=print, **kw):
        log("hello from plan")
        return {"out_file": "x"}

    monkeypatch.setattr(pipeline, "plan_replay", fake)
    logs = []
    handlers(cfg, ft).plan(make_ctx("plan", {"clip": "a.mcpr"}, logs=logs))
    assert logs == ["hello from plan"]


def test_minecraft_check_reflects_process_state(cfg, ft, monkeypatch):
    h = handlers(cfg, ft)
    assert h.minecraft_closed_check(make_ctx("plan", {})) is True
    monkeypatch.setattr(pipeline, "minecraft_running", lambda: True)
    assert h.minecraft_closed_check(make_ctx("plan", {})) is False


# ---------------------------------------------------------------- render_wait

def rw_ctx(clip="/rec/day1_1.mcpr", out_file=None, stop=None):
    deps = {"j0001": {"out_file": out_file}} if out_file else {}
    return make_ctx("render_wait", {"clip": clip}, deps=list(deps), dep_results=deps, stop=stop)


def test_render_wait_parks_when_file_missing(cfg, ft, tmp_path):
    h = handlers(cfg, ft, stat=lambda p: None)
    with pytest.raises(NeedsManual, match="day1_1"):
        h.render_wait(rw_ctx(out_file=str(tmp_path / "v.mp4")))


def test_render_wait_parks_when_file_empty(cfg, ft):
    h = handlers(cfg, ft, stat=lambda p: FakeStat(0))
    with pytest.raises(NeedsManual, match="empty"):
        h.render_wait(rw_ctx(out_file="/v/a.mp4"))


def test_render_wait_returns_after_stable_window(cfg, ft):
    h = handlers(cfg, ft, stat=lambda p: FakeStat(1234), stable_seconds=5, poll_seconds=1)
    out = h.render_wait(rw_ctx(out_file="/v/a.mp4"))
    assert out == {"video": "/v/a.mp4", "size": 1234}
    assert sum(ft.slept) == pytest.approx(5)


def test_render_wait_result_is_json(cfg, ft):
    h = handlers(cfg, ft, stat=lambda p: FakeStat(10))
    json.dumps(h.render_wait(rw_ctx(out_file="/v/a.mp4")))


def test_render_wait_growing_file_parks(cfg, ft):
    state = {"n": 0}

    def stat(p):
        state["n"] += 1
        return FakeStat(1000 * state["n"])

    h = handlers(cfg, ft, stat=stat, stable_seconds=5, poll_seconds=1, max_watch_seconds=20)
    with pytest.raises(NeedsManual, match="still being written"):
        h.render_wait(rw_ctx(out_file="/v/a.mp4"))


def test_render_wait_size_changes_then_settles(cfg, ft):
    sizes = iter([100, 100, 200, 300, 300, 300, 300, 300, 300, 300, 300])
    last = {"v": 100}

    def stat(p):
        last["v"] = next(sizes, last["v"])
        return FakeStat(last["v"])

    h = handlers(cfg, ft, stat=stat, stable_seconds=3, poll_seconds=1, max_watch_seconds=60)
    out = h.render_wait(rw_ctx(out_file="/v/a.mp4"))
    assert out["size"] == 300
    # the window restarted after growth, so we waited longer than 3s total
    assert sum(ft.slept) > 3


def test_render_wait_mtime_change_alone_resets_window(cfg, ft):
    mt = iter([1, 2, 3, 3, 3, 3, 3, 3, 3])
    last = {"v": 1}

    def stat(p):
        last["v"] = next(mt, last["v"])
        return FakeStat(500, mtime_ns=last["v"])

    h = handlers(cfg, ft, stat=stat, stable_seconds=3, poll_seconds=1, max_watch_seconds=60)
    h.render_wait(rw_ctx(out_file="/v/a.mp4"))
    assert sum(ft.slept) > 3


def test_render_wait_file_vanishes(cfg, ft):
    seq = iter([FakeStat(10), None])
    h = handlers(cfg, ft, stat=lambda p: next(seq), stable_seconds=5)
    with pytest.raises(NeedsManual, match="disappeared"):
        h.render_wait(rw_ctx(out_file="/v/a.mp4"))


def test_render_wait_uses_plan_out_file(cfg, ft):
    seen = []
    h = handlers(cfg, ft, stat=lambda p: (seen.append(str(p)) or FakeStat(5)))
    h.render_wait(rw_ctx(out_file="/custom/place.mp4"))
    assert set(seen) == {"/custom/place.mp4"}


def test_render_wait_default_path_is_videos_dir_stem(cfg, ft):
    h = handlers(cfg, ft)
    assert h.expected_video(rw_ctx(clip="/rec/day1_2.mcpr")) == \
        (Path(cfg.videos) / "day1_2.mp4").resolve()


def test_render_wait_real_files(cfg, ft, tmp_path):
    vid = tmp_path / "v.mp4"
    h = handlers(cfg, ft, stable_seconds=2, poll_seconds=1)
    with pytest.raises(NeedsManual):
        h.render_wait(rw_ctx(out_file=str(vid)))
    vid.write_bytes(b"x" * 64)
    out = h.render_wait(rw_ctx(out_file=str(vid)))
    assert out["video"] == str(vid) and out["size"] == 64


def test_render_wait_honours_check_stop(cfg, ft):
    h = handlers(cfg, ft, stat=lambda p: FakeStat(5))
    with pytest.raises(StopRequested):
        h.render_wait(rw_ctx(out_file="/v/a.mp4", stop=lambda: True))


def test_render_wait_stop_mid_watch(cfg, ft):
    h = handlers(cfg, ft, stat=lambda p: FakeStat(5), stable_seconds=10)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(StopRequested):
        h.render_wait(rw_ctx(out_file="/v/a.mp4", stop=stop))


def test_render_ready_check(cfg, ft):
    exists = {"v": None}
    h = handlers(cfg, ft, stat=lambda p: exists["v"])
    ctx = rw_ctx(out_file="/v/a.mp4")
    assert h.render_ready(ctx) is False
    exists["v"] = FakeStat(0)
    assert h.render_ready(ctx) is False
    exists["v"] = FakeStat(9)
    assert h.render_ready(ctx) is True


# ---------------------------------------------------------------- edit

def edit_ctx(params=None, videos=("a", "b"), stop=None):
    deps = ["j1", "j2", "j3"][:len(videos)]
    results = {d: {"video": v} for d, v in zip(deps, videos)}
    return make_ctx("edit", params or {}, deps=deps, dep_results=results, stop=stop)


@pytest.fixture
def edit_env(monkeypatch, tmp_path):
    calls = {}

    def fake_finish(cfg, clips, **kw):
        calls["clips"], calls["kw"] = clips, kw
        return tmp_path / "final.mp4", []

    monkeypatch.setattr(pipeline, "finish_video", fake_finish)
    return calls


def real_videos(tmp_path, *names):
    out = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"v")
        out.append(str(p))
    return out


def test_edit_joins_in_depends_on_order(cfg, ft, edit_env, tmp_path):
    vids = real_videos(tmp_path, "one.mp4", "two.mp4", "three.mp4")
    out = handlers(cfg, ft).edit(edit_ctx(videos=vids))
    assert [str(c) for c in edit_env["clips"]] == vids
    assert out["clips"] == 3


def test_edit_forwards_params(cfg, ft, edit_env, tmp_path):
    vids = real_videos(tmp_path, "one.mp4")
    params = {"output": str(tmp_path / "o.mp4"), "title": "T", "subtitle": "S",
              "music_ids": ["m1"], "xfade": 0.5}
    handlers(cfg, ft).edit(edit_ctx(params, videos=vids))
    kw = edit_env["kw"]
    assert (kw["output"], kw["title"], kw["subtitle"]) == (params["output"], "T", "S")
    assert kw["music_ids"] == ["m1"] and kw["xfade"] == 0.5


def test_edit_empty_values_become_none(cfg, ft, edit_env, tmp_path):
    vids = real_videos(tmp_path, "one.mp4")
    handlers(cfg, ft).edit(edit_ctx({"output": None, "title": "", "music_ids": []}, videos=vids))
    kw = edit_env["kw"]
    assert kw["output"] is None and kw["title"] is None and kw["music_ids"] is None


def test_edit_result_json_and_credits(cfg, ft, monkeypatch, tmp_path):
    vids = real_videos(tmp_path, "one.mp4")
    track = SimpleNamespace(needs_attribution=True, credit=lambda: "Song by X")
    free = SimpleNamespace(needs_attribution=False, credit=lambda: "nope")
    monkeypatch.setattr(pipeline, "finish_video",
                        lambda cfg, clips, **kw: (tmp_path / "f.mp4", [track, free]))
    out = handlers(cfg, ft).edit(edit_ctx(videos=vids))
    json.dumps(out)
    assert out == {"output": str(tmp_path / "f.mp4"), "clips": 1, "credits": ["Song by X"]}


def test_edit_no_video_dependency(cfg, ft, edit_env):
    ctx = make_ctx("edit", {}, deps=["j1"], dep_results={"j1": {"summary": 1}})
    with pytest.raises(ValueError, match="no rendered video"):
        handlers(cfg, ft).edit(ctx)


def test_edit_missing_clip_file(cfg, ft, edit_env, tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.mp4"):
        handlers(cfg, ft).edit(edit_ctx(videos=[str(tmp_path / "nope.mp4")]))
    assert "clips" not in edit_env


def test_edit_ignores_dependencies_without_video(cfg, ft, edit_env, tmp_path):
    vids = real_videos(tmp_path, "one.mp4")
    ctx = make_ctx("edit", {}, deps=["j1", "j2"],
                   dep_results={"j1": None, "j2": {"video": vids[0]}})
    assert handlers(cfg, ft).edit(ctx)["clips"] == 1


def test_edit_honours_check_stop(cfg, ft, edit_env, tmp_path):
    vids = real_videos(tmp_path, "one.mp4")
    with pytest.raises(StopRequested):
        handlers(cfg, ft).edit(edit_ctx(videos=vids, stop=lambda: True))
    assert "clips" not in edit_env


# ---------------------------------------------------------------- upload

def up_ctx(params=None, output="x", stop=None, logs=None):
    return make_ctx("upload", params or {"title": "Vid"}, deps=["j9"],
                    dep_results={"j9": {"output": output}}, stop=stop, logs=logs)


@pytest.fixture
def up_env(monkeypatch):
    calls = {}
    response = {"id": "abc123", "etag": "zzz", "snippet": {"title": "Vid"}}

    def fake(cfg, video, title, **kw):
        calls.update(video=video, title=title, kw=kw)
        return response

    monkeypatch.setattr(pipeline, "upload_private", fake)
    return calls


def test_upload_calls_upload_private(cfg, ft, up_env, tmp_path):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"v")
    params = {"title": "My Montage", "description": "d", "tags": ["a", "b"],
              "music_ids": ["m1"]}
    out = handlers(cfg, ft).upload(up_ctx(params, output=str(f)))
    assert up_env["video"] == f and up_env["title"] == "My Montage"
    kw = up_env["kw"]
    assert kw["description"] == "d" and kw["tags"] == ["a", "b"] and kw["music_ids"] == ["m1"]
    assert out["video_id"] == "abc123"
    assert out["privacy"] == "private"
    assert out["watch_url"].endswith("abc123")
    assert "studio.youtube.com" in out["studio_url"]


def test_upload_never_passes_privacy_or_credentials(cfg, ft, up_env, tmp_path):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"v")
    handlers(cfg, ft).upload(up_ctx({"title": "t", "privacy": "public"}, output=str(f)))
    kw = up_env["kw"]
    assert "privacy" not in kw
    for k in kw:
        assert not jq.is_secret_name(k)


def test_upload_result_holds_no_raw_response_or_secrets(cfg, ft, up_env, tmp_path):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"v")
    out = handlers(cfg, ft).upload(up_ctx(output=str(f)))
    json.dumps(out)
    assert jq.find_secrets(out) == []
    assert "etag" not in out and "snippet" not in out


def test_upload_title_falls_back_to_file_stem(cfg, ft, up_env, tmp_path):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"v")
    handlers(cfg, ft).upload(up_ctx({"title": "  "}, output=str(f)))
    assert up_env["title"] == "final"


def test_upload_requires_edit_output(cfg, ft, up_env):
    ctx = make_ctx("upload", {"title": "t"}, deps=["j1"], dep_results={"j1": {}})
    with pytest.raises(ValueError, match="no edited video"):
        handlers(cfg, ft).upload(ctx)


def test_upload_missing_file(cfg, ft, up_env, tmp_path):
    with pytest.raises(FileNotFoundError):
        handlers(cfg, ft).upload(up_ctx(output=str(tmp_path / "gone.mp4")))
    assert "video" not in up_env


def test_upload_honours_check_stop(cfg, ft, up_env, tmp_path):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"v")
    with pytest.raises(StopRequested):
        handlers(cfg, ft).upload(up_ctx(output=str(f), stop=lambda: True))
    assert "video" not in up_env


def test_upload_progress_checks_stop_and_logs(cfg, ft, monkeypatch, tmp_path):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"v")
    stop = {"v": False}
    logs = []

    def fake(cfg, video, title, progress=None, **kw):
        progress(0.1)
        progress(0.5)
        stop["v"] = True
        progress(0.9)
        return {"id": "x"}

    monkeypatch.setattr(pipeline, "upload_private", fake)
    with pytest.raises(StopRequested):
        handlers(cfg, ft).upload(up_ctx(output=str(f), stop=lambda: stop["v"], logs=logs))
    assert any("10%" in m for m in logs) and any("50%" in m for m in logs)


def test_upload_through_real_upload_private_is_private_and_uses_default_credentials(
        cfg, ft, monkeypatch, tmp_path):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"v")
    seen = {}

    def fake_upload(video, meta, **kw):
        seen.update(video=video, meta=meta, kw=kw)
        return {"id": "vid1"}

    monkeypatch.setattr(yt_mod, "upload", fake_upload)
    out = handlers(cfg, ft).upload(up_ctx({"title": "T", "description": "body"}, output=str(f)))
    assert seen["meta"].privacy == "private"
    # credentials flow stays inside youtube.upload (defaults), not passed in
    assert set(seen["kw"]) <= {"progress"}
    assert out["video_id"] == "vid1"


def test_upload_propagates_pipeline_error(cfg, ft, monkeypatch, tmp_path):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"v")

    def boom(*a, **k):
        raise pipeline.PipelineError("YouTube rejected the upload")

    monkeypatch.setattr(pipeline, "upload_private", boom)
    with pytest.raises(pipeline.PipelineError):
        handlers(cfg, ft).upload(up_ctx(output=str(f)))


# ---------------------------------------------------------------- autoplan

class FakeHL:
    def __init__(self, kind, s, e, score=1.0, label="x"):
        self.kind, self.start_ms, self.end_ms, self.score, self.label = kind, s, e, score, label


@pytest.fixture
def auto_env(monkeypatch):
    calls = {}
    hl = [FakeHL("death", 1000, 5000, 0.9, "died"), FakeHL("kill", 9000, 12000, 0.5, "kill")]

    def fake_find(cfg, path, retrack=False, **kw):
        calls["find"] = (path, retrack)
        return hl, {"t": 1}, 7

    def fake_choose(found, total, each, min_gap_s=2.0, max_count=None):
        calls["choose"] = (total, each, max_count)
        return list(found)

    def fake_plan_jobs(cfg, path, chosen, tracks, each, out_dir=None):
        return [{"settings": {"outputFile": Path("/v/s%d.mp4" % i)}, "timeline": {}}
                for i, _ in enumerate(chosen)]

    def fake_write(path, jobs, append=False):
        calls["write"] = (path, len(jobs), append)
        return len(jobs) + (1 if append else 0)

    monkeypatch.setattr(auto_mod, "find_highlights", fake_find)
    monkeypatch.setattr(auto_mod, "choose", fake_choose)
    monkeypatch.setattr(auto_mod, "plan_highlight_jobs", fake_plan_jobs)
    monkeypatch.setattr(auto_mod, "write_jobs", fake_write)
    return calls


def test_autoplan_happy_path(cfg, ft, auto_env):
    out = handlers(cfg, ft).autoplan(
        make_ctx("autoplan", {"replay": "r.mcpr", "total": 40, "seconds_each": 5}))
    json.dumps(out)
    assert out["jobs"] == 2 and out["found"] == 2
    assert out["outputs"] == ["/v/s0.mp4", "/v/s1.mp4"]
    assert out["highlights"][0] == {"kind": "death", "start_ms": 1000, "end_ms": 5000,
                                    "score": 0.9, "label": "died"}
    assert auto_env["choose"] == (40.0, 5.0, None)
    assert auto_env["write"] == (Path("r.mcpr"), 2, False)


def test_autoplan_forwards_options(cfg, ft, auto_env):
    out = handlers(cfg, ft).autoplan(make_ctx(
        "autoplan", {"replay": "r.mcpr", "max_count": 1, "append": True, "retrack": True}))
    assert auto_env["choose"][2] == 1
    assert auto_env["find"][1] is True
    assert auto_env["write"][2] is True
    assert out["jobs"] == 3


def test_autoplan_defaults(cfg, ft, auto_env):
    handlers(cfg, ft).autoplan(make_ctx("autoplan", {"replay": "r.mcpr"}))
    assert auto_env["choose"][:2] == (60.0, 8.0)


def test_autoplan_no_highlights(cfg, ft, auto_env, monkeypatch):
    monkeypatch.setattr(auto_mod, "choose", lambda *a, **k: [])
    with pytest.raises(auto_mod.AutoError, match="no highlights"):
        handlers(cfg, ft).autoplan(make_ctx("autoplan", {"replay": "r.mcpr"}))
    assert "write" not in auto_env


def test_autoplan_parks_when_minecraft_running(cfg, ft, auto_env, monkeypatch):
    monkeypatch.setattr(pipeline, "minecraft_running", lambda: True)
    with pytest.raises(NeedsManual):
        handlers(cfg, ft).autoplan(make_ctx("autoplan", {"replay": "r.mcpr"}))
    assert "find" not in auto_env


def test_autoplan_requires_replay(cfg, ft, auto_env):
    with pytest.raises(ValueError, match="replay"):
        handlers(cfg, ft).autoplan(make_ctx("autoplan", {}))


def test_autoplan_honours_check_stop(cfg, ft, auto_env):
    with pytest.raises(StopRequested):
        handlers(cfg, ft).autoplan(make_ctx("autoplan", {"replay": "r.mcpr"}, stop=lambda: True))
    assert "write" not in auto_env


# ---------------------------------------------------------------- registry / integration

def test_registry_covers_every_kind_enqueue_emits(cfg):
    reg = qh.build_registry(cfg)
    assert reg.kinds() == ["autoplan", "edit", "plan", "render_wait", "upload"]


def test_registry_checks_registered_for_manual_kinds(cfg):
    reg = qh.build_registry(cfg)
    assert reg.check_for("render_wait") is not None
    assert reg.check_for("plan") is not None
    assert reg.check_for("edit") is None


def test_import_is_lazy():
    code = ("import sys, mcmontage.queue_handlers as q; q.build_registry(object());"
            "bad=[m for m in ('mcmontage.pipeline','mcmontage.auto','mcmontage.youtube',"
            "'googleapiclient','mcmontage.edit') if m in sys.modules];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_jsonable_converts_paths_and_tuples():
    out = qh.jsonable({"a": Path("x"), "b": (1, Path("y")), 3: {1, }, "c": object})
    json.dumps(out)
    assert out["a"] == "x" and out["b"] == [1, "y"]


def test_full_chain_plan_wait_edit_upload(cfg, ft, monkeypatch, tmp_path):
    clip = tmp_path / "day1_1.mcpr"
    clip.write_bytes(b"zip")
    video = tmp_path / "videos" / "day1_1.mp4"
    final = tmp_path / "out" / "final.mp4"

    monkeypatch.setattr(replay_mod, "inspect", lambda p: SimpleNamespace(duration_ms=600_000))
    monkeypatch.setattr(pipeline, "plan_replay",
                        lambda cfg, p, **kw: {"replay": p, "out_file": str(video)})

    def fake_finish(cfg, clips, **kw):
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"f")
        return final, []

    monkeypatch.setattr(pipeline, "finish_video", fake_finish)
    monkeypatch.setattr(pipeline, "upload_private",
                        lambda cfg, v, t, **kw: {"id": "VID"})

    q = JobQueue(tmp_path / "work" / "queue.json")
    enqueue_session_batch(q, [clip], 10, max_speedup=30, title="Night", upload=True)
    reg = qh.build_registry(cfg, clock=ft.clock, sleep=ft.sleep, stable_seconds=2)

    r1 = q.run(reg)
    assert [q.get(j).kind for j in r1.done] == ["plan"]
    assert len(r1.waiting) == 1 and q.get(r1.waiting[0]).kind == "render_wait"
    assert "render" in q.get(r1.waiting[0]).message

    video.parent.mkdir(parents=True)
    video.write_bytes(b"m" * 32)
    r2 = q.run(reg)
    assert r2.ok and not r2.waiting
    kinds = {j.kind: j for j in q.jobs()}
    assert all(j.status == jq.DONE for j in kinds.values())
    assert kinds["upload"].result["video_id"] == "VID"
    assert kinds["edit"].result["output"] == str(final)

    raw = (tmp_path / "work" / "queue.json").read_text()
    for jd in json.loads(raw)["jobs"]:
        assert jq.find_secrets(jd["params"]) == []
        assert jq.find_secrets(jd["result"]) == []


def test_chain_stays_parked_until_render_exists(cfg, ft, monkeypatch, tmp_path):
    clip = tmp_path / "a.mcpr"
    clip.write_bytes(b"z")
    monkeypatch.setattr(replay_mod, "inspect", lambda p: SimpleNamespace(duration_ms=1000))
    monkeypatch.setattr(pipeline, "plan_replay",
                        lambda cfg, p, **kw: {"out_file": str(tmp_path / "nope.mp4")})
    q = JobQueue(tmp_path / "work" / "queue.json")
    enqueue_session_batch(q, [clip], 5)
    reg = qh.build_registry(cfg, clock=ft.clock, sleep=ft.sleep)
    for _ in range(2):
        rep = q.run(reg)
        assert len(rep.waiting) == 1
    assert q.summary()["pending"] == 1  # the edit, blocked behind the render


# ---------------------------------------------------------------- CLI

def make_mcpr(path: Path, when: datetime.datetime, minutes=10):
    meta = {"date": int(when.timestamp() * 1000), "duration": minutes * 60_000}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metaData.json", json.dumps(meta))
    return path


@pytest.fixture
def env(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    # session 1: two clips on one day, session 2: one clip two days later
    make_mcpr(rec / "20260101_120000.mcpr", datetime.datetime(2026, 1, 1, 12, 0))
    make_mcpr(rec / "20260101_130000.mcpr", datetime.datetime(2026, 1, 1, 13, 0))
    make_mcpr(rec / "20260103_120000.mcpr", datetime.datetime(2026, 1, 3, 12, 0))
    conf = tmp_path / "config.toml"
    conf.write_text(
        "[paths]\n"
        f"profile = '{tmp_path / 'profile'}'\n"
        f"recordings = '{rec}'\n"
        f"videos = '{tmp_path / 'videos'}'\n"
        f"output = '{tmp_path / 'out'}'\n"
        f"work = '{tmp_path / 'work'}'\n")
    return SimpleNamespace(conf=conf, qpath=tmp_path / "work" / "queue.json",
                           rec=rec, tmp=tmp_path)


def run_cli(env, *argv):
    return cli.main(["--config", str(env.conf), *argv])


def queue_of(env):
    return JobQueue(env.qpath)


def test_cli_docstring_documents_queue_actions():
    for word in ("enqueue", "run", "cancel", "resume", "requeue"):
        assert word in cli.__doc__


def test_cli_status_empty(env, capsys):
    run_cli(env, "queue", "status")
    assert "queue is empty" in capsys.readouterr().out


def test_cli_enqueue_dry_run_writes_nothing(env, capsys):
    run_cli(env, "queue", "enqueue", "--dry-run", "--upload", "--title", "T")
    out = capsys.readouterr().out
    assert "would queue plan + render_wait" in out
    assert "would queue edit" in out and "would queue upload (private)" in out
    assert "dry run" in out
    assert out.count("would queue plan") == 3
    assert not env.qpath.exists()


def test_cli_enqueue_creates_chains_per_session(env, capsys):
    run_cli(env, "queue", "enqueue")
    q = queue_of(env)
    kinds = [j.kind for j in q.jobs()]
    assert kinds.count("plan") == 3 and kinds.count("render_wait") == 3
    assert kinds.count("edit") == 2 and "upload" not in kinds
    assert "session 1" in capsys.readouterr().out


def test_cli_enqueue_is_idempotent(env, capsys):
    run_cli(env, "queue", "enqueue")
    n = len(queue_of(env))
    capsys.readouterr()
    run_cli(env, "queue", "enqueue")
    assert len(queue_of(env)) == n
    assert "queued 0 new job(s)" in capsys.readouterr().out


def test_cli_enqueue_single_session(env):
    run_cli(env, "queue", "enqueue", "--session", "2")
    q = queue_of(env)
    plans = [j for j in q.jobs() if j.kind == "plan"]
    assert len(plans) == 1 and plans[0].params["clip"].endswith("20260103_120000.mcpr")


def test_cli_enqueue_unknown_session(env):
    with pytest.raises(SystemExit, match="no session 9"):
        run_cli(env, "queue", "enqueue", "--session", "9")


def test_cli_enqueue_session_clips_in_recording_order(env):
    run_cli(env, "queue", "enqueue", "--session", "1")
    q = queue_of(env)
    clips = [j.params["clip"] for j in q.jobs() if j.kind == "render_wait"]
    assert [Path(c).name for c in clips] == ["20260101_120000.mcpr", "20260101_130000.mcpr"]


def test_cli_enqueue_params(env):
    run_cli(env, "queue", "enqueue", "--session", "1", "--seconds", "12",
            "--max-speedup", "40", "--radius", "9", "--title", "Hello",
            "--subtitle", "Sub", "--xfade", "0.5")
    q = queue_of(env)
    plan = next(j for j in q.jobs() if j.kind == "plan")
    assert plan.params["seconds"] == 12 and plan.params["max_speedup"] == 40
    assert plan.params["camera"]["radius"] == 9
    edit = next(j for j in q.jobs() if j.kind == "edit")
    assert edit.params["title"] == "Hello" and edit.params["subtitle"] == "Sub"
    assert edit.params["xfade"] == 0.5


def test_cli_enqueue_zero_speedup_means_whole_replays(env):
    run_cli(env, "queue", "enqueue", "--session", "1", "--max-speedup", "0")
    plan = next(j for j in queue_of(env).jobs() if j.kind == "plan")
    assert plan.params["max_speedup"] is None


def test_cli_enqueue_upload_needs_title(env):
    with pytest.raises(SystemExit, match="--title"):
        run_cli(env, "queue", "enqueue", "--upload")
    assert not env.qpath.exists()


def test_cli_enqueue_upload_job_depends_on_edit(env):
    run_cli(env, "queue", "enqueue", "--session", "1", "--upload", "--title", "T",
            "--description", "desc", "--tags", "a, b,,c")
    q = queue_of(env)
    up = next(j for j in q.jobs() if j.kind == "upload")
    edit = next(j for j in q.jobs() if j.kind == "edit")
    assert up.depends_on == [edit.id]
    assert up.params["tags"] == ["a", "b", "c"] and up.params["description"] == "desc"
    assert "privacy" not in up.params
    assert jq.find_secrets(up.params) == []


def test_cli_enqueue_music_validated_and_stored(env, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, "resolve_tracks", lambda ids: seen.append(ids) or [])
    run_cli(env, "queue", "enqueue", "--session", "1", "--music", "a", "--music", "b")
    assert seen == [["a", "b"]]
    edit = next(j for j in queue_of(env).jobs() if j.kind == "edit")
    assert edit.params["music_ids"] == ["a", "b"]


def test_cli_enqueue_bad_music_aborts(env, monkeypatch):
    from mcmontage import music as music_mod

    def bad(ids):
        raise music_mod.MusicError("unknown track nope")

    monkeypatch.setattr(pipeline, "resolve_tracks", bad)
    with pytest.raises(SystemExit, match="unknown track"):
        run_cli(env, "queue", "enqueue", "--music", "nope")
    assert not env.qpath.exists()


def test_cli_enqueue_out_with_many_sessions_refused(env):
    with pytest.raises(SystemExit, match="--out"):
        run_cli(env, "queue", "enqueue", "--out", "x.mp4")


def test_cli_enqueue_out_single_session(env):
    run_cli(env, "queue", "enqueue", "--session", "2", "--out", "final.mp4")
    edit = next(j for j in queue_of(env).jobs() if j.kind == "edit")
    assert edit.params["output"] == "final.mp4"


def test_cli_enqueue_no_recordings(env, tmp_path):
    for f in env.rec.glob("*.mcpr"):
        f.unlink()
    with pytest.raises(SystemExit, match="no .mcpr"):
        run_cli(env, "queue", "enqueue")


def test_cli_cancel_resume_requeue(env, capsys):
    run_cli(env, "queue", "enqueue", "--session", "2")
    q = queue_of(env)
    plan = next(j for j in q.jobs() if j.kind == "plan")
    run_cli(env, "queue", "cancel", plan.id)
    assert queue_of(env).get(plan.id).status == jq.CANCELLED
    assert "cancelled" in capsys.readouterr().out
    run_cli(env, "queue", "requeue", plan.id)
    assert queue_of(env).get(plan.id).status == jq.PENDING


def test_cli_resume_waiting_job(env, capsys):
    q = JobQueue(env.qpath)
    j = q.add("render_wait", {"clip": "a"})
    reg = Registry().register("render_wait", lambda ctx: (_ for _ in ()).throw(NeedsManual("do it")))
    q.run(reg)
    assert q.get(j.id).status == jq.WAITING
    run_cli(env, "queue", "resume", j.id)
    assert queue_of(env).get(j.id).status == jq.PENDING
    assert queue_of(env).get(j.id).resumed is True
    assert "pending" in capsys.readouterr().out


def test_cli_resume_non_waiting_errors(env):
    JobQueue(env.qpath).add("plan", {"clip": "a"})
    with pytest.raises(SystemExit, match="not waiting"):
        run_cli(env, "queue", "resume", "j0001")


def test_cli_cancel_requires_id(env):
    with pytest.raises(SystemExit, match="job id"):
        run_cli(env, "queue", "cancel")


def test_cli_cancel_unknown_id(env):
    with pytest.raises(SystemExit, match="no such job"):
        run_cli(env, "queue", "cancel", "j9999")


def test_cli_cancel_done_job_refused(env):
    q = JobQueue(env.qpath)
    j = q.add("a")
    q.run(Registry().register("a", lambda ctx: {}))
    with pytest.raises(SystemExit, match="already done"):
        run_cli(env, "queue", "cancel", j.id)


def fake_registry(monkeypatch, **handlers_):
    reg = Registry()
    for k, h in handlers_.items():
        reg.register(k, h)
    monkeypatch.setattr(qh, "build_registry", lambda cfg, **kw: reg)
    return reg


def test_cli_run_prints_progress_and_report(env, monkeypatch, capsys):
    q = JobQueue(env.qpath)
    q.add("a", {"clip": "/x/foo.mcpr"})
    fake_registry(monkeypatch, a=lambda ctx: (ctx.log("working"), {"ok": 1})[1])
    run_cli(env, "queue", "run")
    out = capsys.readouterr().out
    assert "[start  ] j0001 a" in out and "foo.mcpr" in out
    assert "[done   ] j0001" in out
    assert "    working" in out
    assert "done 1  failed 0" in out
    assert queue_of(env).get("j0001").status == jq.DONE


def test_cli_run_failure_exits_nonzero(env, monkeypatch, capsys):
    q = JobQueue(env.qpath)
    q.add("a", max_attempts=1)
    fake_registry(monkeypatch, a=lambda ctx: 1 / 0)
    with pytest.raises(SystemExit) as e:
        run_cli(env, "queue", "run")
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "[failed ]" in out and "ZeroDivisionError" in out and "failed 1" in out


def test_cli_run_stop_on_error(env, monkeypatch):
    q = JobQueue(env.qpath)
    q.add("a", max_attempts=1)
    q.add("b")
    ran = []
    fake_registry(monkeypatch, a=lambda ctx: 1 / 0, b=lambda ctx: ran.append(1) or {})
    with pytest.raises(SystemExit):
        run_cli(env, "queue", "run", "--stop-on-error")
    assert ran == []
    assert queue_of(env).get("j0002").status == jq.PENDING


def test_cli_run_without_stop_on_error_continues(env, monkeypatch):
    q = JobQueue(env.qpath)
    q.add("a", max_attempts=1)
    q.add("b")
    ran = []
    fake_registry(monkeypatch, a=lambda ctx: 1 / 0, b=lambda ctx: ran.append(1) or {})
    with pytest.raises(SystemExit):
        run_cli(env, "queue", "run")
    assert ran == [1]


def test_cli_run_waiting_prints_hint(env, monkeypatch, capsys):
    q = JobQueue(env.qpath)
    q.add("r")

    def park(ctx):
        raise NeedsManual("render it")

    fake_registry(monkeypatch, r=park)
    run_cli(env, "queue", "run")
    out = capsys.readouterr().out
    assert "[waiting] j0001" in out and "render it" in out
    assert "waiting 1" in out and "manual step" in out


def test_cli_run_ctrl_c_stops_cleanly(env, monkeypatch, capsys):
    q = JobQueue(env.qpath)
    q.add("a")
    q.add("b")
    ran = []

    def first(ctx):
        os.kill(os.getpid(), signal.SIGINT)   # simulate the user pressing Ctrl-C
        ran.append("a")
        return {}

    fake_registry(monkeypatch, a=first, b=lambda ctx: ran.append("b") or {})
    run_cli(env, "queue", "run")
    out = capsys.readouterr().out
    assert ran == ["a"]
    assert "Stopping after the current step" in out and "stopped early" in out
    assert queue_of(env).get("j0002").status == jq.PENDING


@pytest.mark.skipif(os.name == "nt", reason="posix signals")
def test_cli_run_restores_sigint_handler(env, monkeypatch):
    JobQueue(env.qpath).add("a")
    fake_registry(monkeypatch, a=lambda ctx: {})
    before = signal.getsignal(signal.SIGINT)
    run_cli(env, "queue", "run")
    assert signal.getsignal(signal.SIGINT) is before


def test_cli_run_second_runner_refused(env, monkeypatch):
    q = JobQueue(env.qpath)
    q.add("a")
    fake_registry(monkeypatch, a=lambda ctx: {})
    q._runner_lock.acquire()
    try:
        with pytest.raises(SystemExit, match="locked"):
            run_cli(env, "queue", "run")
    finally:
        q._runner_lock.release()


def test_cli_run_with_real_registry_unknown_kind(env):
    JobQueue(env.qpath).add("mystery")
    with pytest.raises(SystemExit, match="mystery"):
        run_cli(env, "queue", "run")


def test_cli_run_empty_queue(env, capsys):
    run_cli(env, "queue", "run")
    assert "done 0  failed 0" in capsys.readouterr().out


def test_cli_existing_actions_still_work(env, capsys):
    q = JobQueue(env.qpath)
    q.add("a", max_attempts=1)
    q.run(Registry().register("a", lambda ctx: 1 / 0))
    run_cli(env, "queue", "retry")
    assert "revived 1" in capsys.readouterr().out
    q2 = queue_of(env)
    q2.run(Registry().register("a", lambda ctx: {}))
    run_cli(env, "queue", "prune")
    assert "pruned 1" in capsys.readouterr().out
    run_cli(env, "queue", "reset")
    assert "queue cleared" in capsys.readouterr().out


def test_cli_enqueue_then_run_end_to_end(env, monkeypatch, capsys):
    monkeypatch.setattr(replay_mod, "inspect", lambda p: SimpleNamespace(duration_ms=600_000))
    monkeypatch.setattr(pipeline, "plan_replay",
                        lambda cfg, p, **kw: {"out_file": str(env.tmp / "videos" / (Path(p).stem + ".mp4"))})
    run_cli(env, "queue", "enqueue", "--session", "2")
    capsys.readouterr()
    run_cli(env, "queue", "run")
    out = capsys.readouterr().out
    assert "[done   ] j0001 plan" in out
    assert "[waiting] j0002 render_wait" in out
    assert "20260103_120000.mp4" in out or "render" in out
