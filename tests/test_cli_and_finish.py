import json
from datetime import datetime

import pytest

import synthetic as syn
from golden import assert_golden, norm_tmp
from mcmontage import cli, music, pipeline, youtube


@pytest.fixture
def env(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    conf = tmp_path / "config.toml"
    conf.write_text(
        '[paths]\nprofile="%s"\nrecordings="%s"\nvideos="%s"\noutput="%s"\nwork="%s"\n'
        % tuple(str(tmp_path / n).replace("\\", "/") for n in
                ("prof", "rec", "vids", "out", "work")))
    return tmp_path, conf


def run(conf, *argv):
    return cli.main(["--config", str(conf), *map(str, argv)])


def norm(text, tmp):
    return norm_tmp(text, tmp)


def test_cli_plan_static_then_inspect_golden(env, capsys):
    tmp, conf = env
    p = syn.make_mcpr(tmp / "rec" / "a.mcpr")
    run(conf, "plan", p, "-x", 10, "-y", 64, "-z", -3, "--keyframes", 5, "--seconds", 20)
    capsys.readouterr()
    run(conf, "inspect", p)
    assert_golden("cli_inspect_after_plan.txt", norm(capsys.readouterr().out, tmp))


def test_cli_plan_static_without_centre_exits(env):
    tmp, conf = env
    p = syn.make_mcpr(tmp / "rec" / "a.mcpr")
    with pytest.raises(SystemExit, match="X/Y/Z|-x -y -z"):
        run(conf, "plan", p)


def test_cli_plan_zero_coordinates_are_valid(env):
    tmp, conf = env
    p = syn.make_mcpr(tmp / "rec" / "a.mcpr")
    run(conf, "plan", p, "-x", 0, "-y", 0, "-z", 0, "--keyframes", 3)


def test_cli_picks_newest_replay_when_none_given(env, capsys):
    import os
    tmp, conf = env
    old = syn.make_mcpr(tmp / "rec" / "old.mcpr")
    new = syn.make_mcpr(tmp / "rec" / "new.mcpr")
    os.utime(old, (1000, 1000)); os.utime(new, (2000, 2000))
    run(conf, "inspect")
    assert "new.mcpr" in capsys.readouterr().out


def test_cli_no_replays_message(env):
    _, conf = env
    with pytest.raises(SystemExit, match="no .mcpr files"):
        run(conf, "inspect")


def test_cli_inspect_dump_omits_private_raw_field(env):
    tmp, conf = env
    p = syn.make_mcpr(tmp / "rec" / "a.mcpr")
    run(conf, "plan", p, "-x", 0, "-y", 0, "-z", 0, "--keyframes", 3)
    run(conf, "inspect", p, "--dump", tmp / "dump.json")
    dumped = json.loads((tmp / "dump.json").read_text())
    assert dumped and all("_timeline_raw" not in j for j in dumped)
    assert isinstance(dumped[0]["timeline"], dict)


def _three_day_setup(tmp):
    for name, when in (("r1", datetime(2026, 9, 1, 20)), ("r2", datetime(2026, 9, 2, 1)),
                       ("r3", datetime(2026, 9, 5, 21))):
        syn.make_mcpr(tmp / "rec" / f"{name}.mcpr", date=when, duration_ms=1_800_000)


def test_cli_sessions_golden(env, capsys):
    tmp, conf = env
    _three_day_setup(tmp)
    run(conf, "sessions")
    assert_golden("cli_sessions.txt", norm(capsys.readouterr().out, tmp))


def test_cli_batch_dry_run_writes_nothing(env, capsys):
    tmp, conf = env
    _three_day_setup(tmp)
    before = {p.name: p.read_bytes() for p in (tmp / "rec").iterdir()}
    run(conf, "batch", "--dry-run", "--rename", "--seconds", 20, "--max-speedup", 30)
    after = {p.name: p.read_bytes() for p in (tmp / "rec").iterdir()}
    assert before == after
    assert_golden("cli_batch_dry_run.txt", norm(capsys.readouterr().out, tmp))


def test_cli_batch_max_speedup_zero_means_whole_replay(env, capsys):
    tmp, conf = env
    _three_day_setup(tmp)
    run(conf, "batch", "--dry-run", "--max-speedup", 0, "--seconds", 10)
    out = capsys.readouterr().out
    assert "30.0 min -> 10s  (x180)" in out


def test_cli_batch_unknown_session(env):
    tmp, conf = env
    _three_day_setup(tmp)
    with pytest.raises(SystemExit, match="no session 9"):
        run(conf, "batch", "--session", 9, "--dry-run")


# ---- finish_video / upload ---------------------------------------------------

@pytest.fixture
def library(tmp_path, monkeypatch):
    f1 = syn.make_tone(tmp_path / "lib" / "one.wav", 1.0) if syn.have_ffmpeg() else tmp_path / "one.wav"
    f2 = syn.make_tone(tmp_path / "lib" / "two.wav", 1.0, freq=660) if syn.have_ffmpeg() else tmp_path / "two.wav"
    lib = [
        music.Track("one", f1, "One", "Ann", "src", "CC-BY-4.0", url="https://x.test/1"),
        music.Track("two", f2, "Two", "Bob", "src", "CC0"),
    ]
    monkeypatch.setattr(music, "load_library", lambda *a, **k: lib)
    for t in lib:
        t.file.parent.mkdir(parents=True, exist_ok=True)
        t.file.touch()
    return lib


def test_resolve_tracks_accepts_str_list_none(library):
    assert pipeline.resolve_tracks(None) == []
    assert pipeline.resolve_tracks([]) == []
    assert [t.id for t in pipeline.resolve_tracks("one")] == ["one"]
    assert [t.id for t in pipeline.resolve_tracks(["two", "one"])] == ["two", "one"]
    with pytest.raises(music.MusicError):
        pipeline.resolve_tracks(["nope"])


def test_description_always_credits_music(library):
    d = pipeline.build_upload_description("My trip", ["one", "two"])
    assert_golden("description_with_credits.txt", d)


def test_description_without_body_or_music(library):
    assert pipeline.build_upload_description("", None) == ""
    assert pipeline.build_upload_description("   ", None) == ""


def test_upload_private_is_always_private(tmp_path, cfg, library, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x" * 10)
    seen = {}

    def fake_upload(path, meta, progress=None):
        seen["meta"] = meta
        return {"id": "abc123"}

    monkeypatch.setattr(youtube, "upload", fake_upload)
    resp = pipeline.upload_private(cfg, v, "  Title  ", "desc", tags=[" a ", "", "b"],
                                   music_ids=["one"], log=lambda m: None)
    m = seen["meta"]
    assert m.privacy == "private"
    assert m.title == "Title" and m.tags == ["a", "b"]
    assert "Ann" in m.description and m.description.startswith("desc")
    assert youtube.watch_url(resp) == "https://www.youtube.com/watch?v=abc123"
    assert youtube.studio_url(resp).endswith("/abc123/edit")


def test_upload_private_validation_errors(tmp_path, cfg, library):
    with pytest.raises(pipeline.PipelineError, match="no such video"):
        pipeline.upload_private(cfg, tmp_path / "nope.mp4", "t")
    v = tmp_path / "v.mp4"; v.write_bytes(b"x")
    with pytest.raises(pipeline.PipelineError, match="title"):
        pipeline.upload_private(cfg, v, "   ")
    with pytest.raises(pipeline.PipelineError, match="limit"):
        pipeline.upload_private(cfg, v, "x" * 101)


def test_videometa_body_shape():
    body = youtube.VideoMeta("T", "D", ["a"]).to_body()
    assert body == {"snippet": {"title": "T", "description": "D", "tags": ["a"],
                                "categoryId": "20"},
                    "status": {"privacyStatus": "private",
                               "selfDeclaredMadeForKids": False}}


def test_videometa_limits():
    for kw in (dict(title=""), dict(title="x" * 101), dict(title="t", description="d" * 5001),
               dict(title="t", privacy="secret"), dict(title="t", tags=["x" * 501])):
        with pytest.raises(youtube.UploadError):
            youtube.VideoMeta(**kw).validate()
    youtube.VideoMeta("x" * 100, "d" * 5000).validate()


def test_url_helpers_without_id():
    assert youtube.watch_url({}) == "" and youtube.watch_url(None) == ""
    assert youtube.studio_url({}) == ""


@pytest.mark.ffmpeg
def test_finish_video_end_to_end(tmp_path, cfg, library):
    from mcmontage import edit
    a = syn.make_clip(tmp_path / "a.mp4", seconds=4)
    b = syn.make_clip(tmp_path / "b.mp4", seconds=4, pattern="testsrc")
    cfg.xfade, cfg.crf = 1.0, 35
    lines = []
    out, tracks = pipeline.finish_video(cfg, [a, b], music_ids=["one", "two"],
                                        log=lines.append)
    assert out == cfg.output / "a_final.mp4"
    p = edit.probe(out)
    assert p.has_audio and p.duration == pytest.approx(7, abs=0.4)
    assert [t.id for t in tracks] == ["one", "two"]
    assert any("Required credit" in l for l in lines)
    assert any("Ann" in l for l in lines)


def test_finish_video_no_clips(cfg):
    with pytest.raises(pipeline.PipelineError, match="no clips"):
        pipeline.finish_video(cfg, [])
