"""tracks.dump() / plan_replay track caching, with a fake `java`.

The real extractor needs the mod jar; what is testable (and where bugs would
destroy a recording) is the plumbing around it: it must run against a COPY,
pass the right argv, and surface failures.
"""

import json
import os
import stat
import sys

import pytest

import synthetic as syn
from mcmontage import pipeline, tracks


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    (build / "cp.txt").write_text('-cp "x"\n')
    (build / "DumpTracks.class").write_bytes(b"\xca\xfe")
    monkeypatch.setattr(tracks, "CP_FILE", build / "cp.txt")
    monkeypatch.setattr(tracks, "CLASSES", build)

    log = tmp_path / "java_calls.json"
    script = tmp_path / "fakejava.py"
    script.write_text(f'''
import json, sys
argv = sys.argv[1:]
mode = {{"fail": False}}
copy, out = argv[-2], argv[-1]
with open(r"{log}", "a") as fh:
    fh.write(json.dumps(argv) + "\\n")
# like EntityPositionTracker: scribbles a cache entry into the replay it is given
with open(copy, "ab") as fh:
    fh.write(b"SCRIBBLE")
import os
if os.environ.get("FAKE_JAVA_FAIL"):
    sys.stderr.write("boom\\nstack trace line\\n"); sys.exit(3)
json.dump({{"7": [[0,0,64,0,0,0],[1000,5,64,0,0,0]]}}, open(out, "w"))
''')
    if os.name == "nt":
        launcher = tmp_path / "java.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        launcher = tmp_path / "java"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return str(launcher), log


def calls(log):
    return [json.loads(l) for l in log.read_text().splitlines()]


def test_dump_runs_against_a_copy_never_the_original(tmp_path, fake_env):
    java, log = fake_env
    mcpr = syn.make_mcpr(tmp_path / "rec.mcpr")
    before = mcpr.read_bytes()
    out = tmp_path / "w" / "rec.tracks.json"
    tracks.dump(mcpr, out, java=java)
    assert mcpr.read_bytes() == before, "original recording was modified"
    (argv,) = calls(log)
    assert argv[0].startswith("@") and argv[0].endswith("cp.txt")
    assert argv[1] == "DumpTracks"
    assert argv[2] != str(mcpr), "extractor must be handed a temp copy"
    assert argv[3] == str(out)
    assert set(tracks.load(out)) == {7}


def test_dump_failure_reports_stderr_tail(tmp_path, fake_env, monkeypatch):
    java, _ = fake_env
    monkeypatch.setenv("FAKE_JAVA_FAIL", "1")
    mcpr = syn.make_mcpr(tmp_path / "rec.mcpr")
    with pytest.raises(tracks.TrackError, match="exit 3") as e:
        tracks.dump(mcpr, tmp_path / "o.json", java=java)
    assert "stack trace line" in str(e.value)


def test_dump_requires_compiled_class(tmp_path, fake_env):
    java, _ = fake_env
    (tracks.CLASSES / "DumpTracks.class").unlink()
    with pytest.raises(tracks.TrackError, match="javac"):
        tracks.dump(syn.make_mcpr(tmp_path / "r.mcpr"), tmp_path / "o.json", java=java)


def test_plan_follow_uses_cache_and_retrack_refreshes(tmp_path, cfg, monkeypatch):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", duration_ms=60_000)
    n = {"dumps": 0}

    def fake_dump(path, out_json, java="java"):
        n["dumps"] += 1
        syn.write_tracks(out_json, syn.standard_tracks())
        return out_json

    monkeypatch.setattr(tracks, "dump", fake_dump)
    kw = dict(follow=True, keyframes=5, log=lambda m: None)
    pipeline.plan_replay(cfg, mcpr, **kw)
    pipeline.plan_replay(cfg, mcpr, **kw)
    assert n["dumps"] == 1, "second plan should reuse work/a.tracks.json"
    pipeline.plan_replay(cfg, mcpr, retrack=True, **kw)
    assert n["dumps"] == 2


def test_plan_follow_failure_leaves_recording_untouched(tmp_path, cfg, monkeypatch):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", duration_ms=60_000)
    before = mcpr.read_bytes()

    def boom(*a, **k):
        raise tracks.TrackError("no jar")

    monkeypatch.setattr(tracks, "dump", boom)
    with pytest.raises(tracks.TrackError):
        pipeline.plan_replay(cfg, mcpr, follow=True, log=lambda m: None)
    assert mcpr.read_bytes() == before


def test_batch_continues_past_one_track_extraction_failure(tmp_path, cfg, monkeypatch):
    from datetime import datetime
    paths = [syn.make_mcpr(tmp_path / "rec" / f"r{i}.mcpr", duration_ms=60_000,
                           date=datetime(2026, 9, 1, 20 + i)) for i in range(3)]

    def flaky(path, out_json, java="java"):
        if path.name == "r1.mcpr":
            raise tracks.TrackError("DumpTracks failed (exit 1)")
        syn.write_tracks(out_json, syn.standard_tracks())
        return out_json

    monkeypatch.setattr(tracks, "dump", flaky)
    results, failures = pipeline.batch_prepare(
        cfg, paths, seconds=5, follow=True, keyframes=4, log=lambda m: None)
    assert len(results) == 2
    assert [p.name for p, _ in failures] == ["r1.mcpr"]


def test_cli_plan_reports_track_errors_cleanly(tmp_path, monkeypatch):
    from mcmontage import cli
    rec = tmp_path / "rec"
    mcpr = syn.make_mcpr(rec / "a.mcpr", duration_ms=60_000)
    conf = tmp_path / "c.toml"
    conf.write_text('[paths]\nprofile="%s"\nwork="%s"\n' % (
        (tmp_path / "p").as_posix(), (tmp_path / "w").as_posix()))
    monkeypatch.setattr(tracks, "dump",
                        lambda *a, **k: (_ for _ in ()).throw(tracks.TrackError("no jar")))
    with pytest.raises(SystemExit, match="no jar"):
        cli.main(["--config", str(conf), "plan", str(mcpr), "--follow"])
