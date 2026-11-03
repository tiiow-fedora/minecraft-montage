"""Structural conformance of generated renderQueue.json to the documented schema.

This encodes the README's "render-queue schema" section as executable rules and
sweeps seeded random plan parameters through it.  Goldens catch *any* change;
this catches *invalid* ones, including inputs nobody wrote a golden for.
"""

import json
import math
import random
import zipfile

import pytest

import synthetic as syn
from mcmontage import pipeline

ALLOWED_PROPS = {"timestamp", "camera:position", "camera:rotation", "spectate"}
RENDER_SETTINGS_KEYS = {"renderMethod", "encodingPreset", "videoWidth", "videoHeight",
                        "framesPerSecond", "bitRate", "outputFile", "renderNameTags",
                        "antiAliasing"}


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_interpolator(ip):
    t = ip["type"]
    if isinstance(t, dict):
        assert t["type"] == "catmull-rom-spline" and _is_num(t["alpha"])
    else:
        assert t in ("linear", "cubic-spline")
    assert ip["properties"] and set(ip["properties"]) <= ALLOWED_PROPS


def check_path(path, kind):
    assert set(path) == {"keyframes", "segments", "interpolators"}
    kfs = path["keyframes"]
    assert len(kfs) >= 2
    times = [k["time"] for k in kfs]
    assert all(isinstance(t, int) for t in times), "time must be an int (ms)"
    assert times == sorted(times) and times[0] == 0
    assert len(path["segments"]) == len(kfs) - 1
    for seg in path["segments"]:
        assert seg is None or 0 <= seg < len(path["interpolators"])
    for ip in path["interpolators"]:
        check_interpolator(ip)
    for k in kfs:
        props = k["properties"]
        assert set(props) <= ALLOWED_PROPS
        if kind == "time":
            assert set(props) == {"timestamp"} and isinstance(props["timestamp"], int)
        else:
            assert set(props) == {"camera:position", "camera:rotation"}
            assert len(props["camera:position"]) == 3
            assert all(_is_num(v) for v in props["camera:position"])
            yaw, pitch, roll = props["camera:rotation"]
            assert all(_is_num(v) for v in (yaw, pitch, roll))
            assert -90.0 <= pitch <= 90.0
            assert roll == 0.0
    if kind == "camera":
        yaws = [k["properties"]["camera:rotation"][0] for k in kfs]
        assert all(abs(b - a) <= 180 for a, b in zip(yaws, yaws[1:])), \
            "yaw must be unwrapped, or interpolation spins the long way round"


def check_queue(raw_json: bytes):
    queue = json.loads(raw_json)
    assert isinstance(queue, list) and queue
    for job in queue:
        assert set(job) == {"timeline", "settings"}
        assert isinstance(job["timeline"], str), "timeline must be a nested JSON STRING"
        tl = json.loads(job["timeline"])
        assert list(tl) == [""]
        time_path, pos_path = tl[""]          # order: TIME=0, POSITION=1
        check_path(time_path, "time")
        check_path(pos_path, "camera")
        assert time_path["keyframes"][-1]["time"] == pos_path["keyframes"][-1]["time"]
        s = job["settings"]
        assert set(s) == RENDER_SETTINGS_KEYS
        assert isinstance(s["outputFile"], str)
        for k in ("videoWidth", "videoHeight", "framesPerSecond", "bitRate"):
            assert isinstance(s[k], int) and s[k] > 0


def written_queue(mcpr):
    with zipfile.ZipFile(mcpr) as z:
        assert z.testzip() is None
        return z.read("renderQueue.json")


@pytest.mark.parametrize("seed", range(40))
def test_random_static_plans_conform(tmp_path, cfg, seed):
    rng = random.Random(seed)
    dur = rng.randrange(5_000, 4_000_000)
    start = rng.randrange(0, dur - 1000)
    end = rng.randrange(start + 1, dur + 1)
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", duration_ms=dur)
    pipeline.plan_replay(
        cfg, mcpr, seconds=rng.choice([1, 7.5, 20, 90]), start=start, end=end,
        follow=False,
        centre=(rng.uniform(-3e4, 3e4), rng.uniform(-60, 300), rng.uniform(-3e4, 3e4)),
        radius=rng.choice([0.5, 12, 80]), height=rng.uniform(-10, 40),
        revolutions=rng.choice([0.25, 0.5, 1, 2.5, 7]),
        keyframes=rng.choice([2, 3, 24, 90]), log=lambda m: None)
    check_queue(written_queue(mcpr))


@pytest.mark.parametrize("seed", range(20))
def test_random_follow_plans_conform(tmp_path, cfg, seed):
    rng = random.Random(1000 + seed)
    dur = rng.randrange(20_000, 600_000)
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr", duration_ms=dur)
    tr = {1: syn.walk_points(0, dur, 50, start=(rng.uniform(-500, 500), 64, 0),
                             velocity=(rng.uniform(-8, 8), 0, rng.uniform(-8, 8)),
                             jitter=rng.uniform(0, 1), seed=seed),
          2: syn.circle_points(0, dur, 1000)}
    syn.write_tracks(cfg.work / "a.tracks.json", tr)
    pipeline.plan_replay(
        cfg, mcpr, seconds=rng.choice([5, 20, 60]), follow=True,
        radius=rng.choice([6, 12, 40]), height=rng.uniform(0, 20),
        revolutions=rng.choice([0.25, 0.5, 1]), keyframes=rng.choice([5, 30, 90]),
        smooth=rng.choice([None, 0, 1000, 4000]), log=lambda m: None)
    check_queue(written_queue(mcpr))


def test_append_conforms(tmp_path, cfg):
    mcpr = syn.make_mcpr(tmp_path / "a.mcpr")
    for i in range(3):
        pipeline.plan_replay(cfg, mcpr, follow=False, centre=(i, 64, i), append=i > 0,
                             out_file=f"o{i}.mp4", log=lambda m: None)
    raw = written_queue(mcpr)
    check_queue(raw)
    assert len(json.loads(raw)) == 3


def test_checker_rejects_known_historic_mistake():
    """README: 'timestamp:timestamp' was once emitted and silently rejected."""
    bad_tl = {"": [
        {"keyframes": [{"time": 0, "properties": {"timestamp:timestamp": 0}},
                       {"time": 10, "properties": {"timestamp:timestamp": 5}}],
         "segments": [0], "interpolators": [{"type": "linear",
                                             "properties": ["timestamp:timestamp"]}]},
        {"keyframes": [], "segments": [], "interpolators": []}]}
    raw = json.dumps([{"timeline": json.dumps(bad_tl), "settings": {}}]).encode()
    with pytest.raises(AssertionError):
        check_queue(raw)


def test_checker_rejects_unnested_timeline():
    raw = json.dumps([{"timeline": {"": []}, "settings": {}}]).encode()
    with pytest.raises(AssertionError):
        check_queue(raw)
