import json
import math
import zipfile

import pytest

from mcmontage import campath as cp
from mcmontage import replay, timeline
from mcmontage.events import Highlight
from mcmontage.tracks import Track

POS, ROT = timeline.PROP_POSITION, timeline.PROP_ROTATION


# ----------------------------------------------------------------- helpers

def eyes_rots(path):
    return ([k["properties"][POS] for k in path.keyframes],
            [k["properties"][ROT] for k in path.keyframes])


def times(path):
    return [k["time"] for k in path.keyframes]


def look_vec(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return (-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))


def assert_looks_at(path, targets, tol_deg=0.02):
    eyes, rots = eyes_rots(path)
    assert len(eyes) == len(targets)
    for e, r, t in zip(eyes, rots, targets):
        d = [t[i] - e[i] for i in range(3)]
        n = math.sqrt(sum(c * c for c in d))
        v = look_vec(r[0], r[1])
        dot = sum(v[i] * d[i] / n for i in range(3))
        assert math.degrees(math.acos(max(-1.0, min(1.0, dot)))) < tol_deg


def assert_valid(path):
    ts = times(path)
    assert all(isinstance(t, int) for t in ts)
    assert all(b > a for a, b in zip(ts, ts[1:]))
    json.dumps(path.to_json(), allow_nan=False)
    eyes, rots = eyes_rots(path)
    for v in eyes + rots:
        assert all(type(c) is float and math.isfinite(c) for c in v)
    assert all(-90 <= r[1] <= 90 for r in rots)


def hl(kind="death", start=10_000, end=20_000, pos=(100.0, 70.0, -50.0), entity=1, **kw):
    return Highlight(kind=kind, start_ms=start, end_ms=end, score=0.5,
                     entity=entity, pos=pos, **kw)


def synth_track(entity=1, n=200, step=100):
    pts = [(i * step, 100 + i * 0.2, 70.0, -50 + 3 * math.sin(i / 10), 0.0, 0.0)
           for i in range(n)]
    return Track(entity, pts)


ALL_RECIPE_KINDS = ("death", "kill", "speed", "build", "fall", "custom")


# ----------------------------------------------------------------- easing

@pytest.mark.parametrize("name", sorted(cp.EASINGS))
def test_easing_endpoints(name):
    f = cp.get_easing(name)
    assert f(0.0) == 0.0
    assert f(1.0) == 1.0


@pytest.mark.parametrize("name", sorted(set(cp.EASINGS) - cp.OVERSHOOTING))
def test_easing_monotonic_and_bounded(name):
    f = cp.get_easing(name)
    vals = [f(i / 200) for i in range(201)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert all(-1e-12 <= v <= 1 + 1e-12 for v in vals)


@pytest.mark.parametrize("name", sorted(cp.OVERSHOOTING))
def test_back_easings_overshoot(name):
    vals = [cp.get_easing(name)(i / 200) for i in range(201)]
    assert min(vals) < -1e-3 or max(vals) > 1 + 1e-3


def test_easing_clamps_input():
    f = cp.get_easing("ease_in_out_cubic")
    assert f(-3) == 0.0 and f(7) == 1.0


@pytest.mark.parametrize("name", ["ease_in_out_quad", "ease_in_out_cubic",
                                  "ease_in_out_sine", "smoothstep", "smootherstep"])
def test_in_out_symmetric_midpoint(name):
    f = cp.get_easing(name)
    assert f(0.5) == pytest.approx(0.5)
    assert f(0.25) + f(0.75) == pytest.approx(1.0)


def test_ease_in_slow_start_ease_out_fast_start():
    assert cp.get_easing("ease_in_cubic")(0.2) < 0.2
    assert cp.get_easing("ease_out_cubic")(0.2) > 0.2


def test_aliases_registered():
    assert cp.get_easing("ease_in") is cp.get_easing("ease_in_quad")
    assert cp.get_easing("ease_out") is cp.get_easing("ease_out_quad")
    assert cp.get_easing("ease_in_out") is cp.get_easing("ease_in_out_quad")


def test_unknown_easing_raises():
    with pytest.raises(ValueError, match="unknown easing"):
        cp.get_easing("bouncy")
    with pytest.raises(ValueError):
        cp.get_easing(None)


def test_callable_easing_accepted():
    p = cp.orbit_shot((0, 70, 0), 1000, easing=lambda t: t, keyframes=5)
    assert_valid(p)


# ----------------------------------------------------------------- orbit

def test_orbit_distance_constant():
    c = (10.0, 70.0, 20.0)
    p = cp.orbit_shot(c, 4000, radius=9, height=3, revolutions=1.3, keyframes=40)
    eyes, _ = eyes_rots(p)
    for e in eyes:
        assert math.dist(e, c) == pytest.approx(math.hypot(9, 3), abs=1e-3)
    assert_valid(p)


def test_orbit_look_at_consistency():
    c = (10.0, 70.0, 20.0)
    p = cp.orbit_shot(c, 4000, keyframes=30, revolutions=2)
    assert_looks_at(p, [c] * 30)


def test_orbit_times_span_duration():
    p = cp.orbit_shot((0, 70, 0), 3000, keyframes=10)
    ts = times(p)
    assert ts[0] == 0 and ts[-1] == 3000 and len(ts) == 10


def test_orbit_spiral_in_radius_shrinks():
    c = (0.0, 70.0, 0.0)
    p = cp.orbit_shot(c, 3000, radius=20, end_radius=5, height=0, keyframes=20)
    d = [math.dist((e[0], 0, e[2]), (0, 0, 0)) for e in eyes_rots(p)[0]]
    assert d[0] == pytest.approx(20, abs=1e-3)
    assert d[-1] == pytest.approx(5, abs=1e-3)
    assert all(b <= a + 1e-6 for a, b in zip(d, d[1:]))


def test_orbit_height_easing():
    p = cp.orbit_shot((0, 0, 0), 1000, radius=10, height=2, end_height=12, keyframes=9)
    ys = [e[1] for e in eyes_rots(p)[0]]
    assert ys[0] == pytest.approx(2) and ys[-1] == pytest.approx(12)


def test_orbit_eased_angular_velocity():
    c = (0.0, 0.0, 0.0)
    p = cp.orbit_shot(c, 1000, radius=10, height=0, revolutions=0.5,
                      easing="ease_in_out_cubic", keyframes=21, safety=cp.Safety(min_y=-100))
    eyes, _ = eyes_rots(p)
    ang = [math.atan2(e[2], e[0]) for e in eyes]
    steps = [abs(b - a) for a, b in zip(ang, ang[1:])]
    assert steps[0] < steps[len(steps) // 2] and steps[-1] < steps[len(steps) // 2]


def test_orbit_negative_revolutions_reverses():
    a = cp.orbit_shot((0, 0, 0), 1000, revolutions=0.25, keyframes=5, easing="linear",
                      safety=cp.Safety(min_y=-100))
    b = cp.orbit_shot((0, 0, 0), 1000, revolutions=-0.25, keyframes=5, easing="linear",
                      safety=cp.Safety(min_y=-100))
    assert eyes_rots(a)[0][-1][2] > 0 > eyes_rots(b)[0][-1][2]


def test_yaw_unwrap_continuity():
    p = cp.orbit_shot((0, 70, 0), 5000, revolutions=3, keyframes=80)
    yaws = [r[0] for r in eyes_rots(p)[1]]
    assert all(abs(b - a) < 180 for a, b in zip(yaws, yaws[1:]))
    assert abs(yaws[-1] - yaws[0]) > 360  # actually accumulated, not wrapped


def test_orbit_target_fn_moving_target():
    tf = lambda f: (f * 50, 70.0, 0.0)
    p = cp.orbit_shot((0, 70, 0), 2000, target_fn=tf, keyframes=11, radius=8, height=3)
    assert_looks_at(p, [tf(i / 10) for i in range(11)])


def test_orbit_interpolator_choice():
    p = cp.orbit_shot((0, 70, 0), 1000, interpolator=timeline.LINEAR)
    assert p.interpolators[0]["type"] == "linear"
    q = cp.orbit_shot((0, 70, 0), 1000)
    assert q.interpolators[0]["type"]["type"] == "catmull-rom-spline"


def test_orbit_keyframe_density():
    assert len(cp.orbit_shot((0, 70, 0), 1000, keyframes=7).keyframes) == 7


# ----------------------------------------------------------------- other shots

def test_dolly_endpoints_and_look():
    s, e, lk = (0.0, 70.0, 0.0), (20.0, 72.0, 30.0), (5.0, 70.0, 5.0)
    p = cp.dolly_shot(s, e, lk, 2000, keyframes=9)
    eyes, _ = eyes_rots(p)
    assert eyes[0] == pytest.approx(list(s)) and eyes[-1] == pytest.approx(list(e))
    assert_looks_at(p, [lk] * 9)
    assert_valid(p)


def test_dolly_eased_slow_start():
    p = cp.dolly_shot((0, 70, 0), (100, 70, 0), (50, 70, 50), 1000,
                      easing="ease_in_cubic", keyframes=11)
    xs = [e[0] for e in eyes_rots(p)[0]]
    assert xs[1] - xs[0] < xs[-1] - xs[-2]


def test_dolly_arc_bows_sideways():
    p = cp.dolly_shot((0, 70, 0), (40, 70, 0), (20, 70, 30), 1000,
                      easing="linear", arc=6, keyframes=9)
    eyes, _ = eyes_rots(p)
    assert eyes[0][2] == pytest.approx(0) and eyes[-1][2] == pytest.approx(0, abs=1e-6)
    assert abs(eyes[4][2]) == pytest.approx(6, abs=1e-3)


def test_dolly_vertical_move_no_crash():
    p = cp.dolly_shot((0, 70, 0), (0, 90, 0), (0, 80, 10), 1000, arc=2)
    assert_valid(p)


def test_crane_rises_and_looks():
    t = (0.0, 70.0, 0.0)
    p = cp.crane_shot(t, 3000, start_height=2, end_height=20, keyframes=12)
    eyes, _ = eyes_rots(p)
    ys = [e[1] for e in eyes]
    assert ys[0] == pytest.approx(72) and ys[-1] == pytest.approx(90)
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
    assert_looks_at(p, [t] * 12)


def test_flyby_passes_target_and_goes_beyond():
    t = (0.0, 70.0, 0.0)
    p = cp.flyby_shot(t, 2000, pass_distance=8, approach=30, direction_deg=0,
                      keyframes=21, height=0)
    eyes, _ = eyes_rots(p)
    d = [math.dist((e[0], e[2]), (0, 0)) for e in eyes]
    assert min(d) == pytest.approx(8, abs=1e-3)
    assert eyes[0][0] < 0 < eyes[-1][0]          # approaches then passes
    assert d.index(min(d)) == 10                 # closest at the midpoint
    assert_looks_at(p, [t] * 21)


def test_static_shot_two_keyframes_linear():
    p = cp.static_shot((5, 70, 5), (0, 70, 0), 1500)
    assert len(p.keyframes) == 2 and p.interpolators[0]["type"] == "linear"
    assert times(p) == [0, 1500]
    assert_looks_at(p, [(0, 70, 0)] * 2)
    assert cp.hold is cp.static_shot


def test_follow_shot_tracks_moving_target():
    trk = synth_track()
    p = cp.follow_shot(trk, 2000, 12000, 4000, keyframes=20, radius=10, height=4)
    eyes, _ = eyes_rots(p)
    assert eyes[0][0] < eyes[-1][0] - 5      # camera travelled with the player
    assert_valid(p)


def test_follow_shot_look_matches_track():
    trk = synth_track()
    p = cp.follow_shot(trk, 2000, 12000, 4000, keyframes=10, eye_height=1.6)
    eyes, rots = eyes_rots(p)
    # target must be inside the smoothing deviation budget of the real track
    for i, (e, r) in enumerate(zip(eyes, rots)):
        ts = 2000 + 10000 * i / 9
        real = trk.at(ts)
        d = look_vec(*r[:2])
        # ray from eye along look passes within ~8 blocks of the true position
        v = [real[j] + (1.6 if j == 1 else 0) - e[j] for j in range(3)]
        t = sum(v[j] * d[j] for j in range(3))
        perp = math.sqrt(max(0, sum(c * c for c in v) - t * t))
        assert perp < 8


def test_follow_shot_time_map_used():
    trk = synth_track()
    lin = cp.follow_shot(trk, 0, 19000, 4000, keyframes=10)
    late = cp.follow_shot(trk, 0, 19000, 4000, keyframes=10, time_map=lambda ms: 18000)
    assert eyes_rots(lin)[0][1] != eyes_rots(late)[0][1]


def test_follow_shot_errors():
    with pytest.raises(cp.CamPathError):
        cp.follow_shot(Track(1, []), 0, 1000, 1000)
    with pytest.raises(cp.CamPathError):
        cp.follow_shot(synth_track(), 5000, 5000, 1000)


# ----------------------------------------------------------------- safety

def test_floor_clamp():
    p = cp.orbit_shot((0, -60, 0), 1000, radius=10, height=-30, keyframes=8)
    assert all(e[1] >= -64 - 1e-9 for e in eyes_rots(p)[0])


def test_floor_configurable():
    p = cp.orbit_shot((0, 10, 0), 1000, radius=10, height=-30, keyframes=8,
                      safety=cp.Safety(min_y=5))
    assert all(e[1] >= 5 - 1e-9 for e in eyes_rots(p)[0])


def test_max_y_ceiling():
    p = cp.crane_shot((0, 300, 0), 1000, end_height=100, safety=cp.Safety(max_y=320))
    assert all(e[1] <= 320 + 1e-9 for e in eyes_rots(p)[0])


def test_min_distance_enforced():
    c = (0.0, 70.0, 0.0)
    p = cp.orbit_shot(c, 1000, radius=0.5, height=0.5, keyframes=6,
                      safety=cp.Safety(min_distance=4.0))
    for e in eyes_rots(p)[0]:
        assert math.dist(e, c) >= 4.0 - 1e-3


def test_min_distance_zero_distance_eye():
    c = (0.0, 70.0, 0.0)
    p = cp.static_shot(c, c, 1000, safety=cp.Safety(min_distance=3))
    assert all(math.dist(e, c) >= 3 - 1e-3 for e in eyes_rots(p)[0])
    assert_valid(p)


def test_min_distance_with_floor_near_target():
    t = (0.0, -63.0, 0.0)
    p = cp.static_shot((0.0, -63.5, 0.1), t, 1000, safety=cp.Safety(min_distance=5))
    for e in eyes_rots(p)[0]:
        assert e[1] >= -64 - 1e-9
        assert math.dist(e, t) >= 5 - 1e-2


def test_straight_down_pitch_clamped():
    p = cp.static_shot((0, 100, 0), (0, 70, 0), 1000)
    assert eyes_rots(p)[1][0][1] == pytest.approx(90.0)
    assert_valid(p)


def test_clamp_pitch():
    assert cp.clamp_pitch(120) == 90 and cp.clamp_pitch(-95) == -90
    assert cp.clamp_pitch(12.5) == 12.5


def test_times_deduped_when_collapsing():
    # 1 ms across 30 keyframes: many round to the same ms; end must survive
    p = cp.orbit_shot((0, 70, 0), 3, keyframes=30)
    ts = times(p)
    assert ts[-1] == 3 and ts[0] == 0
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_nonfinite_inputs_rejected():
    with pytest.raises(cp.CamPathError):
        cp.orbit_shot((float("nan"), 0, 0), 1000)
    with pytest.raises(cp.CamPathError):
        cp.orbit_shot((0, 0, 0), 1000, radius=float("inf"))
    with pytest.raises(cp.CamPathError):
        cp.static_shot((0, 0), (0, 0, 0), 1000)
    with pytest.raises(cp.CamPathError):
        cp.dolly_shot((0, 0, 0), (1, 1, 1), (0, 0, "x"), 1000)


def test_bad_duration_and_keyframes():
    with pytest.raises(cp.CamPathError):
        cp.orbit_shot((0, 70, 0), 0)
    with pytest.raises(cp.CamPathError):
        cp.orbit_shot((0, 70, 0), 1000, keyframes=1)
    with pytest.raises(cp.CamPathError):
        cp.flyby_shot((0, 70, 0), -5)


def test_camerror_is_valueerror():
    assert issubclass(cp.CamPathError, ValueError)


def test_json_plain_types_with_numpy_like():
    class F(float):
        pass
    p = cp.orbit_shot((F(1), F(70), F(2)), 1000, radius=F(8))
    assert_valid(p)
    json.dumps(p.to_json())


# ----------------------------------------------------------------- speed ramp

def test_speed_ramp_hits_endpoints_exactly():
    p = cp.speed_ramp(10_000, 20_000, 5000, [(0.4, 1.0), (0.3, 0.25), (0.3, 1.0)])
    assert p.keyframes[0] == {"time": 0, "properties": {"timestamp": 10_000}}
    assert p.keyframes[-1] == {"time": 5000, "properties": {"timestamp": 20_000}}


def test_speed_ramp_strictly_monotonic():
    p = cp.speed_ramp(1000, 9000, 4000, [(0.5, 1), (0.2, 0.1), (0.3, 2)], keyframes=60)
    ts = times(p)
    stamps = [k["properties"]["timestamp"] for k in p.keyframes]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    json.dumps(p.to_json())


def test_speed_ramp_slow_section_is_slower():
    p = cp.speed_ramp(0, 10_000, 5000, [(0.4, 1.0), (0.2, 0.1), (0.4, 1.0)], keyframes=101)
    stamps = [k["properties"]["timestamp"] for k in p.keyframes]
    d = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(d) < 0.3 * max(d)
    assert d.index(min(d)) in range(35, 65)


def test_speed_ramp_uniform_equals_linear():
    p = cp.speed_ramp(0, 8000, 4000, [(1, 1)], keyframes=9)
    stamps = [k["properties"]["timestamp"] for k in p.keyframes]
    assert stamps == [i * 1000 for i in range(9)]


def test_speed_ramp_uses_linear_interpolator():
    p = cp.speed_ramp(0, 1000, 1000, [(1, 1)])
    assert p.interpolators == [{"type": "linear", "properties": ["timestamp"]}]


@pytest.mark.parametrize("rates", [[], [(0, 1)], [(1, 0)], [(1, -1)], [(float("nan"), 1)]])
def test_speed_ramp_bad_rates(rates):
    with pytest.raises(cp.CamPathError):
        cp.speed_ramp(0, 1000, 1000, rates)


def test_speed_ramp_bad_spans():
    with pytest.raises(cp.CamPathError):
        cp.speed_ramp(1000, 1000, 1000, [(1, 1)])
    with pytest.raises(cp.CamPathError):
        cp.speed_ramp(0, 1000, 0, [(1, 1)])
    with pytest.raises(cp.CamPathError):
        cp.speed_ramp(0, 1000, 1000, [(1, 1)], keyframes=1)


def test_speed_ramp_tiny_span_still_valid():
    p = cp.speed_ramp(0, 5, 2000, [(1, 1), (1, 0.5)], keyframes=40)
    stamps = [k["properties"]["timestamp"] for k in p.keyframes]
    assert stamps[0] == 0 and stamps[-1] == 5
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


def test_speed_profile_replay_at():
    prof = cp.speed_profile(1000, 5000, 2000, [(1, 1), (1, 3)])
    assert prof.replay_at(-5) == 1000 and prof.replay_at(0) == 1000
    assert prof.replay_at(2000) == 5000 and prof.replay_at(9999) == 5000
    vals = [prof.replay_at(i * 20) for i in range(101)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


# ----------------------------------------------------------------- choose_shot

def test_choose_shot_deterministic():
    for k in ALL_RECIPE_KINDS:
        h = hl(kind=k, start=33_333, end=40_000)
        assert cp.choose_shot(h, synth_track()) == cp.choose_shot(h, synth_track())


def test_choose_shot_varies_with_start():
    recipes = {repr(cp.choose_shot(hl(kind="custom", start=s * 1000, end=s * 1000 + 5000)))
               for s in range(40)}
    assert len(recipes) > 3


@pytest.mark.parametrize("kind", ALL_RECIPE_KINDS)
def test_choose_shot_valid_recipe(kind):
    r = cp.choose_shot(hl(kind=kind), synth_track())
    assert r.shot in cp.SHOT_KINDS
    cp.get_easing(r.easing)


def test_choose_shot_slowmo_for_death_and_kill():
    assert cp.choose_shot(hl("death")).slowmo
    assert cp.choose_shot(hl("kill")).slowmo
    assert not cp.choose_shot(hl("build")).slowmo


def test_choose_shot_follow_requires_track():
    for s in range(60):
        h = hl("speed", start=s * 777, end=s * 777 + 5000)
        assert cp.choose_shot(h, None).shot != "follow"
    shots = {cp.choose_shot(hl("speed", start=s * 777, end=s * 777 + 5000),
                            synth_track()).shot for s in range(60)}
    assert "follow" in shots


def test_choose_shot_unknown_kind_falls_back():
    h = Highlight(kind="weird", start_ms=0, end_ms=1000, score=0.1, pos=(0, 0, 0))
    assert cp.choose_shot(h).shot in cp.SHOT_KINDS


def test_choose_shot_stable_across_processes():
    import subprocess, sys
    code = ("from mcmontage import campath as c; from mcmontage.events import Highlight;"
            "print(c.choose_shot(Highlight('death',12345,20000,.5,pos=(0,0,0))))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(i), "PYTHONPATH": "."}).stdout
            for i in range(3)}
    assert len(outs) == 1 and "ShotRecipe" in outs.pop()


# ----------------------------------------------------------------- plan_highlight

@pytest.mark.parametrize("kind", ALL_RECIPE_KINDS)
def test_plan_highlight_all_kinds(kind):
    h = hl(kind=kind)
    tp, pp = cp.plan_highlight(h, synth_track(), 6.0)
    assert_valid(pp)
    assert times(pp)[0] == 0 and times(pp)[-1] == 6000
    tk = tp.keyframes
    assert tk[0] == {"time": 0, "properties": {"timestamp": 10_000}}
    assert tk[-1] == {"time": 6000, "properties": {"timestamp": 20_000}}
    json.dumps(timeline.build_timeline(tp, pp))


@pytest.mark.parametrize("shot", cp.SHOT_KINDS)
def test_plan_highlight_every_shot_kind(shot):
    params = {"static": dict(distance=6.0, height=2.0),
              "dolly": dict(distance=15.0, end_distance=6.0)}.get(shot, {})
    r = cp.ShotRecipe(shot=shot, easing="smoothstep", params=params)
    tp, pp = cp.plan_highlight(hl(), synth_track(), 3.0, recipe=r)
    assert_valid(pp)
    for e in eyes_rots(pp)[0]:
        assert math.dist(e, (100.0, 71.0, -50.0)) >= 2.0 - 1e-3 or shot == "follow"


def test_plan_highlight_slowmo_adds_time_keyframes():
    r = cp.ShotRecipe(shot="orbit", slowmo=True)
    tp, _ = cp.plan_highlight(hl(), None, 5.0, recipe=r)
    assert len(tp.keyframes) > 2
    tp2, _ = cp.plan_highlight(hl(), None, 5.0, recipe=r, slowmo=False)
    assert len(tp2.keyframes) == 2


def test_plan_highlight_deterministic():
    a = cp.plan_highlight(hl(), synth_track(), 5.0)
    b = cp.plan_highlight(hl(), synth_track(), 5.0)
    assert (a[0].to_json(), a[1].to_json()) == (b[0].to_json(), b[1].to_json())


def test_plan_highlight_focus_from_track_when_no_pos():
    h = hl(pos=None)
    r = cp.ShotRecipe(shot="orbit", params=dict(radius=10.0, height=0.0))
    _, pp = cp.plan_highlight(h, synth_track(), 3.0, recipe=r)
    assert_valid(pp)


def test_plan_highlight_no_pos_no_track_errors():
    with pytest.raises(cp.CamPathError, match="no pos"):
        cp.plan_highlight(hl(pos=None), None, 3.0, recipe=cp.ShotRecipe(shot="orbit"))


def test_plan_highlight_follow_without_track_errors():
    with pytest.raises(cp.CamPathError, match="track"):
        cp.plan_highlight(hl(), None, 3.0, recipe=cp.ShotRecipe(shot="follow"))


@pytest.mark.parametrize("secs", [0, -1, float("nan"), float("inf")])
def test_plan_highlight_bad_seconds(secs):
    with pytest.raises(cp.CamPathError):
        cp.plan_highlight(hl(), None, secs)


def test_plan_highlight_unknown_shot():
    with pytest.raises(cp.CamPathError, match="unknown shot"):
        cp.plan_highlight(hl(), None, 3.0, recipe=cp.ShotRecipe(shot="spin"))


def test_plan_highlight_keyframes_and_interpolator_options():
    r = cp.ShotRecipe(shot="orbit")
    _, pp = cp.plan_highlight(hl(), None, 3.0, recipe=r, keyframes=33,
                              interpolator=timeline.CUBIC)
    assert len(pp.keyframes) == 33 and pp.interpolators[0]["type"] == "cubic-spline"


def test_plan_highlight_floor_respected_underground():
    h = hl(pos=(0.0, -62.0, 0.0), kind="fall")
    _, pp = cp.plan_highlight(h, None, 3.0)
    assert all(e[1] >= -64 - 1e-9 for e in eyes_rots(pp)[0])


def test_plan_highlight_follow_with_slowmo_uses_time_map():
    r = cp.ShotRecipe(shot="follow", slowmo=True)
    tp, pp = cp.plan_highlight(hl("kill", 0, 15000), synth_track(), 4.0, recipe=r)
    assert_valid(pp) and len(tp.keyframes) > 2


# ----------------------------------------------------------------- plan_montage

def montage_hls():
    return [hl("death", 5_000, 10_000, entity=1), hl("kill", 30_000, 36_000, entity=2),
            hl("build", 50_000, 60_000, entity=1), hl("speed", 70_000, 75_000, entity=9)]


def test_plan_montage_names_distinct_and_formatted(tmp_path):
    jobs = cp.plan_montage(montage_hls(), synth_track(), 4.0, tmp_path, stem="clip")
    names = [j["settings"]["outputFile"] for j in jobs]
    assert len(set(names)) == 4
    assert names[0] == str(tmp_path / "clip_h01_death.mp4")
    assert names[1].endswith("clip_h02_kill.mp4")


def test_plan_montage_without_out_dir():
    jobs = cp.plan_montage(montage_hls()[:1], None, 3.0)
    assert jobs[0]["settings"]["outputFile"] == "montage_h01_death.mp4"


def test_plan_montage_settings_passed():
    jobs = cp.plan_montage(montage_hls()[:2], synth_track(), 3.0,
                           settings=dict(width=1280, height=720, fps=30,
                                         encoding_preset="WEBM_CUSTOM"))
    s = jobs[0]["settings"]
    assert (s["videoWidth"], s["videoHeight"], s["framesPerSecond"]) == (1280, 720, 30)
    assert s["outputFile"].endswith(".webm")


def test_plan_montage_track_dict_by_entity():
    tr = {1: synth_track(1), 2: synth_track(2)}
    jobs = cp.plan_montage(montage_hls(), tr, 3.0)
    assert len(jobs) == 4   # entity 9 has no track but has a pos: still plannable


def test_plan_montage_empty():
    assert cp.plan_montage([], None, 3.0) == []


def test_plan_montage_bad_settings():
    with pytest.raises(ValueError):
        cp.plan_montage(montage_hls(), None, 3.0, settings={"anti_aliasing": "X99"})
    with pytest.raises(cp.CamPathError):
        cp.plan_montage(montage_hls(), None, 3.0, settings={"output_file": "x.mp4"})
    with pytest.raises(cp.CamPathError):
        cp.plan_montage(montage_hls(), None, 3.0, stem="a/b")


def test_plan_montage_job_structure():
    jobs = cp.plan_montage(montage_hls(), synth_track(), 3.0)
    for j in jobs:
        assert set(j) == {"timeline", "settings"}
        assert list(j["timeline"]) == [""]
        time_p, pos_p = j["timeline"][""]
        assert time_p["keyframes"][0]["properties"].keys() == {"timestamp"}
        assert pos_p["keyframes"][0]["properties"].keys() == {POS, ROT}
        assert len(pos_p["segments"]) == len(pos_p["keyframes"]) - 1
        json.dumps(j, allow_nan=False)


def make_mcpr(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metaData.json", json.dumps({"duration": 100000}))
        zf.writestr("recording.tmcpr", b"\x00" * 64)


def test_plan_montage_round_trip_mcpr(tmp_path):
    mcpr = tmp_path / "r.mcpr"
    make_mcpr(mcpr)
    jobs = cp.plan_montage(montage_hls(), synth_track(), 3.0, tmp_path / "out")
    replay.write_render_queue(mcpr, jobs)
    back = replay.read_render_queue(mcpr)
    assert len(back) == len(jobs)
    for orig, got in zip(jobs, back):
        assert got["timeline"] == json.loads(json.dumps(orig["timeline"]))
        assert got["settings"] == orig["settings"]
    assert replay.inspect(mcpr).has_render_queue
    with zipfile.ZipFile(mcpr) as zf:
        assert "recording.tmcpr" in zf.namelist()


# ================================================================ review additions
import random
import time


def _dist(eye, tgt):
    return math.dist(eye, tgt)


def test_safe_eye_respects_ceiling_and_min_distance_together():
    # Target hugging the ceiling: the outward push must not leave the allowed band.
    s = cp.Safety(max_y=100.0, min_distance=2.0)
    for eye in [(0, 99.9, 0), (0, 100.0, 0), (0.0, 120.0, 0.0), (0, 99.5, 0)]:
        tgt = (0.0, 99.5, 0.0)
        x, y, z = cp._safe_eye(eye, tgt, s)
        assert y <= 100.0 + 1e-9
        assert _dist((x, y, z), tgt) >= 2.0 - 1e-6


def test_safe_eye_respects_floor_and_ceiling_band_exhaustive():
    r = random.Random(7)
    for _ in range(2000):
        lo = r.uniform(-70, 60)
        hi = lo + r.choice([0.0, 0.5, 3.0, 50.0])
        s = cp.Safety(min_y=lo, max_y=hi, min_distance=r.choice([0.0, 1.0, 2.0, 6.0]))
        tgt = tuple(r.uniform(-5, 5) for _ in range(2)) + (r.uniform(lo - 5, hi + 5),)
        tgt = (tgt[0], tgt[2], tgt[1])
        eye = (tgt[0] + r.choice([0, 0, r.uniform(-3, 3)]), r.uniform(lo - 10, hi + 10),
               tgt[2] + r.choice([0, r.uniform(-3, 3)]))
        x, y, z = cp._safe_eye(eye, tgt, s)
        assert lo - 1e-9 <= y <= hi + 1e-9
        assert _dist((x, y, z), tgt) >= s.min_distance - 1e-6


def test_safety_rejects_inverted_band():
    with pytest.raises(cp.CamPathError):
        cp.Safety(min_y=10.0, max_y=5.0)
    with pytest.raises(cp.CamPathError):
        cp.Safety(min_distance=float("nan"))


def test_speed_profile_extreme_rates_stay_finite_and_monotone():
    for rates in ([(1, 1e308), (1, 1e308)], [(1, 1e-300), (1, 1e300)],
                  [(1, 1e-320), (1, 1)], [(1e-300, 1), (1e300, 1)]):
        path = cp.speed_ramp(0, 100_000, 5000, rates)
        ts = [k["properties"][timeline.PROP_TIMESTAMP] for k in path.keyframes]
        tt = times(path)
        assert ts[0] == 0 and ts[-1] == 100_000 and tt[0] == 0 and tt[-1] == 5000
        assert all(b > a for a, b in zip(ts, ts[1:])) and all(b > a for a, b in zip(tt, tt[1:]))
        json.dumps(path.to_json(), allow_nan=False)


def test_speed_ramp_rejects_non_finite_blend():
    for b in (float("inf"), float("nan")):
        with pytest.raises(cp.CamPathError):
            cp.speed_ramp(0, 1000, 1000, [(1, 1)], blend=b)


def test_keyframes_must_be_integer():
    with pytest.raises(cp.CamPathError):
        cp.orbit_shot((0, 0, 0), 1000, keyframes=2.5)
    with pytest.raises(cp.CamPathError):
        cp.orbit_shot((0, 0, 0), 1000, keyframes=float("nan"))
    with pytest.raises(cp.CamPathError):
        cp.orbit_shot((0, 0, 0), 1000, keyframes=4.0)


def test_overhead_pass_does_not_spin_yaw():
    # radius 0 puts the eye straight above/below the target: yaw is undefined
    # there and must not flip-flop between neighbouring keyframes.
    p = cp.crane_shot((0, 0, 0), 1000, radius=0.0, start_height=5, end_height=-5,
                      sweep_deg=0.0, keyframes=9, safety=cp.Safety(min_distance=0.5))
    yaws = [r[0] for r in eyes_rots(p)[1]]
    assert max(yaws) - min(yaws) < 1e-6
    assert_valid(p)


def test_no_negative_zero_in_output():
    p = cp.orbit_shot((0, 0, 0), 1000, radius=0.0, height=5.0, keyframes=5,
                      safety=cp.Safety(min_distance=0.5))
    for k in p.keyframes:
        for v in k["properties"][POS] + k["properties"][ROT]:
            assert not (v == 0 and math.copysign(1, v) < 0)


def test_follow_shot_long_track_is_fast_and_matches_short_window():
    n = 72_000
    pts = [(i * 50, i * 0.15, 64 + 3 * math.sin(i / 200), 5 * math.sin(i / 900), 0.0, 0.0)
           for i in range(n)]
    tr = Track(1, pts)
    t0 = time.perf_counter()
    path = cp.follow_shot(tr, 1_800_000, 1_812_000, 6000)
    assert time.perf_counter() - t0 < 2.0
    assert_valid(path)
    # same shot against only the relevant slice of the replay must agree closely
    sub = Track(1, [p for p in pts if 1_700_000 <= p[0] <= 1_900_000])
    ref = cp.follow_shot(sub, 1_800_000, 1_812_000, 6000)
    for a, b in zip(eyes_rots(path)[0], eyes_rots(ref)[0]):
        assert math.dist(a, b) < 1.0


def test_follow_shot_drops_non_finite_and_unsorted_points():
    pts = [(i * 100, i * 0.2, 70.0, 0.0, 0.0, 0.0) for i in range(300)]
    bad = list(reversed(pts)) + [(5000, float("nan"), 70.0, 0.0, 0.0, 0.0)]
    a = cp.follow_shot(Track(1, bad), 5000, 15000, 4000)
    b = cp.follow_shot(Track(1, pts), 5000, 15000, 4000)
    assert eyes_rots(a)[0] == eyes_rots(b)[0]


# ---------------------------------------------------------------- property tests

def _rand_vec(r, spread=30.0):
    return (r.uniform(-spread, spread), r.uniform(-20, 120), r.uniform(-spread, spread))


def _rand_safety(r):
    lo = r.choice([-64.0, 0.0, 60.0, r.uniform(-64, 80)])
    hi = r.choice([None, lo + r.choice([0.0, 1.0, 10.0, 300.0])])
    return cp.Safety(min_y=lo, max_y=hi, min_distance=r.choice([0.0, 0.5, 2.0, 8.0]))


def _rand_shot(r):
    dur = r.choice([1, 2, 3, 7, 50, 1000, 5000, 60_000])
    kf = r.choice([2, 3, 5, 24, 90])
    safety = _rand_safety(r)
    ease = r.choice(sorted(cp.EASINGS))
    common = dict(keyframes=kf, safety=safety)
    tgt = r.choice([(0.0, 64.0, 0.0), _rand_vec(r)])
    which = r.choice(["orbit", "dolly", "crane", "flyby", "static"])
    if which == "orbit":
        p = cp.orbit_shot(tgt, dur, radius=r.choice([0.0, 0.1, 5.0, 40.0]),
                          end_radius=r.choice([None, 0.0, 3.0]),
                          height=r.uniform(-10, 20), end_height=r.choice([None, -5.0, 30.0]),
                          revolutions=r.uniform(-3, 3), start_angle_deg=r.uniform(-720, 720),
                          easing=ease, **common)
        tg = None
    elif which == "dolly":
        s, e = _rand_vec(r), r.choice([tgt, _rand_vec(r)])
        p = cp.dolly_shot(s, e, tgt, dur, easing=ease, arc=r.uniform(-8, 8), **common)
    elif which == "crane":
        p = cp.crane_shot(tgt, dur, radius=r.choice([0.0, 4.0, 15.0]),
                          start_height=r.uniform(-8, 20), end_height=r.uniform(-8, 40),
                          angle_deg=r.uniform(-360, 360), sweep_deg=r.uniform(-90, 90),
                          easing=ease, **common)
    elif which == "flyby":
        p = cp.flyby_shot(tgt, dur, pass_distance=r.choice([0.0, 3.0, 20.0]),
                          height=r.uniform(-5, 10), approach=r.choice([0.0, 30.0]),
                          direction_deg=r.uniform(-360, 360), easing=ease, **common)
    else:
        p = cp.static_shot(_rand_vec(r), tgt, dur, keyframes=kf, safety=safety)
    return p, dur, safety, which


@pytest.mark.parametrize("seed", range(60))
def test_property_shots_are_valid_safe_and_look_at_something(seed):
    r = random.Random(seed)
    for _ in range(8):
        p, dur, safety, which = _rand_shot(r)
        assert_valid(p)
        ts = times(p)
        assert ts[0] == 0 and ts[-1] == dur
        eyes, rots = eyes_rots(p)
        for e in eyes:
            assert e[1] >= safety.min_y - 1e-3
            if safety.max_y is not None:
                assert e[1] <= safety.max_y + 1e-3
        yaws = [x[0] for x in rots]
        assert all(abs(b - a) <= 180.0 + 1e-3 for a, b in zip(yaws, yaws[1:]))


@pytest.mark.parametrize("seed", range(40))
def test_property_orbit_keeps_min_distance_and_aim(seed):
    r = random.Random(500 + seed)
    for _ in range(10):
        safety = cp.Safety(min_y=-64.0, max_y=r.choice([None, 300.0]),
                           min_distance=r.choice([1.0, 2.0, 5.0]))
        tgt = _rand_vec(r)
        p = cp.orbit_shot(tgt, r.choice([1, 40, 4000]), radius=r.choice([0.0, 1.0, 12.0]),
                          height=r.uniform(-3, 3), revolutions=r.uniform(-2, 2),
                          keyframes=r.randrange(2, 40), safety=safety)
        assert_valid(p)
        eyes, _ = eyes_rots(p)
        for e in eyes:
            assert math.dist(e, tgt) >= safety.min_distance - 1e-3
        assert_looks_at(p, [tgt] * len(p.keyframes), tol_deg=0.05)


@pytest.mark.parametrize("seed", range(60))
def test_property_speed_ramp_monotone_and_pinned(seed):
    r = random.Random(900 + seed)
    for _ in range(10):
        n = r.randrange(1, 5)
        rates = [(10 ** r.uniform(-6, 6), 10 ** r.uniform(-9, 9)) for _ in range(n)]
        start = r.choice([0, 1234, 10**9])
        span = r.choice([1, 2, 5, 100, 10_000, 10**7])
        dur = r.choice([1, 2, 10, 500, 5000, 120_000])
        kf = r.choice([2, 3, 40, 200])
        blend = r.choice([0.0, 0.08, 0.5, 3.0])
        try:
            path = cp.speed_ramp(start, start + span, dur, rates, keyframes=kf, blend=blend)
        except cp.CamPathError:
            assert span < 1 or dur < 1   # only legitimate refusals
            continue
        ts = [k["properties"][timeline.PROP_TIMESTAMP] for k in path.keyframes]
        tt = times(path)
        assert (tt[0], ts[0]) == (0, start) and (tt[-1], ts[-1]) == (dur, start + span)
        assert all(b > a for a, b in zip(tt, tt[1:]))
        assert all(b > a for a, b in zip(ts, ts[1:]))
        prof = cp.speed_profile(start, start + span, dur, rates, blend=blend)
        assert all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in prof.table)
        assert all(b >= a for a, b in zip(prof.table, prof.table[1:]))
        prev = -1.0
        for i in range(0, 201):
            v = prof.replay_at(dur * i / 200)
            assert start <= v <= start + span and v >= prev
            prev = v


@pytest.mark.parametrize("seed", range(30))
def test_property_plan_highlight_any_kind_any_short_duration(seed):
    r = random.Random(3000 + seed)
    tr = synth_track()
    for _ in range(6):
        kind = r.choice(ALL_RECIPE_KINDS)
        s = r.randrange(0, 15_000)
        h = Highlight(kind, s, s + r.choice([1, 2, 500, 9000]), 0.5, entity=1,
                      pos=r.choice([None, _rand_vec(r)]))
        secs = r.choice([0.001, 0.002, 0.05, 1.0, 6.0])
        tpath, ppath = cp.plan_highlight(h, tr, secs)
        dur = int(round(secs * 1000))
        assert_valid(ppath)
        tt = times(tpath)
        assert tt[0] == 0 and tt[-1] == dur
        assert times(ppath)[-1] == dur
        ts = [k["properties"][timeline.PROP_TIMESTAMP] for k in tpath.keyframes]
        assert ts[0] == h.start_ms and ts[-1] == h.end_ms
        assert all(b > a for a, b in zip(ts, ts[1:]))
