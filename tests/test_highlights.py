import json
import math

import pytest

from mcmontage.events import Event, EventError, Highlight
from mcmontage.highlights import (
    DetectConfig, detect_all, detect_builds, detect_deaths, detect_kills,
    detect_rapid_movement, load_highlights, save_highlights, select,
)
from mcmontage.tracks import Track

CFG = DetectConfig()
P = 1  # player entity id


def mk(eid, t0, t1, fn, dt=50):
    pts = []
    t = t0
    while t <= t1:
        x, y, z, yaw, pitch = fn(t)
        pts.append((t, x, y, z, yaw, pitch))
        t += dt
    return Track(eid, pts)


def still(eid=P, t0=0, t1=60000, at=(0, 64, 0)):
    return mk(eid, t0, t1, lambda t: (*at, 0.0, 0.0), dt=1000)


def walk(speed, t0=0, t1=60000, eid=P):
    return mk(eid, t0, t1, lambda t: (speed * t / 1000.0, 64, 0, 90.0, 0.0))


def hl(kind, s, e, score, **kw):
    return Highlight(kind, s, e, score, **kw)


def all_finite(hs):
    for h in hs:
        assert math.isfinite(h.score) and 0.0 <= h.score <= 1.0
        if h.pos:
            assert all(math.isfinite(v) for v in h.pos)


# ---------------------------------------------------------------- deaths

def test_deaths_from_events():
    ev = [Event(10000, "death", entity=P, pos=(1, 2, 3))]
    out = detect_deaths({P: still()}, P, ev, CFG)
    assert len(out) == 1
    assert out[0].kind == "death" and out[0].start_ms == 10000
    assert out[0].pos == (1, 2, 3)
    assert out[0].meta["source"] == "event" and out[0].score == 1.0


def test_deaths_ignore_other_entities():
    ev = [Event(10000, "death", entity=99)]
    assert detect_deaths({P: still()}, P, ev, CFG) == []


def test_deaths_dedupe_close_events():
    ev = [Event(10000, "death", entity=P), Event(10500, "death", entity=P),
          Event(20000, "death", entity=P)]
    assert [h.start_ms for h in detect_deaths({}, P, ev, CFG)] == [10000, 20000]


def test_deaths_event_without_pos_uses_track():
    ev = [Event(5000, "death", entity=P)]
    out = detect_deaths({P: still(at=(7, 64, 9))}, P, ev, CFG)
    assert out[0].pos == (7.0, 64.0, 9.0)


def test_deaths_events_suppress_inference():
    pts = [(0, 0, 64, 0, 0, 0), (1000, 0, 64, 0, 0, 0), (9000, 500, 64, 0, 0, 0),
           (10000, 500, 64, 0, 0, 0)]
    out = detect_deaths({P: Track(P, pts)}, P, [Event(300, "death", entity=P)], CFG)
    assert [h.start_ms for h in out] == [300]


def test_deaths_inferred_gap_then_teleport():
    pts = [(0, 0, 64, 0, 0, 0), (1000, 1, 64, 0, 0, 0), (1050, 1, 64, 0, 0, 0),
           (6000, 300, 70, 300, 0, 0), (7000, 301, 70, 300, 0, 0)]
    out = detect_deaths({P: Track(P, pts)}, P, [], CFG)
    assert len(out) == 1
    h = out[0]
    assert h.meta["source"] == "inferred" and h.meta["gap"] and not h.meta["fall"]
    assert h.start_ms == 1050 and h.pos == (1.0, 64.0, 0.0)
    assert h.score == pytest.approx(CFG.death_inferred_confidence)


def test_deaths_inferred_fall_then_teleport_without_gap():
    pts = [(t, 0, 100 - t / 50.0, 0, 0, 0) for t in range(0, 2000, 50)]  # drop 40
    last = pts[-1][0]
    pts.append((last + 50, 400, 70, 400, 0, 0))
    out = detect_deaths({P: Track(P, pts)}, P, [], CFG)
    assert len(out) == 1 and out[0].meta["fall"] and not out[0].meta["gap"]


def test_deaths_plain_teleport_is_not_a_death():
    pts = [(0, 0, 64, 0, 0, 0), (50, 0, 64, 0, 0, 0), (100, 500, 64, 0, 0, 0),
           (150, 500, 64, 0, 0, 0)]
    assert detect_deaths({P: Track(P, pts)}, P, [], CFG) == []


def test_deaths_gap_without_teleport_is_not_a_death():
    pts = [(0, 0, 64, 0, 0, 0), (9000, 1, 64, 0, 0, 0)]
    assert detect_deaths({P: Track(P, pts)}, P, [], CFG) == []


def test_deaths_no_player_or_track():
    assert detect_deaths({}, None, [Event(1, "death", entity=P)], CFG) == []
    assert detect_deaths({}, P, [], CFG) == []


# ----------------------------------------------------------------- kills

def kill(t, target=100, pos=None):
    return Event(t, "kill", entity=P, target=target, pos=pos)


def test_single_kill():
    out = detect_kills([kill(5000)], P, CFG)
    assert len(out) == 1 and out[0].label == "Kill"
    assert out[0].score == pytest.approx(CFG.kill_base_score)


def test_kills_streak_scores_rise():
    base = detect_kills([kill(1000)], P, CFG)[0].score
    two = detect_kills([kill(1000), kill(3000)], P, CFG)
    three = detect_kills([kill(1000), kill(3000), kill(5000)], P, CFG)
    assert len(two) == 1 and len(three) == 1
    assert base < two[0].score < three[0].score
    assert three[0].label == "Triple kill" and three[0].meta["kills"] == 3


def test_kills_split_on_gap():
    out = detect_kills([kill(1000), kill(1000 + CFG.kill_streak_gap_ms + 1)], P, CFG)
    assert len(out) == 2


def test_kills_ignore_other_killers_and_kinds():
    ev = [Event(1000, "kill", entity=5), Event(2000, "damage", entity=P)]
    assert detect_kills(ev, P, CFG) == []


def test_kills_unsorted_input_and_span():
    out = detect_kills([kill(5000), kill(1000), kill(3000)], P, CFG)
    assert len(out) == 1 and out[0].start_ms == 1000 and out[0].end_ms == 5001


def test_kills_score_capped_at_one():
    evs = [kill(1000 + 1000 * i) for i in range(20)]
    assert detect_kills(evs, P, CFG)[0].score == 1.0


def test_kills_no_player():
    assert detect_kills([kill(1)], None, CFG) == []


def test_kills_pos_from_last_positioned():
    out = detect_kills([kill(1000, pos=(1, 2, 3)), kill(2000)], P, CFG)
    assert out[0].pos == (1.0, 2.0, 3.0) or out[0].pos == (1, 2, 3)


# ----------------------------------------------------------------- speed

def test_speed_detected_for_fast_travel():
    tr = mk(P, 0, 30000, lambda t: ((t / 1000.0 * 15) if 10000 <= t <= 20000 else
                                    (150 if t > 20000 else 0), 64, 0, 0, 0))
    # stationary, 15 b/s for 10s, stationary again
    out = [h for h in detect_rapid_movement(tr, CFG) if h.kind == "speed"]
    assert len(out) == 1
    h = out[0]
    assert 9000 <= h.start_ms <= 11000 and 19000 <= h.end_ms <= 21000
    assert not h.meta["elytra_like"]


def test_speed_walking_is_not_highlight():
    assert detect_rapid_movement(walk(5.0), CFG) == []


def test_speed_elytra_flag_and_higher_score():
    fast = detect_rapid_movement(walk(40.0, t1=20000), CFG)
    slow = detect_rapid_movement(walk(12.0, t1=20000), CFG)
    assert fast[0].meta["elytra_like"] and "Elytra" in fast[0].label
    assert fast[0].score > slow[0].score


def test_speed_short_burst_ignored():
    # a 0.5s dash at 30 b/s averages only 7.5 b/s over the 2s window
    tr = mk(P, 0, 20000, lambda t: ((30 * (min(t, 10500) - 10000) / 1000.0
                                     if t > 10000 else 0), 64, 0, 0, 0))
    assert detect_rapid_movement(tr, CFG) == []


def test_speed_teleport_does_not_fake_speed():
    pts = [(t, 0, 64, 0, 0, 0) for t in range(0, 5000, 50)]
    pts += [(t, 200, 64, 0, 0, 0) for t in range(5000, 10000, 50)]
    assert detect_rapid_movement(Track(P, pts), CFG) == []


def test_speed_adjacent_windows_merge_into_one():
    out = detect_rapid_movement(walk(20.0, t1=30000), CFG)
    assert len([h for h in out if h.kind == "speed"]) == 1
    assert out[0].end_ms - out[0].start_ms >= 28000


def test_fall_detected():
    tr = mk(P, 0, 12000, lambda t: (0, 200 - max(0, min(t - 4000, 3000)) * 0.03
                                    if t < 7000 else 110, 0, 0, 0))
    # 4s-7s: drops 90 blocks at 30 b/s
    falls = [h for h in detect_rapid_movement(tr, CFG) if h.kind == "fall"]
    assert len(falls) == 1
    assert falls[0].meta["drop"] >= 80
    assert falls[0].start_ms >= 3500 and falls[0].end_ms <= 7500


def test_fall_small_drop_ignored():
    tr = mk(P, 0, 6000, lambda t: (0, 70 - (t / 1000) * 3 if t < 2000 else 64, 0, 0, 0))
    assert [h for h in detect_rapid_movement(tr, CFG) if h.kind == "fall"] == []


def test_fall_slow_descent_ignored():
    tr = mk(P, 0, 30000, lambda t: (0, 100 - t / 1000 * 2, 0, 0, 0))  # 2 b/s, 60 blocks
    assert detect_rapid_movement(tr, CFG) == []


def test_fall_bigger_drop_scores_higher():
    def fall_track(h):
        v = 30.0
        return mk(P, 0, 8000, lambda t: (0, max(100 - h, 100 - v * max(0, t - 2000) / 1000.0), 0, 0, 0))
    small = [x for x in detect_rapid_movement(fall_track(20), CFG) if x.kind == "fall"][0]
    big = [x for x in detect_rapid_movement(fall_track(80), CFG) if x.kind == "fall"][0]
    assert big.score > small.score


def test_rapid_empty_and_single_point():
    assert detect_rapid_movement(Track(P, []), CFG) == []
    assert detect_rapid_movement(Track(P, [(0, 0, 0, 0, 0, 0)]), CFG) == []


def test_rapid_nan_points_dropped():
    tr = walk(20.0, t1=10000)
    pts = list(tr.points)
    pts[10] = (pts[10][0], float("nan"), 64, 0, 0, 0)
    out = detect_rapid_movement(Track(P, pts), CFG)
    all_finite(out)
    assert out


def test_rapid_unsorted_points():
    tr = walk(20.0, t1=10000)
    shuffled = list(reversed(tr.points))
    assert detect_rapid_movement(Track(P, shuffled), CFG) == detect_rapid_movement(tr, CFG)


# ---------------------------------------------------------------- builds

def places(n, t0=1000, dt=500, centre=(10, 64, 10), eid=P):
    return [Event(t0 + i * dt, "block_place", entity=eid,
                  pos=(centre[0] + (i % 4), centre[1] + (i // 8), centre[2] + (i % 3)))
            for i in range(n)]


def test_builds_from_place_events():
    out = detect_builds(still(), places(30), CFG)
    assert len(out) == 1
    h = out[0]
    assert h.kind == "build" and h.meta["blocks"] == 30 and h.meta["source"] == "event"
    assert h.start_ms == 1000 and h.end_ms == 1000 + 29 * 500 + 1
    assert h.pos is not None and abs(h.pos[0] - 11.5) < 2


def test_builds_too_few_blocks():
    assert detect_builds(still(), places(CFG.build_min_blocks - 1), CFG) == []


def test_builds_split_by_time_gap():
    evs = places(20, t0=0) + places(20, t0=40000)
    assert len(detect_builds(still(), evs, CFG)) == 2


def test_builds_split_by_distance():
    evs = places(15, t0=0, dt=100) + places(15, t0=1600, dt=100, centre=(500, 64, 500))
    out = detect_builds(still(), sorted(evs, key=lambda e: e.t), CFG)
    assert len(out) == 2


def test_builds_more_blocks_higher_score():
    a = detect_builds(still(), places(15), CFG)[0]
    b = detect_builds(still(), places(90), CFG)[0]
    assert b.score > a.score


def test_builds_events_of_other_entity_ignored_falls_back_to_track():
    out = detect_builds(still(), places(30, eid=77), CFG)
    assert out == []  # still track has no activity, so heuristic finds nothing


def test_builds_unsorted_events():
    evs = list(reversed(places(20)))
    assert len(detect_builds(still(), evs, CFG)) == 1


def fidget(t0=10000, t1=50000):
    """Mostly idle, but within [t0,t1] a lot of small look/position changes."""
    def fn(t):
        if t0 <= t <= t1:
            k = t / 50.0
            return (5 + 0.3 * math.sin(k), 64, 5 + 0.3 * math.cos(k),
                    (k * 7) % 60, 20 * math.sin(k / 3))
        return (5.0, 64.0, 5.0, 0.0, 0.0)
    pts = []
    for t in range(0, 70000, 50):
        if t0 <= t <= t1 or t % 5000 == 0:
            pts.append((t, *fn(t)))
    return Track(P, pts)


def test_builds_fallback_finds_fidgeting_session():
    out = detect_builds(fidget(), [], CFG)
    assert len(out) == 1
    h = out[0]
    assert h.meta["source"] == "heuristic" and "Possible" in h.label
    assert h.start_ms >= 9000 and h.end_ms <= 52000
    assert h.end_ms - h.start_ms >= 30000


def test_builds_fallback_ignores_idle_player():
    assert detect_builds(still(t1=120000), [], CFG) == []


def test_builds_fallback_ignores_travelling_player():
    assert detect_builds(walk(5.0, t1=60000), [], CFG) == []


def test_builds_fallback_short_activity_ignored():
    assert detect_builds(fidget(10000, 15000), [], CFG) == []


def test_builds_empty_track():
    assert detect_builds(Track(P, []), [], CFG) == []


# ------------------------------------------------------------- detect_all

def scenario():
    """60s replay: kill streak, a death, a fast run and a fall."""
    def fn(t):
        x = 0.0
        y = 64.0
        if 20000 <= t <= 30000:
            x = (t - 20000) / 1000.0 * 18
        elif t > 30000:
            x = 180.0
        if 40000 <= t <= 42000:
            y = 164.0 - (t - 40000) / 1000.0 * 40
        elif t > 42000:
            y = 84.0
        return (x, y, 0.0, 0.0, 0.0)
    tr = mk(P, 0, 60000, fn)
    other = mk(2, 0, 60000, lambda t: (1, 64, 1, 0, 0), dt=5000)
    evs = [kill(5000), kill(7000), Event(52000, "death", entity=P)]
    return {P: tr, 2: other}, evs


def test_detect_all_sorted_and_normalised():
    tracks, evs = scenario()
    out = detect_all(tracks, evs)
    kinds = {h.kind for h in out}
    assert {"kill", "death", "speed", "fall"} <= kinds
    assert [h.start_ms for h in out] == sorted(h.start_ms for h in out)
    all_finite(out)


def test_detect_all_autoidentifies_player():
    tracks, evs = scenario()
    assert detect_all(tracks, evs) == detect_all(tracks, evs, player_id=P)


def test_detect_all_padding_applied():
    tracks, evs = scenario()
    death = [h for h in detect_all(tracks, evs) if h.kind == "death"][0]
    pre, post = CFG.pad_ms["death"]
    assert death.start_ms == 52000 - pre and death.end_ms == 52001 + post


def test_detect_all_clamps_to_bounds():
    tr = still(t1=10000)
    out = detect_all({P: tr}, [Event(500, "death", entity=P), Event(9900, "death", entity=P)])
    assert all(0 <= h.start_ms and h.end_ms <= 10000 for h in out)
    assert out[0].start_ms == 0 and out[-1].end_ms == 10000


def test_detect_all_merges_overlapping_same_kind():
    tr = still(t1=60000)
    evs = [Event(20000, "death", entity=P), Event(23000, "death", entity=P)]
    out = detect_all({P: tr}, evs)
    assert len(out) == 1 and out[0].meta["merged"] == 2


def test_detect_all_does_not_merge_different_kinds():
    tr = still(t1=60000)
    evs = [Event(20000, "death", entity=P), kill(20500)]
    out = detect_all({P: tr}, evs)
    assert {h.kind for h in out} == {"death", "kill"}


def test_detect_all_kind_weights_order_scores():
    tracks, evs = scenario()
    out = detect_all(tracks, evs)
    death = [h for h in out if h.kind == "death"][0]
    speed = [h for h in out if h.kind == "speed"][0]
    assert death.score > speed.score


def test_detect_all_empty():
    assert detect_all({}) == []
    assert detect_all({}, []) == []


def test_detect_all_empty_tracks_with_events_only():
    out = detect_all({}, [Event(1000, "death", entity=P)], player_id=P)
    assert len(out) == 1 and out[0].start_ms == 0  # clamped at replay start 0


def test_detect_all_single_point_track():
    assert detect_all({P: Track(P, [(0, 0, 0, 0, 0, 0)])}) == []


def test_detect_all_accepts_track_list():
    tracks, evs = scenario()
    assert detect_all(list(tracks.values()), evs) == detect_all(tracks, evs)


def test_detect_all_deterministic():
    tracks, evs = scenario()
    a = [h.to_json() for h in detect_all(tracks, evs)]
    b = [h.to_json() for h in detect_all(tracks, list(reversed(evs)))]
    assert a == b


def test_detect_all_scores_are_json_exact():
    tracks, evs = scenario()
    for h in detect_all(tracks, evs):
        assert Highlight.from_json(h.to_json()) == h


def test_detect_all_unknown_event_kinds_ignored():
    tr = still()
    assert detect_all({P: tr}, [Event(5, "weird", entity=P)]) == []


# ---------------------------------------------------------------- select

def test_select_respects_budget():
    hs = [hl("kill", 0, 10000, 0.9), hl("kill", 20000, 30000, 0.8),
          hl("kill", 40000, 50000, 0.7)]
    out = select(hs, 20000)
    assert sum(h.duration_ms for h in out) <= 20000
    assert [h.score for h in out] == [0.9, 0.8]


def test_select_chronological_output():
    hs = [hl("kill", 40000, 45000, 0.9), hl("kill", 0, 5000, 0.5), hl("kill", 20000, 25000, 0.7)]
    out = select(hs, 100000)
    assert [h.start_ms for h in out] == [0, 20000, 40000]


def test_select_non_overlapping_prefers_higher_score():
    hs = [hl("kill", 0, 10000, 0.5), hl("death", 5000, 15000, 0.9)]
    out = select(hs, 100000, min_gap_ms=0)
    assert out == [hs[1]]


def test_select_min_gap_enforced():
    a, b = hl("kill", 0, 5000, 0.9), hl("kill", 6000, 11000, 0.8)
    assert select([a, b], 100000, min_gap_ms=2000) == [a]
    assert select([a, b], 100000, min_gap_ms=1000) == [a, b]
    assert select([a, b], 100000, min_gap_ms=0) == [a, b]


def test_select_max_count():
    hs = [hl("kill", i * 20000, i * 20000 + 5000, 0.1 * (i + 1)) for i in range(5)]
    out = select(hs, 10**9, max_count=2)
    assert len(out) == 2 and {h.score for h in out} == {0.5, 0.4}


def test_select_skips_too_long_but_takes_smaller():
    big = hl("build", 0, 50000, 0.95)
    small = hl("kill", 100000, 105000, 0.3)
    assert select([big, small], 10000) == [small]


def test_select_tie_break_earlier_start():
    a, b = hl("kill", 50000, 55000, 0.5), hl("kill", 0, 5000, 0.5)
    assert select([a, b], 5000) == [b]


def test_select_edge_budgets():
    hs = [hl("kill", 0, 5000, 0.5)]
    assert select(hs, 0) == []
    assert select(hs, -5) == []
    assert select([], 1000) == []
    assert select(hs, 100, ) == []
    assert select(hs, 1000, max_count=0) == []


def test_select_does_not_mutate_input():
    hs = [hl("kill", 9000, 9500, 0.2), hl("kill", 0, 500, 0.9)]
    copy = list(hs)
    select(hs, 10000)
    assert hs == copy


def test_select_on_detect_all_output():
    tracks, evs = scenario()
    out = select(detect_all(tracks, evs), 20000)
    assert sum(h.duration_ms for h in out) <= 20000
    for a, b in zip(out, out[1:]):
        assert a.end_ms <= b.start_ms


# ----------------------------------------------------------------- json

def test_save_load_roundtrip(tmp_path):
    tracks, evs = scenario()
    hs = detect_all(tracks, evs)
    p = tmp_path / "h.json"
    save_highlights(p, hs)
    assert load_highlights(p) == hs
    assert json.loads(p.read_text())["version"] == 1


def test_save_load_empty(tmp_path):
    p = tmp_path / "h.json"
    save_highlights(p, [])
    assert load_highlights(p) == []


def test_load_accepts_bare_list(tmp_path):
    p = tmp_path / "h.json"
    h = hl("custom", 0, 10, 0.5, label="x")
    p.write_text(json.dumps([h.to_json()]))
    assert load_highlights(p) == [h]


def test_load_bad_entry_raises(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"highlights": [{"kind": "kill"}]}))
    with pytest.raises(EventError):
        load_highlights(p)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_highlights(tmp_path / "nope.json")
