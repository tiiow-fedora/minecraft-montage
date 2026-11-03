import pytest

import synthetic as syn
from mcmontage import tracks
from mcmontage.tracks import Track


def line(n=11, dx=1.0):
    return Track(1, [(i * 1000, i * dx, 64.0, 0.0, 0.0, 0.0) for i in range(n)])


def test_at_interpolates_and_clamps():
    t = line()
    assert t.at(500) == pytest.approx((0.5, 64, 0))
    assert t.at(-100) == (0.0, 64.0, 0.0)
    assert t.at(10**9) == (10.0, 64.0, 0.0)
    assert t.at(3000) == pytest.approx((3.0, 64, 0))


def test_at_duplicate_timestamps_no_division_by_zero():
    t = Track(1, [(0, 0, 0, 0, 0, 0), (1000, 5, 0, 0, 0, 0), (1000, 9, 0, 0, 0, 0),
                  (2000, 9, 0, 0, 0, 0)])
    assert t.at(1000)[0] in (5, 9)
    assert t.at(1500) == pytest.approx((9, 0, 0))


def test_distance_travelled():
    assert line().distance_travelled() == pytest.approx(10)


def test_smoothed_preserves_straight_line_in_the_interior():
    t = line(n=61)
    s = t.smoothed(window_ms=4000)
    # symmetric window on a straight constant-speed line returns the same point
    assert s.at(30_000)[0] == pytest.approx(30.0, abs=0.01)
    assert s.start == t.start and s.end == t.end


def test_smoothed_reduces_jitter():
    pts = syn.walk_points(0, 20_000, 50, velocity=(1, 0, 0), jitter=1.0, seed=3)
    raw = Track(1, pts)
    sm = raw.smoothed(window_ms=2000)
    def roughness(tr):
        xs = [tr.at(t)[2] for t in range(0, 20_000, 250)]
        return sum(abs(b - a) for a, b in zip(xs, xs[1:]))
    assert roughness(sm) < roughness(raw) / 2


def test_smoothed_empty_track():
    with pytest.raises(tracks.TrackError):
        Track(1, []).smoothed()


def test_load_drops_short_tracks_and_sorts(tmp_path):
    p = syn.write_tracks(tmp_path / "t.json", {
        5: [(2000, 1, 1, 1, 0, 0), (1000, 0, 0, 0, 0, 0)],
        6: [(0, 0, 0, 0, 0, 0)],
    })
    got = tracks.load(p)
    assert set(got) == {5}
    assert got[5].times == [1000, 2000]


def test_identify_player_picks_most_sampled_mover():
    got = {k: Track(k, v) for k, v in syn.standard_tracks().items()}
    assert tracks.identify_player(got).entity_id == 100


def test_identify_player_ignores_stationary_spammer():
    spam = Track(9, syn.walk_points(0, 60_000, 10))                       # 6000 samples, never moves
    walker = Track(1, syn.walk_points(0, 60_000, 500, velocity=(1, 0, 0)))  # 121 samples, moves 60 blocks
    assert tracks.identify_player({9: spam, 1: walker}).entity_id == 1


def test_identify_player_falls_back_when_nobody_moves():
    a = Track(1, syn.walk_points(0, 1000, 100))
    b = Track(2, syn.walk_points(0, 1000, 10))
    assert tracks.identify_player({1: a, 2: b}).entity_id == 2


def test_identify_player_no_tracks():
    with pytest.raises(tracks.TrackError):
        tracks.identify_player({})


def test_max_deviation_zero_for_identical():
    t = line()
    assert tracks.max_deviation(t, t, 0, 10_000, 20) == 0


def test_choose_smoothing_shrinks_window_for_fast_corners():
    """A square-wave route: big windows cut corners far more than the budget."""
    pts = []
    for i in range(0, 120_001, 100):
        leg = (i // 10_000) % 4
        s = (i % 10_000) / 10_000 * 100
        x, z = [(s, 0), (100, s), (100 - s, 100), (0, 100 - s)][leg]
        pts.append((i, x, 64.0, z, 0.0, 0.0))
    t = Track(1, pts)
    w_tight, dev_tight = tracks.choose_smoothing(t, 0, 120_000, 60, 2.0)
    w_loose, dev_loose = tracks.choose_smoothing(t, 0, 120_000, 60, 40.0)
    assert dev_tight <= 2.0 and dev_loose <= 40.0
    assert w_tight < w_loose
    assert w_loose in tracks.SMOOTH_CANDIDATES


def test_choose_smoothing_returns_largest_window_on_stationary():
    t = Track(1, syn.walk_points(0, 30_000, 250))
    assert tracks.choose_smoothing(t, 0, 30_000, 30, 2.0) == (16000, 0.0)


def test_choose_smoothing_gives_up_with_zero():
    t = Track(1, [(0, 0, 0, 0, 0, 0), (1000, 1000, 0, 0, 0, 0), (2000, 0, 0, 0, 0, 0),
                  (3000, 1000, 0, 0, 0, 0)] * 1)
    w, dev = tracks.choose_smoothing(t, 0, 3000, 30, 0.001)
    assert w == 0 and dev == 0.0


def test_rank_orders_by_samples():
    got = {k: Track(k, v) for k, v in syn.standard_tracks().items()}
    rows = tracks.rank(got)
    assert rows[0][0] == 100
    assert tracks.rank(got, limit=1) == rows[:1]


def test_dump_reports_missing_pieces(tmp_path, monkeypatch):
    with pytest.raises(tracks.TrackError, match="no such replay"):
        tracks.dump(tmp_path / "x.mcpr", tmp_path / "o.json")
    p = syn.make_mcpr(tmp_path / "a.mcpr")
    monkeypatch.setattr(tracks, "CP_FILE", tmp_path / "nope.txt")
    with pytest.raises(tracks.TrackError, match="make_cp"):
        tracks.dump(p, tmp_path / "o.json")
