import time

from mcmontage.tracks import Track


def line(n):
    return Track(1, [(i * 50, float(i), 0.0, 0.0, 0.0, 0.0) for i in range(n)])


def test_at_interpolates_and_clamps():
    t = line(10)
    assert t.at(-5) == (0.0, 0.0, 0.0)
    assert t.at(25) == (0.5, 0.0, 0.0)
    assert t.at(10**9) == (9.0, 0.0, 0.0)
    assert t.at(100) == (2.0, 0.0, 0.0)


def test_at_is_fast_on_long_tracks():
    t = line(72000)
    start = time.perf_counter()
    for k in range(20000):
        t.at(k * 150)
    assert time.perf_counter() - start < 1.0


def test_smoothed_long_track_is_fast():
    t = line(72000)
    start = time.perf_counter()
    sm = t.smoothed(window_ms=4000)
    assert len(sm.points) > 1000
    assert time.perf_counter() - start < 3.0
