import math

import pytest

from golden import assert_golden
from mcmontage import timeline as tl


def test_look_at_cardinal_directions():
    # Minecraft: yaw 0 faces +Z, yaw -90 faces +X (x = -sin(yaw)), pitch>0 looks down
    assert tl.look_at((0, 0, 0), (0, 0, 5)) == pytest.approx((0, 0))
    assert tl.look_at((0, 0, 0), (5, 0, 0))[0] == pytest.approx(-90)
    assert tl.look_at((0, 0, 0), (-5, 0, 0))[0] == pytest.approx(90)
    assert abs(tl.look_at((0, 0, 0), (0, 0, -5))[0]) == pytest.approx(180)
    assert tl.look_at((0, 10, 0), (0, 0, 0))[1] == pytest.approx(90)
    assert tl.look_at((0, 0, 0), (0, 10, 0))[1] == pytest.approx(-90)


def test_look_at_same_point_is_zero():
    assert tl.look_at((1, 2, 3), (1, 2, 3)) == (0.0, 0.0)


def test_unwrap_removes_jumps():
    out = tl._unwrap([170, -170, -150, 170])
    assert out == pytest.approx([170, 190, 210, 170])
    for a, b in zip(out, out[1:]):
        assert abs(b - a) <= 180


def test_unwrap_empty():
    assert tl._unwrap([]) == []


def test_orbit_full_circle_closes_and_yaw_is_monotonic():
    p = tl.orbit_path((0, 64, 0), 10, 5, 10_000, revolutions=1.0, keyframes=13)
    kfs = p.keyframes
    assert kfs[0]["time"] == 0 and kfs[-1]["time"] == 10_000
    first = kfs[0]["properties"][tl.PROP_POSITION]
    last = kfs[-1]["properties"][tl.PROP_POSITION]
    assert first == pytest.approx(last, abs=1e-3)
    yaws = [k["properties"][tl.PROP_ROTATION][0] for k in kfs]
    # continuous rotation: no jump larger than half a turn, and ends a full turn away
    assert all(abs(b - a) < 180 for a, b in zip(yaws, yaws[1:]))
    assert abs(yaws[-1] - yaws[0]) == pytest.approx(360, abs=1.0)


def test_orbit_camera_always_faces_centre():
    centre = (3.0, 70.0, -4.0)
    p = tl.orbit_path(centre, 8, 4, 5000, revolutions=0.75, keyframes=9)
    for k in p.keyframes:
        eye = k["properties"][tl.PROP_POSITION]
        yaw, pitch, roll = k["properties"][tl.PROP_ROTATION]
        assert roll == 0.0
        exp_yaw, exp_pitch = tl.look_at(eye, centre)
        d = (yaw - exp_yaw + 180) % 360 - 180
        assert d == pytest.approx(0, abs=0.01)
        assert pitch == pytest.approx(exp_pitch, abs=0.01)


def test_orbit_rejects_too_few_keyframes():
    with pytest.raises(ValueError):
        tl.orbit_path((0, 0, 0), 5, 1, 1000, keyframes=1)
    with pytest.raises(ValueError):
        tl.dolly_path((0, 0, 0), (1, 1, 1), (0, 0, 5), 1000, keyframes=1)


def test_path_segments_match_keyframes():
    for n in (2, 5, 90):
        j = tl.orbit_path((0, 0, 0), 5, 1, 1000, keyframes=n).to_json()
        assert len(j["segments"]) == n - 1
        assert set(j["segments"]) == {0}
        assert len(j["interpolators"]) == 1


def test_time_path_rejects_empty_span():
    with pytest.raises(ValueError):
        tl.time_path(1000, 1000, 5000)
    with pytest.raises(ValueError):
        tl.time_path(2000, 1000, 5000)


def test_speedup_factor():
    assert tl.speedup_factor(0, 600_000, 30_000) == 20.0


def test_render_settings_validation():
    with pytest.raises(ValueError):
        tl.render_settings("x.mp4", anti_aliasing="X3")
    with pytest.raises(ValueError):
        tl.render_settings("x.mp4", encoding_preset="GIF")
    with pytest.raises(ValueError):
        tl.render_settings("x.mp4", render_method="FISHEYE")


def test_property_keys_are_the_validated_ones():
    # README: "timestamp" and "spectate" are bare; camera props are grouped.
    assert tl.PROP_TIMESTAMP == "timestamp"
    assert tl.PROP_SPECTATE == "spectate"
    assert tl.PROP_POSITION == "camera:position"
    assert tl.PROP_ROTATION == "camera:rotation"


def test_interpolator_encodings():
    assert tl.LINEAR == "linear"
    assert tl.CUBIC == "cubic-spline"
    assert tl.catmull_rom() == {"type": "catmull-rom-spline", "alpha": 0.5}


# ---- golden files -----------------------------------------------------------

def _tl_json(tp, pp):
    return tl.build_timeline(tp, pp)


def test_golden_static_orbit_timeline():
    tp = tl.time_path(0, 600_000, 30_000)
    pp = tl.orbit_path((100.5, 64.0, -20.25), 12, 6, 30_000, revolutions=0.5,
                       keyframes=7)
    assert_golden("timeline_static_orbit.json", _tl_json(tp, pp))


def test_golden_dolly_timeline():
    tp = tl.time_path(10_000, 70_000, 20_000)
    pp = tl.dolly_path((0, 70, 0), (40, 75, 40), (20, 64, 20), 20_000, keyframes=5)
    assert_golden("timeline_dolly.json", _tl_json(tp, pp))


def test_golden_render_settings_defaults():
    assert_golden("render_settings_default.json",
                  tl.render_settings("/videos/clip.mp4"))


def test_golden_render_settings_overrides():
    assert_golden("render_settings_overrides.json", tl.render_settings(
        "C:\\videos\\clip.mp4", width=1280, height=720, fps=30, bitrate=8_000_000,
        encoding_preset="MP4_POTATO", render_method="EQUIRECTANGULAR",
        name_tags=True, anti_aliasing="X4"))
