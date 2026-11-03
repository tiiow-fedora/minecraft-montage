import pytest
from PIL import Image, ImageDraw, ImageFilter

from mcmontage import thumbnail as th
from tests.conftest import needs_ffmpeg


def checker(size=(640, 360), cell=8):
    img = Image.new("RGB", size, "black")
    d = ImageDraw.Draw(img)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                d.rectangle([x, y, x + cell - 1, y + cell - 1], fill="white")
    return img


def test_sharp_beats_blurry():
    sharp = checker()
    blurry = sharp.filter(ImageFilter.GaussianBlur(6))
    assert th.score_frame(sharp).sharpness > th.score_frame(blurry).sharpness
    assert th.score_frame(sharp).score > th.score_frame(blurry).score


def test_high_contrast_beats_flat():
    flat = Image.new("RGB", (640, 360), (120, 120, 120))
    assert th.score_frame(checker()).contrast > th.score_frame(flat).contrast
    assert th.score_frame(checker()).score > th.score_frame(flat).score


def test_black_and_white_frames_penalised():
    black = Image.new("RGB", (640, 360), (0, 0, 0))
    white = Image.new("RGB", (640, 360), (255, 255, 255))
    assert th.score_frame(black).score == 0
    assert th.score_frame(white).score == 0
    # same pattern, crushed to near-black, must lose to the normal exposure
    dark = checker().point(lambda v: v // 10)
    assert th.score_frame(dark).score < th.score_frame(checker()).score


def test_score_independent_of_resolution():
    a = th.score_frame(checker((640, 360), 8)).score
    b = th.score_frame(checker((1280, 720), 16)).score
    assert abs(a - b) < 0.15


def test_sample_times_skip_margins_and_are_sorted():
    ts = th.sample_times(100, 5)
    assert ts[0] == pytest.approx(5) and ts[-1] == pytest.approx(95)
    assert ts == sorted(ts)
    assert th.sample_times(100, 1) == [50]
    with pytest.raises(th.ThumbnailError):
        th.sample_times(0, 3)
    with pytest.raises(th.ThumbnailError):
        th.sample_times(10, 0)


def _cand(t, s):
    return th.Candidate(t, None, th.FrameScore(0, 0, 128, s))


def test_pick_frames_respects_gap_and_order():
    cands = [_cand(1, .9), _cand(2, .85), _cand(10, .5), _cand(20, .4)]
    picked = th.pick_frames(cands, k=3, min_gap=5)
    assert [c.time for c in picked] == [1, 10, 20]
    assert [c.time for c in th.pick_frames(cands, k=2)] == [1, 2]


def test_crop_to_16x9_from_various_aspects():
    for size in [(1920, 1080), (1000, 1000), (3000, 1000), (640, 360)]:
        out = th.crop_to_16x9(Image.new("RGB", size, "red"))
        assert out.size == th.THUMB_SIZE


def test_wrap_text_truncates():
    img = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(img)
    f = th.find_font(60)
    lines = th.wrap_text(d, "word " * 60, f, 500, max_lines=3)
    assert len(lines) == 3 and lines[-1].endswith("…")


def test_unknown_template():
    with pytest.raises(th.ThumbnailError):
        th.get_template("nope")


@pytest.mark.parametrize("name", sorted(th.TEMPLATES))
def test_every_template_renders_and_draws_text(name):
    base = Image.new("RGB", (1280, 720), (30, 90, 40))
    out = th.render(base, "Building the Nether Hub", "Day 3", name, badge="Day 3")
    assert out.size == th.THUMB_SIZE
    assert out.tobytes() != th.render(base, "", "", name).tobytes()


def test_text_overlay_changes_expected_region():
    base = Image.new("RGB", (1280, 720), (30, 90, 40))
    plain = th.render(base, "", "", "minimal")
    texty = th.render(base, "Hello", "", "minimal")
    # bottom-left anchored: top half must be untouched
    top = (0, 0, 1280, 300)
    assert plain.crop(top).tobytes() == texty.crop(top).tobytes()
    assert plain.crop((0, 400, 1280, 720)).tobytes() != texty.crop((0, 400, 1280, 720)).tobytes()


def test_very_long_title_still_fits_canvas():
    base = Image.new("RGB", (1280, 720), (30, 90, 40))
    out = th.render(base, "supercalifragilistic " * 12, "sub " * 30, "bold")
    assert out.size == th.THUMB_SIZE


def test_save_png_and_size_cap(tmp_path):
    p = th.save(Image.new("RGB", (1280, 720), "blue"), tmp_path / "a.png")
    assert p.suffix == ".png" and p.stat().st_size < th.MAX_BYTES
    noisy = Image.effect_noise((1280, 720), 80).convert("RGB")
    p2 = th.save(noisy, tmp_path / "b.png", max_bytes=400_000)
    assert p2.suffix == ".jpg" and p2.stat().st_size <= 400_000
    assert Image.open(p2).width >= th.MIN_WIDTH
    with pytest.raises(th.ThumbnailError):
        th.save(noisy, tmp_path / "c.jpg", max_bytes=100)


@needs_ffmpeg
def test_best_frame_skips_the_black_half(sample_video):
    best = th.best_frames(sample_video, k=2, samples=10)
    assert best and all(c.time > 5 for c in best)


@needs_ffmpeg
def test_make_thumbnail_end_to_end(sample_video, tmp_path):
    out = th.make_thumbnail(sample_video, tmp_path / "t.png",
                            "Test Title", "Sub", "banner")
    img = Image.open(out)
    assert img.size == th.THUMB_SIZE
    assert out.stat().st_size < th.MAX_BYTES


def test_missing_video(tmp_path):
    with pytest.raises(th.ThumbnailError):
        th.sample_frames(tmp_path / "nope.mp4")
