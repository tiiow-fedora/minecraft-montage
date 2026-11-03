"""Tests for mcmontage.beatsync using synthesised audio (no fixtures on disk)."""

from __future__ import annotations

import cmath
import json
import math
import random
import shutil
import subprocess
import wave
from array import array

import pytest

from mcmontage import beatsync as bs
from mcmontage.beatsync import BeatError, BeatGrid, Cut

SR = 11025


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------


def synth(
    bpm: float, dur: float = 20.0, kind: str = "click", noise: float = 0.0,
    swing: bool = False, accent: bool = False, seed: int = 1, off: float = 0.3,
    sr: int = SR, lead_silence: float = 0.0,
) -> tuple[array, list[float]]:
    """Return (samples, true_beat_times). Beats are 1 kHz clicks or 60 Hz
    decaying thumps; ``swing`` adds quiet 2/3-position hats; ``accent`` makes
    every 4th beat louder so downbeats are recoverable."""
    rng = random.Random(seed)
    n = int(dur * sr)
    x = [0.0] * n
    per = 60.0 / bpm

    def burst(t0: float, f: float, decay: float, amp: float) -> None:
        s = int(t0 * sr)
        for i in range(min(int(0.25 * sr), n - s)):
            x[s + i] += amp * math.sin(2 * math.pi * f * i / sr) * math.exp(-i / sr / decay)

    truth = []
    k = 0
    while True:
        t = off + lead_silence + k * per
        if t >= dur - 0.1:
            break
        amp = 0.8 * (0.6 if accent and k % 4 else 1.0)
        if kind == "click":
            burst(t, 1000, 0.01, amp)
        else:
            burst(t, 60, 0.08, amp)
        if swing:
            burst(t + per * 2 / 3, 6000, 0.005, 0.25)
        truth.append(t)
        k += 1
    if noise:
        x = [v + rng.uniform(-noise, noise) for v in x]
    pcm = array("h", (max(-32767, min(32767, int(v * 32767))) for v in x))
    return pcm, truth


def match(detected: list[float], truth: list[float], tol: float = 0.05) -> tuple[float, float]:
    rec = sum(any(abs(d - t) <= tol for d in detected) for t in truth) / len(truth)
    pre = sum(any(abs(d - t) <= tol for t in truth) for d in detected) / max(1, len(detected))
    return rec, pre


def write_wav(path, samples, sr: int = SR) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def octave_equiv(a: float, b: float, tol: float = 0.02) -> bool:
    return any(abs(a * f - b) / b < tol for f in (0.25, 0.5, 1, 2, 4))


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4, 8, 64, 256])
def test_fft_matches_naive_dft(n):
    rng = random.Random(n)
    x = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
    for a, b in zip(bs.fft(x), bs.dft(x)):
        assert abs(a - b) < 1e-9


def test_fft_pure_tone_peaks_at_bin():
    n = 64
    x = [math.cos(2 * math.pi * 5 * t / n) for t in range(n)]
    mags = [abs(c) for c in bs.fft(x)]
    assert max(range(n), key=lambda k: mags[k]) in (5, n - 5)
    assert mags[5] == pytest.approx(n / 2)


@pytest.mark.parametrize("n", [0, 3, 6, 100])
def test_fft_rejects_non_power_of_two(n):
    with pytest.raises(ValueError):
        bs.fft([0] * n)


def test_fft_parseval():
    rng = random.Random(3)
    x = [rng.uniform(-1, 1) for _ in range(128)]
    X = bs.fft(x)
    assert sum(abs(c) ** 2 for c in X) / 128 == pytest.approx(sum(v * v for v in x))


# ---------------------------------------------------------------------------
# WAV / decode
# ---------------------------------------------------------------------------


def test_wav_roundtrip(tmp_path):
    pcm, _ = synth(120, 2)
    p = tmp_path / "a.wav"
    write_wav(p, pcm)
    got, sr = bs.samples_from_wav(p)
    assert sr == SR and list(got) == list(pcm)


def test_wav_stereo_is_mixed_to_mono(tmp_path):
    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(array("h", [100, 300, -100, -300]).tobytes())
    got, sr = bs.samples_from_wav(p)
    assert sr == 8000 and list(got) == [200, -200]


def test_wav_missing_and_garbage(tmp_path):
    with pytest.raises(BeatError):
        bs.samples_from_wav(tmp_path / "nope.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    with pytest.raises(BeatError):
        bs.samples_from_wav(bad)


def test_wav_8bit_rejected(tmp_path):
    p = tmp_path / "u8.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(bytes(100))
    with pytest.raises(BeatError):
        bs.samples_from_wav(p)


def test_decode_pcm_missing_file(tmp_path):
    with pytest.raises(BeatError, match="no such"):
        bs.decode_pcm(tmp_path / "x.mp3")


def test_decode_pcm_missing_ffmpeg(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p, synth(120, 1)[0])
    with pytest.raises(BeatError, match="ffmpeg not found"):
        bs.decode_pcm(p, ffmpeg="definitely-not-ffmpeg-xyz")


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


@needs_ffmpeg
def test_decode_pcm_garbage_file_errors(tmp_path):
    p = tmp_path / "junk.mp3"
    p.write_bytes(b"\x00\x01garbage" * 50)
    with pytest.raises(BeatError, match="ffmpeg failed"):
        bs.decode_pcm(p)


@needs_ffmpeg
def test_decode_pcm_resamples_wav(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p, synth(120, 2, sr=22050)[0], sr=22050)
    got = bs.decode_pcm(p, sample_rate=8000)
    assert abs(len(got) - 16000) < 200


# ---------------------------------------------------------------------------
# Onset envelope
# ---------------------------------------------------------------------------


def test_envelope_empty_for_silence_and_tiny():
    assert bs.onset_envelope(array("h", [0] * 20000), SR) == []
    assert bs.onset_envelope(array("h", [3, -3] * 10000), SR) == []
    assert bs.onset_envelope(array("h", [1000] * 50), SR) == []


def test_envelope_peaks_at_clicks():
    pcm, truth = synth(120, 6)
    env = bs.onset_envelope(pcm, SR)
    rate = bs.frame_rate(SR)
    peaks = bs.pick_peaks(bs.normalise_novelty(env, rate), rate)
    times = [(p + 0.5) / rate for p in peaks]
    rec, pre = match(times, truth, 0.03)
    assert rec == 1.0 and pre == 1.0


def test_frame_rate_about_100hz():
    for sr in (8000, 11025, 22050, 44100):
        assert 90 < bs.frame_rate(sr) < 111


def test_normalise_novelty_unit_variance_and_nonnegative():
    flux = [0.0, 1.0, 0.0, 0.0, 5.0, 0.0] * 50
    nov = bs.normalise_novelty(flux, 100.0)
    assert min(nov) >= 0
    assert math.sqrt(sum(v * v for v in nov) / len(nov)) == pytest.approx(1.0)


def test_normalise_novelty_flat_is_zero():
    assert not any(bs.normalise_novelty([2.0] * 100, 100.0))
    assert bs.normalise_novelty([], 100.0) == []


def test_pick_peaks_respects_min_gap_and_flat():
    env = [0.0] * 100
    env[20], env[22] = 5.0, 8.0
    env[60] = 5.0
    assert bs.pick_peaks(env, 100.0, min_gap_s=0.05) == [22, 60]
    assert bs.pick_peaks([0.0] * 50, 100.0) == []
    assert bs.pick_peaks([1.0], 100.0) == []


def test_pick_peaks_ignores_low_ripple():
    env = [0.1 + 0.02 * (i % 2) for i in range(200)]
    env[100] = 10.0
    assert bs.pick_peaks(env, 100.0) == [100]


def test_flux_method_finds_clicks():
    pcm, truth = synth(120, 5)
    g = bs.analyse_samples(pcm, SR, method="flux")
    assert abs(g.bpm - 120) < 3
    rec, pre = match(g.beats, truth, 0.06)
    assert rec > 0.9 and pre > 0.9


def test_unknown_method_raises():
    pcm, _ = synth(120, 3)
    with pytest.raises(ValueError):
        bs.onset_envelope(pcm, SR, method="magic")


# ---------------------------------------------------------------------------
# Tempo / beats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bpm", [90, 120, 140])
@pytest.mark.parametrize(
    "kind,noise,swing",
    [("click", 0.0, False), ("click", 0.1, True), ("kick", 0.05, True)],
)
def test_bpm_and_beat_positions(bpm, kind, noise, swing):
    pcm, truth = synth(bpm, 20, kind=kind, noise=noise, swing=swing)
    g = bs.analyse_samples(pcm, SR)
    assert g.bpm == pytest.approx(bpm, rel=0.01)
    rec, pre = match(g.beats, truth, 0.05)
    assert rec >= 0.95 and pre >= 0.95
    assert g.confidence > 0.5


@pytest.mark.parametrize("bpm", [75, 100, 160])
def test_other_tempos(bpm):
    pcm, truth = synth(bpm, 20)
    g = bs.analyse_samples(pcm, SR)
    assert g.bpm == pytest.approx(bpm, rel=0.015)
    rec, pre = match(g.beats, truth)
    assert rec >= 0.95 and pre >= 0.95


@pytest.mark.parametrize("bpm", [60, 180])
def test_octave_extremes_return_equivalent_tempo(bpm):
    """At the range edges the felt tempo is ambiguous; any octave is valid."""
    pcm, _ = synth(bpm, 20)
    g = bs.analyse_samples(pcm, SR)
    assert octave_equiv(g.bpm, bpm)
    assert 60 <= g.bpm <= 180


def test_octave_range_forces_fast_tempo():
    pcm, truth = synth(80, 20)
    g = bs.analyse_samples(pcm, SR, min_bpm=140, max_bpm=190)
    assert g.bpm == pytest.approx(160, rel=0.02)


def test_prior_prefers_mid_tempo_over_double():
    pcm, _ = synth(90, 20)
    assert bs.analyse_samples(pcm, SR).bpm == pytest.approx(90, rel=0.01)


def test_beats_are_monotonic_and_regularly_spaced():
    pcm, _ = synth(120, 20, noise=0.05, swing=True)
    g = bs.analyse_samples(pcm, SR)
    gaps = [b - a for a, b in zip(g.beats, g.beats[1:])]
    assert all(0.4 < d < 0.6 for d in gaps)


def test_lead_in_silence_is_trimmed():
    pcm, truth = synth(120, 20, lead_silence=3.0, off=0.0)
    g = bs.analyse_samples(pcm, SR)
    assert g.beats[0] > 2.8
    rec, pre = match(g.beats, truth)
    assert rec >= 0.95 and pre >= 0.95


def test_trailing_silence_is_trimmed():
    pcm, truth = synth(120, 12)
    pcm.extend([0] * (SR * 5))
    g = bs.analyse_samples(pcm, SR)
    assert g.beats[-1] < 12.5
    assert g.bpm == pytest.approx(120, rel=0.01)


def test_tempo_survives_rare_dropout():
    pcm, truth = synth(120, 20)
    for i in range(int(8 * SR), int(9.5 * SR)):
        pcm[i] = 0
    g = bs.analyse_samples(pcm, SR)
    assert g.bpm == pytest.approx(120, rel=0.01)
    kept = [t for t in truth if not 8 <= t < 9.5]
    rec, _ = match(g.beats, kept)
    assert rec >= 0.9


def test_silence_gives_empty_grid():
    g = bs.analyse_samples(array("h", [0] * (SR * 10)), SR)
    assert g.beats == [] and g.bpm == 0 and g.confidence == 0


def test_empty_and_very_short_inputs():
    for n in (0, 1, 50, 500, SR // 2, SR):
        g = bs.analyse_samples(array("h", [1000, -1000] * (n // 2)), SR)
        assert g.beats == [] and g.confidence == 0


def test_short_but_real_input_does_not_crash():
    pcm, _ = synth(120, 2.5)
    g = bs.analyse_samples(pcm, SR)
    assert isinstance(g, BeatGrid)


def test_pure_noise_has_low_confidence():
    rng = random.Random(7)
    pcm = array("h", (int(rng.gauss(0, 3000)) for _ in range(SR * 15)))
    g = bs.analyse_samples(pcm, SR)
    assert g.confidence < 0.5


def test_other_sample_rates():
    pcm, truth = synth(120, 10, sr=22050)
    g = bs.analyse_samples(pcm, 22050)
    assert g.bpm == pytest.approx(120, rel=0.01)
    assert match(g.beats, truth)[0] >= 0.95


def test_downbeat_detected_from_accents():
    pcm, truth = synth(120, 20, accent=True, off=0.3)
    g = bs.analyse_samples(pcm, SR)
    first = min(range(len(truth)), key=lambda i: abs(truth[i] - g.beats[0]))
    expected = (-first) % 4
    assert g.downbeat_offset == expected
    db = g.downbeats
    assert all(any(abs(d - truth[k]) < 0.05 for k in range(0, len(truth), 4)) for d in db)


def test_downbeat_offset_function():
    s = [1.0, 0.5, 0.5, 0.5] * 4
    assert bs.estimate_downbeat_offset(s) == 0
    assert bs.estimate_downbeat_offset([0.5] + s[:-1]) == 1
    assert bs.estimate_downbeat_offset([1.0] * 16) == 0  # no clear winner
    assert bs.estimate_downbeat_offset([1, 2, 3]) == 0  # too little data


def test_track_beats_edge_cases():
    assert bs.track_beats([], 50) == []
    assert bs.track_beats([1.0] * 10, 1) == []


def test_estimate_tempo_too_short():
    assert bs.estimate_tempo([1.0, 0.0] * 10, 100.0) == (0.0, 0.0)


def test_extrapolate_beats():
    g = BeatGrid(120, [1.0, 1.5, 2.0], 1.0)
    ext = bs.extrapolate_beats(g, 4.0)
    assert ext[:3] == g.beats and ext[-1] == pytest.approx(4.0)
    assert bs.extrapolate_beats(BeatGrid(0, [1.0], 0), 10) == [1.0]
    assert g.beats == [1.0, 1.5, 2.0]  # input not mutated


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------

B = [0.5, 1.0, 1.5, 2.0, 2.5]


def test_snap_nearest():
    assert bs.snap_to_beat(1.2, B) == 1.0
    assert bs.snap_to_beat(1.3, B) == 1.5


def test_snap_tie_goes_earlier():
    assert bs.snap_to_beat(1.25, B) == 1.0


def test_snap_outside_range():
    assert bs.snap_to_beat(-5, B) == 0.5
    assert bs.snap_to_beat(99, B) == 2.5


def test_snap_tolerance_leaves_far_times_alone():
    assert bs.snap_to_beat(1.2, B, tolerance=0.1) == 1.2
    assert bs.snap_to_beat(1.2, B, tolerance=0.2) == 1.0
    assert bs.snap_to_beat(99, B, tolerance=1.0) == 99


def test_snap_empty_and_single():
    assert bs.snap_to_beat(3.3, []) == 3.3
    assert bs.snap_to_beat(3.3, [7.0]) == 7.0
    assert bs.nearest_beat_index(1, []) == -1


def test_snap_exact_beat():
    assert bs.snap_to_beat(1.5, B, tolerance=0.0) == 1.5


# ---------------------------------------------------------------------------
# plan_cuts
# ---------------------------------------------------------------------------


def grid(bpm=120.0, n=200, off=0.25):
    return [off + i * 60.0 / bpm for i in range(n)]


def check_plan(cuts, beats, total, lo, hi):
    assert cuts[0].start_s == 0.0
    assert cuts[-1].end_s == pytest.approx(total)
    for a, b in zip(cuts, cuts[1:]):
        assert a.end_s == pytest.approx(b.start_s)
    for c in cuts:
        assert lo - 1e-6 <= c.duration <= hi + 1e-6
        assert c.end_s > c.start_s
    for c in cuts[1:]:
        assert c.beat_index >= 0
        assert beats[c.beat_index] == pytest.approx(c.start_s)
    assert cuts[0].beat_index == -1


@pytest.mark.parametrize("bpm", [70, 100, 128, 174])
@pytest.mark.parametrize("total", [10.0, 33.3, 61.0])
@pytest.mark.parametrize("lo,hi", [(1.0, 4.0), (2.0, 6.0), (3.0, 4.5)])
def test_plan_cuts_invariants(bpm, total, lo, hi):
    beats = grid(bpm, 400)
    try:
        cuts = bs.plan_cuts(beats, total, lo, hi)
    except BeatError:
        pytest.skip("unsatisfiable combination")
    check_plan(cuts, beats, total, lo, hi)


def test_plan_cuts_prefers_downbeats():
    beats = grid(120, 100, off=0.0)
    cuts = bs.plan_cuts(beats, 40.0, 2.0, 8.0, prefer_downbeats=True)
    inner = cuts[1:]
    on_bar = sum(c.beat_index % 4 == 0 for c in inner)
    assert on_bar >= 0.8 * len(inner)


def test_plan_cuts_downbeat_offset_respected():
    beats = grid(120, 100, off=0.0)
    cuts = bs.plan_cuts(beats, 40.0, 2.0, 8.0, downbeat_offset=2)
    inner = cuts[1:]
    assert sum((c.beat_index - 2) % 4 == 0 for c in inner) >= 0.8 * len(inner)


def test_plan_cuts_without_downbeat_pref_tracks_target():
    beats = grid(120, 100, off=0.0)
    cuts = bs.plan_cuts(beats, 30.0, 1.0, 5.0, prefer_downbeats=False,
                        clip_durations=[3.0])
    assert all(abs(c.duration - 3.0) < 1e-6 for c in cuts)


def test_plan_cuts_follows_clip_durations():
    beats = grid(120, 100, off=0.0)
    cuts = bs.plan_cuts(beats, 30.0, 1.0, 8.0, prefer_downbeats=False,
                        clip_durations=[2.0, 4.0])
    check_plan(cuts, beats, 30.0, 1.0, 8.0)
    assert cuts[0].duration == pytest.approx(2.0)
    assert cuts[1].duration == pytest.approx(4.0)


def test_plan_cuts_no_beats_short_total_single_cut():
    cuts = bs.plan_cuts([], 3.0, 1.0, 5.0)
    assert cuts == [Cut(0.0, 3.0, -1)]


def test_plan_cuts_no_beats_long_total_raises():
    with pytest.raises(BeatError):
        bs.plan_cuts([], 30.0, 1.0, 5.0)


def test_plan_cuts_total_below_min_returns_single():
    assert bs.plan_cuts(grid(), 0.5, 1.0, 4.0) == [Cut(0.0, 0.5, -1)]


def test_plan_cuts_beats_beyond_total_are_ignored():
    beats = grid(120, 400)
    cuts = bs.plan_cuts(beats, 12.0, 1.0, 4.0)
    assert all(c.end_s <= 12.0 + 1e-9 for c in cuts)


def test_plan_cuts_sparse_beats_unsatisfiable():
    with pytest.raises(BeatError):
        bs.plan_cuts([5.0, 25.0], 30.0, 1.0, 4.0)


def test_plan_cuts_bad_args():
    for kw in ({"total_s": 0}, {"total_s": -1}, {"min_clip_s": 0},
               {"min_clip_s": 5, "max_clip_s": 2}, {"clip_durations": []},
               {"clip_durations": [1, -1]}):
        args = {"beats": grid(), "total_s": 10.0, "min_clip_s": 1.0, "max_clip_s": 4.0}
        args.update(kw)
        with pytest.raises(BeatError):
            bs.plan_cuts(**args)
    with pytest.raises(BeatError):
        bs.plan_cuts([2.0, 1.0], 10.0)


def test_plan_cuts_end_remainder_not_stranded():
    # A greedy fill would leave a 0.3s tail; the DP must avoid it.
    beats = [i * 1.0 for i in range(1, 30)]
    cuts = bs.plan_cuts(beats, 10.3, 2.0, 3.0)
    check_plan(cuts, beats, 10.3, 2.0, 3.0)


def test_plan_cuts_from_detected_grid():
    pcm, _ = synth(120, 20)
    g = bs.analyse_samples(pcm, SR)
    cuts = bs.plan_cuts(g.beats, 18.0, 1.5, 5.0, downbeat_offset=g.downbeat_offset)
    check_plan(cuts, g.beats, 18.0, 1.5, 5.0)


# ---------------------------------------------------------------------------
# fit_clips_to_beats
# ---------------------------------------------------------------------------


def boundaries(durs, start=0.0):
    out, t = [], start
    for d in durs:
        t += d
        out.append(t)
    return out


def test_fit_clips_land_on_beats_and_stay_close():
    beats = grid(120, 100, off=0.0)
    durs = [2.2, 3.7, 1.4, 2.9, 4.1]
    fitted = bs.fit_clips_to_beats(durs, beats)
    ends = boundaries(fitted)
    assert len(fitted) == len(durs)
    for e in ends:
        assert min(abs(e - b) for b in beats) < 1e-9
    ideal = boundaries(durs)
    assert all(abs(e - i) <= 0.5 for e, i in zip(ends, ideal))
    assert all(d > 0 for d in fitted)


def test_fit_clips_already_aligned_unchanged():
    beats = grid(120, 100, off=0.0)
    durs = [1.0, 2.0, 0.5, 3.0]
    assert bs.fit_clips_to_beats(durs, beats) == pytest.approx(durs)


def test_fit_clips_drift_does_not_accumulate():
    beats = grid(120, 400, off=0.0)
    durs = [0.74] * 60  # each rounds the same way; naive rounding would drift
    ends = boundaries(bs.fit_clips_to_beats(durs, beats))
    ideal = boundaries(durs)
    assert max(abs(e - i) for e, i in zip(ends, ideal)) <= 0.5


def test_fit_clips_shorter_than_beat_get_distinct_beats():
    beats = grid(60, 100, off=0.0)
    fitted = bs.fit_clips_to_beats([0.2] * 10, beats)
    assert all(d >= 1.0 - 1e-9 for d in fitted)  # at least one beat each


def test_fit_clips_start_offset():
    beats = grid(120, 100, off=0.0)
    fitted = bs.fit_clips_to_beats([1.0, 1.0], beats, start_s=5.1)
    ends = boundaries(fitted, 5.1)
    assert ends[0] > 5.1 and all(min(abs(e - b) for b in beats) < 1e-9 for e in ends)


def test_fit_clips_errors_and_empty():
    assert bs.fit_clips_to_beats([], [1.0]) == []
    with pytest.raises(BeatError):
        bs.fit_clips_to_beats([1.0, 1.0, 1.0], [1.0, 2.0])
    with pytest.raises(BeatError):
        bs.fit_clips_to_beats([1.0, 0.0], grid())
    with pytest.raises(BeatError):
        bs.fit_clips_to_beats([1.0], [])


# ---------------------------------------------------------------------------
# cuts_for_highlights
# ---------------------------------------------------------------------------


def test_highlights_cover_and_align():
    beats = grid(120, 200, off=0.0)
    scores = [1, 5, 2, 8, 1]
    cuts = bs.cuts_for_highlights(scores, beats, 40.0)
    assert len(cuts) == 5
    assert cuts[0].start_s == 0 and cuts[-1].end_s == 40.0
    for a, b in zip(cuts, cuts[1:]):
        assert a.end_s == b.start_s
    for c in cuts[1:]:
        assert any(abs(c.start_s - b) < 1e-9 for b in beats)
        assert beats[c.beat_index] == c.start_s


def test_highlights_higher_score_gets_longer_slot():
    beats = grid(120, 200, off=0.0)
    cuts = bs.cuts_for_highlights([1, 10, 1], beats, 12.0)
    assert cuts[1].duration > cuts[0].duration
    assert cuts[1].duration > cuts[2].duration


def test_highlights_equal_scores_roughly_equal():
    beats = grid(120, 200, off=0.0)
    cuts = bs.cuts_for_highlights([3, 3, 3, 3], beats, 32.0)
    durs = [c.duration for c in cuts]
    assert max(durs) - min(durs) <= 0.5 + 1e-9


def test_highlights_respect_min_beats():
    beats = grid(120, 200, off=0.0)
    cuts = bs.cuts_for_highlights([0, 100, 0], beats, 30.0, min_beats=4)
    assert all(c.duration >= 4 * 0.5 - 1e-9 for c in cuts)


def test_highlights_respect_max_beats_when_possible():
    beats = grid(120, 200, off=0.0)
    cuts = bs.cuts_for_highlights([1, 100, 1, 1], beats, 20.0, max_beats=12)
    assert all(c.duration <= 12 * 0.5 + 1e-9 for c in cuts)


def test_highlights_bad_scores_tolerated():
    beats = grid(120, 200, off=0.0)
    cuts = bs.cuts_for_highlights([float("nan"), -3, 2], beats, 20.0)
    assert len(cuts) == 3 and cuts[-1].end_s == 20.0


def test_highlights_empty_and_too_few_beats():
    assert bs.cuts_for_highlights([], grid(), 10) == []
    with pytest.raises(BeatError):
        bs.cuts_for_highlights([1, 1, 1, 1], [1.0, 2.0], 10.0)


def test_highlights_tight_beats_reduce_min():
    beats = [1.0, 2.0, 3.0, 4.0]
    cuts = bs.cuts_for_highlights([1, 1, 1, 1, 1], beats, 5.0, min_beats=3)
    assert len(cuts) == 5 and cuts[-1].end_s == 5.0


# ---------------------------------------------------------------------------
# Serialisation and cache
# ---------------------------------------------------------------------------


def sample_grid():
    return BeatGrid(123.4, [0.5, 1.0, 1.5], 0.8, [1.0, 2.0, 3.0], 2)


def test_grid_json_roundtrip():
    g = sample_grid()
    assert BeatGrid.loads(g.dumps()) == g
    assert BeatGrid.from_dict(g.to_dict()) == g


def test_grid_from_dict_defaults_optional_fields():
    g = BeatGrid.from_dict({"bpm": 100, "beats": [1, 2], "confidence": 0.5})
    assert g.strengths == [] and g.downbeat_offset == 0


@pytest.mark.parametrize("bad", ["{", "[]", '{"bpm": 1}', '{"bpm":"x","beats":[],"confidence":0}',
                                 '{"bpm":1,"beats":5,"confidence":0}'])
def test_grid_loads_rejects_malformed(bad):
    with pytest.raises(BeatError):
        BeatGrid.loads(bad)


def test_grid_properties():
    g = BeatGrid(120, [float(i) for i in range(10)], 1.0, downbeat_offset=1)
    assert g.downbeats == [1.0, 5.0, 9.0]
    assert g.period == 0.5
    assert BeatGrid(0, [], 0).period == 0


def test_cache_path(tmp_path):
    assert bs.cache_path_for(tmp_path / "song.mp3").name == "song.mp3.beats.json"
    assert bs.cache_path_for("a/b.wav").as_posix() == "a/b.wav.beats.json"


def _music(tmp_path, content=b"abc"):
    p = tmp_path / "m.mp3"
    p.write_bytes(content)
    return p


def test_load_or_detect_caches(tmp_path):
    p = _music(tmp_path)
    calls = []

    def det(path):
        calls.append(path)
        return sample_grid()

    assert bs.load_or_detect(p, det) == sample_grid()
    assert bs.cache_path_for(p).is_file()
    assert bs.load_or_detect(p, det) == sample_grid()
    assert len(calls) == 1


def test_cache_invalidated_by_size_change(tmp_path):
    p = _music(tmp_path)
    n = []
    det = lambda path: (n.append(1), sample_grid())[1]
    bs.load_or_detect(p, det)
    p.write_bytes(b"abcdef")
    bs.load_or_detect(p, det)
    assert len(n) == 2


def test_cache_invalidated_by_mtime_change(tmp_path):
    import os
    p = _music(tmp_path)
    n = []
    det = lambda path: (n.append(1), sample_grid())[1]
    bs.load_or_detect(p, det)
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    bs.load_or_detect(p, det)
    assert len(n) == 2


def test_cache_force(tmp_path):
    p = _music(tmp_path)
    n = []
    det = lambda path: (n.append(1), sample_grid())[1]
    bs.load_or_detect(p, det)
    bs.load_or_detect(p, det, force=True)
    assert len(n) == 2


def test_corrupt_cache_is_ignored_and_rewritten(tmp_path):
    p = _music(tmp_path)
    bs.cache_path_for(p).write_text("{not json")
    assert bs.load_cache(p) is None
    assert bs.load_or_detect(p, lambda path: sample_grid()) == sample_grid()
    assert bs.load_cache(p) == sample_grid()


def test_cache_wrong_version_ignored(tmp_path):
    p = _music(tmp_path)
    bs.save_cache(p, sample_grid())
    cp = bs.cache_path_for(p)
    d = json.loads(cp.read_text())
    d["version"] = 999
    cp.write_text(json.dumps(d))
    assert bs.load_cache(p) is None


def test_cache_unwritable_dir_is_not_fatal(tmp_path, monkeypatch):
    p = _music(tmp_path)

    def boom(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(bs.os, "replace", boom)
    assert bs.load_or_detect(p, lambda path: sample_grid()) == sample_grid()


def test_load_or_detect_missing_file(tmp_path):
    with pytest.raises(BeatError):
        bs.load_or_detect(tmp_path / "gone.mp3", lambda p: sample_grid())


# ---------------------------------------------------------------------------
# ffmpeg integration
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_detect_beats_end_to_end_with_lavfi_click_track(tmp_path):
    bpm, dur = 120, 12
    p = tmp_path / "clicks.wav"
    expr = "0.8*sin(2*PI*1000*t)*lt(mod(t\\,0.5)\\,0.02)*exp(-30*mod(t\\,0.5))"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"aevalsrc={expr}:s=22050:d={dur}", str(p)],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr
    g = bs.detect_beats(p)
    assert g.bpm == pytest.approx(bpm, rel=0.015)
    truth = [i * 0.5 for i in range(dur * 2 - 1)]
    # A constant sub-frame decode offset is possible; compare after aligning phase.
    shift = min((g.beats[0] - t for t in truth[:3]), key=abs)
    assert abs(shift) < 0.05
    rec, pre = match(g.beats, truth, 0.05)
    assert rec >= 0.9 and pre >= 0.9
    g2 = bs.load_or_detect(p)
    assert bs.load_or_detect(p) == g2
    assert bs.cache_path_for(p).is_file()


# ===========================================================================
# Adversarial review additions: realistic drum grooves, drift, meter, MP3,
# a full-length track for performance, and robustness of the helpers.
# ===========================================================================

import functools
import os
import statistics
import time


@functools.lru_cache(maxsize=4)
def _drum_templates(sr: int):
    rng = random.Random(7)

    def kick():
        out, ph = [], 0.0
        for i in range(int(0.25 * sr)):
            t = i / sr
            ph += 2 * math.pi * (45 + 90 * math.exp(-t / 0.03)) / sr
            out.append(math.sin(ph) * math.exp(-t / 0.09))
        return out

    def snare():
        return [(0.6 * rng.uniform(-1, 1) + 0.4 * math.sin(2 * math.pi * 190 * i / sr))
                * math.exp(-i / sr / 0.05) for i in range(int(0.2 * sr))]

    def hat():
        return [rng.uniform(-1, 1) * math.exp(-i / sr / 0.012) for i in range(int(0.06 * sr))]

    return {"k": kick(), "s": snare(), "h": hat()}


# (position within the bar in beats, instrument, amplitude)
ROCK_44 = [(0, "k", 1.0), (0.5, "h", .3), (1, "s", .8), (1.5, "h", .3), (2, "k", .9),
           (2.5, "k", .7), (2.5, "h", .3), (3, "s", .8), (3.5, "h", .3)]
# kick/snare on the beats with 16th-note syncopation around them
FUNK_44 = [(0, "k", 1.0), (0.5, "h", .25), (0.75, "k", .6), (1, "s", .9), (1.5, "h", .25),
           (2, "k", .9), (2.5, "h", .25), (2.75, "k", .6), (3, "s", .9), (3.5, "h", .25),
           (3.75, "h", .25)]
WALTZ_34 = [(0, "k", 1.0), (0.5, "h", .3), (1, "s", .6), (1.5, "h", .3), (2, "s", .6), (2.5, "h", .3)]
# every hit between the beats: no energy on the beat itself
OFFBEAT_44 = [(0.5, "h", .6), (1.5, "s", .8), (2.5, "h", .6), (3.5, "s", .8), (1.75, "k", .7)]


def groove(bpm, dur, pattern, per_bar=4, drift=0.0, drift_period=30.0, sr=SR,
           seed=1, noise=0.02, lead=0.5, jitter=0.0):
    """Drum-machine style track. Tempo may wander sinusoidally by +-``drift``.
    Returns (samples, true quarter-note times)."""
    rng = random.Random(seed)
    tp = _drum_templates(sr)
    n = int(dur * sr)
    x = [0.0] * n
    beats, t = [], lead
    while t < dur - 0.3:
        beats.append(t)
        t += 60.0 / (bpm * (1 + drift * math.sin(2 * math.pi * t / drift_period)))
    for bi, bt in enumerate(beats):
        nxt = beats[bi + 1] if bi + 1 < len(beats) else bt + 60.0 / bpm
        for pos, inst, amp in pattern:
            if int(pos) != bi % per_bar:
                continue
            s = int((bt + (pos - int(pos)) * (nxt - bt) + rng.gauss(0, jitter)) * sr)
            tpl = tp[inst]
            for i in range(min(len(tpl), n - s)):
                x[s + i] += amp * tpl[i]
    if noise:
        x = [v + rng.uniform(-noise, noise) for v in x]
    pk = max(abs(v) for v in x) or 1.0
    return array("h", (int(v / pk * 0.8 * 32767) for v in x)), beats


@functools.lru_cache(maxsize=32)
def _groove_grid(bpm, dur, pattern_name, per_bar, drift, drift_period, jitter, seed):
    pat = {"rock": ROCK_44, "funk": FUNK_44, "waltz": WALTZ_34, "offbeat": OFFBEAT_44}[pattern_name]
    x, truth = groove(bpm, dur, pat, per_bar, drift, drift_period, jitter=jitter, seed=seed)
    return bs.analyse_samples(x, SR), truth


@pytest.mark.parametrize("bpm", [95, 105, 120, 135, 145, 155, 165, 175])
def test_drum_groove_tempo_and_beats_constant_tempo(bpm):
    """Regression: with eighth-note hats, 145-175 BPM used to come out as 2/3 of
    the true tempo (a 1.5-beat period) because integer-lag autocorrelation
    favours periods whose multiples land near whole frames."""
    g, truth = _groove_grid(bpm, 40, "rock", 4, 0.0, 30.0, 0.004, bpm)
    assert g.bpm == pytest.approx(bpm, rel=0.02)
    rec, pre = match(g.beats, truth, 0.07)
    assert rec >= 0.95 and pre >= 0.95


@pytest.mark.parametrize("bpm", [90, 110, 128, 160])
@pytest.mark.parametrize("drift", [0.02, -0.02])
def test_drum_groove_follows_two_percent_tempo_drift(bpm, drift):
    g, truth = _groove_grid(bpm, 50, "rock", 4, drift, 25.0, 0.004, 3)
    # on slow tracks a double-time answer (octave) is acceptable by contract
    assert octave_equiv(g.bpm, bpm, 0.03)
    if abs(g.bpm - bpm) / bpm < 0.03:
        rec, pre = match(g.beats, truth, 0.07)
        assert rec >= 0.95 and pre >= 0.95
    else:  # double time: every true beat must still be hit
        rec, _ = match(g.beats, truth, 0.07)
        assert rec >= 0.95


@pytest.mark.parametrize("bpm", [100, 128, 140])
def test_syncopated_funk_groove_locks_on_the_beat(bpm):
    g, truth = _groove_grid(bpm, 45, "funk", 4, 0.0, 30.0, 0.005, 5)
    assert g.bpm == pytest.approx(bpm, rel=0.02)
    rec, pre = match(g.beats, truth, 0.07)
    assert rec >= 0.95 and pre >= 0.95


def test_offbeat_only_groove_has_right_tempo_and_regular_grid():
    """No energy on the beat at all: locking to the off-beat is the only thing an
    onset tracker can do. Tempo and regularity must still be right, and the
    phase must be on the beat or exactly half a beat away, never arbitrary."""
    g, truth = _groove_grid(120, 40, "offbeat", 4, 0.0, 30.0, 0.0, 2)
    assert g.bpm == pytest.approx(120, rel=0.02)
    per = 0.5
    gaps = [b - a for a, b in zip(g.beats, g.beats[1:])]
    assert max(gaps) - min(gaps) < 0.03
    phase = statistics.median([((b - truth[0] + per / 2) % per) - per / 2 for b in g.beats])
    assert min(abs(phase), abs(abs(phase) - per / 2)) < 0.03


@pytest.mark.parametrize("bpm", [90, 120, 150])
def test_three_four_meter_beats_are_found(bpm):
    g, truth = _groove_grid(bpm, 45, "waltz", 3, 0.0, 30.0, 0.004, 9)
    assert g.bpm == pytest.approx(bpm, rel=0.02)
    rec, pre = match(g.beats, truth, 0.07)
    assert rec >= 0.95 and pre >= 0.95
    # Downbeats are documented as 4/4-only; for 3/4 they must still be a valid,
    # in-range selection that planning can consume without error.
    assert 0 <= g.downbeat_offset < bs.BEATS_PER_BAR
    cuts = bs.plan_cuts(g.beats, 30.0, 1.0, 6.0, downbeat_offset=g.downbeat_offset)
    assert cuts[0].start_s == 0.0 and cuts[-1].end_s == 30.0


def test_four_minute_track_analyses_quickly():
    x, truth = groove(124, 240, ROCK_44, 4, drift=0.02, drift_period=70.0, seed=11)
    t0 = time.perf_counter()
    g = bs.analyse_samples(x, SR)
    took = time.perf_counter() - t0
    assert took < 5.0, f"4-minute analysis took {took:.1f}s"
    assert g.bpm == pytest.approx(124, rel=0.03)
    rec, pre = match(g.beats, truth, 0.07)
    assert rec >= 0.95 and pre >= 0.95
    assert len(g.beats) == len(g.strengths) > 400


def test_flux_method_gets_tempo_on_a_groove_but_phase_is_not_guaranteed():
    """Known limitation: log-spectral flux weighs broadband hats as heavily as
    the kick/snare, so on hat-heavy grooves it may lock to the off-beat. Tempo
    and regularity must still be right (the method is unused by the pipeline)."""
    x, truth = groove(120, 20, ROCK_44, 4, seed=4)
    g = bs.analyse_samples(x, SR, method="flux")
    assert g.bpm == pytest.approx(120, rel=0.03)
    gaps = [b - a for a, b in zip(g.beats, g.beats[1:])]
    assert max(gaps) - min(gaps) < 0.03


def _has_mp3_encoder():
    if shutil.which("ffmpeg") is None:
        return False
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return "libmp3lame" in r.stdout


needs_mp3 = pytest.mark.skipif(not _has_mp3_encoder(), reason="ffmpeg libmp3lame missing")


def _encode(tmp_path, samples, sr, name, *extra):
    wav = tmp_path / "src.wav"
    write_wav(wav, samples, sr)
    out = tmp_path / name
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "libmp3lame",
                        "-b:a", "128k", *extra, str(out)], capture_output=True)
    assert r.returncode == 0, r.stderr
    return out


@needs_mp3
def test_real_mp3_round_trip_beats_stay_aligned_with_the_audio(tmp_path):
    """MP3 adds encoder delay (1105 samples) and decoder delay (529). A file with
    the LAME/Xing header is decoded gaplessly by ffmpeg, so beat times must stay
    within a few ms of the source."""
    x, truth = groove(120, 30, ROCK_44, 4, sr=22050, seed=3, jitter=0.003)
    mp3 = _encode(tmp_path, x, 22050, "song.mp3")
    g = bs.detect_beats(mp3)
    assert g.bpm == pytest.approx(120, rel=0.01)
    rec, pre = match(g.beats, truth, 0.03)
    assert rec >= 0.95 and pre >= 0.95
    off = statistics.median([((b - truth[0] + 0.25) % 0.5) - 0.25 for b in g.beats])
    assert abs(off) < 0.012
    # and the decoded length matches the source (no leading/trailing padding kept)
    pcm = bs.decode_pcm(mp3)
    assert abs(len(pcm) / bs.DEFAULT_SAMPLE_RATE - 30.0) < 0.03


@needs_mp3
def test_mp3_without_gapless_header_is_late_by_the_encoder_delay_only(tmp_path):
    """Without a Xing/LAME header the ~1.6k-sample codec delay cannot be known:
    beats come out late by that delay and no more (bounded, known limitation)."""
    x, truth = groove(120, 30, ROCK_44, 4, sr=22050, seed=3, jitter=0.003)
    mp3 = _encode(tmp_path, x, 22050, "raw.mp3", "-write_xing", "0")
    g = bs.detect_beats(mp3)
    assert g.bpm == pytest.approx(120, rel=0.01)
    off = statistics.median([((b - truth[0] + 0.25) % 0.5) - 0.25 for b in g.beats])
    assert 0.0 <= off < 0.06
    rec, _ = match(g.beats, [t + off for t in truth], 0.03)
    assert rec >= 0.95


@needs_mp3
def test_four_minute_mp3_end_to_end_under_budget(tmp_path):
    x, truth = groove(132, 240, FUNK_44, 4, sr=SR, drift=0.015, drift_period=80.0, seed=21)
    mp3 = _encode(tmp_path, x, SR, "long.mp3")
    t0 = time.perf_counter()
    g = bs.detect_beats(mp3)
    took = time.perf_counter() - t0
    assert took < 5.0, f"decode+analysis took {took:.1f}s"
    assert g.bpm == pytest.approx(132, rel=0.03)
    rec, pre = match(g.beats, truth, 0.08)
    assert rec >= 0.93 and pre >= 0.93


# ---- helper robustness -----------------------------------------------------

def test_extrapolate_beats_zero_period_terminates():
    g = BeatGrid(120, [1.0, 1.0, 1.0, 1.0], 1.0)
    assert bs.extrapolate_beats(g, 100.0) == [1.0, 1.0, 1.0, 1.0]


def test_extrapolate_beats_refuses_to_build_absurd_lists():
    g = BeatGrid(120, [1.0, 1.0 + 1e-6, 1.0 + 2e-6], 1.0)
    with pytest.raises(BeatError):
        bs.extrapolate_beats(g, 1e6)


@pytest.mark.parametrize("bad", [
    {"bpm": float("nan"), "beats": [1.0], "confidence": 0.5},
    {"bpm": 120.0, "beats": [1.0, float("inf")], "confidence": 0.5},
    {"bpm": 120.0, "beats": [2.0, 1.0], "confidence": 0.5},
    {"bpm": 120.0, "beats": [1.0], "confidence": float("nan")},
    {"bpm": -5.0, "beats": [1.0], "confidence": 0.5},
    {"bpm": 120.0, "beats": [1.0], "confidence": 0.5, "strengths": [float("nan")]},
])
def test_grid_from_dict_rejects_non_finite_or_unsorted(bad):
    with pytest.raises(BeatError):
        BeatGrid.from_dict(bad)


def test_cache_with_poisoned_grid_is_ignored(tmp_path):
    p = _music(tmp_path)
    bs.save_cache(p, sample_grid())
    cp = bs.cache_path_for(p)
    d = json.loads(cp.read_text())
    d["grid"]["beats"] = [2.0, 1.0]
    cp.write_text(json.dumps(d))
    assert bs.load_cache(p) is None


def test_cache_not_poisoned_when_file_changes_during_detection(tmp_path):
    """The fingerprint must describe the file that was analysed, not whatever is
    on disk when detection finishes."""
    p = _music(tmp_path)

    def detect(path):
        path.write_bytes(b"replaced while we were analysing")
        return sample_grid()

    bs.load_or_detect(p, detect)
    assert bs.load_cache(p) is None


def test_save_cache_failure_leaves_no_tmp_debris(tmp_path, monkeypatch):
    p = _music(tmp_path)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(bs.os, "replace", boom)
    assert bs.save_cache(p, sample_grid()) is None
    monkeypatch.undo()
    assert sorted(os.listdir(tmp_path)) == ["m.mp3"]


@pytest.mark.parametrize("total", [float("nan"), float("inf"), -1.0, 0.0])
def test_plan_cuts_rejects_unusable_total(total):
    with pytest.raises(BeatError):
        bs.plan_cuts([1.0, 2.0, 3.0], total)


@pytest.mark.parametrize("lo,hi", [(float("nan"), 4.0), (1.0, float("nan")), (1.0, float("inf"))])
def test_plan_cuts_rejects_non_finite_limits(lo, hi):
    with pytest.raises(BeatError):
        bs.plan_cuts([1.0, 2.0, 3.0], 10.0, lo, hi)


def test_fit_clips_rejects_non_finite_durations():
    with pytest.raises(BeatError):
        bs.fit_clips_to_beats([1.0, float("nan")], [0.5 * i for i in range(1, 20)])


def test_cuts_for_highlights_rejects_non_finite_total():
    with pytest.raises(BeatError):
        bs.cuts_for_highlights([1.0, 2.0], [1.0, 2.0, 3.0], float("nan"))
