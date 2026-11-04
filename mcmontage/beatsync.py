"""Beat detection and beat-synced cut planning, in pure standard-library Python.

Why this exists: a montage that cuts on the beat feels edited; one that cuts
randomly feels like a slideshow. The runtime environment has no numpy/scipy/
librosa, so everything here is plain Python and tuned to be cheap: audio is
decoded by ffmpeg to ~11 kHz mono, reduced to a 100 Hz onset-strength envelope,
and tempo/beats are derived from that envelope (autocorrelation for tempo,
Ellis-2007 dynamic programming for beat placement).

Pipeline
    decode_pcm / samples_from_wav -> onset_envelope -> normalise_novelty
    -> estimate_tempo -> track_beats -> BeatGrid
Planning
    plan_cuts / fit_clips_to_beats / cuts_for_highlights / snap_to_beat

Caveats (also in the docstrings): tempo is assumed roughly constant; the
reported tempo is folded toward 120 BPM by a prior, so an "octave" of the
musically-felt tempo is a legitimate result; downbeat detection is a heuristic
based on accent energy and is only meaningful for 4/4 music.
"""

from __future__ import annotations

import cmath
import json
import math
import os
import subprocess
import sys
import tempfile
import wave
from array import array
from bisect import bisect_left
from dataclasses import dataclass, field
from operator import add, mul, sub
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_SAMPLE_RATE = 11025
BEATS_PER_BAR = 4
CACHE_VERSION = 1
_MAX_EXTRAPOLATED_BEATS = 200_000
_SILENCE_PEAK = 16  # int16 peak below this is treated as digital silence


class BeatError(RuntimeError):
    """Raised for decode failures, bad input, or unsatisfiable cut plans."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BeatGrid:
    """Result of beat analysis.

    ``strengths`` (onset strength at each beat) and ``downbeat_offset`` are kept
    on the grid so downbeat-aware planning works from a cached grid without
    re-reading the audio. ``confidence`` is 0..1; below ~0.2 the music probably
    has no steady pulse and callers should fall back to unsynced cutting.
    """

    bpm: float
    beats: list[float]
    confidence: float
    strengths: list[float] = field(default_factory=list)
    downbeat_offset: int = 0

    @property
    def downbeats(self) -> list[float]:
        return self.beats[self.downbeat_offset % BEATS_PER_BAR :: BEATS_PER_BAR]

    @property
    def period(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm,
            "beats": list(self.beats),
            "confidence": self.confidence,
            "strengths": list(self.strengths),
            "downbeat_offset": self.downbeat_offset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeatGrid":
        try:
            beats = [float(b) for b in d["beats"]]
            grid = cls(
                bpm=float(d["bpm"]),
                beats=beats,
                confidence=float(d["confidence"]),
                strengths=[float(s) for s in d.get("strengths", [])],
                downbeat_offset=int(d.get("downbeat_offset", 0)),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as e:
            raise BeatError(f"malformed BeatGrid data: {e}") from e
        # JSON happily round-trips NaN/Infinity, and a hand-edited or damaged
        # cache must never feed planners an unsorted or non-finite grid.
        numbers = [grid.bpm, grid.confidence, *grid.beats, *grid.strengths]
        if not all(math.isfinite(v) for v in numbers):
            raise BeatError("malformed BeatGrid data: non-finite number")
        if grid.bpm < 0:
            raise BeatError("malformed BeatGrid data: negative bpm")
        if any(b2 < b1 for b1, b2 in zip(grid.beats, grid.beats[1:])):
            raise BeatError("malformed BeatGrid data: beats not sorted")
        return grid

    def dumps(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def loads(cls, text: str) -> "BeatGrid":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as e:
            raise BeatError(f"invalid BeatGrid JSON: {e}") from e


@dataclass(frozen=True)
class Cut:
    """A contiguous slice [start_s, end_s) of the timeline.

    ``beat_index`` is the index (into the beats list given to the planner) of
    the beat the cut *starts* on; -1 means it starts at t=0 off-beat.
    """

    start_s: float
    end_s: float
    beat_index: int

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# Audio input
# ---------------------------------------------------------------------------


def decode_pcm(
    path, sample_rate: int = DEFAULT_SAMPLE_RATE, ffmpeg: str = "ffmpeg"
) -> array:
    """Decode any audio file to mono signed-16-bit samples via ffmpeg.

    Low sample rate is deliberate: beat tracking needs only the rhythmic
    envelope, and 11 kHz keeps pure-Python processing fast.
    """
    path = Path(path)
    if not path.is_file():
        raise BeatError(f"no such audio file: {path}")
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-vn",
        "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as e:
        raise BeatError(f"ffmpeg not found ({ffmpeg!r})") from e
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise BeatError(
            f"ffmpeg failed decoding {path.name} (exit {proc.returncode}): "
            + " | ".join(tail)
        )
    out = proc.stdout
    out = out[: len(out) - (len(out) % 2)]
    samples = array("h")
    samples.frombytes(out)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def samples_from_wav(path) -> tuple[array, int]:
    """Read a 16-bit PCM WAV (any channel count, mixed to mono) with ``wave``.

    Exists so tests and tools can analyse audio without an ffmpeg binary.
    Returns (samples, sample_rate) at the file's native rate.
    """
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise BeatError("only 16-bit PCM WAV is supported")
            ch, sr = w.getnchannels(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError) as e:
        raise BeatError(f"cannot read WAV {path}: {e}") from e
    data = array("h")
    data.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        data.byteswap()
    if ch > 1:
        data = array(
            "h",
            (sum(data[i : i + ch]) // ch for i in range(0, len(data) - ch + 1, ch)),
        )
    return data, sr


# ---------------------------------------------------------------------------
# FFT (only needed by the optional spectral-flux onset method)
# ---------------------------------------------------------------------------


def fft(x: Sequence[complex]) -> list[complex]:
    """Iterative radix-2 Cooley-Tukey FFT; len(x) must be a power of two."""
    n = len(x)
    if n == 0 or n & (n - 1):
        raise ValueError("fft length must be a power of two")
    a = [complex(v) for v in x]
    j = 0
    for i in range(1, n):  # bit-reversal permutation
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    size = 2
    while size <= n:
        w_step = cmath.exp(-2j * math.pi / size)
        half = size // 2
        for start in range(0, n, size):
            w = 1 + 0j
            for k in range(start, start + half):
                u, v = a[k], a[k + half] * w
                a[k], a[k + half] = u + v, u - v
                w *= w_step
        size <<= 1
    return a


def dft(x: Sequence[complex]) -> list[complex]:
    """Naive O(n^2) DFT; the reference the FFT is tested against."""
    n = len(x)
    return [
        sum(x[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))
        for k in range(n)
    ]


# ---------------------------------------------------------------------------
# Onset strength
# ---------------------------------------------------------------------------


def hop_for(sample_rate: int) -> int:
    """Hop in samples giving a ~100 Hz envelope."""
    return max(1, round(sample_rate / 100))


def frame_rate(sample_rate: int) -> float:
    return sample_rate / hop_for(sample_rate)


def onset_envelope(
    samples: Sequence[int], sample_rate: int, method: str = "energy"
) -> list[float]:
    """Raw onset strength per hop (half-wave-rectified log-energy difference).

    "energy" (default) uses non-overlapping frames, so an onset lands in
    exactly one frame (<= 5 ms timing error) and costs one pass over the audio.
    It watches both the signal and its first difference (a cheap high-pass) so
    kicks and hats both register. "flux" is classic spectral flux via ``fft``:
    better on dense mixes, roughly 20x slower.
    """
    hop = hop_for(sample_rate)
    n_frames = len(samples) // hop
    if n_frames < 2:
        return []
    peak = max(max(samples), -min(samples))
    if peak < _SILENCE_PEAK:
        return []
    if method == "flux":
        return _flux_spectral(samples, hop, n_frames, peak)
    if method != "energy":
        raise ValueError(f"unknown onset method {method!r}")
    inv_low = 1.0 / (hop * peak * peak)
    inv_high = 1.0 / (hop * 4.0 * peak * peak)
    out = [0.0] * n_frames
    prev_lo = prev_hi = None
    for i in range(n_frames):
        seg = samples[i * hop : (i + 1) * hop]
        e_lo = sum(map(mul, seg, seg))
        d = list(map(sub, seg[1:], seg[:-1]))
        e_hi = sum(map(mul, d, d))
        lo = math.log1p(1e4 * e_lo * inv_low)
        hi = math.log1p(1e4 * e_hi * inv_high)
        if prev_lo is not None:
            out[i] = max(0.0, lo - prev_lo) + max(0.0, hi - prev_hi)
        prev_lo, prev_hi = lo, hi
    return out


def _flux_spectral(samples: Sequence[int], hop: int, n_frames: int, peak: int) -> list[float]:
    n_fft = 1
    while n_fft < 2 * hop:
        n_fft <<= 1
    win = [0.5 - 0.5 * math.cos(2 * math.pi * k / n_fft) for k in range(n_fft)]
    inv = 1.0 / peak
    out = [0.0] * n_frames
    prev: list[float] | None = None
    total = len(samples)
    for i in range(n_frames):
        centre = i * hop + hop // 2
        s0 = centre - n_fft // 2
        frame = [
            (samples[s0 + k] * inv if 0 <= s0 + k < total else 0.0) * win[k]
            for k in range(n_fft)
        ]
        spec = fft(frame)[: n_fft // 2]
        mag = [math.log1p(100.0 * abs(c)) for c in spec]
        if prev is not None:
            out[i] = sum(max(0.0, a - b) for a, b in zip(mag, prev))
        prev = mag
    return out


def normalise_novelty(flux: Sequence[float], rate: float, window_s: float = 0.4) -> list[float]:
    """Adaptive-threshold the raw flux and scale it to unit variance.

    Subtracting a moving average removes the noise floor so a steady hiss or a
    loud passage doesn't swamp quieter beats; scaling makes the DP tightness
    constant independent of loudness. All-zero output means "no onsets".
    """
    n = len(flux)
    if n == 0:
        return []
    w = max(3, int(rate * window_s))
    csum = [0.0]
    for v in flux:
        csum.append(csum[-1] + v)
    half = w // 2
    out = []
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out.append(max(0.0, flux[i] - (csum[b] - csum[a]) / (b - a)))
    var = sum(v * v for v in out) / n
    if var < 1e-12:
        return [0.0] * n
    sd = math.sqrt(var)
    return [v / sd for v in out]


def pick_peaks(
    env: Sequence[float], rate: float, threshold: float = 0.5, min_gap_s: float = 0.05,
    window_s: float = 0.5,
) -> list[int]:
    """Indices of onset peaks: local maxima above an adaptive threshold.

    The threshold is ``threshold`` * the local mean (over ``window_s``) plus a
    fraction of the global max noise guard, so quiet passages still yield peaks
    but ripples in noise do not. Peaks closer than ``min_gap_s`` keep the larger.
    """
    n = len(env)
    if n < 3:
        return []
    half = max(1, int(rate * window_s / 2))
    gmax = max(env)
    if gmax <= 0:
        return []
    csum = [0.0]
    for v in env:
        csum.append(csum[-1] + v)
    gap = max(1, int(rate * min_gap_s))
    peaks: list[int] = []
    for i in range(1, n - 1):
        v = env[i]
        if v <= 0 or v < env[i - 1] or v <= env[i + 1]:
            continue
        a, b = max(0, i - half), min(n, i + half + 1)
        local = (csum[b] - csum[a]) / (b - a)
        if v < local * (1.0 + threshold) + 0.05 * gmax:
            continue
        if peaks and i - peaks[-1] < gap:
            if v > env[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


# ---------------------------------------------------------------------------
# Tempo and beat tracking
# ---------------------------------------------------------------------------


def _smooth(env: Sequence[float]) -> list[float]:
    """Tiny triangular smoothing so integer-lag autocorrelation tolerates
    a beat period that isn't a whole number of frames."""
    n = len(env)
    k = (1, 2, 3, 2, 1)
    out = []
    for i in range(n):
        s = 0.0
        for o, wgt in zip(range(-2, 3), k):
            j = i + o
            if 0 <= j < n:
                s += wgt * env[j]
        out.append(s / 9.0)
    return out


def estimate_tempo(
    novelty: Sequence[float],
    rate: float,
    min_bpm: float = 60.0,
    max_bpm: float = 180.0,
    prior_bpm: float = 120.0,
    prior_octaves: float = 0.9,
) -> tuple[float, float]:
    """Return (coarse_bpm, autocorr_strength 0..1) from the novelty envelope.

    Octave errors (half/double tempo) are the classic failure: an autocorrelation
    peaks at every multiple of the beat period. Two guards: each candidate lag
    is credited with its 2x-lag peak (so the true period beats its sub-octave
    only when the pulse genuinely repeats), and a log-Gaussian prior around
    ``prior_bpm`` breaks the remaining ties, as in Ellis 2007.
    Returns (0, 0) when the input is too short to judge.
    """
    x = list(novelty[: int(rate * 120)])
    m = len(x)
    lag_lo = max(2, int(rate * 60.0 / max_bpm) - 1)
    lag_hi = int(math.ceil(rate * 60.0 / min_bpm)) + 1
    if m < 3 * lag_hi:
        return 0.0, 0.0
    mean = sum(x) / m
    x = [v - mean for v in x]
    a0 = sum(v * v for v in x) / m
    if a0 <= 1e-12:
        return 0.0, 0.0
    acf: dict[int, float] = {}
    for lag in range(max(1, lag_lo - 1), 4 * lag_hi + 2):
        if lag >= m // 2:
            acf[lag] = 0.0
        else:
            acf[lag] = sum(map(mul, x, x[lag:])) / (m - lag) / a0

    def peak(lag: int) -> float:
        # A beat period is rarely a whole number of frames. Sampling the ACF at
        # integer lags alone favours periods whose multiples happen to fall
        # near an integer (e.g. 1.5 beats at 145 BPM beats 1 beat, giving a
        # 3:2 tempo error), so read the local peak height instead.
        return max(acf.get(lag - 1, 0.0), acf.get(lag, 0.0), acf.get(lag + 1, 0.0))

    def score(lag: int) -> float:
        bpm = 60.0 * rate / lag
        prior = math.exp(-0.5 * (math.log2(bpm / prior_bpm) / prior_octaves) ** 2)
        return prior * (peak(lag) + 0.5 * peak(2 * lag) + 0.25 * peak(4 * lag))

    def raw(lag: int) -> float:
        return peak(lag) + 0.5 * peak(2 * lag) + 0.25 * peak(4 * lag)

    best = max(range(lag_lo, lag_hi + 1), key=score)
    # 3:2 guard. A period of 1.5 beats is as periodic as the beat itself in any
    # groove with eighth-note hats, and the 120-BPM prior then tips the choice
    # toward it (e.g. 155 BPM reported as 103). Unlike a 2x octave this is never
    # a musically valid answer, so prefer the shorter period when it is about
    # as strong without the prior.
    alt = int(round(best * 2.0 / 3.0))
    if alt >= lag_lo:
        alt = max(range(alt - 1, alt + 2), key=raw)
        if alt >= lag_lo and raw(alt) >= 0.95 * raw(best):
            best = alt
    frac = float(best)
    if lag_lo < best < lag_hi:
        y0, y1, y2 = score(best - 1), score(best), score(best + 1)
        den = y0 - 2 * y1 + y2
        if den < 0:
            frac = best + 0.5 * (y0 - y2) / den
    return 60.0 * rate / frac, max(0.0, min(1.0, peak(best)))


def track_beats(
    novelty: Sequence[float], period_frames: float, tightness: float = 100.0
) -> list[int]:
    """Ellis (2007) dynamic-programming beat tracker; returns frame indices.

    Maximises sum(novelty at beats) minus a penalty on log-deviation of each
    inter-beat interval from the target period. Compared with picking a fixed
    grid, it follows onset peaks (so swing/human timing is respected) yet
    stays regular through sections with no onsets.
    """
    n = len(novelty)
    p = period_frames
    if n == 0 or p < 2:
        return []
    local = _smooth(novelty)
    lo, hi = max(1, int(round(p / 2))), int(round(2 * p))
    pens_rev = [-tightness * math.log(d / p) ** 2 for d in range(hi, lo - 1, -1)]
    cum = [0.0] * n
    back = [-1] * n
    for t in range(n):
        a, b = max(0, t - hi), t - lo
        if b >= a:
            seg = cum[a : b + 1]
            off = a - (t - hi)
            vals = list(map(add, seg, pens_rev[off : off + len(seg)]))
            m = max(vals)
            if m > 0:
                cum[t] = local[t] + m
                back[t] = a + vals.index(m)
                continue
        cum[t] = local[t]
    tail_start = max(0, n - hi)
    last = max(range(tail_start, n), key=lambda i: (cum[i], -i))
    frames = []
    while last >= 0:
        frames.append(last)
        last = back[last]
    frames.reverse()
    return frames


def estimate_downbeat_offset(strengths: Sequence[float], beats_per_bar: int = BEATS_PER_BAR) -> int:
    """Which beat phase (0..beats_per_bar-1) carries the most accent energy.

    Downbeats in most music are the loudest/most onset-rich beat of the bar.
    With fewer than two bars of data, or no clear winner, returns 0.
    """
    if len(strengths) < 2 * beats_per_bar:
        return 0
    sums = [0.0] * beats_per_bar
    counts = [0] * beats_per_bar
    for i, s in enumerate(strengths):
        sums[i % beats_per_bar] += s
        counts[i % beats_per_bar] += 1
    means = [s / c for s, c in zip(sums, counts)]
    best = max(range(beats_per_bar), key=lambda k: (means[k], -k))
    runner = max(m for k, m in enumerate(means) if k != best)
    if means[best] < runner * 1.05:
        return 0
    return best


def analyse_samples(
    samples: Sequence[int],
    sample_rate: int,
    min_bpm: float = 60.0,
    max_bpm: float = 180.0,
    method: str = "energy",
) -> BeatGrid:
    """Detect tempo and beats in raw mono PCM. The testable core of the module.

    Silence, noise-free emptiness and inputs too short for a couple of bars
    yield an empty grid (bpm 0, no beats, confidence 0) rather than an error,
    so callers can treat "no beat" as an ordinary fallback.
    """
    empty = BeatGrid(0.0, [], 0.0)
    flux = onset_envelope(samples, sample_rate, method)
    if not flux:
        return empty
    rate = frame_rate(sample_rate)
    nov = normalise_novelty(flux, rate)
    if not any(nov):
        return empty
    nov_s = _smooth(nov)
    coarse, conf = estimate_tempo(nov_s, rate, min_bpm, max_bpm)
    if coarse <= 0:
        return empty
    frames = track_beats(nov, rate * 60.0 / coarse)
    if len(frames) >= 3:
        # Trim edge beats the DP extrapolated through silence (Ellis 2007).
        strength = [max(nov[max(0, f - 2) : f + 3]) for f in frames]
        thr = 0.25 * sum(strength) / len(strength)
        while len(frames) > 2 and strength[0] < thr:
            frames.pop(0)
            strength.pop(0)
        while len(frames) > 2 and strength[-1] < thr:
            frames.pop()
            strength.pop()
    else:
        strength = [max(nov[max(0, f - 2) : f + 3]) for f in frames]
    hop = hop_for(sample_rate)
    beats = [(f + 0.5) * hop / sample_rate for f in frames]
    bpm = coarse
    if len(beats) >= 4:
        k0 = beats[0]
        idx = [round((b - k0) / (60.0 / coarse)) for b in beats]
        n = len(beats)
        mi, mt = sum(idx) / n, sum(beats) / n
        sxx = sum((i - mi) ** 2 for i in idx)
        if sxx > 0:
            slope = sum((i - mi) * (b - mt) for i, b in zip(idx, beats)) / sxx
            if slope > 0 and abs(60.0 / slope - coarse) / coarse < 0.08:
                bpm = 60.0 / slope
    return BeatGrid(
        bpm=bpm,
        beats=beats,
        confidence=conf,
        strengths=strength,
        downbeat_offset=estimate_downbeat_offset(strength),
    )


def detect_beats(
    path, sample_rate: int = DEFAULT_SAMPLE_RATE, ffmpeg: str = "ffmpeg", **kw
) -> BeatGrid:
    """Decode ``path`` with ffmpeg and run ``analyse_samples``."""
    return analyse_samples(decode_pcm(path, sample_rate, ffmpeg), sample_rate, **kw)


def extrapolate_beats(grid: BeatGrid, until_s: float) -> list[float]:
    """Continue the beat list past its last beat at a constant period.

    Music is often shorter than the montage (and loops); planning needs beats
    to cover the whole timeline.
    """
    beats = list(grid.beats)
    if len(beats) < 2:
        return beats
    gaps = sorted(b - a for a, b in zip(beats, beats[1:]))
    period = gaps[len(gaps) // 2]
    if not period > 0 or not math.isfinite(until_s):
        return beats          # degenerate grid: nothing sensible to continue
    if (until_s - beats[-1]) / period > _MAX_EXTRAPOLATED_BEATS:
        raise BeatError(
            f"refusing to extrapolate {(until_s - beats[-1]) / period:.0f} beats "
            f"at a {period:.6f}s period")
    while beats[-1] + period <= until_s:
        beats.append(beats[-1] + period)
    return beats


# ---------------------------------------------------------------------------
# Snapping and cut planning
# ---------------------------------------------------------------------------


def nearest_beat_index(t: float, beats: Sequence[float]) -> int:
    """Index of the beat closest to ``t`` (earlier beat wins ties); -1 if none."""
    if not beats:
        return -1
    i = bisect_left(beats, t)
    if i == 0:
        return 0
    if i == len(beats):
        return len(beats) - 1
    return i - 1 if t - beats[i - 1] <= beats[i] - t else i


def snap_to_beat(t: float, beats: Sequence[float], tolerance: float | None = None) -> float:
    """Nearest beat time to ``t``, or ``t`` itself if no beat is within
    ``tolerance`` seconds (None = always snap). Empty beats return ``t``."""
    i = nearest_beat_index(t, beats)
    if i < 0:
        return t
    if tolerance is not None and abs(beats[i] - t) > tolerance:
        return t
    return beats[i]


def _check_sorted(beats: Sequence[float]) -> None:
    if any(b2 < b1 for b1, b2 in zip(beats, beats[1:])):
        raise BeatError("beats must be sorted ascending")


def plan_cuts(
    beats: Sequence[float],
    total_s: float,
    min_clip_s: float = 1.0,
    max_clip_s: float = 8.0,
    prefer_downbeats: bool = True,
    clip_durations: Sequence[float] | None = None,
    downbeat_offset: int = 0,
    downbeat_weight: float = 0.1,
) -> list[Cut]:
    """Partition [0, total_s] into cuts whose inner boundaries are beats.

    A dynamic program (not greedy) is used because greedy snapping can strand a
    final remainder shorter than ``min_clip_s``; the DP guarantees every cut
    length is in [min_clip_s, max_clip_s] whenever that is possible, else raises
    BeatError. The first cut starts at 0 and the last ends at ``total_s``
    (neither need be on a beat). Preferred lengths come from ``clip_durations``
    (cycled, e.g. the source clips' natural lengths) or default to the midpoint
    of min/max. With ``prefer_downbeats``, boundaries off the bar line cost
    ``downbeat_weight`` extra (in squared-relative-length units).
    If total_s < min_clip_s a single cut is returned (min can't be honoured).
    """
    if not (math.isfinite(total_s) and total_s > 0):
        raise BeatError("total_s must be positive and finite")
    if not (math.isfinite(min_clip_s) and math.isfinite(max_clip_s)
            and 0 < min_clip_s <= max_clip_s):
        raise BeatError("need 0 < min_clip_s <= max_clip_s (finite)")
    if clip_durations is not None and (
        not clip_durations or any(d <= 0 for d in clip_durations)
    ):
        raise BeatError("clip_durations must be positive")
    _check_sorted(beats)
    if total_s < min_clip_s:
        return [Cut(0.0, total_s, -1)]
    eps = 1e-9
    nodes = [(-1, 0.0)] + [
        (i, t) for i, t in enumerate(beats) if eps < t < total_s - eps
    ] + [(-2, total_s)]
    n = len(nodes)
    mid = (min_clip_s + max_clip_s) / 2.0
    m = len(clip_durations) if clip_durations else 0
    # states[j][k] = (cost, prev_j, prev_k); k = segment ordinal (0 if no durations)
    states: list[dict[int, tuple[float, int, int]]] = [dict() for _ in range(n)]
    states[0][0] = (0.0, -1, -1)
    for j in range(n - 1):
        tj = nodes[j][1]
        for k, (c, _, _) in list(states[j].items()):
            target = clip_durations[k % m] if m else mid
            for j2 in range(j + 1, n):
                dur = nodes[j2][1] - tj
                if dur > max_clip_s + eps:
                    break
                if dur < min_clip_s - eps:
                    continue
                cost = c + ((dur - target) / target) ** 2
                bi = nodes[j2][0]
                if prefer_downbeats and bi >= 0 and (bi - downbeat_offset) % BEATS_PER_BAR:
                    cost += downbeat_weight
                k2 = k + 1 if m else 0
                if k2 not in states[j2] or cost < states[j2][k2][0]:
                    states[j2][k2] = (cost, j, k)
    if not states[n - 1]:
        raise BeatError(
            f"cannot split {total_s:.2f}s into beat-aligned cuts of "
            f"{min_clip_s}-{max_clip_s}s with the given beats"
        )
    k = min(states[n - 1], key=lambda kk: states[n - 1][kk][0])
    j = n - 1
    path = []
    while j > 0:
        path.append(j)
        _, pj, pk = states[j][k]
        j, k = pj, pk
    path.append(0)
    path.reverse()
    return [
        Cut(nodes[a][1], nodes[b][1], nodes[a][0]) for a, b in zip(path, path[1:])
    ]


def fit_clips_to_beats(
    durations: Sequence[float], beats: Sequence[float], start_s: float = 0.0, window: int = 4
) -> list[float]:
    """Retime clip lengths so every clip *end* lands on a beat.

    Minimises total drift sum|boundary_k - ideal_k| (ideal = cumulative sum of
    the requested durations from ``start_s``) plus half the per-clip length
    error, with boundaries strictly increasing. Each boundary may choose among
    the ``window`` beats either side of its ideal; the window widens
    automatically when clips are shorter than a beat so distinct beats exist.
    Returns new durations (same count); raises BeatError if there are too few
    beats after ``start_s``.
    """
    if not durations:
        return []
    if any(not (math.isfinite(d) and d > 0) for d in durations):
        raise BeatError("durations must be positive and finite")
    if not math.isfinite(start_s):
        raise BeatError("start_s must be finite")
    _check_sorted(beats)
    K = len(durations)
    first = bisect_left(beats, start_s + 1e-9)
    nb = len(beats)
    if nb - first < K:
        raise BeatError(f"need {K} beats after {start_s:.2f}s, have {nb - first}")
    ideal, acc = [], start_s
    for d in durations:
        acc += d
        ideal.append(acc)
    cands: list[list[int]] = []
    for k in range(K):
        i0 = max(first, nearest_beat_index(ideal[k], beats))
        lo = max(i0 - window, first + k)
        hi = min(max(i0 + window, lo + window), nb - 1 - (K - 1 - k))
        if lo > hi:
            raise BeatError("not enough beats to retime clips")
        cands.append(list(range(lo, hi + 1)))
    prev_cost = {i: (abs(beats[i] - ideal[0]) + 0.5 * abs(beats[i] - start_s - durations[0]), -1)
                 for i in cands[0]}
    table = [prev_cost]
    for k in range(1, K):
        cur = {}
        for i in cands[k]:
            best = None
            for pi, (pc, _) in table[-1].items():
                if beats[pi] >= beats[i] - 1e-9:
                    continue
                c = pc + abs(beats[i] - ideal[k]) + 0.5 * abs(beats[i] - beats[pi] - durations[k])
                if best is None or c < best[0]:
                    best = (c, pi)
            if best is not None:
                cur[i] = best
        if not cur:
            raise BeatError("not enough distinct beats to retime clips")
        table.append(cur)
    i = min(table[-1], key=lambda ii: table[-1][ii][0])
    chosen = []
    for k in range(K - 1, -1, -1):
        chosen.append(i)
        i = table[k][i][1]
    chosen.reverse()
    times = [start_s] + [beats[i] for i in chosen]
    return [b - a for a, b in zip(times, times[1:])]


def _allocate(weights: Sequence[float], total: int, lo: int, hi: int) -> list[int]:
    """Integer shares summing to ``total``, each in [lo, hi], roughly
    proportional to ``weights`` (largest-deficit-first)."""
    n = len(weights)
    hi = max(hi, math.ceil(total / n))
    ws = sum(weights)
    shares = [lo] * n
    for _ in range(total - n * lo):
        i = max(
            (i for i in range(n) if shares[i] < hi),
            key=lambda i: (total * weights[i] / ws - shares[i], -i),
        )
        shares[i] += 1
    return shares


def cuts_for_highlights(
    highlight_scores: Sequence[float],
    beats: Sequence[float],
    total_s: float,
    min_beats: int = 2,
    max_beats: int = 16,
) -> list[Cut]:
    """One beat-aligned slot per highlight; higher scores get longer slots.

    Slot lengths are counted in beats (so they scale with tempo), proportional
    to score but each within [min_beats, max_beats] where the available beats
    allow. Returns len(scores) contiguous cuts covering [0, total_s]; inner
    boundaries are beats. Raises BeatError if there are fewer usable beats than
    highlights.
    """
    n = len(highlight_scores)
    if n == 0:
        return []
    if not (math.isfinite(total_s) and total_s > 0):
        raise BeatError("total_s must be positive and finite")
    _check_sorted(beats)
    usable = [(i, t) for i, t in enumerate(beats) if 1e-9 < t < total_s - 1e-9]
    steps = len(usable) + 1  # beat-to-beat intervals, counting 0 and total_s as ends
    if steps < n:
        raise BeatError(f"{len(usable)} beats in {total_s:.1f}s is too few for {n} highlights")
    lo = max(1, min(min_beats, steps // n))
    weights = [max(0.0, s) if math.isfinite(s) else 0.0 for s in highlight_scores]
    weights = [w + 1e-6 for w in weights]
    shares = _allocate(weights, steps, lo, max(max_beats, lo))
    cuts, pos = [], 0  # pos: boundary position, 0 = time 0, p = usable[p-1]
    for sh in shares:
        nxt = pos + sh
        start_idx, start_t = (-1, 0.0) if pos == 0 else usable[pos - 1]
        end_t = total_s if nxt >= steps else usable[nxt - 1][1]
        cuts.append(Cut(start_t, end_t, start_idx))
        pos = nxt
    return cuts


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def cache_path_for(music_path) -> Path:
    """Sidecar cache next to the music file: ``song.mp3`` -> ``song.mp3.beats.json``."""
    p = Path(music_path)
    return p.with_name(p.name + ".beats.json")


def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def save_cache(music_path, grid: BeatGrid, fingerprint: dict | None = None) -> Path | None:
    """Write the cache atomically; returns None (not an error) if the
    directory is read-only, since caching is only an optimisation.

    ``fingerprint`` should be the file's fingerprint from *before* analysis
    started: stamping it afterwards would bless a grid computed from different
    bytes if the file was replaced mid-analysis."""
    music_path = Path(music_path)
    cp = cache_path_for(music_path)
    tmp = None
    try:
        payload = {"version": CACHE_VERSION,
                   "source": fingerprint or _fingerprint(music_path), "grid": grid.to_dict()}
        fd, tmp_name = tempfile.mkstemp(dir=str(cp.parent), prefix=cp.name + ".", suffix=".tmp")
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload))
        os.replace(tmp, cp)
        tmp = None
    except OSError:
        return None
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
    return cp


def load_cache(music_path) -> BeatGrid | None:
    """Cached grid if present, well-formed, and matching the file's size+mtime."""
    music_path = Path(music_path)
    try:
        payload = json.loads(cache_path_for(music_path).read_text(encoding="utf-8"))
        if payload.get("version") != CACHE_VERSION:
            return None
        if payload.get("source") != _fingerprint(music_path):
            return None
        return BeatGrid.from_dict(payload["grid"])
    except (OSError, ValueError, AttributeError, BeatError):
        return None


def load_or_detect(
    music_path,
    detect: Callable[..., BeatGrid] = detect_beats,
    force: bool = False,
) -> BeatGrid:
    """Return the cached analysis when still valid, else detect and cache it.

    Keyed on mtime+size rather than a content hash so checking is O(1) on
    multi-minute tracks; replacing the file in place invalidates the cache.
    """
    music_path = Path(music_path)
    if not music_path.is_file():
        raise BeatError(f"no such audio file: {music_path}")
    if not force:
        cached = load_cache(music_path)
        if cached is not None:
            return cached
    try:
        before = _fingerprint(music_path)
    except OSError as e:
        raise BeatError(f"cannot stat audio file {music_path}: {e}") from e
    grid = detect(music_path)
    save_cache(music_path, grid, before)
    return grid
