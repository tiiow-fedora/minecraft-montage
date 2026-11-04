"""Tests for mcmontage.transitions: string-level first, then real ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from mcmontage import transitions as tr
from mcmontage.transitions import (
    EASINGS, TRANSITIONS, ClipSpec, FilterGraph, Transition,
    atempo_chain, available_transitions, build_audio_graph, build_video_graph,
    pick_transitions, render_graph_args, timeline, validate_transitions,
)

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe")
                   and tr.ffmpeg_has_xfade())
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/xfade missing")


def clips(*durs, audio=False):
    return [ClipSpec(i, d, has_audio=audio) for i, d in enumerate(durs)]


# ---------------------------------------------------------------- registry

def test_transitions_registry_unique_and_lowercase():
    assert len(set(TRANSITIONS)) == len(TRANSITIONS)
    assert all(t == t.lower() and t.isalnum() for t in TRANSITIONS)


@pytest.mark.parametrize("name", ["fade", "wipeleft", "slideright",
                                  "circleopen", "dissolve", "pixelize",
                                  "radial", "fadeblack", "fadewhite",
                                  "distance", "smoothleft"])
def test_required_names_in_registry(name):
    assert name in TRANSITIONS


def test_min_ffmpeg_note_present():
    assert tr.MIN_FFMPEG and all(isinstance(v, str) for v in tr.MIN_FFMPEG.values())


# -------------------------------------------------------------- Transition

def test_transition_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown transition"):
        Transition("spinny", 0.5)


def test_transition_rejects_tiny_or_negative_duration():
    for d in (0, -1, 0.01):
        with pytest.raises(ValueError):
            Transition("fade", d)


def test_transition_custom_requires_expr():
    with pytest.raises(ValueError, match="needs an expr"):
        Transition("custom", 0.5)


def test_transition_expr_forces_custom():
    t = Transition("fade", 0.5, "A*P+B*(1-P)")
    assert t.name == "custom"


def test_transition_duration_rounded_to_ms():
    assert Transition("fade", 0.12345).duration == 0.123


def test_eased_known_and_unknown():
    for k in EASINGS:
        assert Transition.eased(k, 0.5).expr == EASINGS[k]
    with pytest.raises(ValueError):
        Transition.eased("bouncy")


def test_filter_args_plain():
    assert Transition("wipeleft", 0.5).filter_args(1.5) == \
        "xfade=transition=wipeleft:duration=0.5:offset=1.5"


def test_filter_args_custom_escapes_commas():
    s = Transition("fade", 0.5, "if(gt(P,0.5),A,B)").filter_args(2)
    assert "expr=if(gt(P\\,0.5)\\,A\\,B)" in s
    assert "transition=custom" in s


def test_esc_expr_specials():
    assert tr.esc_expr("a,b:c;d[e]'f") == r"a\,b\:c\;d\[e\]\'f"


def test_with_duration_keeps_name_and_expr():
    t = Transition("fade", 0.5, "A").with_duration(0.2)
    assert (t.name, t.duration, t.expr) == ("custom", 0.2, "A")


# ---------------------------------------------------------------- ClipSpec

def test_clipspec_length_with_speed():
    assert ClipSpec(0, 4.0, speed=2.0).length == 2.0
    assert ClipSpec(0, 3.0, speed=0.5).length == 6.0


@pytest.mark.parametrize("kw", [dict(duration=0), dict(duration=-1),
                                dict(duration=1, speed=0),
                                dict(duration=1, index=-1)])
def test_clipspec_validation(kw):
    kw.setdefault("index", 0)
    with pytest.raises(ValueError):
        ClipSpec(**kw)


# ----------------------------------------------------------------- timeline

def test_timeline_offsets_and_total():
    cs = clips(4, 4, 4)
    ts = [Transition("fade", 1.0), Transition("fade", 0.5)]
    offsets, total = timeline(cs, ts)
    assert offsets == [3.0, 6.5]
    assert total == 4 + 3 + 3.5 - 0  # 4, +3 (4-1), +3.5 (4-0.5)
    assert total == 10.5


def test_timeline_single_clip():
    assert timeline(clips(5), []) == ([], 5.0)


def test_timeline_with_speed():
    cs = [ClipSpec(0, 4, speed=2), ClipSpec(1, 2, speed=1)]
    offsets, total = timeline(cs, [Transition("fade", 0.5)])
    assert offsets == [1.5] and total == 3.5


# --------------------------------------------------------------- validation

def test_validate_count_mismatch():
    with pytest.raises(ValueError, match="need 2 transitions"):
        validate_transitions(clips(2, 2, 2), [Transition()])


def test_validate_too_long_for_neighbour():
    with pytest.raises(ValueError, match="too long"):
        validate_transitions(clips(2, 0.4), [Transition("fade", 0.5)])


def test_validate_adjacent_transitions_must_fit_in_middle_clip():
    cs = clips(3, 1.0, 3)
    ts = [Transition("fade", 0.6), Transition("fade", 0.6)]
    with pytest.raises(ValueError, match="too long"):
        validate_transitions(cs, ts)


def test_validate_exact_fit_allowed():
    cs = clips(3, 1.0, 3)
    ts = [Transition("fade", 0.5), Transition("fade", 0.5)]
    assert validate_transitions(cs, ts) == ts


def test_clamp_shortens_instead_of_raising():
    out = validate_transitions(clips(2, 0.4), [Transition("fade", 0.5)],
                               clamp=True)
    assert out[0].duration <= 0.4 and out[0].duration >= tr.MIN_DURATION


def test_clamp_middle_clip_sum():
    cs = clips(3, 1.0, 3)
    ts = [Transition("fade", 0.8), Transition("fade", 0.8)]
    out = validate_transitions(cs, ts, clamp=True)
    assert out[0].duration + out[1].duration <= 1.0 + 1e-9


def test_clamp_raises_when_impossible():
    with pytest.raises(ValueError, match="cannot clamp"):
        validate_transitions(clips(2, 0.02), [Transition("fade", 0.5)],
                             clamp=True)


def test_empty_clips_rejected():
    with pytest.raises(ValueError):
        build_video_graph([], [])


def test_duplicate_input_index_rejected():
    with pytest.raises(ValueError, match="used twice"):
        build_video_graph([ClipSpec(0, 2), ClipSpec(0, 2)],
                          [Transition("fade", 0.5)])


# ----------------------------------------------------------- pick_transitions

def test_pick_count_and_determinism():
    a = pick_transitions(8, "dynamic", seed=3)
    assert len(a) == 7
    assert a == pick_transitions(8, "dynamic", seed=3)


def test_pick_seed_changes_result():
    assert pick_transitions(12, "dynamic", seed=1) != \
        pick_transitions(12, "dynamic", seed=2)


def test_pick_no_adjacent_repeats_for_multi_pools():
    for style in ("subtle", "dynamic"):
        names = [t.name for t in pick_transitions(60, style, seed=9)]
        assert all(a != b for a, b in zip(names, names[1:]))


def test_pick_fade_style_is_all_fade():
    assert {t.name for t in pick_transitions(5, "fade")} == {"fade"}


def test_pick_names_valid_and_pools_in_registry():
    for pool in tr.STYLE_POOLS.values():
        assert set(pool) <= set(TRANSITIONS)


def test_pick_edge_counts_and_duration_override():
    assert pick_transitions(0) == [] and pick_transitions(1) == []
    assert all(t.duration == 0.25 for t in pick_transitions(4, duration=0.25))


def test_pick_unknown_style():
    with pytest.raises(ValueError):
        pick_transitions(3, "chaotic")


# ------------------------------------------------------------- graph strings

def test_single_clip_passthrough_has_no_xfade():
    g = build_video_graph(clips(3), [], 160, 90, 15)
    assert "xfade" not in g.filter_complex
    assert g.label == "v0" and g.duration == 3.0
    assert len(g.chains) == 1


def test_normalisation_chain_contents():
    g = build_video_graph(clips(3), [], 1280, 720, 30)
    c = g.chains[0]
    for part in ("[0:v]", "trim=duration=3", "scale=1280:720:force_original_aspect_ratio=decrease",
                 "pad=1280:720:(ow-iw)/2:(oh-ih)/2", "fps=30", "format=yuv420p",
                 "setsar=1", "[v0]"):
        assert part in c


def test_xfade_chain_offsets_and_labels():
    cs = clips(4, 4, 4)
    ts = [Transition("fade", 1.0), Transition("wipeleft", 0.5)]
    g = build_video_graph(cs, ts, 160, 90, 15)
    assert "[v0][v1]xfade=transition=fade:duration=1:offset=3[x1]" in g.chains
    assert "[x1][v2]xfade=transition=wipeleft:duration=0.5:offset=6.5[x2]" in g.chains
    assert g.label == "x2" and g.duration == 10.5


def test_prefix_applies_to_all_labels():
    g = build_video_graph(clips(2, 2), [Transition()], 160, 90, 15, prefix="m_")
    assert g.label == "m_x1"
    assert "[m_v0][m_v1]xfade" in g.filter_complex


def test_speed_changes_pts_and_total():
    cs = [ClipSpec(0, 4, speed=2.0), ClipSpec(1, 2)]
    g = build_video_graph(cs, [Transition("fade", 0.5)], 160, 90, 15)
    assert "setpts=(PTS-STARTPTS)/2" in g.chains[0]
    assert "offset=1.5" in g.chains[2]
    assert g.duration == 3.5


def test_input_index_respected():
    g = build_video_graph([ClipSpec(5, 2), ClipSpec(3, 2)], [Transition()],
                         160, 90, 15)
    assert g.chains[0].startswith("[5:v]") and g.chains[1].startswith("[3:v]")


def test_default_transitions_picked_by_style_seed():
    a = build_video_graph(clips(2, 2, 2), None, 160, 90, 15, style="dynamic", seed=4)
    b = build_video_graph(clips(2, 2, 2), None, 160, 90, 15, style="dynamic", seed=4)
    assert a == b


def test_graph_is_deterministic_and_filter_complex_joined():
    a = build_video_graph(clips(2, 2), [Transition()], 160, 90, 15)
    b = build_video_graph(clips(2, 2), [Transition()], 160, 90, 15)
    assert a.filter_complex == b.filter_complex == ";".join(a.chains)


def test_clamp_option_flows_through_build():
    with pytest.raises(ValueError):
        build_video_graph(clips(2, 0.3), [Transition("fade", 0.5)], 160, 90, 15)
    g = build_video_graph(clips(2, 0.3), [Transition("fade", 0.5)], 160, 90, 15,
                          clamp=True)
    assert g.duration == pytest.approx(2.3 - 0.3, abs=0.01)


def test_video_only_by_default_for_silent_clips():
    g = build_video_graph(clips(2, 2), [Transition()], 160, 90, 15)
    assert g.audio_label is None and g.audio_map is None
    assert "acrossfade" not in g.filter_complex


def test_audio_auto_included_when_clips_have_audio():
    g = build_video_graph(clips(2, 2, audio=True), [Transition("fade", 0.5)],
                          160, 90, 15)
    assert g.audio_label == "ax1" and g.audio_map == "[ax1]"
    assert "acrossfade=d=0.5" in g.filter_complex


def test_audio_graph_silence_for_silent_clip():
    cs = [ClipSpec(0, 2, has_audio=True), ClipSpec(1, 2, has_audio=False)]
    g = build_audio_graph(cs, [Transition("fade", 0.5)])
    assert any(c.startswith("anullsrc") for c in g.chains)
    assert g.kind == "audio" and g.duration == 3.5 and g.audio_map == "[ax1]"


def test_audio_graph_speed_uses_atempo():
    cs = [ClipSpec(0, 4, speed=2.0, has_audio=True)]
    g = build_audio_graph(cs, [])
    assert "atempo=2" in g.chains[0]
    assert g.label == "a0"


@pytest.mark.parametrize("speed,expected", [
    (1.0, ["atempo=1"]), (1.5, ["atempo=1.5"]), (2.0, ["atempo=2"]),
    (4.0, ["atempo=2.0", "atempo=2"]), (0.25, ["atempo=0.5", "atempo=0.5"]),
    (3.0, ["atempo=2.0", "atempo=1.5"]),
])
def test_atempo_chain(speed, expected):
    assert atempo_chain(speed) == expected


def test_filtergraph_helpers():
    g = FilterGraph(["a", "b"], "out", 3.0)
    assert g.chains == ("a", "b") and g.video_map == "[out]"
    assert g.extend(["c"]).filter_complex == "a;b;c"
    assert g.extend(["c"]).label == "out"


# ------------------------------------------------------------- render args

def test_render_args_with_audio_and_trim():
    g = build_video_graph(clips(2, 2, audio=True), [Transition()], 160, 90, 15)
    a = render_graph_args(g, ["a.mp4", "b.mp4"], "out.mp4")
    assert a[0] == "ffmpeg" and a[-1] == "out.mp4"
    assert a.count("-i") == 2 and "-filter_complex" in a
    assert a[a.index("-filter_complex") + 1] == g.filter_complex
    assert "[x1]" in a and "[ax1]" in a and "-an" not in a
    assert a[a.index("-t") + 1] == "3.5"


def test_render_args_silent_uses_an():
    g = build_video_graph(clips(2), [], 160, 90, 15)
    a = render_graph_args(g, ["a.mp4"], "o.mp4", overwrite=False,
                          trim_to_duration=False)
    assert "-an" in a and "-n" in a and "-y" not in a and "-t" not in a


def test_render_args_audio_only_graph():
    g = build_audio_graph(clips(2, audio=True), [])
    a = render_graph_args(g, ["a.mp4"], "o.m4a")
    assert "libx264" not in a and "aac" in a


# ------------------------------------------------------------- integration

def _make(path, src, dur=1.0, audio=False):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
           f"{src}=s=160x90:r=15:d={dur}"] if src == "testsrc2" else \
          ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
           f"color=c={src}:s=160x90:r=15:d={dur}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f=440:d={dur}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(path)]
    subprocess.run(cmd, check=True)
    return path


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    return float(d["format"]["duration"]), d["streams"]


def _render(graph, inputs, out):
    args = render_graph_args(graph, inputs, out, preset="ultrafast")
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return _duration(out)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return {
        "a": _make(d / "a.mp4", "testsrc2"),
        "b": _make(d / "b.mp4", "red"),
        "c": _make(d / "c.mp4", "blue"),
        "a_snd": _make(d / "a_snd.mp4", "testsrc2", audio=True),
        "b_snd": _make(d / "b_snd.mp4", "green", audio=True),
        "wide": _make(d / "wide.mp4", "testsrc2", dur=1.0),
        "dir": d,
    }


@needs_ffmpeg
@pytest.mark.parametrize("name", TRANSITIONS)
def test_every_transition_renders_with_predicted_duration(media, name):
    cs = [ClipSpec(0, 1.0), ClipSpec(1, 1.0)]
    g = build_video_graph(cs, [Transition(name, 0.4)], 160, 90, 15)
    dur, streams = _render(g, [media["a"], media["b"]],
                           media["dir"] / f"out_{name}.mp4")
    assert dur == pytest.approx(g.duration, abs=0.15) and g.duration == 1.6
    assert streams[0]["width"] == 160 and streams[0]["height"] == 90


@needs_ffmpeg
def test_available_transitions_covers_registry_on_this_build():
    got = available_transitions()
    assert got == TRANSITIONS, (
        f"this ffmpeg lacks: {sorted(set(TRANSITIONS) - set(got))}")


@needs_ffmpeg
def test_three_clip_chain_with_mixed_transitions(media):
    cs = [ClipSpec(0, 1.0), ClipSpec(1, 1.0), ClipSpec(2, 1.0)]
    ts = [Transition("circleopen", 0.3), Transition("slideleft", 0.4)]
    g = build_video_graph(cs, ts, 160, 90, 15)
    dur, _ = _render(g, [media["a"], media["b"], media["c"]],
                     media["dir"] / "three.mp4")
    assert g.duration == pytest.approx(2.3)
    assert dur == pytest.approx(g.duration, abs=0.15)


@needs_ffmpeg
def test_picked_transitions_render(media):
    cs = [ClipSpec(i, 1.0) for i in range(3)]
    g = build_video_graph(cs, None, 160, 90, 15, style="dynamic", seed=11)
    dur, _ = _render(g, [media["a"], media["b"], media["c"]],
                     media["dir"] / "picked.mp4")
    assert dur == pytest.approx(g.duration, abs=0.15)


@needs_ffmpeg
def test_custom_expr_and_easing_render(media):
    cs = [ClipSpec(0, 1.0), ClipSpec(1, 1.0)]
    for t in (Transition.eased("inout", 0.4),
              Transition("fade", 0.4, "if(gt(P,0.5),A,B)")):
        g = build_video_graph(cs, [t], 160, 90, 15)
        dur, _ = _render(g, [media["a"], media["b"]],
                         media["dir"] / "custom.mp4")
        assert dur == pytest.approx(1.6, abs=0.15)


@needs_ffmpeg
def test_single_clip_passthrough_renders_and_trims(media):
    g = build_video_graph([ClipSpec(0, 0.6)], [], 160, 90, 15)
    dur, _ = _render(g, [media["a"]], media["dir"] / "single.mp4")
    assert dur == pytest.approx(0.6, abs=0.15)


@needs_ffmpeg
def test_speed_up_renders_shorter(media):
    cs = [ClipSpec(0, 1.0, speed=2.0), ClipSpec(1, 1.0)]
    g = build_video_graph(cs, [Transition("fade", 0.2)], 160, 90, 15)
    dur, _ = _render(g, [media["a"], media["b"]], media["dir"] / "fast.mp4")
    assert g.duration == pytest.approx(1.3)
    assert dur == pytest.approx(1.3, abs=0.15)


@needs_ffmpeg
def test_letterboxing_of_mismatched_input(media):
    # a 160x90 source into a 120x120 canvas must be padded, not rejected
    g = build_video_graph([ClipSpec(0, 1.0), ClipSpec(1, 1.0)],
                          [Transition("fade", 0.3)], 120, 120, 15)
    _, streams = _render(g, [media["a"], media["b"]], media["dir"] / "box.mp4")
    assert (streams[0]["width"], streams[0]["height"]) == (120, 120)


@needs_ffmpeg
def test_audio_crossfade_matches_video_duration(media):
    cs = [ClipSpec(0, 1.0, has_audio=True), ClipSpec(1, 1.0, has_audio=True)]
    g = build_video_graph(cs, [Transition("fade", 0.4)], 160, 90, 15)
    dur, streams = _render(g, [media["a_snd"], media["b_snd"]],
                           media["dir"] / "snd.mp4")
    kinds = {s["codec_type"] for s in streams}
    assert kinds == {"video", "audio"}
    assert dur == pytest.approx(1.6, abs=0.15)


@needs_ffmpeg
def test_mixed_audio_and_silent_clip(media):
    cs = [ClipSpec(0, 1.0, has_audio=True), ClipSpec(1, 1.0)]
    g = build_video_graph(cs, [Transition("fade", 0.4)], 160, 90, 15)
    dur, streams = _render(g, [media["a_snd"], media["b"]],
                           media["dir"] / "mixed.mp4")
    assert any(s["codec_type"] == "audio" for s in streams)
    assert dur == pytest.approx(1.6, abs=0.15)


@needs_ffmpeg
def test_audio_only_graph_renders(media):
    cs = [ClipSpec(0, 1.0, has_audio=True), ClipSpec(1, 1.0, has_audio=True)]
    g = build_audio_graph(cs, [Transition("fade", 0.4)])
    dur, streams = _render(g, [media["a_snd"], media["b_snd"]],
                           media["dir"] / "only.m4a")
    assert dur == pytest.approx(1.6, abs=0.15)
    assert streams[0]["codec_type"] == "audio"
