"""Glue for the automatic-montage flow.

    detect  ->  select  ->  plan camera  ->  (render in Minecraft)  ->  assemble

Each stage is a plain function over the pure modules (highlights, campath,
beatsync, transitions, titlecards) so the CLI, the GUI and the job queue share
one implementation. Nothing here is imported by the older pipeline, so the
manual workflow is unaffected.
"""

from __future__ import annotations

import dataclasses
from bisect import bisect_right
from pathlib import Path

from . import beatsync, campath, highlights as hl_mod, replay as replay_mod
from . import tracks as tracks_mod
from . import titlecards as tc_mod
from . import transitions as tr_mod
from .events import Highlight, load_events


class AutoError(RuntimeError):
    pass


def events_path(cfg, replay_path) -> Path:
    return Path(cfg.work) / (Path(replay_path).stem + ".events.json")


def load_tracks(cfg, replay_path, retrack=False):
    """Entity tracks for a replay, extracting (and caching) them on demand."""
    replay_path = Path(replay_path)
    cache = Path(cfg.work) / (replay_path.stem + ".tracks.json")
    if retrack or not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        tracks_mod.dump(replay_path, cache)
    return tracks_mod.load(cache)


def find_highlights(cfg, replay_path, *, retrack=False, detect_cfg=None, tracks=None):
    """Detect highlights; returns (highlights, tracks, player_id)."""
    tracks = tracks if tracks is not None else load_tracks(cfg, replay_path, retrack)
    if not tracks:
        raise AutoError("replay has no usable entity tracks")
    player = tracks_mod.identify_player(tracks)
    events = load_events(events_path(cfg, replay_path))
    found = hl_mod.detect_all(tracks, events, player_id=player.entity_id, cfg=detect_cfg)
    return found, tracks, player.entity_id


def choose(found, total_seconds, seconds_each, min_gap_s=2.0, max_count=None):
    """Pick the best non-overlapping highlights for a montage of `total_seconds`.

    The budget is in *replay* time because highlights are replay spans; each
    one is then compressed/slowed to `seconds_each` of video by the planner.
    """
    n = max(1, int(total_seconds // seconds_each))
    if max_count is not None:
        n = min(n, max_count)
    budget = sum(h.duration_ms for h in found) if found else 0
    return hl_mod.select(found, budget, min_gap_ms=int(min_gap_s * 1000), max_count=n)


def plan_highlight_jobs(cfg, replay_path, chosen, tracks, seconds_each, out_dir=None):
    """Render jobs (one per highlight) using the richer camera shots."""
    out_dir = Path(out_dir) if out_dir else Path(cfg.videos)
    return campath.plan_montage(
        chosen, tracks, seconds_each, out_dir=out_dir,
        stem=Path(replay_path).stem,
        settings=dict(width=cfg.width, height=cfg.height, fps=cfg.fps,
                      bitrate=cfg.bitrate),
    )


def write_jobs(replay_path, jobs, append=False):
    """Write jobs into the replay's render queue (caller checks Minecraft is closed)."""
    existing = []
    if append:
        existing = [{"timeline": j["timeline"], "settings": j["settings"]}
                    for j in replay_mod.read_render_queue(replay_path)]
    replay_mod.write_render_queue(replay_path, existing + list(jobs))
    return len(existing) + len(jobs)


def beat_slots(n_clips, beats, xfade, total_s=None, min_clip_s=2.0, max_clip_s=12.0,
               *, start_s=0.0, targets=None, limits=None, trailing_xfade=0.0):
    """Per-clip lengths (seconds of source) so the montage's cuts land on beats.

    Meaning of the result: with `slots[i]` as the length of clip i and an xfade
    of `xfade` between neighbours, the xfade into clip i+1 *starts* (xfade
    offset) on a beat, and the montage ends on a beat (unless `trailing_xfade`
    adds an overlap into an outro card, in which case that overlap starts on
    the beat). Concretely, with F_i the beat-aligned boundaries,
    ``slots[i] == F_i - F_(i-1) + (xfade if i < n-1 else trailing_xfade)``,
    so cumulative sums of ``slots[i] - xfade`` reproduce the beat times.

    start_s   -- timeline time at which clip 0 starts (e.g. after a title
                 card); beats are interpreted on the *final* video's clock.
    targets   -- desired boundary-to-boundary lengths, one per clip.  Default:
                 `total_s / n_clips` when total_s is given, else the midpoint
                 of min/max (clamped to [min_clip_s, max_clip_s]).  There is
                 always exactly one slot per clip: nothing is cycled/truncated.
    limits    -- longest each slot may be (the clip's real duration).  A
                 boundary that would overrun is pulled back to the latest
                 earlier beat, so clips are never asked for footage they lack
                 (which would shift every later cut off the beat).
    """
    if n_clips < 1:
        raise AutoError("no clips")
    if len(beats) < 2:
        raise AutoError("not enough beats to sync to")
    if targets is not None and len(targets) != n_clips:
        raise AutoError("targets needs one value per clip")
    if limits is not None and len(limits) != n_clips:
        raise AutoError("limits needs one value per clip")
    rel = [b - start_s for b in beats if b - start_s > 1e-9]
    if len(rel) < n_clips:
        raise AutoError(f"need {n_clips} beats after {start_s:.1f}s, music has {len(rel)}")
    if targets is None:
        per = total_s / n_clips if total_s else (min_clip_s + max_clip_s) / 2
        targets = [min(max(per, min_clip_s), max_clip_s)] * n_clips
    try:
        fitted = beatsync.fit_clips_to_beats(list(targets), rel)
    except beatsync.BeatError as e:
        raise AutoError(f"cannot sync clips to beats: {e}") from e
    xs = [xfade] * (n_clips - 1) + [trailing_xfade]
    bounds, acc = [], 0.0
    for d in fitted:
        acc += d
        bounds.append(rel[beatsync.nearest_beat_index(acc, rel)])
    out, cursor = [], 0.0
    for i, b in enumerate(bounds):
        if limits is not None:
            cap = cursor + limits[i] - xs[i]
            if b > cap + 1e-6:
                k = bisect_right(rel, cap + 1e-6) - 1
                b = rel[k] if k >= 0 and rel[k] > cursor + 1e-6 else max(cap, cursor + 0.05)
        b = max(b, cursor + 0.05)
        out.append(b - cursor + xs[i])
        cursor = b
    return out


#: card transitions are plain crossfades whatever the montage style is
_CARD_TRANSITION = "fade"

def _card_xfade(xfade, card_s):
    return max(tr_mod.MIN_DURATION, min(float(xfade), card_s / 2))


def _credit_lines(credits):
    lines = []
    for c in credits:
        lines += str(c).replace("\r\n", "\n").split("\n")
    lines = [ln.strip() for ln in lines]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _join(a, a_dur, b, trans, out):
    """xfade chain joining stream `a` (a_dur long) to stream `b`."""
    if trans.duration > a_dur + 1e-6:
        raise AutoError("transition longer than the segment it follows")
    off = round(a_dur - trans.duration, 3)
    return f"[{a}][{b}]{trans.filter_args(off)}[{out}]"


def assemble(cfg, clips, music, output, *, xfade=None, style="dynamic", seed=0,
             width=None, height=None, fps=None, sync=True, dry_run=False,
             title=None, subtitle=None, credits=None, lower_thirds=None,
             intro_s=3.0, outro_s=5.0, card_style=None, on_missing_font=None,
             credits_heading=None, work_dir=None):
    """Join clips with varied transitions, cutting to the music's beat.

    Returns the ffmpeg argument list (one ffmpeg invocation).  Music is mixed
    in as a single bed (attribution is the caller's job unless `credits`).

    title / subtitle -- generated title card, first segment, crossfaded into
                        the first clip.  `subtitle` needs `title`.
    credits          -- list of strings (blank = gap; embedded newlines split,
                        so ``music.attribution_block(...)`` works) shown on a
                        final card.  More than 12 lines scroll.
    lower_thirds     -- [(text, start_s, end_s)] overlaid on the finished
                        video; times are on the FINAL timeline (title card
                        included).  end_s is clipped to the video length; one
                        starting at/after the end raises AutoError.
    intro_s/outro_s  -- card lengths in seconds (only used when a card exists).
    card_style       -- titlecards.CardStyle for cards and lower thirds.
    on_missing_font  -- "error" (default: AutoError) or "blank" (cards without
                        text, lower thirds skipped).  Overrides card_style.
    credits_heading  -- optional bold first line on the credits card.
    work_dir         -- where drawtext text files go (default cfg.work/assemble).

    Beat sync applies to the montage clips only: the cards keep their length
    and the clip cuts are timed on the final video's clock, so they land on
    beats even after a title card.  The music bed spans cards too.  Clip audio
    is always dropped.
    """
    from . import edit as edit_mod

    clips = [Path(c) for c in clips]
    if not clips:
        raise AutoError("no clips")
    has_title = bool(title and title.strip())
    credit_lines = _credit_lines(credits) if credits else []
    has_credits = any(credit_lines)
    if subtitle and not has_title:
        raise AutoError("subtitle needs a title")
    lts = list(lower_thirds or [])
    for lt in lts:
        if len(lt) != 3 or not str(lt[0]).strip() or not (0 <= lt[1] < lt[2]):
            raise AutoError(f"bad lower third {lt!r}: want (text, start_s, end_s) "
                            "with 0 <= start < end")
    if (has_title and intro_s <= 0) or (has_credits and outro_s <= 0):
        raise AutoError("intro_s/outro_s must be > 0")

    probes = [edit_mod.probe(c) for c in clips]
    xfade = cfg.xfade if xfade is None else xfade
    # libx264/yuv420p needs even dimensions
    width = (width or probes[0].width) // 2 * 2
    height = (height or probes[0].height) // 2 * 2
    fps = fps or (round(probes[0].fps) or 60)

    cstyle = card_style or tc_mod.CardStyle()
    if on_missing_font:
        cstyle = dataclasses.replace(cstyle, on_missing_font=on_missing_font)
    needs_text = has_title or has_credits or bool(lts)
    if needs_text and not tc_mod.drawtext_available():
        if cstyle.on_missing_font != "blank":
            raise AutoError("this ffmpeg has no drawtext filter (needs libfreetype); "
                            "use on_missing_font='blank' to render without text")
        # force the text-less path in titlecards
        cstyle = dataclasses.replace(cstyle, font="/nonexistent/no-drawtext")
    work = Path(work_dir) if work_dir else Path(cfg.work) / "assemble" / Path(output).stem
    if needs_text:
        work.mkdir(parents=True, exist_ok=True)

    t_in = _card_xfade(xfade, intro_s) if has_title else 0.0
    t_out = _card_xfade(xfade, outro_s) if has_credits else 0.0
    # where the first montage frame lands on the final timeline
    start_s = (intro_s - t_in) if has_title else 0.0

    durs = [p.duration for p in probes]
    if sync and music:
        grid = beatsync.load_or_detect(music)
        if grid.beats and grid.confidence >= 0.3:
            try:
                want = beat_slots(len(clips), grid.beats, xfade, start_s=start_s,
                                  targets=[max(d, 0.1) for d in durs],
                                  limits=durs, trailing_xfade=t_out)
                durs = [min(w, p.duration) for w, p in zip(want, probes)]
            except AutoError:
                pass        # music too short / irregular: keep natural lengths

    try:
        specs = [tr_mod.ClipSpec(i, d) for i, d in enumerate(durs)]
        trans = (tr_mod.pick_transitions(len(specs), style=style, seed=seed,
                                         duration=xfade) if len(specs) > 1 else [])
        card_t = lambda d: tr_mod.Transition(_CARD_TRANSITION, d)  # noqa: E731
        # validate cards + clips together: a clip between two transitions must
        # host both, and that includes the title/credits crossfades
        seq_specs = ([tr_mod.ClipSpec(0, intro_s)] if has_title else []) + specs \
            + ([tr_mod.ClipSpec(0, outro_s)] if has_credits else [])
        seq_trans = ([card_t(t_in)] if has_title else []) + list(trans) \
            + ([card_t(t_out)] if has_credits else [])
        seq_trans = tr_mod.validate_transitions(seq_specs, seq_trans, clamp=True) \
            if len(seq_specs) > 1 else []
        k0 = 1 if has_title else 0
        trans = seq_trans[k0:k0 + len(trans)]
        graph = tr_mod.build_video_graph(specs, trans, width, height, fps,
                                         clamp=True,
                                         prefix="m" if needs_text else "")
        chains = list(graph.chains)
        cur, cur_dur = graph.label, graph.duration
        if has_title:
            card = tc_mod.build_card_graph(title, subtitle or None, intro_s, width,
                                           height, fps, cstyle, work, prefix="tc")
            t = seq_trans[0]
            chains = list(card.chains) + chains
            chains.append(_join(card.label, card.duration, cur, t, "jin"))
            cur, cur_dur = "jin", card.duration + cur_dur - t.duration
        if has_credits:
            card = tc_mod.credits_card(
                credit_lines, outro_s, width, height, fps, cstyle, work,
                mode="scroll" if len(credit_lines) > 12 else "stack",
                heading=credits_heading, prefix="cr")
            t = seq_trans[-1]
            chains += list(card.chains)
            chains.append(_join(cur, cur_dur, card.label, t, "jout"))
            cur, cur_dur = "jout", cur_dur + card.duration - t.duration
        for i, (text, a, b) in enumerate(lts):
            if a >= cur_dur:
                raise AutoError(f"lower third {text!r} starts at {a}s, after the "
                                f"video ends ({cur_dur:.1f}s)")
            lt = tc_mod.lower_third(str(text), a, min(b, cur_dur), width=width,
                                    height=height, work_dir=work, in_label=cur,
                                    out_label=f"lt{i}out", style=cstyle,
                                    prefix=f"lt{i}")
            chains += list(lt.chains)
            cur = lt.label
    except tc_mod.FontNotFoundError as e:
        raise AutoError(str(e)) from e
    except ValueError as e:
        raise AutoError(str(e)) from e

    graph = tr_mod.FilterGraph(tuple(chains), cur, cur_dur)
    args = tr_mod.render_graph_args(graph, list(clips), output, crf=cfg.crf,
                                    trim_to_duration=True)
    if music:
        # append the music input and a gain/fade chain, mapped as the audio
        args = _with_music(args, graph, music, len(clips), cfg)
    if dry_run:
        return args
    edit_mod._run(args, "ffmpeg assemble")
    return args


def _with_music(args, graph, music, n_inputs, cfg):
    """Insert a music input + level/fade filter into render_graph_args output.

    The bed covers the whole graph (cards included) and loops if the track is
    shorter.
    """
    i = args.index("-filter_complex")
    total = graph.duration
    fade_in = min(1.5, total / 2)
    fade_out = min(cfg.music_fade_out, total / 2)
    chain = (f"[{n_inputs}:a]atrim=0:{total:.3f},asetpts=N/SR/TB,"
             f"volume={cfg.music_gain_db}dB,afade=t=in:st=0:d={fade_in:.3f},"
             f"afade=t=out:st={max(0.0, total - fade_out):.3f}"
             f":d={fade_out:.3f}[aout]")
    fc = args[i + 1] + ";" + chain
    out = args[:i] + ["-stream_loop", "-1", "-i", str(music), "-filter_complex", fc] + args[i + 2:]
    # replace "-an" with the music map
    if "-an" in out:
        k = out.index("-an")
        out[k:k + 1] = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    return out
