"""Glue for the automatic-montage flow.

    detect  ->  select  ->  plan camera  ->  (render in Minecraft)  ->  assemble

Each stage is a plain function over the pure modules (highlights, campath,
beatsync, transitions, titlecards) so the CLI, the GUI and the job queue share
one implementation. Nothing here is imported by the older pipeline, so the
manual workflow is unaffected.
"""

from __future__ import annotations

from pathlib import Path

from . import beatsync, campath, highlights as hl_mod, replay as replay_mod
from . import tracks as tracks_mod
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


def beat_slots(n_clips, beats, xfade, total_s=None, min_clip_s=2.0, max_clip_s=12.0):
    """Source-clip lengths whose *transitions* fall on beats.

    Each clip but the last overlaps its successor by `xfade`, so the visible
    boundary sits `xfade` before the clip's nominal end. Slots are therefore
    fitted to the beat grid and every non-final clip is lengthened by `xfade`.
    """
    if n_clips < 1:
        raise AutoError("no clips")
    if len(beats) < 2:
        raise AutoError("not enough beats to sync to")
    if total_s is None:
        total_s = min(beats[-1], n_clips * (min_clip_s + max_clip_s) / 2)
    cuts = beatsync.plan_cuts(beats, total_s, min_clip_s, max_clip_s)
    slots = [c.duration for c in cuts]
    # Reuse slots cyclically / truncate so there is exactly one per clip.
    slots = [slots[i % len(slots)] for i in range(n_clips)]
    fitted = beatsync.fit_clips_to_beats(slots, beats)
    return [d + (xfade if i < n_clips - 1 else 0.0) for i, d in enumerate(fitted)]


def assemble(cfg, clips, music, output, *, xfade=None, style="dynamic", seed=0,
             width=None, height=None, fps=None, sync=True, dry_run=False):
    """Join clips with varied transitions, cutting to the music's beat.

    Returns the ffmpeg argument list. Music is mixed in as a single bed
    (attribution is the caller's job, as in pipeline.finish_video).
    """
    from . import edit as edit_mod

    clips = [Path(c) for c in clips]
    probes = [edit_mod.probe(c) for c in clips]
    xfade = cfg.xfade if xfade is None else xfade
    width = width or probes[0].width
    height = height or probes[0].height
    fps = fps or (round(probes[0].fps) or 60)

    durs = [p.duration for p in probes]
    if sync and music:
        grid = beatsync.load_or_detect(music)
        if grid.beats and grid.confidence >= 0.3:
            want = beat_slots(len(clips), grid.beats, xfade)
            durs = [min(w, p.duration) for w, p in zip(want, probes)]

    specs = [tr_mod.ClipSpec(i, d) for i, d in enumerate(durs)]
    trans = tr_mod.pick_transitions(len(specs), style=style, seed=seed,
                                    duration=xfade) if len(specs) > 1 else []
    graph = tr_mod.build_video_graph(specs, trans, width, height, fps, clamp=True)
    inputs = list(clips)
    args = tr_mod.render_graph_args(graph, inputs, output, crf=cfg.crf,
                                    trim_to_duration=True)
    if music:
        # append the music input and a gain/fade chain, mapped as the audio
        args = _with_music(args, graph, music, len(clips), cfg)
    if dry_run:
        return args
    edit_mod._run(args, "ffmpeg assemble")
    return args


def _with_music(args, graph, music, n_inputs, cfg):
    """Insert a music input + level/fade filter into render_graph_args output."""
    i = args.index("-filter_complex")
    chain = (f"[{n_inputs}:a]atrim=0:{graph.duration:.3f},asetpts=N/SR/TB,"
             f"volume={cfg.music_gain_db}dB,afade=t=in:st=0:d=1.5,"
             f"afade=t=out:st={max(0.0, graph.duration - cfg.music_fade_out):.3f}"
             f":d={cfg.music_fade_out}[aout]")
    fc = args[i + 1] + ";" + chain
    out = args[:i] + ["-stream_loop", "-1", "-i", str(music), "-filter_complex", fc] + args[i + 2:]
    # replace "-an" with the music map
    if "-an" in out:
        k = out.index("-an")
        out[k:k + 1] = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    return out
