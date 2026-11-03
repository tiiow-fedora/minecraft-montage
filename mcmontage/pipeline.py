"""The two pipeline stages either side of the manual render, as plain functions.

The render itself happens inside Minecraft and cannot be automated (see README),
so the work splits in two:

    plan_replay()   before the render -- write a camera path into the .mcpr
    finish_video()  after the render  -- music, title, fades

Both the CLI and the GUI call these, so there is one implementation of each
rather than two that drift apart.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import edit as edit_mod
from . import music as music_mod
from . import replay as replay_mod
from . import timeline as tl_mod

# How far the smoothed camera target may drift from where the player really
# was, as a fraction of the orbit radius. Past this the subject starts
# sliding out of frame.
SMOOTH_DEVIATION_FRACTION = 0.35


class PipelineError(RuntimeError):
    pass


def minecraft_running() -> bool:
    """True if a Minecraft (javaw) process is up.

    Matters because ReplayMod holds recordings open: writing a render queue into
    a .mcpr while the game is running fails with WinError 5 on the final atomic
    replace. The original survives -- the write is temp-file-then-replace -- but
    the plan does not land.
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq javaw.exe", "/NH"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15,
        ).stdout or ""
    except (OSError, subprocess.SubprocessError):
        return False
    return "javaw.exe" in out.lower()


def plan_replay(
    cfg,
    replay_path,
    seconds=30.0,
    start=None,
    end=None,
    follow=True,
    follow_entity=None,
    smooth=None,
    retrack=False,
    centre=None,
    radius=12.0,
    height=6.0,
    revolutions=0.5,
    keyframes=90,
    out_file=None,
    append=False,
    log=print,
):
    """Write a camera path into a replay's renderQueue.json. Returns a summary."""
    replay_path = Path(replay_path)
    info = replay_mod.inspect(replay_path)

    replay_len = info.duration_ms or 0
    start = 0 if start is None else int(start)
    end = replay_len if end is None else int(end)
    if end <= start:
        raise PipelineError(
            f"replay span is empty (start={start}, end={end}); the replay "
            f"reports duration={replay_len}ms"
        )

    if start < 0:
        raise PipelineError(f"replay span starts before the recording (start={start}ms)")
    if replay_len and end > replay_len:
        raise PipelineError(
            f"replay span ends after the recording (end={end}ms, the replay "
            f"is {replay_len}ms long)"
        )
    duration_ms = int(seconds * 1000)
    if duration_ms <= 0:
        raise PipelineError(f"video length must be positive (got {seconds}s)")
    speedup = tl_mod.speedup_factor(start, end, duration_ms)
    tp = tl_mod.time_path(start, end, duration_ms)

    summary = {
        "replay": replay_path,
        "start": start,
        "end": end,
        "seconds": seconds,
        "speedup": speedup,
    }

    if follow or follow_entity is not None:
        from . import tracks as tracks_mod

        cache = Path(cfg.work) / (replay_path.stem + ".tracks.json")
        if not cache.is_file() or retrack:
            log("Extracting entity tracks (parsing the packet stream)...")
            cache.parent.mkdir(parents=True, exist_ok=True)
            tracks_mod.dump(replay_path, cache)
        all_tracks = tracks_mod.load(cache)

        if follow_entity is not None:
            if follow_entity not in all_tracks:
                raise PipelineError(f"entity {follow_entity} has no track")
            target = all_tracks[follow_entity]
            how = "specified"
        else:
            target = tracks_mod.identify_player(all_tracks)
            how = "auto-detected"

        # Pick the largest smoothing window that still keeps you in frame --
        # see tracks.choose_smoothing for why no fixed or speed-up-scaled
        # window works. The budget is a fraction of the orbit radius, since
        # that is what decides whether the subject drifts out of shot.
        budget = max(2.0, radius * SMOOTH_DEVIATION_FRACTION)
        if smooth is None:
            smooth_ms, dev = tracks_mod.choose_smoothing(
                target, start, end, keyframes, budget)
            log("Smoothing %d ms of replay time (worst drift %.1f blocks, "
                "budget %.1f)." % (smooth_ms, dev, budget))
        else:
            smooth_ms = int(smooth)
            sm = target.smoothed(window_ms=smooth_ms) if smooth_ms else target
            dev = tracks_mod.max_deviation(target, sm, start, end, keyframes)
            log("Smoothing %d ms of replay time (worst drift %.1f blocks)."
                % (smooth_ms, dev))
            if dev > radius * 2:
                log("WARNING: that drift exceeds twice the orbit radius -- you "
                    "will leave the shot. Try a smaller --smooth.")

        aimed_at = target.smoothed(window_ms=smooth_ms) if smooth_ms else target
        pp = tl_mod.follow_orbit_path(
            aimed_at,
            start, end, duration_ms,
            radius=radius, height=height,
            revolutions=revolutions, keyframes=keyframes,
        )
        summary.update(
            mode="follow", entity=target.entity_id, entity_how=how,
            samples=len(target.points),
            travelled=target.distance_travelled(), smooth_ms=smooth_ms,
        )
        log("Following entity %d (%s): %d samples, %.0f blocks travelled."
            % (target.entity_id, how, len(target.points),
               target.distance_travelled()))
    else:
        if not centre or len(centre) != 3 or any(c is None for c in centre):
            raise PipelineError("a static orbit needs an X/Y/Z centre")
        pp = tl_mod.orbit_path(
            tuple(float(c) for c in centre),
            radius=radius, height=height, duration_ms=duration_ms,
            revolutions=revolutions, keyframes=keyframes,
        )
        summary.update(mode="static", centre=tuple(centre))
        log("Static orbit around (%.1f, %.1f, %.1f)." % tuple(centre))

    out_file = out_file or str((Path(cfg.videos) / (replay_path.stem + ".mp4")).resolve())
    settings = tl_mod.render_settings(
        str(out_file), width=cfg.width, height=cfg.height, fps=cfg.fps,
        bitrate=cfg.bitrate,
    )
    job = tl_mod.render_job(tl_mod.build_timeline(tp, pp), settings)

    existing = (
        replay_mod.read_render_queue(replay_path) if info.has_render_queue else []
    )
    jobs = (
        [{"timeline": j["timeline"], "settings": j["settings"]} for j in existing]
        if append else []
    )
    jobs.append(job)

    if minecraft_running():
        raise PipelineError(
            "Minecraft is running, and it holds replay files open -- the write "
            "would fail. Quit Minecraft to the desktop, then try again."
        )

    replay_mod.write_render_queue(replay_path, jobs)
    summary["out_file"] = out_file
    summary["jobs"] = len(jobs)
    log("Wrote camera path into %s (%d job%s)."
        % (replay_path.name, len(jobs), "" if len(jobs) == 1 else "s"))
    return summary


def centred_span(duration_ms, seconds, max_speedup):
    """Pick a source span so the timelapse never exceeds `max_speedup`.

    A whole 73-minute replay squeezed into 18s is x244, far too frantic to
    follow anyone through. Capping the speed-up takes a centred slice instead --
    centred because the start of a recording is usually spawning in and the end
    is usually logging out, so the middle is the part worth watching.

    Returns (start_ms, end_ms), or (None, None) to mean the whole replay.
    """
    if not max_speedup or not duration_ms:
        return None, None
    wanted = int(seconds * max_speedup * 1000)
    if wanted >= duration_ms:
        return None, None
    middle = duration_ms // 2
    half = wanted // 2
    return max(0, middle - half), min(duration_ms, middle + half)


def batch_prepare(
    cfg,
    clips,
    seconds=20.0,
    max_speedup=None,
    rename_scheme=None,
    sessions_obj=None,
    log=print,
    **camera
):
    """Prepare a whole session's worth of replays in one pass.

    Renaming happens FIRST when asked for, never after: the output filename is
    baked into renderQueue.json at prepare time, so renaming afterwards leaves
    every replay pointing at its old name.

    Failures are collected rather than aborting the batch -- one unreadable
    replay should not cost you the other five. Returns (results, failures).
    """
    from . import sessions as sessions_mod

    if minecraft_running():
        raise PipelineError(
            "Minecraft is running, and it holds replay files open. Quit it to "
            "the desktop, then run this again."
        )

    paths = [Path(c) for c in clips]

    if rename_scheme:
        groups = sessions_obj or sessions_mod.group(paths)
        plan = sessions_mod.rename_plan(groups, rename_scheme)
        if plan:
            log("Renaming %d file%s:" % (len(plan), "" if len(plan) == 1 else "s"))
            for src, dst in sessions_mod.apply_renames(plan):
                log("  %-16s -> %s" % (src.name, dst.name))
            moved = {src: dst for src, dst in plan}
            paths = [moved.get(p, p) for p in paths]
        else:
            log("Names are already correct, nothing to rename.")

    results, failures = [], []
    for i, path in enumerate(paths, 1):
        log("")
        log("[%d/%d] %s" % (i, len(paths), path.name))
        try:
            info = replay_mod.inspect(path)
            start, end = centred_span(info.duration_ms or 0, seconds, max_speedup)
            if start is not None:
                log("  Capping at x%g: using %.1f min from the middle of %.1f."
                    % (max_speedup, (end - start) / 60000.0,
                       (info.duration_ms or 0) / 60000.0))
            summary = plan_replay(
                cfg, path, seconds=seconds, start=start, end=end,
                log=lambda m: log("  " + m), **camera
            )
            results.append(summary)
        except (PipelineError, replay_mod.ReplayError, OSError) as exc:
            log("  FAILED: %s" % exc)
            failures.append((path, str(exc)))

    log("")
    log("Prepared %d of %d." % (len(results), len(paths)))
    if failures:
        log("Failed: " + ", ".join(p.name for p, _ in failures))
    return results, failures


def resolve_tracks(music_ids):
    """Track objects for one id, several ids, or None. Validates each."""
    if not music_ids:
        return []
    if isinstance(music_ids, str):
        music_ids = [music_ids]
    library = music_mod.load_library()
    tracks = []
    for mid in music_ids:
        t = music_mod.get(mid, library)
        t.validate()
        tracks.append(t)
    return tracks


def build_upload_description(body, music_ids=None):
    """Final description text, with every required music credit appended."""
    from . import youtube as yt_mod

    return yt_mod.build_description(body or "",
                                    music_tracks=resolve_tracks(music_ids))


def upload_private(
    cfg,
    video,
    title,
    description="",
    tags=None,
    music_ids=None,
    progress=None,
    log=print,
):
    """Upload to YouTube as PRIVATE. Returns the created video resource.

    Privacy is hard-coded, not a parameter. An unverified API project has its
    uploads permanently locked to private with no appeal, so offering a "public"
    switch here would be a lie -- publishing happens in YouTube Studio.
    """
    from . import youtube as yt_mod

    video = Path(video)
    if not video.is_file():
        raise PipelineError(f"no such video: {video}")

    full_description = build_upload_description(description, music_ids)
    meta = yt_mod.VideoMeta(
        title=(title or "").strip(),
        description=full_description,
        tags=[t.strip() for t in (tags or []) if t.strip()],
        privacy="private",
    )
    try:
        meta.validate()
    except yt_mod.UploadError as e:
        raise PipelineError(str(e)) from e

    log("Uploading %s (%.1f MB) as PRIVATE..."
        % (video.name, video.stat().st_size / 1e6))
    log("If this is the first upload, a browser window will open for Google "
        "sign-in -- approve it there and this will continue.")

    try:
        response = yt_mod.upload(video, meta, progress=progress)
    except yt_mod.UploadError as e:
        raise PipelineError(str(e)) from e

    log("")
    log("Uploaded, and it is PRIVATE.")
    log("Watch : %s" % yt_mod.watch_url(response))
    log("Studio: %s" % yt_mod.studio_url(response))
    log("Nothing is public until you publish it yourself in Studio.")
    return response


def finish_video(
    cfg,
    clips,
    output=None,
    title=None,
    subtitle=None,
    music_ids=None,
    xfade=None,
    log=print,
):
    """Assemble rendered clips into the finished video. Returns (path, track).

    Multiple clips are joined in order with a crossfade, so each join costs
    `xfade` seconds of total runtime.
    """
    clips = [Path(c) for c in clips]
    if not clips:
        raise PipelineError("no clips selected")

    tracks = resolve_tracks(music_ids)
    if tracks:
        log("Music: %d track%s, sharing the runtime equally."
            % (len(tracks), "" if len(tracks) == 1 else "s"))
        for t in tracks:
            log("  %s - %s (%s)" % (t.title, t.artist, t.license))

    if output is None:
        output = Path(cfg.output) / (clips[0].stem + "_final.mp4")
    output = Path(output)

    log("Assembling %d clip%s with ffmpeg..."
        % (len(clips), "" if len(clips) == 1 else "s"))
    edit_mod.build(
        clips, output,
        music=[t.file for t in tracks] or None,
        title=title or None,
        subtitle=subtitle or None,
        xfade=cfg.xfade if xfade is None else xfade,
        music_gain_db=cfg.music_gain_db,
        music_fade_out=cfg.music_fade_out,
        crf=cfg.crf,
    )
    p = edit_mod.probe(output)
    log("Done: %s -- %.1fs, %dx%d @ %.0f fps, audio=%s"
        % (output, p.duration, p.width, p.height, p.fps, p.has_audio))
    needed = [t for t in tracks if t.needs_attribution]
    if needed:
        log("")
        log("Required credit%s (auto-added to any upload description):"
            % ("" if len(needed) == 1 else "s"))
        for t in needed:
            log("  " + t.credit())
    return output, tracks
