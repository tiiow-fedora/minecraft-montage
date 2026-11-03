"""Command line entry point.

    python -m mcmontage status              what is set up and what is not
    python -m mcmontage inspect [REPLAY]    dump a .mcpr, validate the schema
    python -m mcmontage plan REPLAY ...     write a camera path into the replay
    python -m mcmontage edit CLIP...        stitch clips, add music and titles
    python -m mcmontage highlights [REPLAY] detect highlights
    python -m mcmontage autoplan [REPLAY]   camera shots for the best highlights
    python -m mcmontage assemble CLIP...    transitions + beat-synced cuts
    python -m mcmontage queue ACTION [ID]   persistent batch job queue:
        status | retry | prune | reset      inspect / maintain
        enqueue [--session N] [--upload]    queue plan -> render -> edit [-> upload]
                                            for a play session (idempotent;
                                            --dry-run shows what would be queued)
        run [--stop-on-error]               execute the queue; Ctrl-C stops
                                            cleanly after the current step
        cancel ID | resume ID | requeue ID  steer one job (resume = "I did the
                                            manual render step")
    python -m mcmontage upload VIDEO ...    upload (lands private)
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

from . import config as config_mod
from . import edit as edit_mod
from . import music as music_mod
from . import replay as replay_mod
from . import pipeline as pipeline_mod
from . import timeline as tl_mod


def _pick_replay(cfg, arg):
    if arg:
        return Path(arg)
    found = replay_mod.find_replays(cfg.recordings)
    if not found:
        raise SystemExit(
            f"no .mcpr files in {cfg.recordings}\n"
            "Record one first. ReplayMod auto-starts recording when you join a "
            "server (settings autoStartRecording / recordServer), and the .mcpr "
            "is finalised when you disconnect -- so join, build, then quit to "
            "the menu."
        )
    return found[0]


def cmd_status(args):
    cfg = config_mod.load(args.config)
    print("profile     :", cfg.profile)
    print("  exists    :", cfg.profile.is_dir())
    mod = cfg.profile / "mods" / "reforgedplaymod-1.21.1-0.3.jar"
    print("  mod jar   :", "present" if mod.is_file() else "MISSING")
    print("recordings  :", cfg.recordings, "" if cfg.recordings.is_dir() else "(not created yet)")
    print("videos      :", cfg.videos, "" if cfg.videos.is_dir() else "(not created yet)")

    replays = replay_mod.find_replays(cfg.recordings)
    print("replays     :", len(replays))
    for r in replays[:5]:
        print("   ", r.name, "(%.1f MB)" % (r.stat().st_size / 1e6))

    print("ffmpeg      :", shutil.which("ffmpeg") or "NOT ON PATH")
    print("ffprobe     :", shutil.which("ffprobe") or "NOT ON PATH")

    try:
        lib = music_mod.load_library()
        print("music       : %d track(s)" % len(lib))
    except music_mod.MusicError as e:
        print("music       :", e)

    secret = Path("secrets/client_secret.json")
    token = Path("secrets/token.json")
    print("youtube     : client_secret=%s token=%s" % (
        "yes" if secret.is_file() else "no", "yes" if token.is_file() else "no"))
    print("schema      : timeline=%s render-queue=%s" % (
        "validated" if replay_mod.TIMELINE_SCHEMA_VALIDATED else "bytecode-only",
        "validated" if replay_mod.RENDER_QUEUE_SCHEMA_VALIDATED else "bytecode-only"))


def cmd_inspect(args):
    cfg = config_mod.load(args.config)
    path = _pick_replay(cfg, args.replay)
    info = replay_mod.inspect(path)

    print("replay   :", info.path)
    print("size     : %.1f MB" % (info.path.stat().st_size / 1e6))
    dur = info.duration_ms
    if dur:
        print("duration : %d ms (%.1f min)" % (dur, dur / 60000.0))
    if info.server_name:
        print("server   :", info.server_name)
    print("entries  :", ", ".join(sorted(info.entries)))
    print("renderQueue.json:", "present" if info.has_render_queue else "absent")
    print("timelines.json  :", "present" if info.has_timelines else "absent")

    if info.has_render_queue:
        jobs = replay_mod.read_render_queue(path)
        print("\n--- render queue: %d job(s) ---" % len(jobs))
        for i, job in enumerate(jobs):
            print("\n[job %d] settings:" % i)
            print(json.dumps(job["settings"], indent=2))
            paths = job["timeline"].get("", [])
            print("[job %d] timeline: %d path(s)" % (i, len(paths)))
            for pi, p in enumerate(paths):
                label = {0: "TIME", 1: "POSITION"}.get(pi, "path%d" % pi)
                kfs = p.get("keyframes", [])
                print("  %-8s keyframes=%d segments=%s interpolators=%s"
                      % (label, len(kfs), p.get("segments"),
                         json.dumps(p.get("interpolators"))))
                for kf in kfs[:3]:
                    print("      ", json.dumps(kf))
                if len(kfs) > 3:
                    print("       ... %d more" % (len(kfs) - 3))
        if args.dump:
            Path(args.dump).write_text(
                json.dumps([{k: v for k, v in j.items() if k != "_timeline_raw"}
                            for j in jobs], indent=2), encoding="utf-8")
            print("\nwrote", args.dump)


def cmd_plan(args):
    cfg = config_mod.load(args.config)
    path = _pick_replay(cfg, args.replay)
    centre = (args.x, args.y, args.z)
    follow = args.follow or args.follow_entity is not None
    if not follow and None in centre:
        raise SystemExit(
            "static orbit needs -x -y -z, or pass --follow to track the player"
        )
    try:
        summary = pipeline_mod.plan_replay(
            cfg, path,
            seconds=args.seconds, start=args.start, end=args.end,
            follow=follow, follow_entity=args.follow_entity,
            smooth=args.smooth, retrack=args.retrack,
            centre=None if follow else centre,
            radius=args.radius, height=args.height,
            revolutions=args.revolutions, keyframes=args.keyframes,
            out_file=args.out, append=args.append,
        )
    except pipeline_mod.PipelineError as e:
        raise SystemExit(str(e))

    print("speed-up    : x%.1f" % summary["speedup"])
    print("output      :", summary["out_file"])
    print("\nNow, in game: Replay Viewer -> open this replay -> Render Queue -> Render.")


def cmd_tracks(args):
    from . import tracks as tracks_mod

    cfg = config_mod.load(args.config)
    path = _pick_replay(cfg, args.replay)
    cache = cfg.work / (path.stem + ".tracks.json")
    if not cache.is_file() or args.retrack:
        print("extracting entity tracks...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        tracks_mod.dump(path, cache)
    all_tracks = tracks_mod.load(cache)
    player = tracks_mod.identify_player(all_tracks)

    print("replay  :", path.name)
    print("entities: %d with a usable track" % len(all_tracks))
    print()
    print("%-8s %8s %9s %9s %10s" % ("entity", "samples", "from_ms", "to_ms", "blocks"))
    for eid, n, t0, t1, dist in tracks_mod.rank(all_tracks, args.limit):
        mark = "  <- player?" if eid == player.entity_id else ""
        print("%-8d %8d %9d %9d %10.1f%s" % (eid, n, t0, t1, dist, mark))
    print()
    print("Pass --follow to orbit the marked entity, or --follow-entity ID to override.")


def _fmt_ms(ms):
    s = int(ms // 1000)
    return "%d:%02d" % (s // 60, s % 60)


def cmd_highlights(args):
    from . import auto as auto_mod
    from . import highlights as hl_mod
    from . import tracks as tracks_mod

    cfg = config_mod.load(args.config)
    path = _pick_replay(cfg, args.replay)
    try:
        found, _tracks, player = auto_mod.find_highlights(
            cfg, path, retrack=args.retrack)
    except (auto_mod.AutoError, tracks_mod.TrackError) as e:
        raise SystemExit(str(e))
    print("replay: %s   player entity: %d   %d highlight(s)"
          % (path.name, player, len(found)))
    for i, h in enumerate(found, 1):
        print("%3d  %-6s %s-%s  score %.2f  %s"
              % (i, h.kind, _fmt_ms(h.start_ms), _fmt_ms(h.end_ms),
                 h.score, h.label))
    if args.out:
        hl_mod.save_highlights(args.out, found)
        print("saved", args.out)


def cmd_autoplan(args):
    from . import auto as auto_mod
    from . import campath as campath_mod

    cfg = config_mod.load(args.config)
    path = _pick_replay(cfg, args.replay)
    try:
        found, tracks, _player = auto_mod.find_highlights(cfg, path)
        chosen = auto_mod.choose(found, args.total, args.seconds_each,
                                 max_count=args.max_count)
        if not chosen:
            raise SystemExit("no highlights found in %s" % path.name)
        jobs = auto_mod.plan_highlight_jobs(
            cfg, path, chosen, tracks, args.seconds_each)
    except (auto_mod.AutoError, campath_mod.CamPathError) as e:
        raise SystemExit(str(e))
    for h, j in zip(chosen, jobs):
        print("  %-6s %s-%s  -> %s" % (h.kind, _fmt_ms(h.start_ms),
              _fmt_ms(h.end_ms), j["settings"]["outputFile"]))
    if args.dry_run:
        print("--- dry run, nothing written ---")
        return
    if pipeline_mod.minecraft_running():
        raise SystemExit("Minecraft is running; quit it first.")
    n = auto_mod.write_jobs(path, jobs, append=args.append)
    print("Wrote %d job(s). Open the replay in game and use Render Queue." % n)


def cmd_assemble(args):
    from . import auto as auto_mod

    cfg = config_mod.load(args.config)
    music = None
    if args.music:
        try:
            music = music_mod.get(args.music).file
        except music_mod.MusicError as e:
            raise SystemExit(str(e))
    out = args.out or (cfg.output / "montage.mp4")
    try:
        argv = auto_mod.assemble(
            cfg, args.clips, music, out, style=args.style, seed=args.seed,
            sync=not args.no_sync, dry_run=args.dry_run)
    except (auto_mod.AutoError, edit_mod.EditError) as e:
        raise SystemExit(str(e))
    if args.dry_run:
        print(edit_mod.describe_command(argv))


def _queue_progress(event, job):
    info = ""
    if event in ("retry", "failed") and job.error:
        info = "  " + job.error.splitlines()[0][:160]
    elif event == "waiting":
        info = "  " + (job.message or "manual step")
    elif event == "skipped":
        info = "  " + (job.error or "")
    subject = ""
    clip = job.params.get("clip") or job.params.get("replay")
    if clip:
        subject = "  " + Path(str(clip)).name
    print("[%-7s] %s %-11s%s%s" % (event, job.id, job.kind, subject, info))
    sys.stdout.flush()


def _print_report(report):
    print()
    print("done %d  failed %d  skipped %d  waiting %d  retried %d%s" % (
        len(report.done), len(report.failed), len(report.skipped),
        len(report.waiting), len(report.retried),
        "  (stopped early)" if report.stopped else ""))
    for label, ids in (("failed", report.failed), ("skipped", report.skipped),
                       ("waiting", report.waiting)):
        if ids:
            print("  %-8s %s" % (label, ", ".join(ids)))
    if report.waiting:
        print("Waiting jobs need a manual step (e.g. render in Minecraft). "
              "Do it, then run `queue run` again, or `queue resume ID`.")


def _queue_run(cfg, q, args):
    from . import jobqueue as jq
    from . import queue_handlers

    registry = queue_handlers.build_registry(cfg)
    stop = {"asked": False}

    def should_stop():
        return stop["asked"]

    def sleep(seconds):
        end = time.monotonic() + seconds
        while not stop["asked"] and time.monotonic() < end:
            time.sleep(min(0.5, max(0.0, end - time.monotonic())))

    def on_sigint(signum, frame):
        if stop["asked"]:
            raise KeyboardInterrupt
        stop["asked"] = True
        print("\nStopping after the current step (Ctrl-C again to abort now)...")

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        report = q.run(
            registry, stop_on_error=args.stop_on_error,
            progress=_queue_progress, should_stop=should_stop, sleep=sleep,
            log=lambda m: print("    " + str(m)))
    except KeyboardInterrupt:
        print("\ninterrupted; the running job was returned to pending")
        raise SystemExit(130)
    except (jq.LockHeldError, jq.UnknownKindError) as e:
        raise SystemExit(str(e))
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    _print_report(report)
    if report.failed:
        raise SystemExit(1)
    return report


def _queue_enqueue(cfg, q, args):
    from . import jobqueue as jq

    groups = _sessions(cfg, args)
    chosen = [s for s in groups if args.session in (None, s.index)]
    if not chosen:
        raise SystemExit("no session %s; run `mcmontage sessions` to list them"
                         % args.session)
    if args.upload and not (args.title or "").strip():
        raise SystemExit("--upload needs --title")
    if args.out and len(chosen) > 1:
        raise SystemExit("--out names one file; pick a single session with --session")
    music = args.music or []
    if music:
        try:
            pipeline_mod.resolve_tracks(music)
        except music_mod.MusicError as e:
            raise SystemExit(str(e))
    max_speedup = args.max_speedup or None
    camera = {"radius": args.radius, "height": args.height,
              "revolutions": args.revolutions, "keyframes": args.keyframes}
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    for s in chosen:
        clips = [c.path for c in s.clips]
        title = args.title
        print("session %d (%s): %d clip(s)" % (s.index, s.day, len(clips)))
        if args.dry_run:
            for c in s.clips:
                start, end = pipeline_mod.centred_span(
                    c.duration_ms, args.seconds, max_speedup)
                span = (c.duration_ms if start is None else end - start) / 60000.0
                print("  would queue plan + render_wait  %-18s %.1f min -> %.0fs"
                      % (c.path.name, span, args.seconds))
            print("  would queue edit   title=%r music=%s output=%s"
                  % (title, ",".join(music) or "-", args.out or "(default)"))
            if args.upload:
                print("  would queue upload (private) title=%r" % title)
            continue
        before = {j.id for j in q.jobs()}
        try:
            batch = jq.enqueue_session_batch(
                q, clips, args.seconds, max_speedup=max_speedup, camera=camera,
                output=args.out, title=title, subtitle=args.subtitle,
                music_ids=music, xfade=args.xfade, upload=args.upload,
                description=args.description or "", tags=tags)
        except jq.QueueError as e:
            raise SystemExit(str(e))
        jobs = batch.plans + batch.renders + [batch.edit] + (
            [batch.upload] if batch.upload else [])
        new = [j for j in jobs if j.id not in before]
        print("  queued %d new job(s), %d already queued: %s"
              % (len(new), len(jobs) - len(new), ", ".join(j.id for j in jobs)))
    if args.dry_run:
        print("--- dry run, nothing queued ---")
    else:
        print("Next: `queue run` (it will pause at each render for you to do "
              "the Minecraft step).")


def cmd_queue(args):
    from . import jobqueue as jq

    cfg = config_mod.load(args.config)
    q = jq.JobQueue(jq.default_queue_path(cfg))
    action = args.action
    if action == "status":
        print(q.format_status())
    elif action == "retry":
        print("revived %d job(s)" % len(q.retry_failed()))
    elif action == "prune":
        print("pruned %d job(s)" % q.prune_done())
    elif action == "reset":
        q.reset()
        print("queue cleared")
    elif action == "run":
        _queue_run(cfg, q, args)
    elif action == "enqueue":
        _queue_enqueue(cfg, q, args)
    elif action in ("cancel", "resume", "requeue"):
        if not args.job_id:
            raise SystemExit("queue %s needs a job id (see `queue status`)" % action)
        try:
            job = getattr(q, action)(args.job_id)
        except jq.QueueError as e:
            raise SystemExit(str(e))
        print("%s: %s is now %s" % (action, job.id, job.status))


def _sessions(cfg, args):
    from . import sessions as sessions_mod

    paths = replay_mod.find_replays(cfg.recordings)
    if not paths:
        raise SystemExit(f"no .mcpr files in {cfg.recordings}")
    return sessions_mod.group(
        paths, cutoff_hour=args.cutoff, max_gap_minutes=args.max_gap)


def cmd_sessions(args):
    cfg = config_mod.load(args.config)
    groups = _sessions(cfg, args)
    print("A play-day runs %02d:00 to %02d:00, so playing past midnight still "
          "counts as the day you started.\n" % (args.cutoff, args.cutoff))
    for s in groups:
        print("session %d  -  play-day %s  -  %d clip(s), %.0f min of footage"
              % (s.index, s.day, len(s.clips), s.total_minutes))
        for c in s.clips:
            print("    %-18s %s  %5.1f min  -> %s"
                  % (c.path.name, c.recorded.strftime("%H:%M"), c.minutes,
                     s.name_for(c, args.scheme) + c.path.suffix))
        print()


def cmd_batch(args):
    cfg = config_mod.load(args.config)
    groups = _sessions(cfg, args)
    chosen = [s for s in groups if args.session in (None, s.index)]
    if not chosen:
        raise SystemExit("no session %s; run `mcmontage sessions` to list them"
                         % args.session)

    clips = [c.path for s in chosen for c in s.clips]
    print("Preparing %d clip(s) from session(s) %s at %.0fs each."
          % (len(clips), ", ".join(str(s.index) for s in chosen), args.seconds))
    if args.dry_run:
        from . import sessions as sessions_mod
        print("\n--- dry run, nothing written ---")
        if args.rename:
            for src, dst in sessions_mod.rename_plan(chosen, args.scheme):
                print("  rename %-18s -> %s" % (src.name, dst.name))
        for s in chosen:
            for c in s.clips:
                start, end = pipeline_mod.centred_span(
                    c.duration_ms, args.seconds, args.max_speedup)
                span = (c.duration_ms if start is None else end - start) / 60000.0
                print("  prepare %-18s %.1f min -> %.0fs  (x%.0f)"
                      % (c.path.name, span, args.seconds,
                         span * 60 / args.seconds))
        return

    try:
        pipeline_mod.batch_prepare(
            cfg, clips,
            seconds=args.seconds,
            max_speedup=args.max_speedup,
            rename_scheme=args.scheme if args.rename else None,
            sessions_obj=chosen,
            follow=True,
            radius=args.radius, height=args.height,
            revolutions=args.revolutions, keyframes=args.keyframes,
        )
    except pipeline_mod.PipelineError as e:
        raise SystemExit(str(e))
    print("\nNow render each one in game, then stitch them in tab 2.")


def cmd_edit(args):
    cfg = config_mod.load(args.config)
    try:
        pipeline_mod.finish_video(
            cfg, args.clips,
            output=args.out or (cfg.output / "montage.mp4"),
            title=args.title, subtitle=args.subtitle,
            music_ids=args.music, xfade=args.xfade,
        )
    except (pipeline_mod.PipelineError, music_mod.MusicError) as e:
        raise SystemExit(str(e))


def cmd_upload(args):
    from . import youtube as yt_mod

    description = pipeline_mod.build_upload_description(
        args.description or "", args.music)
    meta = yt_mod.VideoMeta(
        title=args.title,
        description=description,
        tags=args.tags.split(",") if args.tags else [],
        privacy="private",
    )
    meta.validate()

    print("title      :", meta.title)
    print("privacy    :", meta.privacy, "(unverified API projects cannot publish public)")
    print("description:\n" + "\n".join("  " + l for l in description.splitlines()))
    if not args.yes:
        reply = input("\nUpload this to your YouTube account? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            raise SystemExit("aborted")

    def progress(f):
        sys.stdout.write("\r  uploading %5.1f%%" % (f * 100))
        sys.stdout.flush()

    resp = yt_mod.upload(args.video, meta, progress=progress)
    print("\n\nuploaded (PRIVATE)")
    print("watch :", yt_mod.watch_url(resp))
    print("studio:", yt_mod.studio_url(resp), "  <- review and publish here")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mcmontage")
    ap.add_argument("--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show what is set up").set_defaults(func=cmd_status)

    p = sub.add_parser("inspect", help="dump a .mcpr and its render queue")
    p.add_argument("replay", nargs="?")
    p.add_argument("--dump", help="write the decoded render queue to this file")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("plan", help="write a camera path into a replay")
    p.add_argument("replay", nargs="?")
    p.add_argument("--seconds", type=float, default=30.0, help="output video length")
    p.add_argument("--start", type=int, help="replay start offset in ms")
    p.add_argument("--end", type=int, help="replay end offset in ms")
    p.add_argument("--follow", action="store_true",
                   help="orbit the player as they move (auto-detects the entity)")
    p.add_argument("--follow-entity", type=int,
                   help="follow this entity id instead of auto-detecting")
    p.add_argument("--smooth", type=int, default=None,
                   help="ms window (replay time) for smoothing the followed "
                        "path; by default the largest window whose drift stays "
                        "within %d%% of the orbit radius is chosen by measurement"
                        % int(pipeline_mod.SMOOTH_DEVIATION_FRACTION * 100))
    p.add_argument("--retrack", action="store_true",
                   help="re-extract entity tracks even if cached")
    p.add_argument("-x", type=float, help="static orbit centre X")
    p.add_argument("-y", type=float, help="static orbit centre Y")
    p.add_argument("-z", type=float, help="static orbit centre Z")
    p.add_argument("--radius", type=float, default=30.0)
    p.add_argument("--height", type=float, default=18.0, help="above centre Y")
    p.add_argument("--revolutions", type=float, default=1.0)
    p.add_argument("--keyframes", type=int, default=24)
    p.add_argument("--out", help="rendered video path the mod should write")
    p.add_argument("--append", action="store_true", help="keep existing queue jobs")
    p.set_defaults(func=cmd_plan)

    def _session_args(sp):
        sp.add_argument("--cutoff", type=int, default=4,
                        help="hour a play-day starts (default 4, so playing "
                             "past midnight counts as the previous day)")
        sp.add_argument("--max-gap", type=float, default=None,
                        help="also split a day into separate sessions when "
                             "there is a gap of this many minutes between clips")
        sp.add_argument("--scheme", default="day{session}_{n}",
                        help="naming scheme; {session} {n} {date}")

    p = sub.add_parser("sessions", help="group replays into play sessions")
    _session_args(p)
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("batch",
                       help="rename and prepare a whole session in one go")
    _session_args(p)
    p.add_argument("--session", type=int,
                   help="session number (default: all of them)")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="video length per clip (default 20)")
    p.add_argument("--max-speedup", type=float, default=30.0,
                   help="cap the timelapse rate by taking a centred slice of "
                        "long replays (default 30; 0 to use whole replays)")
    p.add_argument("--rename", action="store_true",
                   help="rename to the scheme first (names are baked into the "
                        "render queue, so this must happen before preparing)")
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--height", type=float, default=6.0)
    p.add_argument("--revolutions", type=float, default=0.5)
    p.add_argument("--keyframes", type=int, default=90)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("tracks", help="list entity tracks in a replay")
    p.add_argument("replay", nargs="?")
    p.add_argument("--retrack", action="store_true")
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(func=cmd_tracks)

    p = sub.add_parser("highlights", help="detect deaths, kills, speed, builds")
    p.add_argument("replay", nargs="?")
    p.add_argument("--retrack", action="store_true")
    p.add_argument("--out", help="save the highlights as JSON")
    p.set_defaults(func=cmd_highlights)

    p = sub.add_parser("autoplan",
                       help="detect highlights and write one camera shot each")
    p.add_argument("replay", nargs="?")
    p.add_argument("--total", type=float, default=60.0, help="montage seconds")
    p.add_argument("--seconds-each", type=float, default=8.0)
    p.add_argument("--max-count", type=int)
    p.add_argument("--append", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_autoplan)

    p = sub.add_parser("assemble",
                       help="join clips with varied transitions, cut to the beat")
    p.add_argument("clips", nargs="+")
    p.add_argument("--music", help="track id from music/tracks.toml")
    p.add_argument("--style", default="dynamic", choices=("subtle", "dynamic", "fade"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-sync", action="store_true", help="skip beat sync")
    p.add_argument("--out")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("queue", help="persistent batch job queue")
    p.add_argument("action", choices=("status", "retry", "prune", "reset", "run",
                                      "enqueue", "cancel", "resume", "requeue"))
    p.add_argument("job_id", nargs="?", help="job id for cancel/resume/requeue")
    p.add_argument("--stop-on-error", action="store_true",
                   help="run: stop at the first job that exhausts its retries")
    _session_args(p)
    p.add_argument("--session", type=int,
                   help="enqueue: session number (default: all, one chain each)")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="enqueue: video length per clip (default 20)")
    p.add_argument("--max-speedup", type=float, default=30.0,
                   help="enqueue: cap the timelapse rate (default 30; 0 = whole replays)")
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--height", type=float, default=6.0)
    p.add_argument("--revolutions", type=float, default=0.5)
    p.add_argument("--keyframes", type=int, default=90)
    p.add_argument("--upload", action="store_true",
                   help="enqueue: also queue a PRIVATE YouTube upload (needs --title)")
    p.add_argument("--title")
    p.add_argument("--subtitle")
    p.add_argument("--description", default="")
    p.add_argument("--tags", default="", help="comma-separated upload tags")
    p.add_argument("--xfade", type=float)
    p.add_argument("--out", help="enqueue: finished video path")
    p.add_argument("--music", action="append", default=None,
                   help="enqueue: track id; repeat for several")
    p.add_argument("--dry-run", action="store_true",
                   help="enqueue: print what would be queued, change nothing")
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("edit",
                       help="assemble the finished video (several clips are "
                            "joined in order with a crossfade)")
    p.add_argument("clips", nargs="+", help="clips in playback order")
    p.add_argument("--xfade", type=float,
                   help="crossfade seconds between clips (default from config)")
    p.add_argument("--out")
    p.add_argument("--title")
    p.add_argument("--subtitle")
    p.add_argument("--music", action="append", default=None,
                   help="track id from music/tracks.toml; repeat for several, "
                        "which then share the runtime equally")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("upload", help="upload to YouTube (lands private)")
    p.add_argument("video")
    p.add_argument("--title", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--tags", default="")
    p.add_argument("--music", action="append", default=None,
                   help="track id whose credit to append; repeat for several")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_upload)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
