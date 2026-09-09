"""Command line entry point.

    python -m mcmontage status              what is set up and what is not
    python -m mcmontage inspect [REPLAY]    dump a .mcpr, validate the schema
    python -m mcmontage plan REPLAY ...     write a camera path into the replay
    python -m mcmontage edit CLIP...        stitch clips, add music and titles
    python -m mcmontage upload VIDEO ...    upload (lands private)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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


def cmd_edit(args):
    cfg = config_mod.load(args.config)
    try:
        pipeline_mod.finish_video(
            cfg, args.clips,
            output=args.out or (cfg.output / "montage.mp4"),
            title=args.title, subtitle=args.subtitle, music_id=args.music,
            xfade=args.xfade,
        )
    except (pipeline_mod.PipelineError, music_mod.MusicError) as e:
        raise SystemExit(str(e))


def cmd_upload(args):
    from . import youtube as yt_mod

    track = music_mod.get(args.music) if args.music else None
    description = yt_mod.build_description(
        args.description or "", music_tracks=[track] if track else None
    )
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

    p = sub.add_parser("tracks", help="list entity tracks in a replay")
    p.add_argument("replay", nargs="?")
    p.add_argument("--retrack", action="store_true")
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(func=cmd_tracks)

    p = sub.add_parser("edit",
                       help="assemble the finished video (several clips are "
                            "joined in order with a crossfade)")
    p.add_argument("clips", nargs="+", help="clips in playback order")
    p.add_argument("--xfade", type=float,
                   help="crossfade seconds between clips (default from config)")
    p.add_argument("--out")
    p.add_argument("--title")
    p.add_argument("--subtitle")
    p.add_argument("--music", help="track id from music/tracks.toml")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("upload", help="upload to YouTube (lands private)")
    p.add_argument("video")
    p.add_argument("--title", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--tags", default="")
    p.add_argument("--music", help="track id, so its credit is appended")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_upload)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
