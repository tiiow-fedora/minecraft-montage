"""CLI subcommands for thumbnails, generated metadata and private publishing.

Kept out of cli.py so that file stays small; cli.main() calls `register(sub)`.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

from . import metadata as md
from . import thumbnail as th
from . import upload_meta as um


def _context(args) -> md.Context:
    from . import edit as edit_mod

    clips = []
    for i, c in enumerate(args.clips or [], start=1):
        p = Path(c)
        clips.append(md.ClipInfo(md.label_from_filename(p.stem, i),
                                 edit_mod.probe(p).duration, str(p)))
    return md.Context(
        world=args.world or "", theme=args.theme or "", series=args.series or "",
        session_index=args.day_index,
        day=datetime.date.fromisoformat(args.date) if args.date else None,
        clips=clips, xfade=args.xfade,
    )


def _meta_args(p):
    p.add_argument("--world", default="")
    p.add_argument("--theme", default="", help="what the video is about")
    p.add_argument("--series", default="")
    p.add_argument("--day-index", type=int, help="'Day N' in the title")
    p.add_argument("--date", help="recording date, YYYY-MM-DD")
    p.add_argument("--xfade", type=float, default=1.0,
                   help="crossfade used when the clips were joined")
    p.add_argument("--clips", nargs="*", help="source clips, in order (for chapters)")


def cmd_thumbnail(args):
    try:
        if args.candidates:
            best = th.best_frames(args.video, k=args.candidates)
            for i, c in enumerate(best, start=1):
                out = Path(args.out).with_name(
                    f"{Path(args.out).stem}_{i}{Path(args.out).suffix}")
                img = th.render(c.image, args.title, args.subtitle, args.template,
                                badge=args.badge)
                print("candidate %d: t=%.1fs score=%.2f -> %s"
                      % (i, c.time, c.score.score, th.save(img, out)))
        else:
            print(th.make_thumbnail(args.video, args.out, args.title,
                                    args.subtitle, args.template, args.badge))
    except th.ThumbnailError as e:
        raise SystemExit(str(e))


def cmd_meta(args):
    ctx = _context(args)
    print("title      :", md.generate_title(ctx))
    print("tags       :", ", ".join(md.generate_tags(ctx)))
    print("description:\n" + md.generate_description(ctx))


def cmd_publish(args):
    ctx = _context(args)
    try:
        plan = um.from_generated(args.video, ctx, thumbnail=args.thumbnail,
                                 title=args.title)
    except um.PlanError as e:
        raise SystemExit(str(e))
    print(um.describe(plan))
    if not args.yes:
        print("\n(dry run -- pass --yes to upload; it will be PRIVATE)")
        return
    result = um.submit(plan, um.YouTubeUploader(), progress=lambda f: (
        sys.stdout.write("\r  uploading %5.1f%%" % (f * 100)), sys.stdout.flush()))
    print("\nuploaded (PRIVATE):", result.watch_url)
    print("studio:", result.studio_url)
    for w in result.warnings:
        print("warning:", w)


def register(sub):
    p = sub.add_parser("thumbnail", help="pick the best frame and make a thumbnail")
    p.add_argument("video")
    p.add_argument("--out", default="out/thumbnail.png")
    p.add_argument("--title", default="")
    p.add_argument("--subtitle", default="")
    p.add_argument("--badge")
    p.add_argument("--template", default="bold", choices=sorted(th.TEMPLATES))
    p.add_argument("--candidates", type=int, default=0,
                   help="write the N best frames as separate options")
    p.set_defaults(func=cmd_thumbnail)

    p = sub.add_parser("meta", help="generate title, description, chapters")
    _meta_args(p)
    p.set_defaults(func=cmd_meta)

    p = sub.add_parser("publish",
                       help="upload with generated metadata (private; dry run "
                            "unless --yes)")
    p.add_argument("video")
    p.add_argument("--title")
    p.add_argument("--thumbnail")
    p.add_argument("--yes", action="store_true")
    _meta_args(p)
    p.set_defaults(func=cmd_publish)
