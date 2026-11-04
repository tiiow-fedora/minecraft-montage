"""The real job handlers for the batch queue.

``jobqueue`` deliberately knows nothing about the pipeline; this module is the
glue that maps its job kinds onto the pipeline functions the CLI and GUI
already use:

    plan        write a camera path into one replay (pipeline.plan_replay)
    render_wait park until the manual Minecraft render has produced its file
    edit        join the rendered clips (pipeline.finish_video)
    upload      upload the result, PRIVATE only (pipeline.upload_private)
    autoplan    detect highlights and write one camera shot each (auto.py)

Rules every handler here follows:

* Heavy modules (ffmpeg wrappers, the Google client, track parsing) are imported
  inside the handler, so ``import mcmontage.queue_handlers`` and building the
  registry stay cheap and work without those dependencies installed.
* ``ctx.check_stop()`` is called between long steps.
* Results are plain JSON (Paths become strings) and contain no credentials.
* The upload handler never sees credentials as data. ``youtube.py`` loads (or
  interactively obtains) them from ``secrets/`` itself; nothing from that flow is
  put in params, results, logs or error text that this module controls.
* Uploads are private. There is no privacy parameter, same as ``upload_private``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

from .jobqueue import JobContext, NeedsManual, Registry

# Camera options a ``plan`` job may carry (see pipeline.plan_replay).
CAMERA_KEYS = frozenset({
    "radius", "height", "revolutions", "keyframes", "smooth", "follow_entity",
    "retrack", "centre", "follow",
})

# A render is "finished" once its file has been the same size for this long.
DEFAULT_STABLE_SECONDS = 5.0
DEFAULT_POLL_SECONDS = 1.0
# Longest a single handler call will sit watching a still-growing file before
# it parks the job instead (the render is evidently still running).
DEFAULT_MAX_WATCH_SECONDS = 120.0


def jsonable(obj: Any) -> Any:
    """Recursively convert to something ``json.dumps`` accepts."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


class Handlers:
    """Bundle of handlers bound to one config, with injectable time/stat."""

    def __init__(
        self,
        cfg: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        stat: Optional[Callable[[Path], Any]] = None,
        stable_seconds: float = DEFAULT_STABLE_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        max_watch_seconds: float = DEFAULT_MAX_WATCH_SECONDS,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self.sleep = sleep
        self.stat = stat or self._default_stat
        self.stable_seconds = stable_seconds
        self.poll_seconds = poll_seconds
        self.max_watch_seconds = max_watch_seconds

    @staticmethod
    def _default_stat(path: Path):
        try:
            return Path(path).stat()
        except OSError:
            return None

    # ---- plan ----------------------------------------------------------
    def plan(self, ctx: JobContext) -> dict:
        from . import pipeline
        from . import replay as replay_mod

        p = ctx.params
        clip = _require_str(p, "clip")
        seconds = float(p.get("seconds") or 20.0)
        camera = dict(p.get("camera") or {})
        unknown = sorted(set(camera) - CAMERA_KEYS)
        if unknown:
            raise ValueError("unknown camera option(s): " + ", ".join(unknown))

        self._require_minecraft_closed(pipeline)
        ctx.check_stop()

        replay_path = Path(clip)
        info = replay_mod.inspect(replay_path)
        start, end = pipeline.centred_span(
            info.duration_ms or 0, seconds, p.get("max_speedup") or None)
        ctx.check_stop()

        summary = pipeline.plan_replay(
            self.cfg, replay_path, seconds=seconds, start=start, end=end,
            log=ctx.log, **camera)
        out = jsonable(summary)
        out.setdefault("clip", str(replay_path))
        return out

    # ---- autoplan ------------------------------------------------------
    def autoplan(self, ctx: JobContext) -> dict:
        from . import auto as auto_mod
        from . import pipeline

        p = ctx.params
        replay = Path(_require_str(p, "replay"))
        total = float(p.get("total") or 60.0)
        each = float(p.get("seconds_each") or 8.0)
        max_count = p.get("max_count")

        self._require_minecraft_closed(pipeline)
        ctx.check_stop()

        found, tracks, _player = auto_mod.find_highlights(
            self.cfg, replay, retrack=bool(p.get("retrack", False)))
        ctx.check_stop()
        chosen = auto_mod.choose(found, total, each, max_count=max_count)
        if not chosen:
            raise auto_mod.AutoError("no highlights found in %s" % replay.name)
        jobs = auto_mod.plan_highlight_jobs(self.cfg, replay, chosen, tracks, each)
        ctx.check_stop()

        n = auto_mod.write_jobs(replay, jobs, append=bool(p.get("append", False)))
        ctx.log("Wrote %d render job(s) into %s" % (n, replay.name))
        return {
            "replay": str(replay),
            "found": len(found),
            "jobs": n,
            "highlights": [
                {"kind": h.kind, "start_ms": h.start_ms, "end_ms": h.end_ms,
                 "score": round(float(h.score), 4), "label": h.label}
                for h in chosen
            ],
            "outputs": [str(j["settings"]["outputFile"]) for j in jobs],
        }

    # ---- render_wait ---------------------------------------------------
    def expected_video(self, ctx: JobContext) -> Path:
        """Where the render will land: the plan's ``out_file`` if it said, else
        the same default ``plan_replay`` uses."""
        for res in ctx.dependency_results.values():
            if isinstance(res, dict) and res.get("out_file"):
                return Path(res["out_file"])
        clip = Path(_require_str(ctx.params, "clip"))
        return (Path(self.cfg.videos) / (clip.stem + ".mp4")).resolve()

    def render_ready(self, ctx: JobContext) -> bool:
        """Cheap check used by the runner to un-park the job: the file exists."""
        st = self.stat(self.expected_video(ctx))
        return st is not None and st.st_size > 0

    def render_wait(self, ctx: JobContext) -> dict:
        path = self.expected_video(ctx)
        st = self.stat(path)
        if st is None:
            raise NeedsManual("render %s in Minecraft (Render Queue); expecting %s"
                              % (Path(_require_str(ctx.params, "clip")).name, path))
        if st.st_size <= 0:
            raise NeedsManual("%s exists but is empty; render still starting" % path.name)

        sig = _signature(st)
        since = self.clock()
        began = since
        while True:
            ctx.check_stop()
            now = self.clock()
            if now - since >= self.stable_seconds:
                return {"video": str(path), "size": int(st.st_size)}
            if now - began >= self.max_watch_seconds:
                raise NeedsManual("%s is still being written (render in progress)"
                                  % path.name)
            self.sleep(max(0.0, min(self.poll_seconds, self.stable_seconds - (now - since))))
            st = self.stat(path)
            if st is None:
                raise NeedsManual("%s disappeared while waiting" % path.name)
            if _signature(st) != sig:
                sig, since = _signature(st), self.clock()

    # ---- edit ----------------------------------------------------------
    def edit(self, ctx: JobContext) -> dict:
        from . import pipeline

        p = ctx.params
        clips = []
        for dep_id in ctx.job.depends_on:
            res = ctx.dependency_results.get(dep_id)
            if isinstance(res, dict) and res.get("video"):
                clips.append(Path(res["video"]))
        if not clips:
            raise ValueError("edit job has no rendered video among its dependencies")
        missing = [c for c in clips if not c.is_file()]
        if missing:
            raise FileNotFoundError("rendered clip(s) missing: "
                                    + ", ".join(str(m) for m in missing))
        ctx.check_stop()

        output, tracks = pipeline.finish_video(
            self.cfg, clips,
            output=p.get("output") or None,
            title=p.get("title") or None,
            subtitle=p.get("subtitle") or None,
            music_ids=list(p.get("music_ids") or []) or None,
            xfade=p.get("xfade"),
            log=ctx.log)
        credits = []
        for t in tracks or []:
            try:
                if t.needs_attribution:
                    credits.append(t.credit())
            except AttributeError:
                pass
        return {"output": str(output), "clips": len(clips), "credits": credits}

    # ---- upload --------------------------------------------------------
    def upload(self, ctx: JobContext) -> dict:
        from . import pipeline
        from . import youtube as yt_mod

        p = ctx.params
        video = None
        for dep_id in ctx.job.depends_on:
            res = ctx.dependency_results.get(dep_id)
            if isinstance(res, dict) and res.get("output"):
                video = Path(res["output"])
        if video is None:
            raise ValueError("upload job has no edited video among its dependencies")
        if not video.is_file():
            raise FileNotFoundError("edited video missing: %s" % video)
        ctx.check_stop()

        title = (p.get("title") or "").strip() or video.stem
        last = {"pct": -10}

        def progress(frac: float) -> None:
            ctx.check_stop()
            pct = int(frac * 100)
            if pct >= last["pct"] + 10 or pct >= 100:
                last["pct"] = pct
                ctx.log("uploading %d%%" % pct)

        # Credentials are loaded inside youtube.upload via its own flow
        # (secrets/client_secret.json -> secrets/token.json); never via params.
        response = pipeline.upload_private(
            self.cfg, video, title,
            description=p.get("description") or "",
            tags=list(p.get("tags") or []),
            music_ids=list(p.get("music_ids") or []) or None,
            progress=progress, log=ctx.log)
        vid = (response or {}).get("id")
        return {
            "video_id": vid,
            "watch_url": yt_mod.watch_url(response),
            "studio_url": yt_mod.studio_url(response),
            "privacy": "private",
            "file": str(video),
        }

    # ---- shared --------------------------------------------------------
    def minecraft_closed_check(self, ctx: JobContext) -> bool:
        from . import pipeline

        return not pipeline.minecraft_running()

    @staticmethod
    def _require_minecraft_closed(pipeline) -> None:
        # Park rather than fail: this is a human action, not an error, and it
        # must not burn the job's retry attempts.
        if pipeline.minecraft_running():
            raise NeedsManual("Minecraft is running and holds replay files open; "
                              "quit it to the desktop, then resume")


def _signature(st: Any) -> tuple:
    return (int(st.st_size), getattr(st, "st_mtime_ns", None))


def _require_str(params: dict, name: str) -> str:
    v = params.get(name)
    if not isinstance(v, str) or not v:
        raise ValueError("job param %r is required" % name)
    return v


def build_registry(cfg: Any, **kwargs: Any) -> Registry:
    """Registry with every real handler bound to ``cfg``.

    Keyword arguments (``clock``, ``sleep``, ``stat``, ``stable_seconds``,
    ``poll_seconds``, ``max_watch_seconds``) go to :class:`Handlers`; they exist
    so tests never wait on real time.
    """
    h = Handlers(cfg, **kwargs)
    reg = Registry()
    reg.register("plan", h.plan, check=h.minecraft_closed_check)
    reg.register("autoplan", h.autoplan, check=h.minecraft_closed_check)
    reg.register("render_wait", h.render_wait, check=h.render_ready)
    reg.register("edit", h.edit)
    reg.register("upload", h.upload)
    return reg
