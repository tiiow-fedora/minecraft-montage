"""Upload-metadata layer: private by default, and hard to make otherwise.

Sits between the generators (metadata.py, thumbnail.py) and the YouTube client
(youtube.py). An `UploadPlan` is everything that would be sent; `submit()` sends
it through an `Uploader`. Real network access lives behind the Uploader
protocol so the whole layer is testable with `FakeUploader` -- there are no
credentials in CI or in cloud sessions.

Rules enforced here (each has a test):
  * privacy is "private" unless the plan carries an explicit
    `allow_non_private=True`. Nothing in this package sets that flag.
  * `submit()` re-checks the final request body, so a plan mutated after
    validation still cannot go out public.
  * title, description and tags are scanned for things that look like
    credentials before anything leaves the machine.
  * the video and thumbnail may not come from a `secrets/` directory.
  * a thumbnail failure never loses the upload: the video id is still returned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol

from . import youtube as yt

PRIVATE = "private"
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
THUMBNAIL_EXTS = (".png", ".jpg", ".jpeg")


class PlanError(RuntimeError):
    pass


# Conservative: only formats that are unambiguous credentials.
_SECRET_PATTERNS = (
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Google OAuth access token", re.compile(r"ya29\.[0-9A-Za-z_\-]{20,}")),
    ("Google OAuth refresh token", re.compile(r"1//[0-9A-Za-z_\-]{30,}")),
    ("Google OAuth client secret", re.compile(r"GOCSPX-[0-9A-Za-z_\-]{20,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"gh[pousr]_[0-9A-Za-z]{30,}")),
    ("JSON secret field",
     re.compile(r'"(?:client_secret|refresh_token|access_token)"\s*:\s*"[^"]{8,}"')),
)


def scan_for_secrets(text: str) -> list:
    """Names of credential-looking patterns found in `text`."""
    return [name for name, pat in _SECRET_PATTERNS if pat.search(text or "")]


def _in_secrets_dir(path) -> bool:
    return any(part.lower() == "secrets" for part in Path(path).parts)


@dataclass
class UploadPlan:
    video: Path
    title: str
    description: str = ""
    tags: list = field(default_factory=list)
    thumbnail: Path = None
    category_id: str = "20"        # Gaming
    made_for_kids: bool = False
    privacy: str = PRIVATE
    allow_non_private: bool = False  # explicit, deliberate opt-out only

    def __post_init__(self):
        self.video = Path(self.video)
        if self.thumbnail:
            self.thumbnail = Path(self.thumbnail)
        self.tags = [t.strip() for t in self.tags if t and t.strip()]
        self.title = (self.title or "").strip()

    # -- validation -------------------------------------------------------

    def problems(self) -> list:
        out = []
        if self.privacy != PRIVATE and not self.allow_non_private:
            out.append(f"privacy is {self.privacy!r} but only 'private' is "
                       "allowed (set allow_non_private to override)")
        if self.privacy not in yt.VALID_PRIVACY:
            out.append(f"unknown privacy {self.privacy!r}")
        try:
            self.to_meta().validate()
        except yt.UploadError as e:
            out.append(str(e))

        blob = "\n".join([self.title, self.description, *self.tags])
        for name in scan_for_secrets(blob):
            out.append(f"metadata looks like it contains a {name}")

        if _in_secrets_dir(self.video):
            out.append("video path is inside a secrets/ directory")
        if self.thumbnail:
            if _in_secrets_dir(self.thumbnail):
                out.append("thumbnail path is inside a secrets/ directory")
            if self.thumbnail.suffix.lower() not in THUMBNAIL_EXTS:
                out.append(f"thumbnail must be one of {', '.join(THUMBNAIL_EXTS)}")
            elif not self.thumbnail.is_file():
                out.append(f"no such thumbnail: {self.thumbnail}")
            elif self.thumbnail.stat().st_size > MAX_THUMBNAIL_BYTES:
                out.append("thumbnail is over YouTube's 2 MB limit")
        if not self.video.is_file():
            out.append(f"no such video: {self.video}")
        return out

    def validate(self) -> None:
        probs = self.problems()
        if probs:
            raise PlanError("; ".join(probs))

    # -- conversion -------------------------------------------------------

    def to_meta(self) -> yt.VideoMeta:
        return yt.VideoMeta(
            title=self.title, description=self.description, tags=list(self.tags),
            category_id=self.category_id, privacy=self.privacy,
            made_for_kids=self.made_for_kids,
        )

    def to_body(self) -> dict:
        return self.to_meta().to_body()

    def manifest(self) -> dict:
        d = asdict(self)
        d["video"] = self.video.name
        d["thumbnail"] = self.thumbnail.name if self.thumbnail else None
        return d


def build_plan(video, title, description="", tags=None, thumbnail=None,
               **kw) -> UploadPlan:
    """The normal way to make a plan: always private, validated immediately."""
    plan = UploadPlan(video=video, title=title, description=description,
                      tags=list(tags or []), thumbnail=thumbnail, **kw)
    plan.validate()
    return plan


def from_generated(video, ctx, thumbnail=None, intro="", title=None) -> UploadPlan:
    """Plan from a metadata.Context: generated title, description and tags."""
    from . import metadata as md

    return build_plan(
        video,
        title or md.generate_title(ctx),
        md.generate_description(ctx, intro=intro),
        md.generate_tags(ctx),
        thumbnail=thumbnail,
    )


def describe(plan: UploadPlan) -> str:
    """Human-readable preview, for dry runs and confirmation prompts."""
    lines = [
        f"video      : {plan.video}",
        f"title      : {plan.title}  ({len(plan.title)}/{yt.MAX_TITLE})",
        f"privacy    : {plan.privacy}",
        f"thumbnail  : {plan.thumbnail or '(none)'}",
        f"tags       : {', '.join(plan.tags) or '(none)'}",
        "description:",
        *("  " + l for l in plan.description.splitlines()),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# uploaders
# --------------------------------------------------------------------------

class Uploader(Protocol):
    def upload_video(self, video: Path, body: dict, progress=None) -> dict: ...
    def set_thumbnail(self, video_id: str, thumbnail: Path) -> None: ...


class FakeUploader:
    """Records calls; used by tests and by --dry-run. Touches no network."""

    def __init__(self, video_id="FAKEID12345", fail_thumbnail=False):
        self.video_id = video_id
        self.fail_thumbnail = fail_thumbnail
        self.uploads = []
        self.thumbnails = []

    def upload_video(self, video, body, progress=None):
        self.uploads.append((Path(video), body))
        if progress:
            progress(1.0)
        return {"id": self.video_id, "status": body.get("status", {})}

    def set_thumbnail(self, video_id, thumbnail):
        if self.fail_thumbnail:
            raise yt.UploadError("thumbnail rejected (simulated)")
        self.thumbnails.append((video_id, Path(thumbnail)))


class YouTubeUploader:
    """The real thing. Needs secrets/client_secret.json; untested in CI."""

    def __init__(self, client_secret=yt.DEFAULT_CLIENT_SECRET,
                 token_path=yt.DEFAULT_TOKEN):
        self.client_secret = client_secret
        self.token_path = token_path

    def upload_video(self, video, body, progress=None):
        s = body["snippet"]
        meta = yt.VideoMeta(
            title=s["title"], description=s["description"], tags=s["tags"],
            category_id=s["categoryId"], privacy=body["status"]["privacyStatus"],
            made_for_kids=body["status"]["selfDeclaredMadeForKids"],
        )
        return yt.upload(video, meta, self.client_secret, self.token_path,
                         progress=progress)

    def set_thumbnail(self, video_id, thumbnail):
        from googleapiclient.discovery import build as build_service
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        creds = yt._credentials(self.client_secret, self.token_path)
        service = build_service("youtube", "v3", credentials=creds,
                                cache_discovery=False)
        try:
            service.thumbnails().set(
                videoId=video_id, media_body=MediaFileUpload(str(thumbnail))
            ).execute()
        except HttpError as e:
            raise yt.UploadError(f"thumbnail rejected: {e}") from e


@dataclass
class SubmitResult:
    response: dict
    video_id: str
    thumbnail_set: bool
    warnings: list = field(default_factory=list)

    @property
    def watch_url(self):
        return yt.watch_url(self.response)

    @property
    def studio_url(self):
        return yt.studio_url(self.response)


def submit(plan: UploadPlan, uploader: Uploader, progress=None,
           write_manifest: bool = True) -> SubmitResult:
    """Validate, upload, set the thumbnail, write a sidecar manifest."""
    plan.validate()
    body = plan.to_body()

    # Defence in depth: check what is actually about to be sent.
    status = body["status"]["privacyStatus"]
    if status != PRIVATE and not plan.allow_non_private:
        raise PlanError(f"refusing to send privacyStatus={status!r}")

    response = uploader.upload_video(plan.video, body, progress=progress)
    vid = (response or {}).get("id", "")
    warnings, thumb_ok = [], False
    if not vid:
        raise PlanError("upload returned no video id")

    if plan.thumbnail:
        try:
            uploader.set_thumbnail(vid, plan.thumbnail)
            thumb_ok = True
        except yt.UploadError as e:
            warnings.append(f"video uploaded but thumbnail failed: {e}")

    if write_manifest:
        side = plan.video.with_suffix(".upload.json")
        data = plan.manifest()
        data.update(video_id=vid, thumbnail_set=thumb_ok)
        side.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    return SubmitResult(response, vid, thumb_ok, warnings)
