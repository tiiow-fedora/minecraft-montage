"""YouTube upload stage.

Uploads land as PRIVATE, deliberately.

Any Google API project that has not passed a Google API audit has its
`videos.insert` uploads permanently locked to private, and per Google's own
policy page that lock cannot be appealed:

    "For videos that have been locked as private due to upload via an
     unverified API service, you will not be able to appeal."
    https://support.google.com/youtube/answer/7300965

So rather than pretend otherwise, this module sets privacyStatus=private
explicitly and treats publishing as a human step in YouTube Studio. That also
gives you a review gate before anything goes out under your name.

Quota, verified against Google's own quota page on 2026-09-08:
    videos.insert costs 1 unit and has its OWN daily bucket of 100 calls/day,
    separate from the 10,000-unit pool used by everything else. The widely
    repeated "1600 units per upload" figure is out of date. Quota will not be
    your constraint.
    https://developers.google.com/youtube/v3/determine_quota_cost

Setup (one-off, and it has to be you, not me -- it needs your Google account):
  1. https://console.cloud.google.com/ -> create a project
  2. APIs & Services -> Library -> enable "YouTube Data API v3"
  3. APIs & Services -> OAuth consent screen -> External; add your own Google
     account under "Test users"
  4. Credentials -> Create credentials -> OAuth client ID -> Desktop app
  5. Download the JSON, save it as secrets/client_secret.json

First run opens a browser for consent and caches a refresh token in
secrets/token.json. Both files are secrets; keep them out of version control.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Anchored to the repo root, not the working directory, so launching the app
# from elsewhere neither loses the token nor writes a new one outside the
# git-ignored secrets/ folder.
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIENT_SECRET = _ROOT / "secrets" / "client_secret.json"
DEFAULT_TOKEN = _ROOT / "secrets" / "token.json"

# YouTube rejects titles over 100 chars and descriptions over 5000.
MAX_TITLE = 100
MAX_DESCRIPTION = 5000
MAX_TAGS_CHARS = 500

VALID_PRIVACY = ("private", "unlisted", "public")


class UploadError(RuntimeError):
    pass


@dataclass
class VideoMeta:
    title: str
    description: str = ""
    tags: list = field(default_factory=list)
    category_id: str = "20"  # 20 = Gaming
    privacy: str = "private"
    made_for_kids: bool = False

    def validate(self) -> None:
        if not self.title.strip():
            raise UploadError("video title is empty")
        if len(self.title) > MAX_TITLE:
            raise UploadError(
                f"title is {len(self.title)} chars, YouTube's limit is {MAX_TITLE}"
            )
        if len(self.description) > MAX_DESCRIPTION:
            raise UploadError(
                f"description is {len(self.description)} chars, limit is "
                f"{MAX_DESCRIPTION}"
            )
        # The API rejects angle brackets in titles and descriptions.
        for label, text in (("title", self.title), ("description", self.description)):
            if "<" in text or ">" in text:
                raise UploadError(
                    f"{label} contains '<' or '>', which YouTube does not allow"
                )
        if self.privacy not in VALID_PRIVACY:
            raise UploadError(f"bad privacy {self.privacy!r}")
        total_tags = sum(len(t) for t in self.tags)
        if total_tags > MAX_TAGS_CHARS:
            raise UploadError(
                f"tags total {total_tags} chars, YouTube's limit is {MAX_TAGS_CHARS}"
            )

    def to_body(self) -> dict:
        return {
            "snippet": {
                "title": self.title,
                "description": self.description,
                "tags": self.tags,
                "categoryId": self.category_id,
            },
            "status": {
                "privacyStatus": self.privacy,
                "selfDeclaredMadeForKids": self.made_for_kids,
            },
        }


def build_description(body, music_tracks=None, extra=None) -> str:
    """Assemble a description with music attribution appended automatically.

    Attribution is not optional and not manual: whatever tracks the video used
    get credited here, in the form their licence requires.
    """
    from . import music as music_mod

    parts = [body.strip()] if body and body.strip() else []
    if music_tracks:
        block = music_mod.attribution_block(music_tracks)
        if block:
            parts.append(block)
    if extra:
        parts.append(extra.strip())
    return "\n\n".join(parts)


def _credentials(client_secret, token_path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_secret = Path(client_secret)
    token_path = Path(token_path)

    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds and creds.valid:
        return creds
    refreshed = False
    if creds and creds.expired and creds.refresh_token:
        from google.auth.exceptions import RefreshError

        try:
            creds.refresh(Request())
            refreshed = True
        except RefreshError:
            # Revoked or expired refresh token (common for "Testing" projects,
            # whose tokens last 7 days): fall through to a fresh consent flow
            # rather than crashing.
            creds = None
    if not refreshed:
        if not client_secret.is_file():
            raise UploadError(
                f"missing OAuth client secret at {client_secret}. See the setup "
                "steps in the module docstring -- this is a one-off you have to "
                "do under your own Google account."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
        creds = flow.run_local_server(port=0)

    _write_private(token_path, creds.to_json())
    return creds


def _write_private(path: Path, text: str) -> None:
    """Write a secret file that is never world-readable, even momentarily.

    The mode is applied at creation (not chmod'd afterwards) and the content is
    swapped in atomically so a crash cannot leave a truncated token behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def upload(
    video_path,
    meta: VideoMeta,
    client_secret=DEFAULT_CLIENT_SECRET,
    token_path=DEFAULT_TOKEN,
    chunk_size=8 * 1024 * 1024,
    progress=None,
    allow_non_private=False,
):
    """Upload a video resumably. Returns the created video resource.

    Refuses non-private uploads unless explicitly overridden in `meta`, because
    an unverified project cannot make them public anyway -- see module docstring.
    """
    from googleapiclient.discovery import build as build_service
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    video_path = Path(video_path)
    if not video_path.is_file():
        raise UploadError(f"no such video: {video_path}")
    if video_path.stat().st_size == 0:
        raise UploadError(f"video is empty: {video_path}")
    meta.validate()
    if meta.privacy != "private" and not allow_non_private:
        raise UploadError(
            f"refusing to upload as {meta.privacy!r}; pass allow_non_private=True "
            "to override (unverified API projects are locked to private anyway)"
        )

    creds = _credentials(client_secret, token_path)
    youtube = build_service("youtube", "v3", credentials=creds, cache_discovery=False)

    media = MediaFileUpload(
        str(video_path), chunksize=chunk_size, resumable=True, mimetype="video/*"
    )
    request = youtube.videos().insert(
        part="snippet,status", body=meta.to_body(), media_body=media
    )

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
        except HttpError as e:
            raise UploadError(f"YouTube rejected the upload: {e}") from e
        if status and progress:
            progress(status.progress())
    if progress:
        progress(1.0)
    return response


def watch_url(response) -> str:
    vid = (response or {}).get("id")
    return f"https://www.youtube.com/watch?v={vid}" if vid else ""


def studio_url(response) -> str:
    """Where to go to review and publish it."""
    vid = (response or {}).get("id")
    return f"https://studio.youtube.com/video/{vid}/edit" if vid else ""
