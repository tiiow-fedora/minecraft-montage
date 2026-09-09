"""Music library with per-track licence records and automatic attribution.

"Non-copyright" music is not public domain. Every track carries a licence, and
some licences (CC-BY in particular) require a specific credit in the video
description. This module makes that credit a hard requirement rather than
something you remember to paste: a track whose licence requires attribution but
carries no attribution text will refuse to be used.

The library lives in music/tracks.toml. See music/tracks.example.toml.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# Licences that legally require a visible credit. CC0 and the YouTube Audio
# Library's default terms do not, but individual YTAL tracks sometimes do --
# which is why attribution_required is per-track and this map is only a default.
ATTRIBUTION_REQUIRED_BY_DEFAULT = {
    "CC-BY-3.0": True,
    "CC-BY-4.0": True,
    "CC-BY-SA-3.0": True,
    "CC-BY-SA-4.0": True,
    "CC0-1.0": False,
    "CC0": False,
    "YOUTUBE_AUDIO_LIBRARY": False,
    "PIXABAY": False,
}

KNOWN_LICENCES = set(ATTRIBUTION_REQUIRED_BY_DEFAULT)


class MusicError(RuntimeError):
    pass


@dataclass(frozen=True)
class Track:
    id: str
    file: Path
    title: str
    artist: str
    source: str
    license: str
    url: str = ""
    attribution: str = ""
    attribution_required: bool | None = None

    @property
    def needs_attribution(self) -> bool:
        if self.attribution_required is not None:
            return self.attribution_required
        return ATTRIBUTION_REQUIRED_BY_DEFAULT.get(self.license.upper(), True)

    def credit(self) -> str:
        """The exact credit line to put in the description."""
        if self.attribution:
            return self.attribution
        bits = [f'"{self.title}" by {self.artist}']
        if self.url:
            bits.append(self.url)
        bits.append(f"Licence: {self.license}")
        return " | ".join(bits)

    def validate(self) -> None:
        if not self.file.is_file():
            raise MusicError(f"track {self.id}: file not found: {self.file}")
        if not self.license:
            raise MusicError(f"track {self.id}: no licence recorded")
        if self.license.upper() not in KNOWN_LICENCES:
            # Not fatal -- but it must be explicit about attribution.
            if self.attribution_required is None:
                raise MusicError(
                    f"track {self.id}: unrecognised licence {self.license!r}; set "
                    "attribution_required = true/false explicitly"
                )
        if self.needs_attribution and not self.attribution and not (
            self.title and self.artist
        ):
            raise MusicError(
                f"track {self.id}: licence {self.license} requires attribution but "
                "there is no attribution text and no title/artist to build one from"
            )


def load_library(path="music/tracks.toml") -> list[Track]:
    path = Path(path)
    if not path.is_file():
        raise MusicError(
            f"no music library at {path} -- copy music/tracks.example.toml to "
            "music/tracks.toml and add your tracks"
        )
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    base = path.parent.parent  # repo root, so file paths can be repo-relative
    tracks = []
    for entry in data.get("track", []):
        missing = [k for k in ("id", "file", "title", "artist", "source", "license")
                   if not entry.get(k)]
        if missing:
            raise MusicError(
                f"track entry {entry.get('id', '<no id>')} missing: {', '.join(missing)}"
            )
        f = Path(entry["file"])
        if not f.is_absolute():
            f = (base / f).resolve()
        tracks.append(
            Track(
                id=entry["id"],
                file=f,
                title=entry["title"],
                artist=entry["artist"],
                source=entry["source"],
                license=entry["license"],
                url=entry.get("url", ""),
                attribution=entry.get("attribution", ""),
                attribution_required=entry.get("attribution_required"),
            )
        )
    if not tracks:
        raise MusicError(f"{path} contains no [[track]] entries")

    ids = [t.id for t in tracks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise MusicError(f"duplicate track ids: {', '.join(sorted(dupes))}")
    return tracks


def get(track_id, library=None) -> Track:
    library = library if library is not None else load_library()
    for t in library:
        if t.id == track_id:
            t.validate()
            return t
    raise MusicError(
        f"no track {track_id!r} in library (have: {', '.join(t.id for t in library)})"
    )


def attribution_block(tracks) -> str:
    """The music credits section for a video description."""
    tracks = [tracks] if isinstance(tracks, Track) else list(tracks)
    if not tracks:
        return ""
    lines = ["Music:"]
    for t in tracks:
        lines.append(f"  {t.credit()}")
    return "\n".join(lines)
