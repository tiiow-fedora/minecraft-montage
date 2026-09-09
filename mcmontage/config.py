"""Configuration. Defaults match the ReForgedPlay/ReplayMod folder layout.

Folder names confirmed from the mod's own constants (com.replaymod.core.Setting
and com.replaymod.render.Setting), not guessed:
    core.recordingPath   ./replay_recordings/
    render.renderPath    ./replay_videos/
    core.cachePath       ./.replay_cache/
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

RECORDINGS_SUBDIR = "replay_recordings"
VIDEOS_SUBDIR = "replay_videos"

DEFAULT_PROFILE = (
    Path(os.environ.get("APPDATA", ""))
    / "ModrinthApp"
    / "profiles"
    / "HarrisonsLongHaulWorld"
)


@dataclass
class Config:
    profile: Path = field(default_factory=lambda: DEFAULT_PROFILE)
    recordings: Path | None = None
    videos: Path | None = None
    output: Path = Path("out")
    work: Path = Path("work")

    width: int = 1920
    height: int = 1080
    fps: int = 60
    bitrate: int = 20_000_000

    xfade: float = 1.0
    music_gain_db: float = -16.0
    music_fade_out: float = 4.0
    crf: int = 18

    def __post_init__(self):
        self.profile = Path(self.profile)
        self.recordings = (
            Path(self.recordings)
            if self.recordings
            else self.profile / RECORDINGS_SUBDIR
        )
        self.videos = (
            Path(self.videos) if self.videos else self.profile / VIDEOS_SUBDIR
        )
        self.output = Path(self.output)
        self.work = Path(self.work)


def load(path="config.toml") -> Config:
    path = Path(path)
    if not path.is_file():
        return Config()
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    paths = data.get("paths", {})
    render = data.get("render", {})
    video = data.get("video", {})
    return Config(
        profile=paths.get("profile") or DEFAULT_PROFILE,
        recordings=paths.get("recordings") or None,
        videos=paths.get("videos") or None,
        output=paths.get("output", "out"),
        work=paths.get("work", "work"),
        width=render.get("width", 1920),
        height=render.get("height", 1080),
        fps=render.get("fps", 60),
        bitrate=render.get("bitrate", 20_000_000),
        xfade=video.get("xfade", 1.0),
        music_gain_db=video.get("music_gain_db", -16.0),
        music_fade_out=video.get("music_fade_out", 4.0),
        crf=video.get("crf", 18),
    )
