import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import synthetic  # noqa: E402
from mcmontage import config as config_mod  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "ffmpeg: needs a real ffmpeg/ffprobe on PATH")


def pytest_collection_modifyitems(config, items):
    if synthetic.have_ffmpeg():
        return
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not on PATH")
    for item in items:
        if "ffmpeg" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def cfg(tmp_path):
    """A Config rooted entirely in tmp_path, independent of the host."""
    return config_mod.Config(
        profile=tmp_path / "profile",
        output=tmp_path / "out",
        work=tmp_path / "work",
    )


@pytest.fixture(autouse=True)
def no_minecraft(monkeypatch):
    """Never shell out to tasklist; individual tests override as needed."""
    from mcmontage import pipeline
    monkeypatch.setattr(pipeline, "minecraft_running", lambda: False)

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg not installed")


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    """10 s video: 5 s of black, then 5 s of a detailed test pattern."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("vid") / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-filter_complex",
         "color=black:s=640x360:d=5:r=10[a];testsrc2=s=640x360:d=5:r=10[b];"
         "[a][b]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-pix_fmt", "yuv420p", str(out)],
        check=True)
    return out
