import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
