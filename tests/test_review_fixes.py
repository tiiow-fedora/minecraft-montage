"""Regression tests for the review/security pass. Run: python -m unittest -v"""
import datetime
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from mcmontage import edit, replay, sessions, youtube


class RenameTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, True)

    def mk(self, name, text):
        p = self.d / name
        p.write_text(text)
        return p

    def test_swap_names(self):
        a, b = self.mk("a.mcpr", "A"), self.mk("b.mcpr", "B")
        sessions.apply_renames([(a, b), (b, a)])
        self.assertEqual((self.d / "a.mcpr").read_text(), "B")
        self.assertEqual((self.d / "b.mcpr").read_text(), "A")

    def test_refuses_to_overwrite_unrelated_file(self):
        a, keep = self.mk("a.mcpr", "A"), self.mk("keep.mcpr", "K")
        with self.assertRaises(sessions.SessionError):
            sessions.apply_renames([(a, keep)])
        self.assertEqual(keep.read_text(), "K")
        self.assertTrue(a.exists())

    def test_duplicate_targets_rejected(self):
        a, b = self.mk("a.mcpr", "A"), self.mk("b.mcpr", "B")
        with self.assertRaises(sessions.SessionError):
            sessions.apply_renames([(a, self.d / "x.mcpr"), (b, self.d / "x.mcpr")])
        self.assertTrue(a.exists() and b.exists())

    def test_scheme_cannot_escape_folder(self):
        clip = sessions.Clip(self.d / "a.mcpr", datetime.datetime(2026, 1, 2, 12), 1000)
        s = sessions.Session(datetime.date(2026, 1, 2), 1, [clip])
        for bad in ("../evil", "a/b", "..", ""):
            with self.assertRaises(sessions.SessionError):
                s.name_for(clip, bad)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, True)
        self.p = self.d / "r.mcpr"
        with zipfile.ZipFile(self.p, "w") as z:
            z.writestr("metaData.json", json.dumps({"duration": 5}))
            z.writestr("recording.tmcpr", b"x" * 100)

    def test_roundtrip_no_tmp_left(self):
        job = {"timeline": {"a": 1}, "settings": {}}
        replay.write_render_queue(self.p, [job])
        self.assertEqual(replay.read_render_queue(self.p)[0]["timeline"], {"a": 1})
        self.assertFalse(self.p.with_suffix(".mcpr.tmp").exists())

    def test_failed_write_keeps_original_and_cleans_tmp(self):
        before = self.p.read_bytes()
        with self.assertRaises(KeyError):
            replay.write_render_queue(self.p, [{"timeline": {}}], backup=False)
        self.assertEqual(self.p.read_bytes(), before)
        self.assertFalse(self.p.with_suffix(".mcpr.tmp").exists())

    def test_non_object_job_is_replay_error(self):
        with zipfile.ZipFile(self.p, "a") as z:
            z.writestr("renderQueue.json", "[1]")
        with self.assertRaises(replay.ReplayError):
            replay.read_render_queue(self.p)


class YouTubeTests(unittest.TestCase):
    def test_angle_brackets_rejected(self):
        for kw in ({"title": "a<b"}, {"title": "ok", "description": "x>y"}):
            with self.assertRaises(youtube.UploadError):
                youtube.VideoMeta(**kw).validate()

    def test_default_secret_paths_anchored_and_ignored_by_git(self):
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(youtube.DEFAULT_TOKEN.parent, root / "secrets")
        r = subprocess.run(["git", "check-ignore", "-q", str(youtube.DEFAULT_TOKEN)],
                           cwd=root)
        self.assertEqual(r.returncode, 0, "secrets/ must be git-ignored")

    @unittest.skipIf(os.name == "nt", "POSIX modes")
    def test_token_written_private(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        t = d / "s" / "token.json"
        youtube._write_private(t, "{}")
        self.assertEqual(stat.S_IMODE(t.stat().st_mode), 0o600)
        self.assertEqual(list(t.parent.iterdir()), [t])

    def test_non_private_upload_refused(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        v = d / "v.mp4"
        v.write_bytes(b"x")
        try:
            import googleapiclient  # noqa: F401
        except ImportError:
            self.skipTest("google client libs not installed")
        with self.assertRaises(youtube.UploadError):
            youtube.upload(v, youtube.VideoMeta(title="t", privacy="public"))


class FfmpegEscapeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_title_filter_runs_in_folder_with_apostrophe(self):
        d = Path(tempfile.mkdtemp()) / "John's, Videos [1]"
        self.addCleanup(shutil.rmtree, d.parent, True)
        d.mkdir()
        fonts = [f for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",)
                 if Path(f).is_file()]
        if not fonts:
            self.skipTest("no font available")
        clip = d / "c.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=blue:s=320x240:d=2:r=10", str(clip)], check=True)
        out = d / "out.mp4"
        edit.build([clip], out, title="It's a test, ok: 100%", font=fonts[0],
                   fade_in=0, fade_out=0)
        self.assertGreater(out.stat().st_size, 0)

    def test_dash_output_name_is_absolutised(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        clip = d / "c.mp4"
        clip.write_bytes(b"x")
        old = Path.cwd()
        os.chdir(d)
        self.addCleanup(os.chdir, old)
        orig = edit.probe
        edit.probe = lambda p: edit.Probe(Path(p), 1.0, 320, 240, 10.0, False)
        self.addCleanup(setattr, edit, "probe", orig)
        args = edit.build([clip], "-evil.mp4", dry_run=True)
        self.assertTrue(os.path.isabs(args[-1]))


if __name__ == "__main__":
    unittest.main()
