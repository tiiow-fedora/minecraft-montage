import json

import pytest

from mcmontage import metadata as md
from mcmontage import upload_meta as um
from mcmontage import youtube as yt


@pytest.fixture
def video(tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"not really a video")
    return v


@pytest.fixture
def thumb(tmp_path):
    t = tmp_path / "t.png"
    t.write_bytes(b"png")
    return t


def test_plan_is_private_by_default(video):
    plan = um.build_plan(video, "Title")
    assert plan.privacy == "private"
    assert plan.to_body()["status"]["privacyStatus"] == "private"
    assert plan.allow_non_private is False


@pytest.mark.parametrize("p", ["public", "unlisted"])
def test_non_private_rejected_without_explicit_optout(video, p):
    with pytest.raises(um.PlanError, match="private"):
        um.build_plan(video, "T", privacy=p)


def test_explicit_optout_is_possible_but_deliberate(video):
    plan = um.build_plan(video, "T", privacy="unlisted", allow_non_private=True)
    assert plan.to_body()["status"]["privacyStatus"] == "unlisted"


def test_unknown_privacy_rejected_even_with_optout(video):
    with pytest.raises(um.PlanError):
        um.build_plan(video, "T", privacy="everyone", allow_non_private=True)


def test_mutation_after_validation_cannot_leak_public(video):
    plan = um.build_plan(video, "T")
    plan.privacy = "public"
    up = um.FakeUploader()
    with pytest.raises(um.PlanError):
        um.submit(plan, up)
    assert up.uploads == []


def test_generated_plan_is_private(video):
    ctx = md.Context(theme="Farm", clips=[md.ClipInfo(f"c{i}", 30) for i in range(3)])
    plan = um.from_generated(video, ctx)
    assert plan.privacy == "private"
    assert "Chapters:" in plan.description and plan.title == "Farm"
    assert "minecraft" in plan.tags


@pytest.mark.parametrize("secret", [
    "AIza" + "a" * 35,
    "ya29." + "b" * 40,
    "1//" + "c" * 50,
    "GOCSPX-" + "d" * 28,
    "-----BEGIN PRIVATE KEY-----",
    "ghp_" + "e" * 36,
    '{"refresh_token": "abcdefghijkl"}',
])
def test_secret_like_text_blocks_upload(video, secret):
    with pytest.raises(um.PlanError, match="looks like"):
        um.build_plan(video, "T", description="hello\n" + secret)
    with pytest.raises(um.PlanError, match="looks like"):
        um.build_plan(video, "T", tags=[secret])


def test_normal_text_is_not_flagged():
    assert um.scan_for_secrets("Day 3: AIza is a word, 1//2 is a fraction") == []
    assert um.scan_for_secrets("") == [] and um.scan_for_secrets(None) == []


def test_secrets_directory_paths_refused(tmp_path):
    d = tmp_path / "secrets"
    d.mkdir()
    v = d / "v.mp4"
    v.write_bytes(b"x")
    with pytest.raises(um.PlanError, match="secrets"):
        um.build_plan(v, "T")


def test_thumbnail_checks(video, tmp_path):
    with pytest.raises(um.PlanError, match="no such thumbnail"):
        um.build_plan(video, "T", thumbnail=tmp_path / "missing.png")
    bad = tmp_path / "t.gif"
    bad.write_bytes(b"x")
    with pytest.raises(um.PlanError, match="must be one of"):
        um.build_plan(video, "T", thumbnail=bad)
    big = tmp_path / "big.jpg"
    big.write_bytes(b"x" * (um.MAX_THUMBNAIL_BYTES + 1))
    with pytest.raises(um.PlanError, match="2 MB"):
        um.build_plan(video, "T", thumbnail=big)


def test_youtube_limits_still_enforced(video):
    with pytest.raises(um.PlanError):
        um.build_plan(video, "")
    with pytest.raises(um.PlanError):
        um.build_plan(video, "x" * 101)
    with pytest.raises(um.PlanError):
        um.build_plan(video, "T", description="d" * 5001)


def test_missing_video(tmp_path):
    with pytest.raises(um.PlanError, match="no such video"):
        um.build_plan(tmp_path / "gone.mp4", "T")


def test_tags_cleaned(video):
    plan = um.build_plan(video, " T ", tags=[" a ", "", "  ", "b"])
    assert plan.tags == ["a", "b"] and plan.title == "T"


def test_submit_uploads_sets_thumbnail_and_writes_manifest(video, thumb):
    plan = um.build_plan(video, "My title", "desc", ["x"], thumbnail=thumb)
    up = um.FakeUploader(video_id="abc123")
    seen = []
    res = um.submit(plan, up, progress=seen.append)
    assert res.video_id == "abc123" and res.thumbnail_set and not res.warnings
    assert up.uploads[0][1]["status"]["privacyStatus"] == "private"
    assert up.thumbnails == [("abc123", thumb)]
    assert seen == [1.0]
    assert res.watch_url.endswith("abc123") and "studio" in res.studio_url
    manifest = json.loads(video.with_suffix(".upload.json").read_text())
    assert manifest["video_id"] == "abc123" and manifest["privacy"] == "private"
    assert manifest["video"] == "v.mp4" and manifest["thumbnail_set"] is True


def test_thumbnail_failure_does_not_lose_upload(video, thumb):
    plan = um.build_plan(video, "T", thumbnail=thumb)
    res = um.submit(plan, um.FakeUploader(fail_thumbnail=True))
    assert res.video_id and not res.thumbnail_set
    assert "thumbnail failed" in res.warnings[0]


def test_no_manifest_when_disabled(video):
    um.submit(um.build_plan(video, "T"), um.FakeUploader(), write_manifest=False)
    assert not video.with_suffix(".upload.json").exists()


def test_upload_without_id_is_an_error(video):
    class Bad(um.FakeUploader):
        def upload_video(self, *a, **k):
            return {}
    with pytest.raises(um.PlanError, match="no video id"):
        um.submit(um.build_plan(video, "T"), Bad())


def test_describe_shows_privacy_and_description(video):
    out = um.describe(um.build_plan(video, "T", "line1\nline2"))
    assert "privacy    : private" in out and "  line2" in out


def test_real_uploader_translates_plan_body(video, monkeypatch):
    captured = {}

    def fake_upload(v, meta, cs, tok, progress=None):
        captured["meta"] = meta
        return {"id": "z"}
    monkeypatch.setattr(yt, "upload", fake_upload)
    plan = um.build_plan(video, "T", "d", ["a"])
    um.submit(plan, um.YouTubeUploader("cs.json", "tok.json"), write_manifest=False)
    assert captured["meta"].privacy == "private" and captured["meta"].tags == ["a"]
