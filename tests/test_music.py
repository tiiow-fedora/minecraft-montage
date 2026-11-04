import pytest

from mcmontage import music
from mcmontage.music import MusicError, Track


def T(tmp_path, **kw):
    f = tmp_path / "t.mp3"
    f.write_bytes(b"x")
    base = dict(id="t", file=f, title="Song", artist="Me", source="src",
                license="CC-BY-4.0")
    base.update(kw)
    return Track(**base)


def lib(tmp_path, body, files=("a.mp3",)):
    (tmp_path / "music").mkdir(exist_ok=True)
    for f in files:
        (tmp_path / "music" / f).write_bytes(b"x")
    p = tmp_path / "music" / "tracks.toml"
    p.write_text(body, encoding="utf-8")
    return p


ENTRY = '''[[track]]
id = "{id}"
file = "music/{file}"
title = "T"
artist = "A"
source = "S"
license = "{lic}"
'''


@pytest.mark.parametrize("lic,needs", [
    ("CC-BY-4.0", True), ("cc-by-3.0", True), ("CC-BY-SA-4.0", True),
    ("CC0", False), ("cc0-1.0", False), ("YOUTUBE_AUDIO_LIBRARY", False),
    ("PIXABAY", False),
])
def test_attribution_defaults(tmp_path, lic, needs):
    assert T(tmp_path, license=lic).needs_attribution is needs


def test_explicit_flag_overrides_licence(tmp_path):
    assert T(tmp_path, license="CC0", attribution_required=True).needs_attribution
    assert not T(tmp_path, license="CC-BY-4.0", attribution_required=False).needs_attribution


def test_credit_prefers_explicit_attribution(tmp_path):
    assert T(tmp_path, attribution="Custom line").credit() == "Custom line"


def test_credit_built_from_fields(tmp_path):
    t = T(tmp_path, url="https://x.test/s")
    assert t.credit() == '"Song" by Me | https://x.test/s | Licence: CC-BY-4.0'
    assert T(tmp_path).credit() == '"Song" by Me | Licence: CC-BY-4.0'


def test_validate_missing_file(tmp_path):
    t = T(tmp_path)
    t.file.unlink()
    with pytest.raises(MusicError, match="not found"):
        t.validate()


def test_validate_unknown_licence_needs_explicit_flag(tmp_path):
    with pytest.raises(MusicError, match="unrecognised"):
        T(tmp_path, license="WEIRD-1.0").validate()
    T(tmp_path, license="WEIRD-1.0", attribution_required=False).validate()


def test_validate_no_licence(tmp_path):
    with pytest.raises(MusicError, match="no licence"):
        T(tmp_path, license="").validate()


def test_validate_requires_creditable_text(tmp_path):
    with pytest.raises(MusicError, match="requires attribution"):
        T(tmp_path, title="", artist="").validate()


def test_load_library_resolves_repo_relative_paths(tmp_path):
    p = lib(tmp_path, ENTRY.format(id="a", file="a.mp3", lic="CC0"))
    tracks = music.load_library(p)
    assert tracks[0].file == (tmp_path / "music" / "a.mp3").resolve()


def test_load_library_errors(tmp_path):
    with pytest.raises(MusicError, match="no music library"):
        music.load_library(tmp_path / "music" / "tracks.toml")
    with pytest.raises(MusicError, match="no \\[\\[track\\]\\]"):
        music.load_library(lib(tmp_path, "# empty\n"))
    with pytest.raises(MusicError, match="missing: license"):
        music.load_library(lib(tmp_path, '[[track]]\nid="a"\nfile="x"\ntitle="t"\nartist="a"\nsource="s"\n'))
    two = ENTRY.format(id="a", file="a.mp3", lic="CC0") * 2
    with pytest.raises(MusicError, match="duplicate"):
        music.load_library(lib(tmp_path, two))


def test_get_validates_and_lists_alternatives(tmp_path):
    p = lib(tmp_path, ENTRY.format(id="a", file="a.mp3", lic="CC0")
            + ENTRY.format(id="b", file="gone.mp3", lic="CC0"))
    library = music.load_library(p)
    assert music.get("a", library).id == "a"
    with pytest.raises(MusicError, match="have: a, b"):
        music.get("zzz", library)
    with pytest.raises(MusicError, match="not found"):
        music.get("b", library)


def test_attribution_block(tmp_path):
    assert music.attribution_block([]) == ""
    t = T(tmp_path)
    assert music.attribution_block(t) == "Music:\n  " + t.credit()
    assert music.attribution_block([t, t]).count("\n") == 2
