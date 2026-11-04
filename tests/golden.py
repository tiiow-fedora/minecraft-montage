"""Tiny golden-file helper.

    assert_golden("name.json", obj)    # compares against tests/golden/name.json
    UPDATE_GOLDEN=1 pytest             # (re)writes goldens instead of comparing

Goldens are reviewed in the diff like any other file: a changed golden means
generated output changed, and that is either the point of the commit or a bug.
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden"


def render(obj) -> str:
    if isinstance(obj, str):
        return obj if obj.endswith("\n") else obj + "\n"
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def assert_golden(name: str, obj) -> None:
    text = render(obj)
    path = GOLDEN_DIR / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    if not path.is_file():
        raise AssertionError(
            f"missing golden file {path}; run UPDATE_GOLDEN=1 pytest to create it")
    want = path.read_text(encoding="utf-8")
    if text != want:
        diff = "".join(difflib.unified_diff(
            want.splitlines(True), text.splitlines(True),
            f"golden/{name}", "actual", n=2))
        raise AssertionError(f"output differs from golden/{name}:\n{diff[:4000]}")


def norm_tmp(value, tmp):
    """Replace a temp dir with <TMP> in a str (or list of str), platform-neutrally.

    On Windows the temp dir has backslashes and a drive colon, which ffmpeg
    filter escaping turns into ``C\\:/Users/...``; fold all spellings so the
    same golden matches on every OS.
    """
    import re
    from pathlib import PurePath

    if isinstance(value, (list, tuple)):
        return [norm_tmp(v, tmp) for v in value]
    tmp = str(tmp)
    spellings = {tmp, tmp.replace("\\", "/"),
                 tmp.replace("\\", "/").replace(":", "\\:"),
                 tmp.replace("\\", "\\\\")}          # JSON-escaped
    for sp in sorted(spellings, key=len, reverse=True):
        value = value.replace(sp, "<TMP>")
    # path remainder after <TMP>: separators are not significant
    return re.sub(r"<TMP>[^\s\"',;\[\]]*",
                  lambda m: re.sub(r"\\+", "/", m.group(0)), value)
