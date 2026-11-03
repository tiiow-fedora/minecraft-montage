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
