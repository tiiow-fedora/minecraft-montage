"""The pure helpers inside gui.py, importable without a display.

gui.py imports tkinter at module level, which is absent on many CI images and
irrelevant to these functions, so a stub tkinter is installed for the import.
"""

import importlib
import sys
import types

import pytest


@pytest.fixture(scope="module")
def App():
    saved = {k: sys.modules.get(k) for k in
             ("tkinter", "tkinter.filedialog", "tkinter.messagebox", "tkinter.ttk",
              "mcmontage.gui")}
    tk = types.ModuleType("tkinter")
    tk.TclError = type("TclError", (Exception,), {})
    for sub in ("filedialog", "messagebox", "ttk"):
        m = types.ModuleType("tkinter." + sub)
        setattr(tk, sub, m)
        sys.modules["tkinter." + sub] = m
    sys.modules["tkinter"] = tk
    sys.modules.pop("mcmontage.gui", None)
    try:
        yield importlib.import_module("mcmontage.gui").App
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.mark.parametrize("text,ms", [
    ("", None), ("   ", None), (None, None),
    ("0", 0), ("90", 90_000), ("1:30", 90_000), ("1:02:03", 3_723_000),
    ("1:30.5", 90_500), ("0:00.001", 1), ("  2:00 ", 120_000), ("100:00", 6_000_000),
])
def test_parse_time_valid(App, text, ms):
    assert App.parse_time(text) == ms


@pytest.mark.parametrize("text", ["abc", "1:2:3:4", "-5", "1:-5", "1::2", "1:30:", "12 34"])
def test_parse_time_rejects_typos(App, text):
    with pytest.raises(ValueError):
        App.parse_time(text)


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "1e999", "infinity"])
def test_parse_time_rejects_non_finite_with_valueerror(App, text):
    """The GUI only catches ValueError; inf used to escape as OverflowError."""
    with pytest.raises(ValueError):
        App.parse_time(text)


@pytest.mark.parametrize("ms,text", [
    (0, "0:00"), (None, "0:00"), (999, "0:01"), (59_499, "0:59"), (60_000, "1:00"),
    (3_599_000, "59:59"), (3_600_000, "1:00:00"), (3_723_000, "1:02:03"),
])
def test_format_time(App, ms, text):
    assert App.format_time(ms) == text


def test_parse_format_roundtrip(App):
    for ms in range(0, 8_000_000, 123_000):
        assert App.parse_time(App.format_time(ms)) == round(ms / 1000) * 1000
