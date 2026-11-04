import pytest
from mcmontage.events import Event, Highlight, EventError, load_events, save_events


def test_highlight_roundtrip():
    h = Highlight("death", 1000, 5000, 0.8, entity=3, pos=(1, 2, 3), label="x")
    assert Highlight.from_json(h.to_json()) == h
    assert h.duration_ms == 4000


def test_highlight_rejects_empty_span():
    with pytest.raises(EventError):
        Highlight("kill", 5, 5, 0.1)


def test_event_roundtrip_and_sidecar(tmp_path):
    evs = [Event(20, "kill", 1, (0, 0, 0), 2), Event(10, "death", 2)]
    p = tmp_path / "e.json"
    save_events(p, evs)
    got = load_events(p)
    assert [e.t for e in got] == [10, 20]
    assert got[1].target == 2


def test_missing_sidecar_is_empty(tmp_path):
    assert load_events(tmp_path / "nope.json") == []


def test_bad_event():
    with pytest.raises(EventError):
        Event.from_json({"kind": "death"})
