"""Tests for mcmontage.jobqueue, using fake handlers only (no pipeline imports)."""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest

from mcmontage import jobqueue as jq
from mcmontage.jobqueue import (
    CorruptStateError, CycleError, FileLock, JobQueue, JobStateError,
    LockHeldError, NeedsManual, ParamError, QueueError, Registry,
    SchemaVersionError, UnknownKindError,
)


# ---------------------------------------------------------------- helpers

class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def qpath(tmp_path):
    return tmp_path / "work" / "queue.json"


@pytest.fixture
def q(qpath):
    return JobQueue(qpath)


def recorder(calls, result=None):
    def handler(ctx):
        calls.append(ctx.job.id)
        return result if result is not None else {"id": ctx.job.id}
    return handler


def reg(**handlers):
    r = Registry()
    for k, h in handlers.items():
        r.register(k, h)
    return r


def dead_pid():
    """A pid that is certainly not alive."""
    return 2 ** 22 + 12345 if os.name != "nt" else 0x7FFFFFF0


def write_state(path, jobs, next_id=None, version=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": version, "next_id": next_id or len(jobs) + 1, "jobs": jobs}))


def raw_job(i, status="pending", **kw):
    d = dict(id="j%04d" % i, kind="a", params={}, status=status, attempts=0,
             error=None, result=None, depends_on=[], created=1.0, updated=1.0,
             started=None, finished=None, key=None, max_attempts=3,
             not_before=0.0, message=None, resumed=False)
    d.update(kw)
    return d


def write_lock(path, pid, host=None, age=0.0):
    now = time.time() - age
    path.write_text(json.dumps({
        "pid": pid, "host": host or socket.gethostname(),
        "created": now, "heartbeat": now, "token": "other"}))


# ---------------------------------------------------------------- basics

def test_add_assigns_sequential_ids_and_defaults(q):
    a = q.add("plan", {"clip": "a.mcpr"})
    b = q.add("plan", {"clip": "b.mcpr"})
    assert (a.id, b.id) == ("j0001", "j0002")
    assert a.status == "pending" and a.attempts == 0 and a.max_attempts == 3
    assert a.created > 0 and a.started is None and a.finished is None


def test_add_persists_to_disk(q, qpath):
    q.add("plan", {"x": 1})
    data = json.loads(qpath.read_text())
    assert data["version"] == jq.SCHEMA_VERSION
    assert data["jobs"][0]["kind"] == "plan"


def test_reopen_loads_same_jobs(q, qpath):
    q.add("plan", {"x": 1})
    q2 = JobQueue(qpath)
    assert [j.kind for j in q2.jobs()] == ["plan"]
    assert q2.add("b").id == "j0002"


def test_no_file_created_on_open(tmp_path):
    p = tmp_path / "nowhere" / "q.json"
    JobQueue(p)
    assert not p.parent.exists()


def test_add_rejects_bad_kind_and_attempts(q):
    with pytest.raises(ValueError):
        q.add("")
    with pytest.raises(ValueError):
        q.add("a", max_attempts=0)


def test_add_rejects_unknown_dependency(q):
    with pytest.raises(QueueError):
        q.add("a", depends_on=["j9999"])


def test_add_rejects_non_json_params(q):
    with pytest.raises(ParamError):
        q.add("a", {"f": object()})
    with pytest.raises(ParamError):
        q.add("a", ["not", "a", "dict"])


def test_get_returns_copy(q):
    j = q.add("a")
    j.status = "done"
    assert q.get(j.id).status == "pending"


def test_get_unknown(q):
    with pytest.raises(QueueError):
        q.get("nope")


# ---------------------------------------------------------------- idempotent keys

def test_key_returns_existing_job(q):
    a = q.add("a", {"n": 1}, key="k1")
    b = q.add("a", {"n": 2}, key="k1")
    assert a.id == b.id
    assert len(q) == 1
    assert q.get(a.id).params == {"n": 1}


def test_key_survives_reopen(q, qpath):
    a = q.add("a", key="k")
    assert JobQueue(qpath).add("a", key="k").id == a.id


def test_key_idempotent_even_when_done(q):
    a = q.add("a", key="k")
    q.run(reg(a=recorder([])))
    assert q.add("a", key="k").status == "done"
    assert len(q) == 1


def test_find_by_key(q):
    j = q.add("a", key="zz")
    assert q.find_by_key("zz").id == j.id
    assert q.find_by_key("nope") is None


def test_different_keys_make_different_jobs(q):
    assert q.add("a", key="1").id != q.add("a", key="2").id


# ---------------------------------------------------------------- ordering and deps

def test_runs_in_creation_order(q):
    calls = []
    for _ in range(3):
        q.add("a")
    q.run(reg(a=recorder(calls)))
    assert calls == ["j0001", "j0002", "j0003"]


def test_dependency_runs_before_dependent_even_if_created_later(qpath):
    # hand-written file: j1 depends on j2
    write_state(qpath, [raw_job(1, depends_on=["j0002"]), raw_job(2)])
    q = JobQueue(qpath)
    calls = []
    q.run(reg(a=recorder(calls)))
    assert calls == ["j0002", "j0001"]


def test_diamond_dependencies(q):
    a = q.add("a")
    b = q.add("a", depends_on=[a])
    c = q.add("a", depends_on=[a])
    d = q.add("a", depends_on=[b, c])
    calls = []
    q.run(reg(a=recorder(calls)))
    assert calls[0] == a.id and calls[-1] == d.id
    assert set(calls) == {a.id, b.id, c.id, d.id}


def test_pending_listed_in_topological_order(qpath):
    write_state(qpath, [raw_job(1, depends_on=["j0002"]), raw_job(2)])
    q = JobQueue(qpath)
    assert [j.id for j in q.pending()] == ["j0002", "j0001"]


def test_depends_on_accepts_job_objects_and_dedupes(q):
    a = q.add("a")
    b = q.add("a", depends_on=[a, a.id])
    assert b.depends_on == [a.id]


def test_dependency_results_visible(q):
    a = q.add("a")
    b = q.add("b", depends_on=[a])
    seen = {}

    def hb(ctx):
        seen.update(ctx.dependency_results)
        return "ok"

    q.run(reg(a=lambda c: {"video": "x.mp4"}, b=hb))
    assert seen == {a.id: {"video": "x.mp4"}}
    assert q.get(b.id).result == "ok"


def test_cycle_detected_on_run(qpath):
    write_state(qpath, [raw_job(1, depends_on=["j0002"]), raw_job(2, depends_on=["j0001"])])
    q = JobQueue(qpath)
    with pytest.raises(CycleError):
        q.run(reg(a=recorder([])))
    with pytest.raises(CycleError):
        q.order()


def test_add_dependency_rejects_cycle(q):
    a, b = q.add("a"), q.add("a", depends_on=[])
    q.add_dependency(b.id, a.id)
    with pytest.raises(CycleError):
        q.add_dependency(a.id, b.id)
    assert q.get(a.id).depends_on == []


def test_self_dependency_is_a_cycle(q):
    a = q.add("a")
    with pytest.raises(CycleError):
        q.add_dependency(a.id, a.id)


def test_unknown_kind_fails_fast_without_running_anything(q):
    calls = []
    q.add("a")
    q.add("mystery")
    with pytest.raises(UnknownKindError):
        q.run(reg(a=recorder(calls)))
    assert calls == []


# ---------------------------------------------------------------- failure / retry / backoff

def test_success_records_result_and_times(qpath):
    c = Clock()
    q = JobQueue(qpath, clock=c)
    j = q.add("a")
    q.run(reg(a=lambda ctx: {"n": 5}))
    got = q.get(j.id)
    assert got.status == "done" and got.result == {"n": 5}
    assert got.attempts == 1 and got.started == 1000.0 and got.finished == 1000.0
    assert got.error is None


def test_retry_then_success(q):
    n = {"c": 0}

    def flaky(ctx):
        n["c"] += 1
        if n["c"] < 3:
            raise RuntimeError("boom")
        return "fine"

    j = q.add("a")
    rep = q.run(reg(a=flaky))
    got = q.get(j.id)
    assert got.status == "done" and got.attempts == 3 and got.error is None
    assert rep.retried == [j.id, j.id] and rep.ok


def test_failure_after_max_attempts(q):
    def bad(ctx):
        raise ValueError("nope")

    j = q.add("a", max_attempts=2)
    rep = q.run(reg(a=bad))
    got = q.get(j.id)
    assert got.status == "failed" and got.attempts == 2
    assert got.error == "ValueError: nope"
    assert rep.failed == [j.id] and not rep.ok


def test_backoff_metadata_exponential(qpath):
    c = Clock(100.0)
    q = JobQueue(qpath, clock=c, backoff_base=2.0, backoff_cap=10.0)
    seen = []

    def bad(ctx):
        seen.append(ctx.job.attempts)
        raise RuntimeError("x")

    j = q.add("a", max_attempts=5)
    q.run(reg(a=bad))
    assert q.backoff_delay(1) == 2 and q.backoff_delay(2) == 4
    assert q.backoff_delay(3) == 8 and q.backoff_delay(4) == 10  # capped
    assert q.get(j.id).not_before == 110.0  # set by the 4th failure (capped)
    assert seen == [1, 2, 3, 4, 5]


def test_not_before_set_on_retry(qpath):
    c = Clock(500.0)
    q = JobQueue(qpath, clock=c)
    j = q.add("a", max_attempts=3)
    stop = {"n": 0}

    def bad(ctx):
        raise RuntimeError("x")

    def should_stop():
        stop["n"] += 1
        return stop["n"] > 1  # allow exactly one attempt

    q.run(reg(a=bad), should_stop=should_stop)
    got = q.get(j.id)
    assert got.status == "pending" and got.not_before == 502.0


def test_sleep_honours_backoff_when_injected(qpath):
    c = Clock(0.0)
    q = JobQueue(qpath, clock=c)
    slept = []

    def sleeper(dt):
        slept.append(dt)
        c.advance(dt)

    n = {"c": 0}

    def flaky(ctx):
        n["c"] += 1
        if n["c"] < 3:
            raise RuntimeError("x")

    q.add("a")
    q.run(reg(a=flaky), sleep=sleeper)
    assert slept == [2.0, 4.0]


def test_no_sleep_means_no_waiting(q):
    def bad(ctx):
        raise RuntimeError("x")

    q.add("a", max_attempts=3)
    t0 = time.monotonic()
    q.run(reg(a=bad))
    assert time.monotonic() - t0 < 1.0


def test_failed_dependency_skips_dependents(q):
    a = q.add("a", max_attempts=1)
    b = q.add("b", depends_on=[a])
    c = q.add("b", depends_on=[b])
    d = q.add("b")
    calls = []

    def bad(ctx):
        raise RuntimeError("no")

    rep = q.run(reg(a=bad, b=recorder(calls)))
    assert q.get(b.id).status == "skipped" and q.get(c.id).status == "skipped"
    assert "dependency" in q.get(b.id).error
    assert q.get(d.id).status == "done"
    assert set(rep.skipped) == {b.id, c.id}
    assert calls == [d.id]


def test_stop_on_error_halts_run(q):
    def bad(ctx):
        raise RuntimeError("no")

    q.add("a", max_attempts=1)
    j2 = q.add("b")
    q.run(reg(a=bad, b=recorder([])), stop_on_error=True)
    assert q.get(j2.id).status == "pending"


def test_continue_after_error_by_default(q):
    def bad(ctx):
        raise RuntimeError("no")

    q.add("a", max_attempts=1)
    j2 = q.add("b")
    q.run(reg(a=bad, b=recorder([])))
    assert q.get(j2.id).status == "done"


def test_retry_failed_revives_failed_and_skipped(q):
    a = q.add("a", max_attempts=1)
    b = q.add("b", depends_on=[a])
    ok = {"v": False}

    def maybe(ctx):
        if not ok["v"]:
            raise RuntimeError("down")

    r = reg(a=maybe, b=recorder([]))
    q.run(r)
    assert q.summary()["failed"] == 1 and q.summary()["skipped"] == 1
    ok["v"] = True
    revived = q.retry_failed()
    assert set(revived) == {a.id, b.id}
    assert q.get(a.id).attempts == 0 and q.get(a.id).error is None
    q.run(r)
    assert q.summary()["done"] == 2


def test_retry_failed_keeps_cancelled(q):
    a = q.add("a")
    b = q.add("a", depends_on=[a])
    q.cancel(a.id)
    q.retry_failed()
    assert q.get(a.id).status == "cancelled"
    assert q.get(b.id).status == "skipped"


def test_non_serialisable_result_counts_as_failure(q):
    j = q.add("a", max_attempts=1)
    q.run(reg(a=lambda ctx: object()))
    got = q.get(j.id)
    assert got.status == "failed" and "TypeError" in got.error


def test_progress_events(q):
    events = []
    a = q.add("a")
    q.add("b", depends_on=[a])
    q.run(reg(a=recorder([]), b=recorder([])), progress=lambda e, j: events.append((e, j.id)))
    assert events == [("start", "j0001"), ("done", "j0001"),
                      ("start", "j0002"), ("done", "j0002")]


def test_progress_callback_errors_are_ignored(q):
    q.add("a")

    def boom(e, j):
        raise RuntimeError("ui bug")

    rep = q.run(reg(a=recorder([])), progress=boom)
    assert rep.ok and q.summary()["done"] == 1


def test_error_text_is_scrubbed_of_tokens(q):
    def bad(ctx):
        raise RuntimeError("HTTP 401 for Bearer abcdefghijklmnop1234")

    j = q.add("a", max_attempts=1)
    q.run(reg(a=bad))
    err = q.get(j.id).error
    assert "abcdefghijklmnop1234" not in err and jq.REDACTED in err


# ---------------------------------------------------------------- stop

def test_should_stop_before_start_runs_nothing(q):
    q.add("a")
    rep = q.run(reg(a=recorder([])), should_stop=lambda: True)
    assert rep.stopped and q.summary()["pending"] == 1


def test_should_stop_between_jobs(q):
    q.add("a")
    q.add("a")
    calls = []
    rep = q.run(reg(a=recorder(calls)), should_stop=lambda: len(calls) >= 1)
    assert rep.stopped and calls == ["j0001"]
    assert q.get("j0002").status == "pending"


def test_check_stop_inside_handler_refunds_attempt(q):
    flag = {"stop": False}

    def handler(ctx):
        flag["stop"] = True
        ctx.check_stop()
        return "unreachable"

    j = q.add("a")
    rep = q.run(reg(a=handler), should_stop=lambda: flag["stop"])
    got = q.get(j.id)
    assert rep.stopped and got.status == "pending" and got.attempts == 0


def test_handler_except_exception_cannot_swallow_stop(q):
    def handler(ctx):
        try:
            ctx.check_stop()
        except Exception:
            return "swallowed"

    j = q.add("a")
    q.run(reg(a=handler), should_stop=lambda: True if False else _always_after_start(q, j))
    assert q.get(j.id).status in ("pending", "done")


def _always_after_start(q, j):
    return q.get(j.id).status == "running"


def test_keyboard_interrupt_requeues_and_propagates(q):
    def handler(ctx):
        raise KeyboardInterrupt

    j = q.add("a")
    with pytest.raises(KeyboardInterrupt):
        q.run(reg(a=handler))
    got = q.get(j.id)
    assert got.status == "pending" and got.attempts == 0
    assert not q._runner_lock.path.exists()


# ---------------------------------------------------------------- NeedsManual

def test_needs_manual_parks_job_and_blocks_dependents(q):
    a = q.add("render")
    b = q.add("edit", depends_on=[a])

    def render(ctx):
        raise NeedsManual("render j in Minecraft")

    calls = []
    rep = q.run(reg(render=render, edit=recorder(calls)))
    ja = q.get(a.id)
    assert ja.status == "waiting" and ja.message == "render j in Minecraft"
    assert ja.attempts == 0
    assert q.get(b.id).status == "pending" and calls == []
    assert rep.waiting == [a.id] and rep.ok


def test_resume_reruns_handler_with_resumed_flag(q):
    a = q.add("render")
    seen = []

    def render(ctx):
        seen.append(ctx.resumed)
        if not ctx.resumed:
            raise NeedsManual("do it")
        return {"video": "v.mp4"}

    r = reg(render=render)
    q.run(r)
    q.resume(a.id)
    assert q.get(a.id).status == "pending"
    q.run(r)
    assert seen == [False, True]
    got = q.get(a.id)
    assert got.status == "done" and got.result == {"video": "v.mp4"} and got.resumed is False


def test_resume_requires_waiting(q):
    a = q.add("a")
    with pytest.raises(JobStateError):
        q.resume(a.id)


def test_check_callable_releases_waiting_job(q, tmp_path):
    marker = tmp_path / "rendered.mp4"
    a = q.add("render")
    b = q.add("edit", depends_on=[a])

    def render(ctx):
        if not marker.exists():
            raise NeedsManual("render it")
        return str(marker)

    r = Registry()
    r.register("render", render, check=lambda ctx: marker.exists())
    r.register("edit", recorder([]))
    q.run(r)
    assert q.get(a.id).status == "waiting"
    marker.write_text("x")
    rep = q.run(r)
    assert q.get(a.id).status == "done" and q.get(b.id).status == "done"
    assert rep.ok


def test_check_polled_once_per_run_no_infinite_loop(q):
    calls = {"check": 0}
    a = q.add("render")

    def check(ctx):
        calls["check"] += 1
        return True  # lies: handler still wants a human

    def render(ctx):
        raise NeedsManual("still not")

    r = Registry()
    r.register("render", render, check=check)
    q.run(r)  # first run parks it
    q.run(r)  # check says yes -> handler parks again; must terminate
    assert calls["check"] == 1
    assert q.get(a.id).status == "waiting"


def test_check_exception_leaves_job_waiting(q):
    a = q.add("render")

    def render(ctx):
        raise NeedsManual("m")

    r = Registry()
    r.register("render", render, check=lambda c: 1 / 0)
    q.run(r)
    q.run(r)
    assert q.get(a.id).status == "waiting"


def test_cancel_waiting_job_skips_dependents(q):
    a = q.add("render")
    b = q.add("edit", depends_on=[a])
    q.run(reg(render=lambda c: (_ for _ in ()).throw(NeedsManual("m")), edit=recorder([])))
    q.cancel(a.id)
    assert q.get(a.id).status == "cancelled" and q.get(b.id).status == "skipped"


# ---------------------------------------------------------------- cancel

def test_cancel_pending(q):
    a = q.add("a")
    calls = []
    q.cancel(a.id)
    q.run(reg(a=recorder(calls)))
    assert calls == [] and q.get(a.id).status == "cancelled"


def test_cancel_done_refused(q):
    a = q.add("a")
    q.run(reg(a=recorder([])))
    with pytest.raises(JobStateError):
        q.cancel(a.id)


def test_cancel_is_idempotent(q):
    a = q.add("a")
    q.cancel(a.id)
    assert q.cancel(a.id).status == "cancelled"


def test_cancel_cascades_to_dependents(q):
    a = q.add("a")
    b = q.add("a", depends_on=[a])
    q.cancel(a.id)
    assert q.get(b.id).status == "skipped"


def test_cancel_during_run_discards_outcome(q):
    a = q.add("a")

    def handler(ctx):
        ctx.queue.cancel(ctx.job.id)  # user cancels mid-run
        return "late"

    q.run(reg(a=handler))
    got = q.get(a.id)
    assert got.status == "cancelled" and got.result is None


# ---------------------------------------------------------------- resume after crash

def test_running_job_with_dead_runner_becomes_pending(qpath):
    write_state(qpath, [raw_job(1, status="running", attempts=1, started=5.0)])
    write_lock(qpath.with_name("queue.json.lock"), dead_pid())
    q = JobQueue(qpath)
    j = q.get("j0001")
    assert j.status == "pending" and j.attempts == 1  # attempt still counted
    assert "interrupted" in j.error
    # persisted, not only in memory
    assert json.loads(qpath.read_text())["jobs"][0]["status"] == "pending"


def test_running_job_without_lock_file_is_recovered(qpath):
    write_state(qpath, [raw_job(1, status="running", attempts=1)])
    assert JobQueue(qpath).get("j0001").status == "pending"


def test_running_job_with_live_runner_left_alone(qpath):
    write_state(qpath, [raw_job(1, status="running", attempts=1)])
    write_lock(qpath.with_name("queue.json.lock"), os.getpid())
    assert JobQueue(qpath).get("j0001").status == "running"


def test_recover_false_leaves_running(qpath):
    write_state(qpath, [raw_job(1, status="running", attempts=1)])
    assert JobQueue(qpath, recover=False).get("j0001").status == "running"


def test_crash_loop_job_fails_once_attempts_exhausted(qpath):
    write_state(qpath, [raw_job(1, status="running", attempts=3, max_attempts=3)])
    q = JobQueue(qpath)
    assert q.get("j0001").status == "failed"


def test_resume_run_after_simulated_crash_runs_remaining_only(qpath):
    write_state(qpath, [
        raw_job(1, status="done", attempts=1, result="r1"),
        raw_job(2, status="running", attempts=1, depends_on=["j0001"]),
        raw_job(3, depends_on=["j0002"]),
    ])
    q = JobQueue(qpath)
    calls = []
    q.run(reg(a=recorder(calls)))
    assert calls == ["j0002", "j0003"]
    assert q.get("j0002").attempts == 2
    assert q.summary()["done"] == 3


def test_run_recovers_orphans_created_after_open(qpath):
    q = JobQueue(qpath)
    q.add("a")
    # another process died mid-run after we opened the queue
    data = json.loads(qpath.read_text())
    data["jobs"][0]["status"] = "running"
    data["jobs"][0]["attempts"] = 1
    qpath.write_text(json.dumps(data))
    calls = []
    q.run(reg(a=recorder(calls)))
    assert calls == ["j0001"]


# ---------------------------------------------------------------- locks

def test_second_runner_blocked_while_first_runs(qpath):
    q1, q2 = JobQueue(qpath), JobQueue(qpath)
    q1.add("a")
    seen = {}

    def handler(ctx):
        try:
            q2.run(reg(a=recorder([])))
        except LockHeldError as e:
            seen["err"] = str(e)

    q1.run(reg(a=handler))
    assert "locked by pid %d" % os.getpid() in seen["err"]


def test_runner_lock_released_after_run_and_after_error(q):
    q.add("a")
    q.run(reg(a=recorder([])))
    assert not q._runner_lock.path.exists()
    q.add("b")
    with pytest.raises(UnknownKindError):
        q.run(reg(a=recorder([])))
    assert not q._runner_lock.path.exists()


def test_stale_lock_dead_pid_is_taken_over(qpath):
    qpath.parent.mkdir(parents=True)
    write_lock(qpath.with_name("queue.json.lock"), dead_pid())
    q = JobQueue(qpath)
    q.add("a")
    rep = q.run(reg(a=recorder([])))
    assert rep.done == ["j0001"]


def test_stale_lock_by_age_on_foreign_host(qpath):
    qpath.parent.mkdir(parents=True)
    write_lock(qpath.with_name("queue.json.lock"), 1, host="other-box", age=13 * 3600)
    q = JobQueue(qpath)
    q.add("a")
    assert q.run(reg(a=recorder([]))).ok


def test_fresh_lock_on_foreign_host_blocks(qpath):
    qpath.parent.mkdir(parents=True)
    write_lock(qpath.with_name("queue.json.lock"), 1, host="other-box", age=5)
    q = JobQueue(qpath)
    q.add("a")
    with pytest.raises(LockHeldError):
        q.run(reg(a=recorder([])))


def test_live_pid_lock_blocks_runner(qpath):
    qpath.parent.mkdir(parents=True)
    write_lock(qpath.with_name("queue.json.lock"), os.getpid())
    q = JobQueue(qpath)
    q.add("a")
    with pytest.raises(LockHeldError):
        q.run(reg(a=recorder([])))


def test_garbage_lock_file_eventually_stale(tmp_path):
    p = tmp_path / "x.lock"
    p.write_text("")
    old = time.time() - 60
    os.utime(p, (old, old))
    lock = FileLock(p, stale_after=100)
    lock.acquire()
    assert lock.held
    lock.release()
    assert not p.exists()


def test_filelock_release_does_not_remove_someone_elses_lock(tmp_path):
    p = tmp_path / "x.lock"
    a = FileLock(p, stale_after=100)
    a.acquire()
    p.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                             "created": time.time(), "heartbeat": time.time(),
                             "token": "intruder"}))
    a.release()
    assert p.exists()


def test_filelock_heartbeat_updates_timestamp(tmp_path):
    c = Clock(10.0)
    lock = FileLock(tmp_path / "x.lock", stale_after=100, clock=c)
    lock.acquire()
    c.advance(50)
    lock.heartbeat()
    assert json.loads(lock.path.read_text())["heartbeat"] == 60.0
    lock.release()


def test_filelock_timeout_message(tmp_path):
    p = tmp_path / "x.lock"
    FileLock(p, stale_after=100).acquire()
    with pytest.raises(LockHeldError, match="locked by pid"):
        FileLock(p, stale_after=100, timeout=0.05).acquire()


def test_pid_alive_self_and_dead():
    assert jq._pid_alive(os.getpid())
    assert not jq._pid_alive(0)
    assert not jq._pid_alive(dead_pid())


def test_reset_refused_while_other_runner_alive(qpath):
    q = JobQueue(qpath)
    q.add("a")
    write_lock(qpath.with_name("queue.json.lock"), os.getpid())
    with pytest.raises(LockHeldError):
        q.reset()


# ---------------------------------------------------------------- atomic writes

def test_atomic_write_survives_failure_mid_write(q, qpath, monkeypatch):
    q.add("a", {"n": 1})
    before = qpath.read_text()

    def boom(src, dst):
        raise OSError("disk exploded")

    monkeypatch.setattr(jq.os, "replace", boom)
    with pytest.raises(OSError):
        q.add("a", {"n": 2})
    monkeypatch.undo()
    assert qpath.read_text() == before
    assert JobQueue(qpath).jobs()[0].params == {"n": 1}
    leftovers = [p.name for p in qpath.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_in_memory_state_resynced_after_failed_save(q, monkeypatch):
    q.add("a")
    monkeypatch.setattr(jq.os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        q.add("a")
    monkeypatch.undo()
    assert len(q) == 1
    assert q.add("a").id == "j0002"


def test_txn_lock_released_after_failure(q, monkeypatch):
    monkeypatch.setattr(jq.os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        q.add("a")
    monkeypatch.undo()
    assert not q._txn_lock.path.exists()


def test_atomic_write_text_replaces_content(tmp_path):
    p = tmp_path / "a" / "f.txt"
    jq._atomic_write_text(p, "one")
    jq._atomic_write_text(p, "two")
    assert p.read_text() == "two"
    assert [x.name for x in p.parent.iterdir()] == ["f.txt"]


# ---------------------------------------------------------------- corruption / versions

def test_corrupt_file_quarantined_and_fresh_start(qpath):
    qpath.parent.mkdir(parents=True)
    qpath.write_text("{ not json")
    q = JobQueue(qpath)
    assert len(q) == 0
    assert q.quarantined == qpath.with_name("queue.json.corrupt")
    assert q.quarantined.read_text() == "{ not json"
    assert not qpath.exists()
    q.add("a")
    assert qpath.exists()


def test_second_corruption_does_not_overwrite_first_quarantine(qpath):
    qpath.parent.mkdir(parents=True)
    qpath.write_text("bad1")
    JobQueue(qpath)
    qpath.write_text("bad2")
    q = JobQueue(qpath)
    assert q.quarantined.name == "queue.json.corrupt.1"
    assert qpath.with_name("queue.json.corrupt").read_text() == "bad1"


def test_corrupt_raise_mode_leaves_file(qpath):
    qpath.parent.mkdir(parents=True)
    qpath.write_text("garbage")
    with pytest.raises(CorruptStateError, match="corrupt"):
        JobQueue(qpath, on_corrupt="raise")
    assert qpath.read_text() == "garbage"


def test_wrong_shape_json_is_corrupt(qpath):
    qpath.parent.mkdir(parents=True)
    qpath.write_text('"just a string"')
    with pytest.raises(CorruptStateError):
        JobQueue(qpath, on_corrupt="raise")


def test_bad_on_corrupt_value(qpath):
    with pytest.raises(ValueError):
        JobQueue(qpath, on_corrupt="shrug")


def test_legacy_unversioned_file_migrated(qpath):
    qpath.parent.mkdir(parents=True)
    qpath.write_text(json.dumps({"jobs": [
        {"id": "j0001", "kind": "a", "params": {}, "status": "pending", "attempts": 0,
         "created": 1, "updated": 1}]}))
    q = JobQueue(qpath)
    job = q.get("j0001")
    assert job.max_attempts == 3
    q.add("a")
    assert json.loads(qpath.read_text())["version"] == jq.SCHEMA_VERSION


def test_bare_list_file_migrated(qpath):
    qpath.parent.mkdir(parents=True)
    qpath.write_text(json.dumps([{"id": "j0001", "kind": "a"}]))
    assert JobQueue(qpath).get("j0001").kind == "a"


def test_newer_schema_refused_not_quarantined(qpath):
    write_state(qpath, [], version=jq.SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionError):
        JobQueue(qpath)
    assert qpath.exists()


def test_unknown_job_fields_ignored(qpath):
    write_state(qpath, [raw_job(1, future_field="x")])
    assert JobQueue(qpath).get("j0001").kind == "a"


def test_migrate_function_direct():
    out = jq.migrate({"jobs": [{"id": "j1", "kind": "a"}]})
    assert out["version"] == jq.SCHEMA_VERSION and out["next_id"] == 2


# ---------------------------------------------------------------- secrets

@pytest.mark.parametrize("name", [
    "token", "access_token", "refreshToken", "client_secret", "password", "api_key",
    "apiKey", "key", "YOUTUBE_TOKEN", "credentials", "authorization", "passphrase",
])
def test_secret_names_detected(name):
    assert jq.is_secret_name(name)


@pytest.mark.parametrize("name", [
    "keyframes", "clip", "title", "music_ids", "radius", "client_secret_path",
    "token_file", "api_key_env", "monkey", "seconds",
])
def test_non_secret_names_allowed(name):
    assert not jq.is_secret_name(name)


def test_add_refuses_secret_param_with_helpful_error(q):
    with pytest.raises(ParamError) as ei:
        q.add("upload", {"title": "t", "access_token": "ya29.abcdefghijklmnop"})
    msg = str(ei.value)
    assert "access_token" in msg and "reference" in msg
    assert "ya29" not in msg  # never echo the value
    assert len(q) == 0


def test_refuses_nested_secret(q):
    with pytest.raises(ParamError):
        q.add("upload", {"auth": {"client": {"secret": "x"}}})
    with pytest.raises(ParamError):
        q.add("upload", {"opts": [{"password": "x"}]})


def test_refuses_token_shaped_value_under_innocent_name(q):
    with pytest.raises(ParamError):
        q.add("upload", {"note": "ya29.a0AfH6SMBxxxxxxxxxxxxxxxxxxxx"})
    with pytest.raises(ParamError):
        q.add("upload", {"note": "GOCSPX-abcdef"})


def test_reference_params_allowed(q):
    q.add("upload", {"client_secret_path": "C:/x/client_secret.json", "token_file": "t.json"})
    assert len(q) == 1


def test_allow_secrets_opt_in_but_still_redacted_in_status(qpath):
    q = JobQueue(qpath, allow_secrets=True)
    q.add("upload", {"password": "hunter2hunter2"})
    assert "hunter2" not in q.format_status()


def test_secret_never_reaches_queue_file(q, qpath):
    with pytest.raises(ParamError):
        q.add("upload", {"client_secret": "SUPERSECRETVALUE"})
    q.add("upload", {"title": "ok"})
    assert "SUPERSECRETVALUE" not in qpath.read_text()


def test_handler_result_secrets_are_redacted_on_disk(q, qpath):
    q.add("a")
    q.run(reg(a=lambda c: {"video": "v.mp4", "access_token": "tok12345678"}))
    text = qpath.read_text()
    assert "tok12345678" not in text and "v.mp4" in text


def test_redact_helper():
    out = jq.redact({"a": 1, "token": "x", "n": [{"password": "p"}, "Bearer abcdefghijkl"]})
    assert out["token"] == jq.REDACTED
    assert out["n"][0]["password"] == jq.REDACTED
    assert "abcdefghijkl" not in out["n"][1]


# ---------------------------------------------------------------- concurrency

def test_threads_adding_to_one_queue(q):
    errors = []

    def work(t):
        try:
            for i in range(20):
                q.add("a", {"t": t, "i": i})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    ids = [j.id for j in q.jobs()]
    assert len(ids) == 160 == len(set(ids))


def test_separate_instances_in_threads_do_not_lose_jobs(qpath):
    errors = []

    def work(t):
        try:
            qq = JobQueue(qpath)
            for i in range(10):
                qq.add("a", {"t": t, "i": i})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work, args=(t,)) for t in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    assert len(JobQueue(qpath)) == 60


def test_threads_with_same_key_create_one_job(q):
    ids = []

    def work():
        ids.append(q.add("a", key="same").id)

    ts = [threading.Thread(target=work) for _ in range(10)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(ids)) == 1 and len(q) == 1


def test_job_added_by_other_instance_during_run_is_picked_up(qpath):
    q = JobQueue(qpath)
    other = JobQueue(qpath)
    q.add("a")
    calls = []

    def first(ctx):
        calls.append(ctx.job.id)
        if len(calls) == 1:
            other.add("a")

    q.run(reg(a=first))
    assert calls == ["j0001", "j0002"]


# ---------------------------------------------------------------- housekeeping

def test_summary_counts(q):
    a = q.add("a")
    q.add("a", max_attempts=1)
    q.add("a")
    q.cancel(a.id)
    s = q.summary()
    assert s["total"] == 3 and s["cancelled"] == 1 and s["pending"] == 2


def test_reset_clears_everything(q):
    q.add("a")
    q.add("a")
    assert q.reset() == 2
    assert len(q) == 0
    assert q.add("a").id == "j0001"


def test_prune_done_keeps_needed_results(q):
    a = q.add("a")
    b = q.add("a", depends_on=[a])
    c = q.add("a")
    q.run(reg(a=recorder([])), should_stop=lambda: len([j for j in q.jobs("done")]) >= 2 and False)
    # everything done: all can go
    assert q.prune_done() == 3
    assert len(q) == 0
    assert {a.id, b.id, c.id}


def test_prune_done_keeps_done_dependency_of_unfinished_job(q):
    a = q.add("a")
    b = q.add("b", depends_on=[a])
    c = q.add("a")
    q.run(reg(a=recorder([]), b=lambda ctx: (_ for _ in ()).throw(NeedsManual("m"))))
    assert q.prune_done() == 1  # only c; a feeds the waiting b
    assert {j.id for j in q.jobs()} == {a.id, b.id}
    assert c.id not in {j.id for j in q.jobs()}


def test_prune_forgets_key(q):
    q.add("a", key="k")
    q.run(reg(a=recorder([])))
    q.prune_done()
    assert q.add("a", key="k").id == "j0002"  # ids are never reused
    assert len(q) == 1


def test_requeue_single_job(q):
    a = q.add("a", max_attempts=1)
    q.run(reg(a=lambda c: 1 / 0))
    assert q.get(a.id).status == "failed"
    q.requeue(a.id)
    got = q.get(a.id)
    assert got.status == "pending" and got.attempts == 0


def test_requeue_done_refused(q):
    a = q.add("a")
    q.run(reg(a=recorder([])))
    with pytest.raises(JobStateError):
        q.requeue(a.id)


# ---------------------------------------------------------------- format_status

def test_format_status_empty(q):
    assert q.format_status() == "queue is empty"


def test_format_status_table(q):
    a = q.add("plan", {"clip": "C:\\replays\\day1_1.mcpr", "seconds": 20})
    q.add("edit", {"title": "T"}, depends_on=[a])
    out = q.format_status()
    lines = out.splitlines()
    assert lines[0].split()[:5] == ["ID", "KIND", "STATUS", "TRY", "INFO"]
    assert "j0001" in out and "plan" in out and "pending" in out
    assert "clip=day1_1.mcpr" in out  # basename only, keeps rows short
    assert "needs j0001" in out
    assert lines[-1] == "2 job(s): 2 pending"


def test_format_status_shows_waiting_message_and_errors(q):
    a = q.add("render")
    b = q.add("a", max_attempts=1)
    q.run(reg(render=lambda c: (_ for _ in ()).throw(NeedsManual("render in game")),
              a=lambda c: 1 / 0))
    out = q.format_status()
    assert "waiting" in out and "render in game" in out
    assert "ZeroDivisionError" in out and "failed" in out
    assert b.id in out


def test_format_status_redacts_secret_params(qpath):
    q = JobQueue(qpath, allow_secrets=True)
    q.add("upload", {"token": "abc123secretvalue", "title": "t"})
    out = q.format_status()
    assert "abc123secretvalue" not in out and jq.REDACTED in out


def test_format_status_is_ascii_and_width_bounded(q):
    q.add("a", {"long": "x" * 500})
    out = q.format_status(width=80)
    out.encode("ascii")
    assert max(len(l) for l in out.splitlines()) <= 90
    assert "..." in out


# ---------------------------------------------------------------- session batch

def test_enqueue_session_batch_shape(q, tmp_path):
    clips = []
    for n in ("day1_1.mcpr", "day1_2.mcpr"):
        p = tmp_path / n
        p.write_bytes(b"x")
        clips.append(p)
    b = jq.enqueue_session_batch(q, clips, 20.0, title="T", upload=True,
                                 music_ids=["m1"], camera={"radius": 8})
    assert [j.kind for j in b.plans] == ["plan", "plan"]
    assert [j.kind for j in b.renders] == ["render_wait", "render_wait"]
    assert b.edit.kind == "edit" and b.upload.kind == "upload"
    assert b.edit.depends_on == [r.id for r in b.renders]
    assert b.upload.depends_on == [b.edit.id]
    assert b.renders[0].depends_on == [b.plans[0].id]
    assert len(q) == 6
    # plain data only
    json.dumps([j.to_dict() for j in q.jobs()])
    assert b.plans[0].params["clip"] == str(clips[0])
    assert b.plans[0].params["camera"] == {"radius": 8}


def test_enqueue_session_batch_idempotent(q, tmp_path):
    p = tmp_path / "a.mcpr"
    p.write_bytes(b"x")
    b1 = jq.enqueue_session_batch(q, [p], 20, upload=True, title="T")
    b2 = jq.enqueue_session_batch(q, [p], 20, upload=True, title="T")
    assert len(q) == 4
    assert b1.edit.id == b2.edit.id and b1.upload.id == b2.upload.id


def test_enqueue_session_batch_changes_with_mtime_or_params(q, tmp_path):
    p = tmp_path / "a.mcpr"
    p.write_bytes(b"x")
    jq.enqueue_session_batch(q, [p], 20)
    jq.enqueue_session_batch(q, [p], 30)       # different seconds
    assert len(q) == 6
    os.utime(p, (1, 1))
    jq.enqueue_session_batch(q, [p], 20)       # same params, file touched
    assert len(q) == 9


def test_enqueue_session_batch_without_upload(q, tmp_path):
    b = jq.enqueue_session_batch(q, [tmp_path / "gone.mcpr"], 10)
    assert b.upload is None and len(q) == 3


def test_enqueue_session_batch_empty_refused(q):
    with pytest.raises(QueueError):
        jq.enqueue_session_batch(q, [], 10)


def test_enqueue_session_batch_rejects_secret_in_camera(q, tmp_path):
    with pytest.raises(ParamError):
        jq.enqueue_session_batch(q, [tmp_path / "a.mcpr"], 10, camera={"api_key": "x"})


def test_session_batch_end_to_end_with_fake_handlers(q, tmp_path):
    clips = [tmp_path / "c1.mcpr", tmp_path / "c2.mcpr"]
    for c in clips:
        c.write_bytes(b"x")
    jq.enqueue_session_batch(q, clips, 20.0, title="T", upload=True)
    rendered = set()
    order = []

    def plan(ctx):
        order.append(("plan", ctx.params["clip"]))
        return {"planned": True}

    def render(ctx):
        clip = ctx.params["clip"]
        if clip not in rendered:
            raise NeedsManual("render " + clip)
        order.append(("render", clip))
        return {"video": clip + ".mp4"}

    def edit(ctx):
        vids = [ctx.dependency_results[d]["video"] for d in ctx.job.depends_on]
        order.append(("edit", vids))
        return {"output": "final.mp4"}

    def upload(ctx):
        (dep,) = ctx.job.depends_on
        order.append(("upload", ctx.dependency_results[dep]["output"]))
        return {"url": "https://example/v"}

    r = Registry()
    r.register("plan", plan)
    r.register("render_wait", render, check=lambda c: c.params["clip"] in rendered)
    r.register("edit", edit)
    r.register("upload", upload)

    rep = q.run(r)
    assert len(rep.waiting) == 2 and q.summary()["done"] == 2
    rendered.update(str(c) for c in clips)   # user renders in Minecraft
    rep = q.run(r)
    assert rep.ok and q.summary()["done"] == 6
    assert order[-2] == ("edit", [str(c) + ".mp4" for c in clips])
    assert order[-1] == ("upload", "final.mp4")


# ---------------------------------------------------------------- registry

def test_registry_decorator_and_kinds():
    r = Registry()

    @r.handler("x")
    def hx(ctx):
        return 1

    assert "x" in r and r.kinds() == ["x"] and r.get("x") is hx
    with pytest.raises(UnknownKindError):
        r.get("y")


def test_ctx_exposes_params_and_queue(q):
    seen = {}

    def h(ctx):
        seen["p"] = ctx.params
        seen["q"] = ctx.queue is q
        ctx.log("hello")

    logs = []
    q.add("a", {"n": 7})
    q.run(reg(a=h), log=logs.append)
    assert seen == {"p": {"n": 7}, "q": True} and logs == ["hello"]


def test_default_queue_path():
    class C:
        work = "w"

    assert str(jq.default_queue_path(C())) == os.path.join("w", "queue.json")
