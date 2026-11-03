"""Persistent, resumable batch job queue.

Why this exists: a montage batch is hours of wall-clock work spread over steps
that need a human in the middle (render each replay inside Minecraft), can fail
for transient reasons (YouTube upload), and runs on a laptop that gets closed
mid-run. The queue records every step in a small JSON file so that re-running
the same command picks up where the last one stopped instead of redoing, or
duplicating, work.

Design choices worth knowing before editing:

* The queue knows nothing about the pipeline. Job *kinds* are strings mapped to
  handler callables in a ``Registry``; tests use fake handlers and the heavy
  imports (ffmpeg, google client) stay in whoever registers the real ones.
* Every mutation is a short transaction: take a tiny cross-process file lock,
  re-read the file, change it, atomically replace it. That lets a second
  process ``add`` jobs or ``cancel`` while a runner is busy without either
  clobbering the other. The *runner* additionally holds a long-lived lock so
  two runners can never execute the same job.
* Locks are plain ``O_CREAT|O_EXCL`` files with pid/host/timestamp contents, so
  they work on Windows (the primary target) as well as POSIX; ``fcntl`` is never
  used. A lock is stale if its owner pid is dead (same host) or it is simply too
  old.
* Secrets are refused at ``add`` time. The queue file lives in the work folder
  next to renders and may be synced, pasted into bug reports, or committed;
  YouTube credentials must never be in it. Pass a *reference* (a path or env
  var name) and let the handler resolve it.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import heapq
import json
import os
import re
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# statuses

PENDING = "pending"
RUNNING = "running"
WAITING = "waiting"      # parked on a manual step (e.g. "render in Minecraft")
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"      # a dependency failed / was cancelled
CANCELLED = "cancelled"

ALL_STATUSES = (PENDING, RUNNING, WAITING, DONE, FAILED, SKIPPED, CANCELLED)
# A dependency in one of these states means the dependent can never run
# (until retry_failed() revives it).
_DEAD_END = frozenset({FAILED, SKIPPED, CANCELLED})


# --------------------------------------------------------------------------
# errors

class QueueError(Exception):
    """Base class for everything this module raises on purpose."""


class ParamError(QueueError, ValueError):
    """Job params are unusable: not JSON, or they contain a secret."""


class LockHeldError(QueueError):
    """Another live process holds the lock."""


class CycleError(QueueError):
    """Job dependencies form a cycle."""


class CorruptStateError(QueueError):
    """The queue file exists but cannot be parsed."""


class SchemaVersionError(QueueError):
    """The queue file was written by a newer version of this program."""


class JobStateError(QueueError):
    """The requested transition is not valid for the job's current status."""


class UnknownKindError(QueueError):
    """A job's kind has no handler in the registry."""


class NeedsManual(Exception):
    """Raised by a handler to park its job until a human (or a check) clears it.

    Not a failure: the attempt is refunded and dependents simply keep waiting.
    """

    def __init__(self, message: str = "manual step required"):
        super().__init__(message)
        self.message = message


class StopRequested(BaseException):
    """Raised by ``ctx.check_stop()``. BaseException so a handler's broad
    ``except Exception`` cannot accidentally swallow a stop request."""


# --------------------------------------------------------------------------
# secret hygiene

_SECRET_TOKENS = frozenset({
    "token", "tokens", "secret", "secrets", "password", "passwords", "passwd",
    "pwd", "passphrase", "key", "keys", "apikey", "credential", "credentials",
    "auth", "authorization", "bearer", "cookie", "cookies",
})
# Fused spellings (accesstoken, clientsecret, userpassword, xapikey, ...). Matched
# at the END of a name part only, so "tokenizer" and "secretary" stay allowed.
_SECRET_SUFFIXES = ("token", "secret", "password", "passwd", "apikey", "passphrase",
                    "secretkey", "privatekey", "accesskey", "authkey", "signingkey")
# Names that are secret only when they are the whole name ("pass" yes, "two_pass" no).
_SECRET_WHOLE_NAMES = frozenset({"pass", "pw", "creds", "cred"})
# A name ending in one of these is a *reference* to a secret, not the secret.
_REFERENCE_SUFFIXES = frozenset({"ref", "path", "file", "env", "name", "id", "url", "var"})
_VALUE_PATTERNS = re.compile(
    r"(ya29\.[0-9A-Za-z_\-]{10,}"            # Google OAuth access token
    r"|1//[0-9A-Za-z_\-]{20,}"               # Google refresh token
    r"|AIza[0-9A-Za-z_\-]{30,}"              # Google API key
    r"|GOCSPX-[0-9A-Za-z_\-]+"               # Google client secret
    r"|(?i:bearer)\s+[0-9A-Za-z._\-]{10,}"
    r"|(?i:authorization):\s*\w+\s+\S{8,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?<=://)[^/\s:@]+:[^/\s@]+@"           # user:password@host
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"          # AWS access key id
    r"|\bgh[pousr]_[0-9A-Za-z]{30,}"            # GitHub tokens
    r"|\bsk-(?:proj-|ant-)?[0-9A-Za-z_\-]{20,}\b"  # OpenAI/Anthropic style keys
    r"|\bxox[abprs]-[0-9A-Za-z\-]{10,}"         # Slack tokens
    r"|\beyJ[0-9A-Za-z_\-]{8,}\.[0-9A-Za-z_\-]{8,}\.[0-9A-Za-z_\-]{4,}"  # JWT
    r")"
)
REDACTED = "***REDACTED***"


def is_secret_name(name: str) -> bool:
    """True for names like ``token``, ``client_secret``, ``apiKey``.

    ``keyframes`` is deliberately NOT secret (it is a real camera option), and
    ``client_secret_path`` is allowed because it is a reference, which is what
    we want callers to pass.
    """
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if p]
    if not parts:
        return False
    if len(parts) > 1 and parts[-1] in _REFERENCE_SUFFIXES:
        return False
    if any(p in _SECRET_TOKENS for p in parts):
        return True
    joined = "".join(parts)
    if joined in _SECRET_WHOLE_NAMES:
        return True
    for p in parts:
        p = p.rstrip("0123456789")
        if p.endswith("s"):
            p = p[:-1]
        if p.endswith(_SECRET_SUFFIXES):
            return True
    return any(joined.endswith(x) for x in _SECRET_SUFFIXES)


def find_secrets(obj: Any, _path: str = "") -> list[str]:
    """Locations (key paths only, never values) that look like secrets."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{_path}.{k}" if _path else str(k)
            if is_secret_name(k):
                found.append(here)
            else:
                found.extend(find_secrets(v, here))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(find_secrets(v, f"{_path}[{i}]"))
    elif isinstance(obj, str) and _VALUE_PATTERNS.search(obj):
        found.append(_path or "<value>")
    return found


def redact(obj: Any) -> Any:
    """Copy of ``obj`` with secret-named values and token-shaped strings masked."""
    if isinstance(obj, dict):
        return {k: (REDACTED if is_secret_name(k) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _VALUE_PATTERNS.sub(REDACTED, obj)
    return obj


# --------------------------------------------------------------------------
# job model

@dataclass
class Job:
    id: str
    kind: str
    params: dict = field(default_factory=dict)
    status: str = PENDING
    attempts: int = 0
    error: Optional[str] = None
    result: Any = None
    depends_on: list = field(default_factory=list)
    created: float = 0.0
    updated: float = 0.0
    started: Optional[float] = None
    finished: Optional[float] = None
    # extras
    key: Optional[str] = None
    max_attempts: int = 3
    not_before: float = 0.0          # backoff metadata: don't retry before this
    message: Optional[str] = None    # why the job is WAITING
    resumed: bool = False            # a human said "I did the manual step"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        names = {f.name for f in dataclasses.fields(cls)}
        job = cls(**{k: v for k, v in d.items() if k in names})
        job.depends_on = list(job.depends_on or [])
        job.params = dict(job.params or {})
        return job

    def copy(self) -> "Job":
        return copy.deepcopy(self)


class Registry:
    """kind -> handler (and optional "is the manual step done yet?" check)."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable] = {}
        self._checks: dict[str, Callable] = {}

    def register(self, kind: str, handler: Callable, check: Optional[Callable] = None) -> "Registry":
        self._handlers[kind] = handler
        if check is not None:
            self._checks[kind] = check
        return self

    def handler(self, kind: str, check: Optional[Callable] = None) -> Callable:
        """Decorator form of ``register``."""
        def deco(fn: Callable) -> Callable:
            self.register(kind, fn, check)
            return fn
        return deco

    def get(self, kind: str) -> Callable:
        try:
            return self._handlers[kind]
        except KeyError:
            raise UnknownKindError(f"no handler registered for job kind {kind!r}") from None

    def check_for(self, kind: str) -> Optional[Callable]:
        return self._checks.get(kind)

    def kinds(self) -> list[str]:
        return sorted(self._handlers)

    def __contains__(self, kind: object) -> bool:
        return kind in self._handlers


@dataclass
class JobContext:
    """What a handler gets. ``job`` is a snapshot; mutate nothing on it."""

    job: Job
    queue: "JobQueue"
    log: Callable[[str], None]
    dependency_results: dict
    _should_stop: Optional[Callable[[], bool]] = None

    @property
    def params(self) -> dict:
        return self.job.params

    @property
    def resumed(self) -> bool:
        return self.job.resumed

    def check_stop(self) -> None:
        """Call between long steps; raises StopRequested if the runner was asked
        to stop. The job returns to pending with its attempt refunded."""
        if self._should_stop is not None and self._should_stop():
            raise StopRequested()


@dataclass
class RunReport:
    done: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    waiting: list = field(default_factory=list)
    retried: list = field(default_factory=list)
    stopped: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed


# --------------------------------------------------------------------------
# cross-platform process / lock helpers

_IS_WINDOWS = os.name == "nt"
_FS_RETRY_DELAY = 0.025
_FS_RETRY_ATTEMPTS = 40          # ~1s: how long a Windows sharing violation may last


def _retry_fs(fn: Callable[[], Any]) -> Any:
    """Run a filesystem call, retrying PermissionError on Windows only.

    Windows refuses to rename/delete a file that another process has open (and
    a freshly deleted file stays "delete pending" until every handle closes),
    which surfaces as PermissionError. Our lock and queue files are read by
    other processes all the time, so these are transient. On POSIX a
    PermissionError is real and propagates immediately.
    """
    for attempt in range(_FS_RETRY_ATTEMPTS):
        try:
            return fn()
        except PermissionError:
            if not _IS_WINDOWS or attempt == _FS_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_FS_RETRY_DELAY)


def _read_text(path: Path) -> str:
    return _retry_fs(lambda: Path(path).read_text(encoding="utf-8"))


def _pid_alive_windows(pid: int, kernel32: Any, get_last_error: Callable[[], int]) -> bool:
    """Windows liveness probe, split out so it can be tested with a fake."""
    import ctypes

    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        # 5 = ERROR_ACCESS_DENIED (exists, not ours); 87 = ERROR_INVALID_PARAMETER (gone)
        return get_last_error() == 5
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _windows_kernel32() -> Any:
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    # Without argtypes/restype ctypes truncates 64-bit HANDLEs to a C int.
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.GetExitCodeProcess.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    return k


def _is_zombie(pid: int) -> bool:
    """Linux: exited but not yet reaped by its parent. kill(pid, 0) succeeds on
    those, which would make a crashed runner's lock look held (e.g. under a
    container PID 1 that never reaps)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            stat = fh.read().decode("ascii", "replace")
        return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")
    except (OSError, IndexError):
        return False


def _pid_alive(pid: int) -> bool:
    """Is a process with this pid running? Works on Windows and POSIX."""
    if not isinstance(pid, int) or pid <= 0 or pid > 0xFFFFFFFF:
        return False
    if _IS_WINDOWS:
        import ctypes

        return _pid_alive_windows(pid, _windows_kernel32(), ctypes.get_last_error)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError, ValueError):
        return False
    return not _is_zombie(pid)


class FileLock:
    """Advisory lock implemented as an exclusively-created file.

    Chosen over fcntl/msvcrt so one code path serves both OSes and so the lock
    carries diagnostics (who holds it, since when) a user can read.

    Taking over a stale lock is done by *renaming* it to a private name and then
    re-checking what we actually moved. Unlinking by path would be a
    check-then-act race: two waiters both judge the same stale file stale, one
    finishes a full takeover, and the other then deletes the new live lock.
    """

    def __init__(
        self,
        path: Path,
        *,
        stale_after: float,
        timeout: float = 0.0,
        poll: float = 0.02,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.stale_after = stale_after
        self.timeout = timeout
        self.poll = poll
        self.clock = clock
        self.token = f"{os.getpid()}-{id(self)}-{time.time_ns()}"
        self.held = False
        self.lost = False   # set when a heartbeat finds someone else owns the file
        # heartbeat() may run on a background thread while release() runs
        self._mutex = threading.RLock()

    # -- inspection
    def _read(self, path: Optional[Path] = None) -> Optional[dict]:
        try:
            raw = _read_text(path or self.path)
        except FileNotFoundError:
            return None
        except OSError:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    def _is_stale(self, info: dict, path: Optional[Path] = None) -> bool:
        now = self.clock()
        if not info:
            # Unreadable/empty: probably mid-creation by another process. Only
            # treat as stale once it has had ample time to be written.
            try:
                age = time.time() - (path or self.path).stat().st_mtime
            except OSError:
                return False
            return age > 2.0
        pid = info.get("pid")
        same_host = info.get("host") == socket.gethostname()
        if same_host and isinstance(pid, int) and not _pid_alive(pid):
            return True
        beat = info.get("heartbeat", info.get("created", 0))
        try:
            return now - float(beat) > self.stale_after
        except (TypeError, ValueError):
            return True

    def holder(self) -> Optional[dict]:
        """Info about the live holder, or None if free or stale."""
        info = self._read()
        if info is None or self._is_stale(info):
            return None
        return info

    # -- acquire / release
    def _payload(self) -> str:
        now = self.clock()
        return json.dumps({
            "pid": os.getpid(), "host": socket.gethostname(),
            "created": now, "heartbeat": now, "token": self.token,
        })

    def _break_stale(self, seen: dict) -> None:
        """Remove the lock file we judged stale, without ever removing a live one."""
        victim = self.path.with_name(f"{self.path.name}.stale-{self.token}")
        try:
            _retry_fs(lambda: os.rename(self.path, victim))
        except FileNotFoundError:
            return                       # someone else already broke it
        except OSError:
            return                       # still in use; the caller re-polls
        moved = self._read(victim)
        if moved is None or self._is_stale(moved, victim):
            try:
                os.unlink(victim)
            except OSError:
                pass
            return
        # We moved a lock that is NOT stale (the owner took over / refreshed it
        # between our check and the rename). Put it back unless someone has
        # already created a new one (os.link never overwrites).
        try:
            os.link(victim, self.path)
        except (OSError, NotImplementedError, AttributeError):
            if not self.path.exists():
                try:
                    os.rename(victim, self.path)
                    return
                except OSError:
                    pass
        try:
            os.unlink(victim)
        except OSError:
            pass

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout
        perm_until: Optional[float] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = self._read()
                if info is None:
                    continue  # vanished between open and read; retry
                if self._is_stale(info):
                    self._break_stale(info)
                    continue
                if time.monotonic() >= deadline:
                    created = info.get("created", 0)
                    try:
                        when = time.strftime("%Y-%m-%d %H:%M:%S",
                                             time.localtime(float(created)))
                    except (TypeError, ValueError, OverflowError, OSError):
                        when = "?"
                    raise LockHeldError(
                        "locked by pid %s on %s since %s (%s)"
                        % (info.get("pid"), info.get("host"), when, self.path)
                    )
                time.sleep(self.poll)
                continue
            except PermissionError:
                # Windows reports "file exists but is delete-pending / open
                # without share-delete" as access denied rather than EEXIST.
                if not _IS_WINDOWS:
                    raise
                now = time.monotonic()
                if perm_until is None:
                    perm_until = now + _FS_RETRY_DELAY * _FS_RETRY_ATTEMPTS
                if now >= max(deadline, perm_until):
                    raise LockHeldError(f"lock file is busy ({self.path})") from None
                time.sleep(max(self.poll, _FS_RETRY_DELAY))
                continue
            try:
                os.write(fd, self._payload().encode("utf-8"))
            finally:
                os.close(fd)
            self.held = True
            self.lost = False
            return

    def heartbeat(self) -> None:
        """Refresh the timestamp so a long-lived holder is never judged stale.
        Sets ``lost`` if the file now belongs to someone else (or is gone)."""
        with self._mutex:
            if not self.held:
                return
            info = self._read()
            if info is None or (info and info.get("token") != self.token):
                self.lost = True
                return
            if not info:
                return                   # unreadable right now; try next beat
            info["heartbeat"] = self.clock()
            try:
                _atomic_write_text(self.path, json.dumps(info), fsync=False)
            except OSError:
                pass

    def release(self) -> None:
        with self._mutex:
            if not self.held:
                return
            self.held = False
            info = self._read()
            if info and info.get("token") == self.token:
                try:
                    _retry_fs(lambda: os.unlink(self.path))
                except OSError:
                    pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _atomic_write_text(path: Path, text: str, *, fsync: bool = True) -> None:
    """Write ``text`` so readers see the old file or the new one, never half.

    Temp file in the same directory (same filesystem, so os.replace is atomic),
    flushed and fsynced before the swap. On any failure the temp file is
    removed and the original is untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        _retry_fs(lambda: os.replace(tmp, path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if fsync and os.name != "nt":
        try:  # make the rename itself durable; not possible on Windows
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass


# --------------------------------------------------------------------------
# schema migration

def _migrate_0_to_1(state: dict) -> dict:
    """Legacy (unversioned) files -> v1. Stub: fills fields v1 requires."""
    jobs = state.get("jobs", [])
    for j in jobs:
        j.setdefault("max_attempts", 3)
        j.setdefault("not_before", 0.0)
        j.setdefault("depends_on", [])
    state["jobs"] = jobs
    state.setdefault("next_id", len(jobs) + 1)
    state["version"] = 1
    return state


MIGRATIONS: dict[int, Callable[[dict], dict]] = {0: _migrate_0_to_1}


def migrate(state: Any) -> dict:
    """Bring any readable state up to ``SCHEMA_VERSION`` or raise."""
    if isinstance(state, list):  # earliest format: a bare list of jobs
        state = {"jobs": state}
    if not isinstance(state, dict):
        raise CorruptStateError("queue file is not a JSON object")
    version = state.get("version", 0)
    if not isinstance(version, int) or version < 0:
        raise CorruptStateError(f"bad schema version {version!r}")
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"queue file is schema v{version}, this program understands up to "
            f"v{SCHEMA_VERSION}; upgrade mcmontage rather than risk losing jobs"
        )
    while version < SCHEMA_VERSION:
        state = MIGRATIONS[version](state)
        version = state["version"]
    return state


# --------------------------------------------------------------------------
# the queue

def default_queue_path(cfg: Any) -> Path:
    """Queue file location: ``<cfg.work>/queue.json``."""
    return Path(cfg.work) / "queue.json"


class JobQueue:
    def __init__(
        self,
        path: Any,
        *,
        on_corrupt: str = "quarantine",
        allow_secrets: bool = False,
        recover: bool = True,
        clock: Callable[[], float] = time.time,
        backoff_base: float = 2.0,
        backoff_cap: float = 300.0,
        runner_stale_after: float = 12 * 3600.0,
        txn_timeout: float = 15.0,
    ) -> None:
        if on_corrupt not in ("quarantine", "raise"):
            raise ValueError("on_corrupt must be 'quarantine' or 'raise'")
        self.path = Path(path)
        self.on_corrupt = on_corrupt
        self.allow_secrets = allow_secrets
        self.clock = clock
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.quarantined: Optional[Path] = None  # where a corrupt file went

        self._jobs: dict[str, Job] = {}
        self._next_id = 1
        self._tlock = threading.RLock()
        self._depth = 0
        self._dirty = False
        self._txn_lock = FileLock(
            self.path.with_name(self.path.name + ".txn.lock"),
            stale_after=30.0, timeout=txn_timeout, clock=clock)
        self._runner_lock = FileLock(
            self.path.with_name(self.path.name + ".lock"),
            stale_after=runner_stale_after, clock=clock)

        with self._tlock:
            self._reload()
            if recover and any(j.status == RUNNING for j in self._jobs.values()):
                # Only if no live runner owns them; otherwise "running" is true.
                # The check must happen INSIDE the txn: a runner that starts
                # between an outside check and the write would otherwise have
                # its legitimately-running job reset and run twice.
                with self._txn():
                    if self._runner_lock.holder() is None:
                        self._recover_running()

    # ---- persistence ----------------------------------------------------
    def _quarantine(self) -> Path:
        base = self.path.with_name(self.path.name + ".corrupt")
        dest, n = base, 0
        while dest.exists():
            n += 1
            dest = base.with_name(f"{base.name}.{n}")
        _retry_fs(lambda: os.replace(self.path, dest))
        self.quarantined = dest
        return dest

    def _quarantine_if_unchanged(self, raw: str) -> bool:
        """Move the corrupt file aside, but only while holding the txn lock and
        only if it still holds exactly the bytes we judged corrupt. Otherwise
        another process has already replaced it with a good file, and moving
        that would destroy its data. Returns False if the file changed."""
        taken = False
        if self._depth == 0:
            self._txn_lock.acquire()
            taken = True
        try:
            try:
                current = _read_text(self.path)
            except FileNotFoundError:
                return True
            if current != raw:
                return False
            self._quarantine()
            return True
        finally:
            if taken:
                self._txn_lock.release()

    def _reload(self) -> None:
        for _ in range(5):
            try:
                raw = _read_text(self.path)
            except FileNotFoundError:
                self._jobs, self._next_id = {}, 1
                return
            try:
                state = migrate(json.loads(raw))
                jobs = [Job.from_dict(d) for d in state["jobs"]]
                next_id = int(state.get("next_id", len(jobs) + 1))
            except SchemaVersionError:
                raise
            except (ValueError, TypeError, KeyError, AttributeError, CorruptStateError) as exc:
                if self.on_corrupt == "raise":
                    raise CorruptStateError(
                        f"{self.path} is corrupt ({exc}); move it aside or open the "
                        f"queue with on_corrupt='quarantine' to start fresh") from exc
                if not self._quarantine_if_unchanged(raw):
                    continue                 # someone fixed it meanwhile; re-read
                self._jobs, self._next_id = {}, 1
                return
            self._jobs = {j.id: j for j in jobs}
            highest = 0
            for jid in self._jobs:
                m = re.fullmatch(r"j(\d+)", jid)
                if m:
                    highest = max(highest, int(m.group(1)))
            self._next_id = max(next_id, highest + 1, 1)
            return
        raise CorruptStateError(f"{self.path} keeps changing while being read")

    def _save(self) -> None:
        state = {
            "version": SCHEMA_VERSION,
            "next_id": self._next_id,
            "saved": self.clock(),
            "jobs": [j.to_dict() for j in self._jobs.values()],
        }
        _atomic_write_text(self.path, json.dumps(state, indent=2, sort_keys=False))

    @contextmanager
    def _txn(self) -> Iterator[None]:
        """Lock, re-read, yield, save if changed. Re-entrant."""
        with self._tlock:
            outer = self._depth == 0
            if outer:
                self._txn_lock.acquire()
            self._depth += 1
            ok = False
            try:
                if outer:
                    self._dirty = False
                    self._reload()
                yield
                if outer and self._dirty:
                    self._save()
                ok = True
            finally:
                self._depth -= 1
                if outer:
                    try:
                        if not ok:
                            # Failure (or failed save): drop half-applied
                            # in-memory changes by re-reading the real file.
                            self._dirty = False
                            try:
                                self._reload()
                            except QueueError:
                                pass
                    finally:
                        self._txn_lock.release()

    @contextmanager
    def _view(self) -> Iterator[None]:
        """Read-only fresh view. Lock-free: replace() makes the file consistent."""
        with self._tlock:
            if self._depth == 0:
                self._reload()
            yield

    def _touch(self, job: Job) -> None:
        job.updated = self.clock()
        self._dirty = True

    # ---- crash recovery -------------------------------------------------
    def _recover_running(self) -> list[str]:
        """Turn orphaned RUNNING jobs back into PENDING (the attempt stays
        counted, so a job that kills its runner every time cannot loop forever).
        Must be called inside a txn and only when no live runner exists."""
        revived = []
        for job in self._jobs.values():
            if job.status != RUNNING:
                continue
            job.error = "interrupted: runner stopped while this job was running"
            if job.attempts >= job.max_attempts:
                job.status = FAILED
                job.finished = self.clock()
            else:
                job.status = PENDING
                job.not_before = 0.0
                revived.append(job.id)
            self._touch(job)
        self._propagate_skips()         # dependents of a job that just failed
        return revived

    # ---- adding / lookup -----------------------------------------------
    def _check_params(self, params: Any) -> dict:
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ParamError("params must be a dict")
        try:
            json.dumps(params)
        except (TypeError, ValueError) as exc:
            raise ParamError(f"params must be JSON-serialisable: {exc}") from exc
        if not self.allow_secrets:
            bad = find_secrets(params)
            if bad:
                raise ParamError(
                    "refusing to store secret-looking params (%s) in the queue "
                    "file. Pass a reference instead, e.g. a file path or an "
                    "environment variable name, and resolve it inside the "
                    "handler." % ", ".join(bad))
        return json.loads(json.dumps(params))

    def add(
        self,
        kind: str,
        params: Optional[dict] = None,
        depends_on: Iterable[Any] = (),
        max_attempts: int = 3,
        key: Optional[str] = None,
    ) -> Job:
        if not isinstance(kind, str) or not kind:
            raise ValueError("kind must be a non-empty string")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        clean = self._check_params(params)
        deps = [d.id if isinstance(d, Job) else str(d) for d in depends_on]
        with self._txn():
            if key is not None:
                for existing in self._jobs.values():
                    if existing.key == key:
                        return existing.copy()
            missing = [d for d in deps if d not in self._jobs]
            if missing:
                raise QueueError("unknown dependency job(s): " + ", ".join(missing))
            now = self.clock()
            job = Job(
                id="j%04d" % self._next_id, kind=kind, params=clean,
                depends_on=list(dict.fromkeys(deps)), created=now, updated=now,
                key=key, max_attempts=max_attempts)
            self._next_id += 1
            self._jobs[job.id] = job
            self._dirty = True
            self._propagate_skips()
            return self._jobs[job.id].copy()

    def add_dependency(self, job_id: str, dep_id: str) -> None:
        """Add an edge after the fact; refuses if it would create a cycle."""
        with self._txn():
            job, dep = self._require(job_id), self._require(dep_id)
            if dep.id in job.depends_on:
                return
            job.depends_on.append(dep.id)
            try:
                self._order()
            except CycleError:
                job.depends_on.remove(dep.id)
                raise
            self._touch(job)

    def _require(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise QueueError(f"no such job: {job_id}") from None

    def get(self, job_id: str) -> Job:
        with self._view():
            return self._require(job_id).copy()

    def find_by_key(self, key: str) -> Optional[Job]:
        with self._view():
            for j in self._jobs.values():
                if j.key == key:
                    return j.copy()
        return None

    def jobs(self, status: Optional[str] = None) -> list[Job]:
        with self._view():
            return [j.copy() for j in self._jobs.values()
                    if status is None or j.status == status]

    def __len__(self) -> int:
        with self._view():
            return len(self._jobs)

    def pending(self) -> list[Job]:
        """Pending jobs in execution (topological) order."""
        with self._view():
            order = self._order()
            return [self._jobs[i].copy() for i in order if self._jobs[i].status == PENDING]

    def summary(self) -> dict:
        with self._view():
            out = {s: 0 for s in ALL_STATUSES}
            for j in self._jobs.values():
                out[j.status] = out.get(j.status, 0) + 1
            out["total"] = len(self._jobs)
            return out

    # ---- ordering -------------------------------------------------------
    def _order(self) -> list[str]:
        """Topological order, stable w.r.t. creation order. Raises CycleError."""
        indeg = {i: 0 for i in self._jobs}
        children: dict[str, list[str]] = {i: [] for i in self._jobs}
        for j in self._jobs.values():
            for d in j.depends_on:
                if d in self._jobs:
                    indeg[j.id] += 1
                    children[d].append(j.id)
        position = {i: n for n, i in enumerate(self._jobs)}
        ready = [(position[i], i) for i, n in indeg.items() if n == 0]
        heapq.heapify(ready)
        out: list[str] = []
        while ready:
            _, cur = heapq.heappop(ready)
            out.append(cur)
            for c in children[cur]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    heapq.heappush(ready, (position[c], c))
        if len(out) != len(self._jobs):
            done_ids = set(out)
            stuck = [i for i in self._jobs if i not in done_ids]
            raise CycleError("dependency cycle among jobs: " + ", ".join(stuck))
        return out

    def order(self) -> list[str]:
        with self._view():
            return self._order()

    # ---- state transitions ---------------------------------------------
    def _propagate_skips(self) -> list[str]:
        """Skip any not-yet-run job whose dependency can never complete."""
        newly: list[str] = []
        try:
            order = self._order()
        except CycleError:
            return newly
        for jid in order:
            job = self._jobs[jid]
            if job.status not in (PENDING, WAITING):
                continue
            for d in job.depends_on:
                dep = self._jobs.get(d)
                if dep is None or dep.status in _DEAD_END:
                    why = "missing" if dep is None else dep.status
                    job.status = SKIPPED
                    job.error = f"dependency {d} {why}"
                    job.finished = self.clock()
                    self._touch(job)
                    newly.append(jid)
                    break
        return newly

    def cancel(self, job_id: str) -> Job:
        """Cancel a job. Cancelling one that is mid-run is allowed: the runner
        discards its outcome when it finishes. Dependents become skipped."""
        with self._txn():
            job = self._require(job_id)
            if job.status == CANCELLED:
                return job.copy()
            if job.status == DONE:
                raise JobStateError(f"{job_id} is already done and cannot be cancelled")
            job.status = CANCELLED
            job.finished = self.clock()
            job.message = None
            self._touch(job)
            self._propagate_skips()
            return job.copy()

    def resume(self, job_id: str) -> Job:
        """Release a WAITING job: the human did the manual step. The handler
        re-runs and can see ``ctx.resumed``."""
        with self._txn():
            job = self._require(job_id)
            if job.status != WAITING:
                raise JobStateError(f"{job_id} is {job.status}, not waiting")
            job.status = PENDING
            job.resumed = True
            job.message = None
            self._touch(job)
            return job.copy()

    def requeue(self, job_id: str) -> Job:
        """Put any non-running, non-done job back to pending with fresh attempts."""
        with self._txn():
            job = self._require(job_id)
            if job.status in (RUNNING, DONE):
                raise JobStateError(f"{job_id} is {job.status}")
            self._revive(job)
            self._propagate_skips()
            return self._jobs[job_id].copy()

    def _revive(self, job: Job) -> None:
        job.status = PENDING
        job.attempts = 0
        job.error = None
        job.not_before = 0.0
        job.finished = None
        self._touch(job)

    def retry_failed(self) -> list[str]:
        """Revive failed jobs and the dependents that were skipped because of
        them. Cancelled jobs stay cancelled (that was a deliberate choice), and
        anything blocked by one is re-skipped. Returns the ids now pending."""
        with self._txn():
            revived = []
            for j in self._jobs.values():
                if j.status in (FAILED, SKIPPED):
                    self._revive(j)
                    revived.append(j.id)
            self._propagate_skips()
            return [i for i in revived if self._jobs[i].status == PENDING]

    def reset(self) -> int:
        """Delete every job. Refuses while a live runner exists."""
        with self._txn():
            if self._runner_lock.holder() is not None and not self._runner_lock.held:
                raise LockHeldError("a runner is active; stop it before reset()")
            n = len(self._jobs)
            self._jobs.clear()
            self._next_id = 1
            self._dirty = True
            return n

    def prune_done(self) -> int:
        """Delete finished jobs nothing unfinished still needs (dependents may
        read their results). Note: pruning forgets the idempotency key too."""
        with self._txn():
            removable = {i for i, j in self._jobs.items() if j.status == DONE}
            changed = True
            while changed:
                changed = False
                for j in self._jobs.values():
                    if j.id not in removable:
                        for d in j.depends_on:
                            if d in removable:
                                removable.discard(d)
                                changed = True
            for i in removable:
                del self._jobs[i]
            if removable:
                self._dirty = True
            return len(removable)

    # ---- backoff --------------------------------------------------------
    def backoff_delay(self, attempts: int) -> float:
        """Seconds to wait after the ``attempts``-th failure (2, 4, 8 ... capped)."""
        return min(self.backoff_cap, self.backoff_base * (2 ** max(0, attempts - 1)))

    # ---- running --------------------------------------------------------
    def run(
        self,
        registry: Registry,
        *,
        stop_on_error: bool = False,
        progress: Optional[Callable[[str, Job], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        log: Optional[Callable[[str], None]] = None,
        lock_timeout: float = 0.0,
    ) -> RunReport:
        """Execute runnable jobs in dependency order until none remain.

        ``sleep``: if None, retry backoff is recorded in ``not_before`` but not
        honoured (retries run straight away) so tests and dry runs never block.
        Pass ``time.sleep`` for real waiting.

        ``progress(event, job)`` gets: start, done, retry, failed, waiting,
        skipped, stopped, discarded. Exceptions in the callback are ignored.
        """
        logf = log or (lambda m: None)
        report = RunReport()

        def emit(event: str, job: Job) -> None:
            if progress is not None:
                try:
                    progress(event, job)
                except Exception:
                    pass

        runner = self._runner_lock
        runner.timeout = lock_timeout
        runner.acquire()
        # A handler may run for hours; without a background beat the lock would
        # look stale (by age) to every other process and be taken over.
        beat_stop = threading.Event()
        beat_every = max(0.05, min(30.0, runner.stale_after / 4.0))

        def beat() -> None:
            while not beat_stop.wait(beat_every):
                runner.heartbeat()

        beater = threading.Thread(target=beat, name="jobqueue-heartbeat", daemon=True)
        beater.start()
        try:
            with self._view():
                order_ids = self._order()  # cycle check up front
                needed = {self._jobs[i].kind for i in order_ids
                          if self._jobs[i].status in (PENDING, WAITING, RUNNING)}
            unknown = sorted(k for k in needed if k not in registry)
            if unknown:
                raise UnknownKindError("no handler registered for job kind(s): "
                                       + ", ".join(unknown))
            with self._txn():
                self._recover_running()
            polled: set[str] = set()

            while True:
                runner.heartbeat()
                if runner.lost:
                    # Another process took over our lock (we were suspended past
                    # the stale timeout). Running on would double-run its jobs.
                    logf("runner lock was taken over by another process; stopping")
                    report.stopped = True
                    break
                if should_stop is not None and should_stop():
                    report.stopped = True
                    break

                with self._txn():
                    for jid in self._propagate_skips():
                        report.skipped.append(jid)
                        emit("skipped", self._jobs[jid].copy())
                    to_poll = [j.copy() for j in self._jobs.values()
                               if j.status == WAITING and j.id not in polled
                               and registry.check_for(j.kind)]
                if to_poll:
                    for j in to_poll:
                        polled.add(j.id)
                        ctx = self._make_ctx(j, logf, should_stop)
                        try:
                            ok = bool(registry.check_for(j.kind)(ctx))
                        except StopRequested:
                            report.stopped = True
                            break
                        except Exception as exc:
                            logf(f"check for {j.id} raised {exc!r}; staying waiting")
                            ok = False
                        if ok:
                            with self._txn():
                                cur = self._jobs.get(j.id)
                                if cur is not None and cur.status == WAITING:
                                    cur.status = PENDING
                                    cur.message = None
                                    self._touch(cur)
                    if report.stopped:
                        break
                    continue

                claimed, wait = self._claim(sleep is not None)
                if claimed is None:
                    if wait is not None and sleep is not None:
                        sleep(wait)
                        continue
                    break

                job, ctx_deps = claimed
                emit("start", job)
                ctx = JobContext(job.copy(), self, logf, ctx_deps, should_stop)
                outcome: tuple
                try:
                    result = registry.get(job.kind)(ctx)
                    # Normalise to exactly what the file will hold (tuple->list,
                    # int keys->str) so dependents see the same value before and
                    # after a reload; also rejects non-serialisable results.
                    outcome = ("done", json.loads(json.dumps(result)))
                except NeedsManual as nm:
                    outcome = ("waiting", nm.message)
                except StopRequested:
                    outcome = ("stopped", None)
                except KeyboardInterrupt:
                    self._finish(job, ("stopped", None), report, emit)
                    raise
                except Exception as exc:
                    outcome = ("error", exc)
                self._finish(job, outcome, report, emit)
                if outcome[0] == "waiting":
                    polled.add(job.id)  # it just told us it is not ready
                if outcome[0] == "stopped":
                    report.stopped = True
                    break
                if outcome[0] == "error" and stop_on_error and report.failed:
                    break
        finally:
            beat_stop.set()
            beater.join(timeout=5.0)
            runner.release()
        with self._view():
            report.waiting = [j.id for j in self._jobs.values() if j.status == WAITING]
        return report

    def _make_ctx(self, job: Job, logf: Callable, should_stop: Optional[Callable]) -> JobContext:
        deps = {d: copy.deepcopy(self._jobs[d].result) for d in job.depends_on if d in self._jobs}
        return JobContext(job, self, logf, deps, should_stop)

    def _claim(self, honour_backoff: bool) -> tuple[Optional[tuple], Optional[float]]:
        """Pick and mark the next runnable job. Returns ((job, dep_results), None),
        or (None, seconds_to_wait) when only backed-off jobs remain, or (None, None)."""
        with self._txn():
            now = self.clock()
            soonest: Optional[float] = None
            for jid in self._order():
                job = self._jobs[jid]
                if job.status != PENDING:
                    continue
                if any(getattr(self._jobs.get(d), "status", None) != DONE for d in job.depends_on):
                    continue
                delay = job.not_before - now
                # A delay longer than the cap cannot have come from our own
                # backoff: the wall clock was stepped back since it was
                # recorded. Waiting it out would stall for hours/days.
                if honour_backoff and 0 < delay <= self.backoff_cap:
                    soonest = delay if soonest is None else min(soonest, delay)
                    continue
                job.status = RUNNING
                job.attempts += 1
                job.started = now
                job.finished = None
                self._touch(job)
                deps = {d: copy.deepcopy(self._jobs[d].result) for d in job.depends_on}
                return (job.copy(), deps), None
            return None, soonest

    def _finish(self, claimed: Job, outcome: tuple, report: RunReport, emit: Callable) -> None:
        jid = claimed.id
        kind, payload = outcome
        with self._txn():
            job = self._jobs.get(jid)
            if (job is None or job.status != RUNNING
                    or job.attempts != claimed.attempts or job.started != claimed.started):
                # Cancelled/removed by someone else while it ran, or re-claimed
                # by a runner that took over our lock (different attempt/start):
                # their word wins, never overwrite their attempt with ours.
                if job is not None:
                    emit("discarded", job.copy())
                return
            now = self.clock()
            if kind == "done":
                job.status = DONE
                job.result = redact(payload) if not self.allow_secrets else payload
                job.error = None
                job.message = None
                job.resumed = False
                job.finished = now
                report.done.append(jid)
                event = "done"
            elif kind == "waiting":
                job.status = WAITING
                job.attempts = max(0, job.attempts - 1)  # not a failed attempt
                job.message = _scrub_error(str(payload))
                job.resumed = False
                event = "waiting"
            elif kind == "stopped":
                job.status = PENDING
                job.attempts = max(0, job.attempts - 1)
                event = "stopped"
            else:
                exc = payload
                job.error = _scrub_error(f"{type(exc).__name__}: {exc}")
                if job.attempts < job.max_attempts:
                    job.status = PENDING
                    job.not_before = now + self.backoff_delay(job.attempts)
                    report.retried.append(jid)
                    event = "retry"
                else:
                    job.status = FAILED
                    job.finished = now
                    report.failed.append(jid)
                    event = "failed"
            self._touch(job)
            snapshot = job.copy()
            if job.status == FAILED:
                for sid in self._propagate_skips():
                    report.skipped.append(sid)
                    emit("skipped", self._jobs[sid].copy())
        emit(event, snapshot)

    # ---- display --------------------------------------------------------
    def format_status(self, width: int = 100) -> str:
        """Plain-ASCII table (Windows consoles choke on box drawing)."""
        with self._view():
            jobs = [self._jobs[i] for i in self._order()] if self._jobs else []
            if not jobs:
                return "queue is empty"
            rows = []
            for j in jobs:
                if j.status == WAITING:
                    info = j.message or "manual step"
                elif j.error:
                    info = j.error
                else:
                    info = ", ".join(f"{k}={_short(v)}" for k, v in redact(j.params).items())
                undone = [d for d in j.depends_on
                          if d in self._jobs and self._jobs[d].status != DONE]
                if j.status == PENDING and undone:
                    info = "needs " + ",".join(undone) + ("; " + info if info else "")
                rows.append([j.id, j.kind, j.status,
                             f"{j.attempts}/{j.max_attempts}", info])
            head = ["ID", "KIND", "STATUS", "TRY", "INFO"]
            widths = [max(len(head[c]), *(len(r[c]) for r in rows)) for c in range(4)]
            room = max(10, width - sum(widths) - 2 * 4)
            lines = []
            fmt = "  ".join("%-" + str(w) + "s" for w in widths) + "  %s"
            lines.append(fmt % tuple(head))
            lines.append(fmt % (*("-" * w for w in widths), "----"))
            for r in rows:
                lines.append(fmt % (*r[:4], _clip(r[4], room)))
            s = self.summary()
            parts = [f"{s[k]} {k}" for k in ALL_STATUSES if s[k]]
            lines.append("")
            lines.append(f"{s['total']} job(s): " + ", ".join(parts))
            return "\n".join(l.rstrip() for l in lines)


def _short(v: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    if isinstance(v, str) and ("/" in s or "\\" in s):
        s = re.split(r"[\\/]", s.rstrip("\\/"))[-1]  # either separator, on any OS
    return _clip(s, 30)


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: max(0, n - 3)] + "..."


def _scrub_error(text: str) -> str:
    return _VALUE_PATTERNS.sub(REDACTED, text)[:2000]


# --------------------------------------------------------------------------
# convenience: session batch -> job chains

@dataclass
class BatchJobs:
    plans: list
    renders: list
    edit: Job
    upload: Optional[Job]


def _fingerprint(path: Any, extra: Any) -> str:
    """Stable id for 'this file in this state with these settings'."""
    p = Path(path)
    try:
        st = p.stat()
        stamp = f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        stamp = "missing"
    blob = json.dumps([str(p.absolute()), stamp, extra], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def enqueue_session_batch(
    queue: JobQueue,
    clips: Sequence[Any],
    seconds: float,
    *,
    max_speedup: Optional[float] = None,
    camera: Optional[dict] = None,
    output: Optional[Any] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    music_ids: Optional[Sequence[str]] = None,
    xfade: Optional[float] = None,
    upload: bool = False,
    description: str = "",
    tags: Optional[Sequence[str]] = None,
) -> BatchJobs:
    """Queue plan -> render (manual wait) -> edit -> [upload] for a set of clips.

    Chain shape: one ``plan`` and one ``render_wait`` per clip, then a single
    ``edit`` depending on every render (clips joined in the given order), then
    optionally ``upload`` depending on the edit.

    Handler contract (for whoever registers real handlers):
      plan        params: clip, seconds, max_speedup, camera
                  -> result: anything JSON (e.g. the prepare summary)
      render_wait params: clip.  Raise NeedsManual until the rendered video
                  exists; return {"video": "<path>"}.
      edit        params: output, title, subtitle, music_ids, xfade; inputs are
                  the ``video`` of each render dependency in ``depends_on``
                  order -> {"output": "<path>"}
      upload      params: title, description, tags, music_ids; input is the
                  edit's ``output``.  Credentials are NOT params: the handler
                  loads them from config/disk (the queue refuses secrets).

    Idempotent: keys derive from clip path + mtime + settings, so running the
    same batch command twice returns the same jobs. Edit the clip or change a
    setting and a fresh chain is created. Paths are stored as strings.
    """
    if not clips:
        raise QueueError("no clips to enqueue")
    camera = dict(camera or {})
    plans, renders = [], []
    plan_settings = {"seconds": seconds, "max_speedup": max_speedup, "camera": camera}
    for clip in clips:
        fp = _fingerprint(clip, plan_settings)
        name = Path(clip).name
        plan = queue.add(
            "plan", {"clip": str(clip), **plan_settings}, key=f"plan:{name}:{fp}")
        render = queue.add(
            "render_wait", {"clip": str(clip)}, depends_on=[plan],
            key=f"render:{name}:{fp}", max_attempts=1000)
        plans.append(plan)
        renders.append(render)

    edit_settings = {
        "output": str(output) if output else None, "title": title,
        "subtitle": subtitle, "music_ids": list(music_ids or []), "xfade": xfade,
    }
    edit_fp = hashlib.sha1(json.dumps(
        [[r.key for r in renders], edit_settings], sort_keys=True).encode()).hexdigest()[:12]
    edit = queue.add("edit", edit_settings, depends_on=renders, key=f"edit:{edit_fp}")

    up = None
    if upload:
        up_settings = {
            "title": title or "", "description": description,
            "tags": list(tags or []), "music_ids": list(music_ids or []),
        }
        up_fp = hashlib.sha1(json.dumps(
            [edit.key, up_settings], sort_keys=True).encode()).hexdigest()[:12]
        up = queue.add("upload", up_settings, depends_on=[edit], key=f"upload:{up_fp}")
    return BatchJobs(plans=plans, renders=renders, edit=edit, upload=up)
