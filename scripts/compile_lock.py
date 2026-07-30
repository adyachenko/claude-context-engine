"""Project-level singleflight lock for knowledge compilation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

STALE_LOCK_SECONDS = 24 * 60 * 60
INCOMPLETE_LOCK_SECONDS = 60
OWNER_FILE = "owner.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LockSnapshot:
    """Identity and ownership data captured while lifecycle changes are guarded."""

    owner: dict[str, Any] | None
    owner_file_exists: bool
    timestamp: float


@contextmanager
def compilation_lock(
    lock_path: Path,
    *,
    now: Callable[[], float] = time.time,
) -> Iterator[bool]:
    """Try to own ``lock_path`` for the duration of the context.

    Yields ``False`` when another live or fresh owner already holds the lock.
    Cleanup failures are logged rather than replacing an exception from the
    compilation body.
    """
    owner = {"pid": os.getpid(), "created_at": now(), "token": uuid.uuid4().hex}
    acquired = _acquire(lock_path, owner=owner, now=now)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                _release(lock_path, owner=owner)
            except OSError:
                logger.exception("Failed to release compilation lock: %s", lock_path)


def _acquire(
    lock_path: Path,
    *,
    now: Callable[[], float] = time.time,
    owner: dict[str, Any] | None = None,
) -> bool:
    """Atomically acquire a lock directory, recovering a stale lock once."""
    owner = owner or {
        "pid": os.getpid(),
        "created_at": now(),
        "token": uuid.uuid4().hex,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with _lifecycle_guard(lock_path):
        if lock_path.exists():
            snapshot = _snapshot(lock_path)
            if snapshot is None or not _snapshot_is_stale(snapshot, now=now):
                return False
            logger.warning("Recovering stale compilation lock: %s", lock_path)
            _remove_lock(lock_path)

        lock_path.mkdir()
        try:
            (lock_path / OWNER_FILE).write_text(json.dumps(owner), encoding="utf-8")
        except Exception:
            try:
                _remove_lock(lock_path)
            except OSError:
                logger.exception("Failed to clean up incomplete compilation lock: %s", lock_path)
            raise
        return True


def _snapshot(lock_path: Path) -> _LockSnapshot | None:
    """Read lock ownership while the lifecycle guard prevents replacement."""
    try:
        lock_stat = lock_path.stat()
    except FileNotFoundError:
        return None

    owner_path = lock_path / OWNER_FILE
    owner: dict[str, Any] | None = None
    owner_file_exists = False
    timestamp = lock_stat.st_mtime
    try:
        raw_owner = owner_path.read_bytes()
        owner_file_exists = True
        timestamp = owner_path.stat().st_mtime
        parsed = json.loads(raw_owner)
        if isinstance(parsed, dict):
            owner = parsed
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        owner = None

    return _LockSnapshot(
        owner=owner,
        owner_file_exists=owner_file_exists,
        timestamp=timestamp,
    )


def _snapshot_is_stale(snapshot: _LockSnapshot, *, now: Callable[[], float]) -> bool:
    """Return whether a captured owner is dead or malformed and expired."""
    owner = snapshot.owner
    if owner is None:
        stale_after = STALE_LOCK_SECONDS if snapshot.owner_file_exists else INCOMPLETE_LOCK_SECONDS
        return now() - snapshot.timestamp > stale_after

    pid = owner.get("pid")
    created_at = owner.get("created_at")
    token = owner.get("token")
    valid = (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and isinstance(created_at, (int, float))
        and not isinstance(created_at, bool)
        and isinstance(token, str)
        and bool(token)
    )
    if not valid:
        return now() - snapshot.timestamp > STALE_LOCK_SECONDS

    # A confirmed live process keeps its lock regardless of age. The age
    # threshold is only a fallback for missing or malformed metadata.
    return not _pid_is_alive(pid)


@contextmanager
def _lifecycle_guard(lock_path: Path) -> Iterator[None]:
    """Serialize lock-directory create, reclaim, and release operations.

    The guard file may remain on disk, but its OS advisory lock is released
    automatically if a process exits or crashes.
    """
    guard_path = lock_path.with_name(f"{lock_path.name}.guard")
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    with guard_path.open("a+b") as guard_file:
        _lock_guard_file(guard_file)
        try:
            yield
        finally:
            _unlock_guard_file(guard_file)


def _lock_guard_file(guard_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        guard_file.seek(0, os.SEEK_END)
        if guard_file.tell() == 0:
            guard_file.write(b"\0")
            guard_file.flush()
        guard_file.seek(0)
        msvcrt.locking(guard_file.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(guard_file.fileno(), fcntl.LOCK_EX)


def _unlock_guard_file(guard_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        guard_file.seek(0)
        msvcrt.locking(guard_file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(guard_file.fileno(), fcntl.LOCK_UN)


def _pid_is_alive(pid: int) -> bool:
    """Check process existence without terminating it on Windows."""
    if os.name == "nt":
        return _windows_pid_is_alive(pid)

    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _windows_pid_is_alive(pid: int) -> bool:
    """Probe a Windows PID through a non-destructive query handle."""
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if handle:
        kernel32.CloseHandle(handle)
        return True

    error = ctypes.get_last_error()
    if error == error_invalid_parameter:
        return False
    if error == error_access_denied:
        return True
    return True


def _release(lock_path: Path, *, owner: dict[str, Any]) -> None:
    """Remove ``lock_path`` only when its metadata still matches this owner."""
    with _lifecycle_guard(lock_path):
        try:
            current_owner = json.loads((lock_path / OWNER_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Compilation lock owner metadata is unreadable: %s", lock_path)
            return

        if current_owner == owner:
            _remove_lock(lock_path)


def _remove_lock(lock_path: Path) -> None:
    """Remove one lock generation while lifecycle operations are serialized."""
    try:
        if lock_path.is_dir():
            shutil.rmtree(lock_path)
        else:
            lock_path.unlink()
    except FileNotFoundError:
        pass
