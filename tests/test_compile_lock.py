"""Tests for project-level compilation locking."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import compile as compile_script  # noqa: E402
from compile_lock import (  # noqa: E402
    INCOMPLETE_LOCK_SECONDS,
    OWNER_FILE,
    STALE_LOCK_SECONDS,
    _acquire,
    compilation_lock,
)


class CompilationLockTests(unittest.TestCase):
    def test_nested_acquisition_is_busy_and_owner_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"

            with compilation_lock(lock_path) as acquired:
                self.assertTrue(acquired)
                with compilation_lock(lock_path) as second:
                    self.assertFalse(second)

            self.assertFalse(lock_path.exists())

    def test_dead_owner_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"
            lock_path.mkdir()
            (lock_path / OWNER_FILE).write_text(
                json.dumps({"pid": 999_999_999, "created_at": 100.0, "token": "old"}),
                encoding="utf-8",
            )

            with mock.patch("compile_lock._pid_is_alive", return_value=False):
                with compilation_lock(lock_path, now=lambda: 101.0) as acquired:
                    self.assertTrue(acquired)

            self.assertFalse(lock_path.exists())

    def test_live_owner_is_not_reclaimed_even_when_old(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"
            lock_path.mkdir()
            (lock_path / OWNER_FILE).write_text(
                json.dumps({"pid": os.getpid(), "created_at": 1.0, "token": "live"}),
                encoding="utf-8",
            )

            with mock.patch("compile_lock._pid_is_alive", return_value=True):
                with compilation_lock(
                    lock_path,
                    now=lambda: STALE_LOCK_SECONDS * 2,
                ) as acquired:
                    self.assertFalse(acquired)

            self.assertTrue(lock_path.exists())

    def test_missing_owner_file_is_recovered_after_short_grace_period(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"
            lock_path.mkdir()
            old_time = 1_000.0
            os.utime(lock_path, (old_time, old_time))

            with compilation_lock(
                lock_path,
                now=lambda: old_time + INCOMPLETE_LOCK_SECONDS + 1,
            ) as acquired:
                self.assertTrue(acquired)

            self.assertFalse(lock_path.exists())

    def test_fresh_malformed_lock_remains_busy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"
            lock_path.mkdir()
            owner_path = lock_path / OWNER_FILE
            owner_path.write_text("not-json", encoding="utf-8")
            fresh_time = 1_000.0
            os.utime(owner_path, (fresh_time, fresh_time))

            with compilation_lock(lock_path, now=lambda: fresh_time + 1) as acquired:
                self.assertFalse(acquired)

            self.assertTrue(lock_path.exists())

    def test_expired_malformed_lock_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"
            lock_path.mkdir()
            owner_path = lock_path / OWNER_FILE
            owner_path.write_text("not-json", encoding="utf-8")
            old_time = 1_000.0
            os.utime(owner_path, (old_time, old_time))

            with compilation_lock(
                lock_path,
                now=lambda: old_time + STALE_LOCK_SECONDS + 1,
            ) as acquired:
                self.assertTrue(acquired)

            self.assertFalse(lock_path.exists())

    def test_two_stale_recoverers_produce_one_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"
            lock_path.mkdir()
            (lock_path / OWNER_FILE).write_text(
                json.dumps({"pid": 999_999_999, "created_at": 100.0, "token": "dead"}),
                encoding="utf-8",
            )
            start = threading.Barrier(2)
            release = threading.Event()

            def contender() -> bool:
                start.wait()
                with compilation_lock(lock_path, now=lambda: 101.0) as acquired:
                    if acquired:
                        release.wait(timeout=2)
                    return acquired

            with (
                mock.patch(
                    "compile_lock._pid_is_alive",
                    side_effect=lambda pid: pid != 999_999_999,
                ),
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                futures = [pool.submit(contender) for _ in range(2)]
                time.sleep(0.1)
                release.set()
                acquired = [future.result(timeout=2) for future in futures]

            self.assertEqual(sum(acquired), 1)
            self.assertFalse(lock_path.exists())

    def test_exception_still_releases_owned_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"

            with self.assertRaisesRegex(RuntimeError, "boom"):
                with compilation_lock(lock_path) as acquired:
                    self.assertTrue(acquired)
                    raise RuntimeError("boom")

            self.assertFalse(lock_path.exists())

    def test_owner_write_cleanup_failure_preserves_write_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"

            with (
                mock.patch.object(Path, "write_text", side_effect=RuntimeError("write failed")),
                mock.patch("compile_lock._remove_lock", side_effect=PermissionError("cleanup")),
                self.assertLogs("compile_lock", level="ERROR"),
                self.assertRaisesRegex(RuntimeError, "write failed"),
            ):
                _acquire(lock_path)

    def test_cleanup_failure_does_not_hide_body_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "compile.lock"

            with (
                mock.patch("compile_lock._release", side_effect=PermissionError("denied")),
                self.assertLogs("compile_lock", level="ERROR"),
                self.assertRaisesRegex(RuntimeError, "original"),
            ):
                with compilation_lock(lock_path) as acquired:
                    self.assertTrue(acquired)
                    raise RuntimeError("original")

    def test_windows_pid_probe_does_not_call_os_kill(self) -> None:
        with (
            mock.patch("compile_lock.os.name", "nt"),
            mock.patch("compile_lock._windows_pid_is_alive", return_value=True) as probe,
            mock.patch("compile_lock.os.kill") as kill,
        ):
            from compile_lock import _pid_is_alive

            self.assertTrue(_pid_is_alive(123))

        probe.assert_called_once_with(123)
        kill.assert_not_called()

    def test_windows_pid_probe_uses_query_handle(self) -> None:
        class FakeFunction:
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __call__(self, *args):
                self.calls.append(args)
                return self.result

        open_process = FakeFunction(1234)
        close_handle = FakeFunction(True)
        kernel32 = types.SimpleNamespace(
            OpenProcess=open_process,
            CloseHandle=close_handle,
        )
        fake_ctypes = types.SimpleNamespace(
            WinDLL=lambda *_args, **_kwargs: kernel32,
            get_last_error=lambda: 0,
            wintypes=types.SimpleNamespace(DWORD=int, BOOL=int, HANDLE=int),
        )

        with mock.patch.dict(sys.modules, {"ctypes": fake_ctypes}):
            from compile_lock import _windows_pid_is_alive

            self.assertTrue(_windows_pid_is_alive(123))

        self.assertEqual(open_process.calls[0][2], 123)
        self.assertEqual(close_handle.calls, [(1234,)])

    def test_compile_skips_all_work_when_lock_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            daily_log = state_dir / "daily.md"
            daily_log.write_text("daily log", encoding="utf-8")
            lock_path = state_dir / "compile.lock"

            compile_daily_log = mock.AsyncMock(return_value=0.0)
            output = io.StringIO()
            with compilation_lock(lock_path) as acquired:
                self.assertTrue(acquired)
                with (
                    mock.patch.object(compile_script, "_STATE_DIR", state_dir),
                    mock.patch.object(compile_script, "list_raw_files", return_value=[daily_log]),
                    mock.patch.object(compile_script, "compile_daily_log", compile_daily_log),
                    mock.patch.object(compile_script, "regenerate_truth"),
                    mock.patch.object(compile_script, "list_wiki_articles", return_value=[]),
                    mock.patch.object(sys, "argv", ["compile.py"]),
                    redirect_stdout(output),
                ):
                    result = compile_script.main()

            self.assertIsNone(result)
            compile_daily_log.assert_not_called()
            self.assertIn("Compilation already running", output.getvalue())


if __name__ == "__main__":
    unittest.main()
