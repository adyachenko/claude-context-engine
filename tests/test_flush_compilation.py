"""Tests for end-of-day compilation decisions in flush.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

with mock.patch("logging.basicConfig"):
    from flush import daily_log_needs_compilation  # noqa: E402
from utils import file_hash  # noqa: E402


class DailyLogNeedsCompilationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.log_path = self.root / "2026-07-30.md"
        self.state_file = self.root / "state.json"
        self.log_path.write_text("daily log", encoding="utf-8")

    def write_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def test_absent_state_needs_compilation(self) -> None:
        self.assertTrue(daily_log_needs_compilation(self.log_path, self.state_file))

    def test_legacy_ingested_state_is_migrated(self) -> None:
        self.write_state({"ingested": {self.log_path.name: {"hash": file_hash(self.log_path)}}})

        self.assertFalse(daily_log_needs_compilation(self.log_path, self.state_file))

    def test_unchanged_ingested_daily_hash_does_not_need_compilation(self) -> None:
        self.write_state(
            {"ingested_daily": {self.log_path.name: {"hash": file_hash(self.log_path)}}}
        )

        self.assertFalse(daily_log_needs_compilation(self.log_path, self.state_file))

    def test_changed_hash_needs_compilation(self) -> None:
        self.write_state({"ingested_daily": {self.log_path.name: {"hash": "old"}}})

        self.assertTrue(daily_log_needs_compilation(self.log_path, self.state_file))

    def test_malformed_state_fails_open(self) -> None:
        self.state_file.write_text("not-json", encoding="utf-8")

        self.assertTrue(daily_log_needs_compilation(self.log_path, self.state_file))

    def test_absent_daily_log_does_not_need_compilation(self) -> None:
        self.log_path.unlink()

        self.assertFalse(daily_log_needs_compilation(self.log_path, self.state_file))


if __name__ == "__main__":
    unittest.main()
