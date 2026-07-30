"""Tests for successful and failed Agent SDK compilation results."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import compile as compile_script  # noqa: E402


class FakeAssistantMessage:
    pass


class FakeTextBlock:
    pass


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResultMessage:
    def __init__(
        self,
        *,
        is_error: bool,
        total_cost_usd: float | None = None,
        result: str | None = None,
        errors: list[str] | None = None,
        subtype: str = "success",
    ) -> None:
        self.is_error = is_error
        self.total_cost_usd = total_cost_usd
        self.result = result
        self.errors = errors
        self.subtype = subtype


def fake_sdk(result: FakeResultMessage) -> types.SimpleNamespace:
    async def query(**_kwargs):
        yield result

    return types.SimpleNamespace(
        AssistantMessage=FakeAssistantMessage,
        ClaudeAgentOptions=FakeClaudeAgentOptions,
        ResultMessage=FakeResultMessage,
        TextBlock=FakeTextBlock,
        query=query,
    )


class CompileResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.log_path = self.root / "2026-07-30.md"
        self.log_path.write_text("# Daily Log: 2026-07-30", encoding="utf-8")
        self.agents_file = self.root / "AGENTS.md"
        self.agents_file.write_text("schema", encoding="utf-8")
        self.compiled_truth = self.root / "missing-compiled-truth.md"

    def run_compile(self, result: FakeResultMessage, state: dict) -> tuple[float | None, mock.Mock]:
        save_state = mock.Mock()
        with (
            mock.patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(result)}),
            mock.patch.object(compile_script, "AGENTS_FILE", self.agents_file),
            mock.patch.object(compile_script, "COMPILED_TRUTH_FILE", self.compiled_truth),
            mock.patch.object(compile_script, "read_wiki_index", return_value="index"),
            mock.patch.object(compile_script, "save_state", save_state),
        ):
            cost = asyncio.run(compile_script.compile_daily_log(self.log_path, state))
        return cost, save_state

    def test_error_result_does_not_mark_log_ingested(self) -> None:
        state: dict = {}
        result = FakeResultMessage(
            is_error=True,
            result="You've hit your limit",
            errors=["rate_limit"],
            subtype="error_max_turns",
        )

        cost, save_state = self.run_compile(result, state)

        self.assertIsNone(cost)
        self.assertNotIn("ingested_daily", state)
        save_state.assert_not_called()

    def test_success_result_updates_state_and_cost(self) -> None:
        state: dict = {}
        result = FakeResultMessage(is_error=False, total_cost_usd=0.42)

        cost, save_state = self.run_compile(result, state)

        self.assertEqual(cost, 0.42)
        self.assertIn(self.log_path.name, state["ingested_daily"])
        self.assertEqual(state["total_cost"], 0.42)
        save_state.assert_called_once_with(state)


if __name__ == "__main__":
    unittest.main()
