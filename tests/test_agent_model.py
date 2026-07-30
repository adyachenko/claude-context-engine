"""Tests for central Claude Agent SDK model selection."""

from __future__ import annotations

import ast
import os
import runpy
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "config.py"
AGENT_SCRIPTS = ("compile.py", "flush.py", "ingest.py", "query.py", "lint.py")


class AgentModelConfigTests(unittest.TestCase):
    def test_default_model_is_sonnet(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WIKI_MODEL", None)
            config = runpy.run_path(str(CONFIG_PATH))

        self.assertEqual(config["AGENT_MODEL"], "sonnet")

    def test_wiki_model_overrides_default(self) -> None:
        with mock.patch.dict(os.environ, {"WIKI_MODEL": "custom-model"}):
            config = runpy.run_path(str(CONFIG_PATH))

        self.assertEqual(config["AGENT_MODEL"], "custom-model")

    def test_every_agent_options_call_uses_central_model(self) -> None:
        calls_checked = 0
        for script_name in AGENT_SCRIPTS:
            script_path = ROOT / "scripts" / script_name
            tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=script_name)
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ClaudeAgentOptions"
            ]
            self.assertEqual(len(calls), 1, f"expected one SDK options call in {script_name}")

            model_keywords = [keyword for keyword in calls[0].keywords if keyword.arg == "model"]
            self.assertEqual(len(model_keywords), 1, f"missing model in {script_name}")
            self.assertIsInstance(model_keywords[0].value, ast.Name)
            self.assertEqual(model_keywords[0].value.id, "AGENT_MODEL")
            calls_checked += 1

        self.assertEqual(calls_checked, 5)


if __name__ == "__main__":
    unittest.main()
