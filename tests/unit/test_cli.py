from __future__ import annotations

import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

from rimrule.models import Vocabulary
from scripts.cli import app, cfg, configure_quiet_http_logging

_OPENAI_CFG = "agent_llm:\n  provider: openai\n  model: gpt-4o\n"


def _vocab() -> Vocabulary:
    return Vocabulary(
        domain=("G",),
        qualifier=("Q",),
        action=("A",),
        strength=("S",),
        tool_category=("T",),
    )


class CliCommandTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _config(self, body: str = _OPENAI_CFG) -> Path:
        path = self.tmp_path / "cfg.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    @patch("scripts.cli.schema_summary", return_value={"samples": 0})
    @patch("scripts.cli.load_toolhop", return_value=[])
    def test_validate_data(self, mock_load, mock_summary):
        config = self._config(
            _OPENAI_CFG + "dataset:\n  path: data.json\n  strict_schema: true\n"
        )
        result = self.runner.invoke(app, ["validate-data", "--config", str(config)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("samples", result.stdout)

    @patch("scripts.cli.collect", return_value=[])
    def test_run_collect(self, mock_collect):
        result = self.runner.invoke(
            app, ["run-collect", "--config", str(self._config())]
        )
        self.assertEqual(result.exit_code, 0)
        mock_collect.assert_called_once()

    @patch("scripts.cli.induce", return_value=[])
    def test_run_induce(self, mock_induce):
        result = self.runner.invoke(
            app, ["run-induce", "--config", str(self._config())]
        )
        self.assertEqual(result.exit_code, 0)

    @patch("scripts.cli.canonicalize")
    def test_run_canonicalize(self, mock_canon):
        mock_canon.return_value = (_vocab(), [])
        result = self.runner.invoke(
            app, ["run-canonicalize", "--config", str(self._config())]
        )
        self.assertEqual(result.exit_code, 0)

    @patch("scripts.cli.consolidate", return_value=[])
    def test_run_consolidate(self, mock_consolidate):
        result = self.runner.invoke(
            app, ["run-consolidate", "--config", str(self._config())]
        )
        self.assertEqual(result.exit_code, 0)

    @patch("scripts.cli.consolidate", return_value=[])
    @patch("scripts.cli.canonicalize")
    @patch("scripts.cli.induce", return_value=[])
    @patch("scripts.cli.collect", return_value=[])
    def test_run_all(self, mock_collect, mock_induce, mock_canon, mock_consolidate):
        mock_canon.return_value = (_vocab(), [])
        result = self.runner.invoke(app, ["run-all", "--config", str(self._config())])
        self.assertEqual(result.exit_code, 0)
        mock_collect.assert_called_once()
        mock_consolidate.assert_called_once()


class CfgOverrideTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_fld_overrides_artifact_directory(self):
        output = self.tmp_path / "experiment-7"
        config = cfg(Path("configs/toolhop.yaml"), fld=output)
        self.assertEqual(config.artifacts_dir, str(output))
        self.assertEqual(config.artifact_path("x.json").parent, output)


class QuietHttpLoggingTest(unittest.TestCase):
    def test_quiet_http_logging(self):
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)
        configure_quiet_http_logging()
        self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertEqual(logging.getLogger().level, logging.WARNING)
