"""Unit tests for EVAL-CORE CLI interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from typer.testing import CliRunner

from eval_core.cli.main import app

runner = CliRunner()


@pytest.fixture
def sample_suite_yaml(tmp_path: Path) -> Path:
    yaml_file = tmp_path / "test_suite.yaml"
    yaml_file.write_text(
        """
cases:
  - id: c1
    input: "Hello"
    expected: "World"
""",
        encoding="utf-8",
    )
    return yaml_file


@patch("eval_core.cli.main.EvaluationEngine")
@patch("eval_core.cli.main.get_runner")
@patch("eval_core.cli.main.get_scorer")
def test_cli_run_happy_path(
    mock_get_scorer,
    mock_get_runner,
    mock_engine_cls,
    sample_suite_yaml: Path,
):
    mock_report = MagicMock()
    mock_report.provider = "openrouter"
    mock_report.model = "gpt-4o"
    mock_report.total_cases = 1
    mock_report.passed = 1
    mock_report.failed = 0
    mock_report.accuracy = 1.0
    mock_report.average_latency_ms = 120.0
    mock_report.total_input_tokens = 10
    mock_report.total_output_tokens = 5
    mock_report.regression_detected = False
    mock_report.regression_details = None

    mock_engine_instance = MagicMock()
    mock_engine_instance.evaluate.return_value = mock_report
    mock_engine_cls.return_value = mock_engine_instance

    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            str(sample_suite_yaml),
            "--provider",
            "openrouter",
            "--model",
            "gpt-4o",
        ],
        env={"OPENROUTER_API_KEY": "sk-or-test"},
    )

    assert result.exit_code == 0
    assert "EVAL-CORE REPORT" in result.stdout
    assert "openrouter" in result.stdout


def test_cli_run_missing_suite():
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_cli_run_missing_provider(sample_suite_yaml: Path):
    result = runner.invoke(
        app,
        ["run", "--suite", str(sample_suite_yaml), "--model", "gpt-4o"],
        env={"EVAL_CORE_PROVIDER": "", "PROVIDER": ""},
    )
    assert result.exit_code == 1
    assert "Provider not specified" in result.stdout


def test_cli_run_missing_api_key(sample_suite_yaml: Path):
    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            str(sample_suite_yaml),
            "--provider",
            "openrouter",
            "--model",
            "gpt-4o",
        ],
        env={
            "OPENROUTER_API_KEY": "",
            "EVAL_CORE_PROVIDER": "",
            "API_KEY": "",
            "GEMINI_API_KEY": "",
        },
    )
    assert result.exit_code == 1
    assert "Missing API key" in result.stdout


@patch("eval_core.cli.main.EvaluationEngine")
@patch("eval_core.cli.main.get_runner")
@patch("eval_core.cli.main.get_scorer")
def test_cli_run_regression_detected_exits_code_1(
    mock_get_scorer,
    mock_get_runner,
    mock_engine_cls,
    sample_suite_yaml: Path,
):
    mock_report = MagicMock()
    mock_report.provider = "openrouter"
    mock_report.model = "gpt-4o"
    mock_report.total_cases = 1
    mock_report.passed = 0
    mock_report.failed = 1
    mock_report.accuracy = 0.0
    mock_report.average_latency_ms = 120.0
    mock_report.total_input_tokens = 10
    mock_report.total_output_tokens = 5
    mock_report.regression_detected = True
    mock_reg = MagicMock()
    mock_reg.cases_regressed = ("c1",)
    mock_reg.accuracy_change = -1.0
    mock_report.regression_details = mock_reg

    mock_engine_instance = MagicMock()
    mock_engine_instance.evaluate.return_value = mock_report
    mock_engine_cls.return_value = mock_engine_instance

    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            str(sample_suite_yaml),
            "--provider",
            "openrouter",
            "--model",
            "gpt-4o",
        ],
        env={"OPENROUTER_API_KEY": "sk-or-test"},
    )

    assert result.exit_code == 1
    assert "YES" in result.stdout
    assert "c1" in result.stdout
