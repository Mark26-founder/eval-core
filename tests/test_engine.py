"""Unit tests for EvaluationEngine in src/eval_core/engine.py."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval_core.cases.models import TestCase
from eval_core.engine import EvaluationEngine
from eval_core.reports import Report, save_report
from eval_core.runners.base import BaseRunner
from eval_core.runners.models import RunnerResponse
from eval_core.scorers.base import BaseScorer
from eval_core.scorers.models import Score


@pytest.fixture
def mock_runner() -> BaseRunner:
    runner = MagicMock(spec=BaseRunner)
    runner.run.return_value = RunnerResponse(
        output="Generated output",
        latency_ms=150.0,
        input_tokens=10,
        output_tokens=20,
    )
    return runner


@pytest.fixture
def mock_scorer() -> BaseScorer:
    scorer = MagicMock(spec=BaseScorer)
    scorer.score.return_value = Score(
        case_id="case-1",
        passed=True,
        accuracy=1.0,
        reasoning="Matches expected",
        latency_ms=150.0,
        input_tokens=10,
        output_tokens=20,
    )
    return scorer


def test_engine_evaluate_end_to_end(mock_runner, mock_scorer, tmp_path: Path):
    cases = [
        TestCase(id="case-1", input="Prompt 1", expected="Generated output"),
    ]

    engine = EvaluationEngine(runner=mock_runner, scorer=mock_scorer)
    output_file = tmp_path / "output_report.json"

    report = engine.evaluate(
        cases=cases,
        provider="openrouter",
        model="gpt-4o",
        output_report_path=output_file,
    )

    mock_runner.run.assert_called_once_with(cases[0])
    mock_scorer.score.assert_called_once()
    assert report.total_cases == 1
    assert report.passed == 1
    assert report.accuracy == 1.0
    assert output_file.is_file()


def test_engine_evaluate_with_previous_report(mock_runner, mock_scorer, tmp_path: Path):
    cases = [
        TestCase(id="case-1", input="Prompt 1", expected="Generated output"),
    ]

    # Create previous report where case-1 passed
    prev_report = Report(
        timestamp="t0",
        provider="openrouter",
        model="gpt-4o",
        total_cases=1,
        passed=1,
        failed=0,
        accuracy=1.0,
        average_latency_ms=100.0,
        total_input_tokens=10,
        total_output_tokens=20,
        cases=(
            Score(
                case_id="case-1",
                passed=True,
                accuracy=1.0,
                reasoning="",
                latency_ms=100.0,
                input_tokens=10,
                output_tokens=20,
            ),
        ),
        regression_detected=False,
    )
    prev_file = tmp_path / "previous_report.json"
    save_report(prev_report, prev_file)

    # Scorer now fails
    mock_scorer.score.return_value = Score(
        case_id="case-1",
        passed=False,
        accuracy=0.0,
        reasoning="Failed",
        latency_ms=150.0,
        input_tokens=10,
        output_tokens=20,
    )

    engine = EvaluationEngine(runner=mock_runner, scorer=mock_scorer)
    report = engine.evaluate(
        cases=cases,
        provider="openrouter",
        model="gpt-4o",
        previous_report_path=prev_file,
    )

    assert report.regression_detected is True
    assert report.regression_details is not None
    assert report.regression_details.cases_regressed == ("case-1",)
