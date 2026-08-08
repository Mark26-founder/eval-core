"""Unit tests for src/eval_core/reports package."""

from pathlib import Path
import pytest

from eval_core.cases.models import TestCase
from eval_core.reports import (
    RegressionDetails,
    Report,
    ReportError,
    compare_reports,
    generate_report,
    load_report,
    save_report,
)
from eval_core.scorers.models import Score


@pytest.fixture
def sample_cases() -> list[TestCase]:
    return [
        TestCase(id="c1", input="in1", expected="out1"),
        TestCase(id="c2", input="in2", expected="out2"),
    ]


@pytest.fixture
def sample_scores() -> list[Score]:
    return [
        Score(
            case_id="c1",
            passed=True,
            accuracy=1.0,
            reasoning="Exact match",
            latency_ms=100.0,
            input_tokens=10,
            output_tokens=5,
        ),
        Score(
            case_id="c2",
            passed=False,
            accuracy=0.0,
            reasoning="Mismatch",
            latency_ms=200.0,
            input_tokens=15,
            output_tokens=10,
        ),
    ]


# ============================================================================
# Report Dataclass & Validation Tests
# ============================================================================


def test_report_validation_valid(sample_scores):
    report = Report(
        timestamp="2026-07-23T00:00:00Z",
        provider="openrouter",
        model="gpt-4o",
        total_cases=2,
        passed=1,
        failed=1,
        accuracy=0.5,
        average_latency_ms=150.0,
        total_input_tokens=25,
        total_output_tokens=15,
        cases=tuple(sample_scores),
        regression_detected=False,
        regression_details=None,
    )
    assert report.total_cases == 2
    assert report.accuracy == 0.5


def test_report_validation_invalid_accuracy(sample_scores):
    with pytest.raises(ReportError, match="accuracy must be between"):
        Report(
            timestamp="2026-07-23T00:00:00Z",
            provider="openrouter",
            model="gpt-4o",
            total_cases=2,
            passed=1,
            failed=1,
            accuracy=1.5,
            average_latency_ms=150.0,
            total_input_tokens=25,
            total_output_tokens=15,
            cases=tuple(sample_scores),
            regression_detected=False,
        )


def test_report_validation_negative_counts(sample_scores):
    with pytest.raises(ReportError, match="total_cases must be non-negative"):
        Report(
            timestamp="2026-07-23T00:00:00Z",
            provider="openrouter",
            model="gpt-4o",
            total_cases=-1,
            passed=0,
            failed=0,
            accuracy=0.0,
            average_latency_ms=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            cases=tuple(sample_scores),
            regression_detected=False,
        )


# ============================================================================
# Report Generation & Aggregation Tests
# ============================================================================


def test_generate_report_aggregation(sample_cases, sample_scores):
    report = generate_report(
        cases=sample_cases,
        scores=sample_scores,
        provider="openrouter",
        model="gpt-4o",
    )

    assert report.provider == "openrouter"
    assert report.model == "gpt-4o"
    assert report.total_cases == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.accuracy == 0.5
    assert report.average_latency_ms == 150.0
    assert report.total_input_tokens == 25
    assert report.total_output_tokens == 15
    assert report.regression_detected is False


def test_generate_report_mismatched_lengths(sample_cases, sample_scores):
    with pytest.raises(ReportError, match="Mismatched cases count"):
        generate_report(
            cases=sample_cases[:1],
            scores=sample_scores,
            provider="openrouter",
            model="gpt-4o",
        )


# ============================================================================
# Regression Detection & Comparison Tests
# ============================================================================


def test_compare_reports_case_regression():
    prev_scores = [
        Score(case_id="c1", passed=True, accuracy=1.0, reasoning="", latency_ms=100.0, input_tokens=5, output_tokens=5),
        Score(case_id="c2", passed=True, accuracy=1.0, reasoning="", latency_ms=100.0, input_tokens=5, output_tokens=5),
    ]
    curr_scores = [
        Score(case_id="c1", passed=True, accuracy=1.0, reasoning="", latency_ms=100.0, input_tokens=5, output_tokens=5),
        Score(case_id="c2", passed=False, accuracy=0.0, reasoning="", latency_ms=100.0, input_tokens=5, output_tokens=5),
    ]

    prev_report = Report(
        timestamp="t1", provider="p", model="m", total_cases=2, passed=2, failed=0, accuracy=1.0,
        average_latency_ms=100.0, total_input_tokens=10, total_output_tokens=10, cases=tuple(prev_scores), regression_detected=False
    )
    curr_report = Report(
        timestamp="t2", provider="p", model="m", total_cases=2, passed=1, failed=1, accuracy=0.5,
        average_latency_ms=100.0, total_input_tokens=10, total_output_tokens=10, cases=tuple(curr_scores), regression_detected=False
    )

    details = compare_reports(curr_report, prev_report)
    assert details.cases_regressed == ("c2",)
    assert details.accuracy_change == -0.5
    assert details.latency_change_ms == 0.0


def test_generate_report_with_previous_report_triggers_regression(sample_cases):
    scores_prev = [
        Score(case_id="c1", passed=True, accuracy=1.0, reasoning="", latency_ms=100.0, input_tokens=5, output_tokens=5),
        Score(case_id="c2", passed=True, accuracy=1.0, reasoning="", latency_ms=100.0, input_tokens=5, output_tokens=5),
    ]
    scores_curr = [
        Score(case_id="c1", passed=True, accuracy=1.0, reasoning="", latency_ms=100.0, input_tokens=5, output_tokens=5),
        Score(case_id="c2", passed=False, accuracy=0.0, reasoning="", latency_ms=100.0, input_tokens=5, output_tokens=5),
    ]

    prev_report = generate_report(sample_cases, scores_prev, "provider", "model")
    curr_report = generate_report(sample_cases, scores_curr, "provider", "model", previous_report=prev_report)

    assert curr_report.regression_detected is True
    assert curr_report.regression_details is not None
    assert curr_report.regression_details.cases_regressed == ("c2",)


# ============================================================================
# Persistence (Save & Load) Tests
# ============================================================================


def test_save_and_load_report_json(tmp_path: Path, sample_cases, sample_scores):
    report = generate_report(sample_cases, sample_scores, "openrouter", "gpt-4o")
    report_file = tmp_path / "sub_dir" / "report.json"

    save_report(report, report_file)
    assert report_file.is_file()

    loaded = load_report(report_file)
    assert loaded.provider == report.provider
    assert loaded.model == report.model
    assert loaded.total_cases == report.total_cases
    assert loaded.passed == report.passed
    assert loaded.accuracy == report.accuracy
    assert len(loaded.cases) == len(report.cases)
    assert loaded.cases[0].case_id == report.cases[0].case_id


def test_load_report_missing_file(tmp_path: Path):
    missing_file = tmp_path / "non_existent.json"
    with pytest.raises(ReportError, match="Report file not found"):
        load_report(missing_file)


def test_load_report_corrupted_json(tmp_path: Path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ corrupted json", encoding="utf-8")
    with pytest.raises(ReportError, match="Failed to load report"):
        load_report(bad_file)
