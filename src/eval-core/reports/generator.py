"""Report generator module."""

from datetime import datetime, timezone
from typing import Sequence

from eval_core.cases.models import TestCase
from eval_core.reports.comparator import compare_reports
from eval_core.reports.exceptions import ReportError
from eval_core.reports.models import RegressionDetails, Report
from eval_core.scorers.models import Score


def generate_report(
    cases: Sequence[TestCase],
    scores: Sequence[Score],
    provider: str,
    model: str,
    previous_report: Report | None = None,
) -> Report:
    """Generates an evaluation report by aggregating case scores.

    Args:
        cases: Sequence of evaluated TestCase objects.
        scores: Sequence of Score objects corresponding to evaluation cases.
        provider: Name of the LLM provider used.
        model: Name of the LLM model evaluated.
        previous_report: Optional baseline Report for comparison.

    Returns:
        An immutable Report object.

    Raises:
        ReportError: If input data is inconsistent or invalid.
    """
    if len(cases) != len(scores):
        raise ReportError(
            f"Mismatched cases count ({len(cases)}) and scores count ({len(scores)})"
        )

    total_cases = len(scores)
    passed = sum(1 for score in scores if score.passed)
    failed = total_cases - passed
    accuracy = round(passed / total_cases, 4) if total_cases > 0 else 0.0

    total_latency_ms = sum(score.latency_ms for score in scores)
    average_latency_ms = (
        round(total_latency_ms / total_cases, 4) if total_cases > 0 else 0.0
    )

    total_input_tokens = sum(score.input_tokens for score in scores)
    total_output_tokens = sum(score.output_tokens for score in scores)

    timestamp = datetime.now(timezone.utc).isoformat()
    tuple_scores = tuple(scores)

    temp_report = Report(
        timestamp=timestamp,
        provider=provider,
        model=model,
        total_cases=total_cases,
        passed=passed,
        failed=failed,
        accuracy=accuracy,
        average_latency_ms=average_latency_ms,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        cases=tuple_scores,
        regression_detected=False,
        regression_details=None,
    )

    if previous_report is not None:
        regression_details = compare_reports(temp_report, previous_report)
        regression_detected = (
            len(regression_details.cases_regressed) > 0
            or regression_details.accuracy_change < 0
        )
        return Report(
            timestamp=timestamp,
            provider=provider,
            model=model,
            total_cases=total_cases,
            passed=passed,
            failed=failed,
            accuracy=accuracy,
            average_latency_ms=average_latency_ms,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            cases=tuple_scores,
            regression_detected=regression_detected,
            regression_details=regression_details,
        )

    return temp_report
