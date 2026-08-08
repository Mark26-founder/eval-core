"""Report comparator module for regression detection."""

from eval_core.reports.models import RegressionDetails, Report


def compare_reports(current: Report, previous: Report) -> RegressionDetails:
    """Compares current report against previous report to detect regressions.

    Regression rules:
    Regression occurs if:
    - A previously passing case now fails
    OR
    - Overall accuracy decreases

    Args:
        current: The current evaluation Report object.
        previous: The baseline/previous evaluation Report object.

    Returns:
        RegressionDetails containing regressed case IDs and metrics changes.
    """
    previous_passed_ids = {
        score.case_id for score in previous.cases if score.passed
    }

    cases_regressed: list[str] = []
    for score in current.cases:
        if score.case_id in previous_passed_ids and not score.passed:
            cases_regressed.append(score.case_id)

    accuracy_change = round(current.accuracy - previous.accuracy, 4)
    latency_change_ms = round(current.average_latency_ms - previous.average_latency_ms, 4)

    return RegressionDetails(
        cases_regressed=tuple(cases_regressed),
        accuracy_change=accuracy_change,
        latency_change_ms=latency_change_ms,
    )
