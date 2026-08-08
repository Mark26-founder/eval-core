"""Persistence module for saving and loading reports as JSON."""

import json
from pathlib import Path
from typing import Any

from eval_core.reports.exceptions import ReportError
from eval_core.reports.models import RegressionDetails, Report
from eval_core.scorers.models import Score


def save_report(report: Report, path: Path | str) -> None:
    """Saves a Report object as JSON to the specified path.

    Creates target directories automatically if missing.

    Args:
        report: The Report object to persist.
        path: File system path for saving the JSON report.

    Raises:
        ReportError: If writing to file fails.
    """
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": report.timestamp,
            "provider": report.provider,
            "model": report.model,
            "total_cases": report.total_cases,
            "passed": report.passed,
            "failed": report.failed,
            "accuracy": report.accuracy,
            "average_latency_ms": report.average_latency_ms,
            "total_input_tokens": report.total_input_tokens,
            "total_output_tokens": report.total_output_tokens,
            "cases": [
                {
                    "case_id": score.case_id,
                    "passed": score.passed,
                    "accuracy": score.accuracy,
                    "reasoning": score.reasoning,
                    "latency_ms": score.latency_ms,
                    "input_tokens": score.input_tokens,
                    "output_tokens": score.output_tokens,
                }
                for score in report.cases
            ],
            "regression_detected": report.regression_detected,
            "regression_details": (
                {
                    "cases_regressed": list(report.regression_details.cases_regressed),
                    "accuracy_change": report.regression_details.accuracy_change,
                    "latency_change_ms": report.regression_details.latency_change_ms,
                }
                if report.regression_details
                else None
            ),
        }

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as err:
        raise ReportError(f"Failed to save report to '{path}': {err}") from err


def load_report(path: Path | str) -> Report:
    """Loads a Report object from a JSON file.

    Args:
        path: Path to the JSON report file.

    Returns:
        The deserialized Report object.

    Raises:
        ReportError: If file loading or parsing fails.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ReportError(f"Report file not found: '{path}'")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ReportError("Report JSON root must be an object")

        raw_cases = data.get("cases", [])
        if not isinstance(raw_cases, list):
            raise ReportError("'cases' must be a list")

        scores: list[Score] = []
        for item in raw_cases:
            if not isinstance(item, dict):
                raise ReportError("Case score item must be an object")
            score = Score(
                case_id=item["case_id"],
                passed=item["passed"],
                accuracy=item["accuracy"],
                reasoning=item["reasoning"],
                latency_ms=item["latency_ms"],
                input_tokens=item["input_tokens"],
                output_tokens=item["output_tokens"],
            )
            scores.append(score)

        raw_reg = data.get("regression_details")
        regression_details: RegressionDetails | None = None
        if raw_reg is not None:
            if not isinstance(raw_reg, dict):
                raise ReportError("'regression_details' must be an object")
            regression_details = RegressionDetails(
                cases_regressed=tuple(raw_reg.get("cases_regressed", [])),
                accuracy_change=raw_reg["accuracy_change"],
                latency_change_ms=raw_reg["latency_change_ms"],
            )

        return Report(
            timestamp=data["timestamp"],
            provider=data["provider"],
            model=data["model"],
            total_cases=data["total_cases"],
            passed=data["passed"],
            failed=data["failed"],
            accuracy=data["accuracy"],
            average_latency_ms=data["average_latency_ms"],
            total_input_tokens=data["total_input_tokens"],
            total_output_tokens=data["total_output_tokens"],
            cases=tuple(scores),
            regression_detected=data["regression_detected"],
            regression_details=regression_details,
        )
    except ReportError:
        raise
    except Exception as err:
        raise ReportError(f"Failed to load report from '{path}': {err}") from err
