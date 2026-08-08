"""Data models for evaluation reports."""

from dataclasses import dataclass
from typing import Sequence

from eval_core.reports.exceptions import ReportError
from eval_core.scorers.models import Score


@dataclass(frozen=True)
class RegressionDetails:
    """Immutable representation of regression details between reports.

    Attributes:
        cases_regressed: List of case IDs that regressed (passed previously, now failed).
        accuracy_change: Difference in overall accuracy (current - previous).
        latency_change_ms: Difference in average latency in ms (current - previous).
    """

    cases_regressed: tuple[str, ...]
    accuracy_change: float
    latency_change_ms: float


@dataclass(frozen=True)
class Report:
    """Immutable representation of an evaluation report.

    Attributes:
        timestamp: ISO-formatted timestamp string.
        provider: Provider name string.
        model: Model name string.
        total_cases: Total number of test cases evaluated.
        passed: Number of passed test cases.
        failed: Number of failed test cases.
        accuracy: Accuracy score between 0.0 and 1.0.
        average_latency_ms: Average latency across cases in milliseconds.
        total_input_tokens: Aggregate input token count.
        total_output_tokens: Aggregate output token count.
        cases: Sequence of Score objects for each case.
        regression_detected: Boolean flag indicating if regression occurred.
        regression_details: Optional details about the regression if evaluated.
    """

    timestamp: str
    provider: str
    model: str
    total_cases: int
    passed: int
    failed: int
    accuracy: float
    average_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    cases: tuple[Score, ...]
    regression_detected: bool
    regression_details: RegressionDetails | None = None

    def __post_init__(self) -> None:
        """Validates report fields.

        Raises:
            ReportError: If field values violate validation rules.
        """
        if not (0.0 <= self.accuracy <= 1.0):
            raise ReportError(f"accuracy must be between 0.0 and 1.0, got {self.accuracy}")
        if self.total_cases < 0:
            raise ReportError(f"total_cases must be non-negative, got {self.total_cases}")
        if self.passed < 0:
            raise ReportError(f"passed must be non-negative, got {self.passed}")
        if self.failed < 0:
            raise ReportError(f"failed must be non-negative, got {self.failed}")
        if self.average_latency_ms < 0:
            raise ReportError(f"average_latency_ms must be non-negative, got {self.average_latency_ms}")
        if self.total_input_tokens < 0:
            raise ReportError(f"total_input_tokens must be non-negative, got {self.total_input_tokens}")
        if self.total_output_tokens < 0:
            raise ReportError(f"total_output_tokens must be non-negative, got {self.total_output_tokens}")
