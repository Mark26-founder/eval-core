"""Data models for evaluation scorers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    """Immutable representation of an evaluation score.

    Attributes:
        case_id: The ID of the evaluated TestCase.
        passed: Whether the test case passed.
        accuracy: The accuracy score (0.0 to 1.0).
        reasoning: A description explaining the score or validation.
        latency_ms: Latency of the model runner in milliseconds.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
    """

    case_id: str
    passed: bool
    accuracy: float
    reasoning: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        """Validates score attributes after initialization.

        Raises:
            ValueError: If validation fails.
        """
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError(f"accuracy must be between 0.0 and 1.0, got {self.accuracy}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
        if self.input_tokens < 0:
            raise ValueError(f"input_tokens must be non-negative, got {self.input_tokens}")
        if self.output_tokens < 0:
            raise ValueError(f"output_tokens must be non-negative, got {self.output_tokens}")

