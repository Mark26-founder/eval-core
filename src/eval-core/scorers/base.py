"""Abstract base class for all scorers."""

from abc import ABC, abstractmethod

from eval_core.cases.models import TestCase
from eval_core.runners.models import RunnerResponse
from eval_core.scorers.models import Score


class BaseScorer(ABC):
    """Abstract base class for all evaluation scorers."""

    @abstractmethod
    def score(
        self,
        expected: str,
        response: RunnerResponse,
        case: TestCase,
    ) -> Score:
        """Scores a response against the expected reference output.

        Args:
            expected: The expected reference output string.
            response: The actual response from the runner.
            case: The original evaluation test case.

        Returns:
            A Score object.

        Raises:
            ScorerError: If validation or evaluation fails.
        """
        ...
