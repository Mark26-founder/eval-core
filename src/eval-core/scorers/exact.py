"""Exact string matching scorer."""

from eval_core.cases.models import TestCase
from eval_core.runners.models import RunnerResponse
from eval_core.scorers.base import BaseScorer
from eval_core.scorers.models import Score


class ExactScorer(BaseScorer):
    """Scorer that compares expected and actual outputs via exact string matching after stripping whitespace."""

    def score(
        self,
        expected: str,
        response: RunnerResponse,
        case: TestCase,
    ) -> Score:
        """Scores a response against expected output using exact string equality.

        Args:
            expected: The expected reference output string.
            response: The actual response from the runner.
            case: The original evaluation test case.

        Returns:
            A Score object.
        """
        norm_expected = expected.strip()
        norm_actual = response.output.strip()

        if norm_expected == norm_actual:
            passed = True
            accuracy = 1.0
            reasoning = "Exact match"
        else:
            passed = False
            accuracy = 0.0
            reasoning = "Outputs differ"

        return Score(
            case_id=case.id,
            passed=passed,
            accuracy=accuracy,
            reasoning=reasoning,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
