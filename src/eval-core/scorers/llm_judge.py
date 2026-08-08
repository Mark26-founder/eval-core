"""LLM Judge scorer implementation."""

from eval_core.cases.models import TestCase
from eval_core.runners.base import BaseRunner
from eval_core.runners.models import RunnerResponse
from eval_core.scorers.base import BaseScorer
from eval_core.scorers.exceptions import ScorerError
from eval_core.scorers.models import Score


class LLMJudgeScorer(BaseScorer):
    """Scorer that uses an LLM as a judge to evaluate semantic correctness."""

    def __init__(self, runner: BaseRunner) -> None:
        """Initializes the LLM Judge Scorer.

        Args:
            runner: The BaseRunner instance used to call the judge model.
        """
        self._runner = runner

    def score(
        self,
        expected: str,
        response: RunnerResponse,
        case: TestCase,
    ) -> Score:
        """Scores a response using a judge LLM to compare actual and expected responses.

        Args:
            expected: The expected reference output string.
            response: The actual response from the runner.
            case: The original evaluation test case.

        Returns:
            A Score object.

        Raises:
            ScorerError: If the runner fails or the judge returns an invalid response.
        """
        prompt = (
            f"Original Input: {case.input}\n"
            f"Expected Output: {expected}\n"
            f"Actual Output: {response.output}\n\n"
            "Is the actual response correct compared to the expected response? "
            "Reply only YES or NO, followed by one sentence explaining why."
        )

        try:
            judge_response = self._runner.run_prompt(prompt)
        except Exception as e:
            raise ScorerError(f"Runner failed during LLM Judge evaluation: {e}") from e

        passed, accuracy, reasoning = self._parse_judge_response(judge_response.output)

        return Score(
            case_id=case.id,
            passed=passed,
            accuracy=accuracy,
            reasoning=reasoning,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    def _parse_judge_response(self, text: str) -> tuple[bool, float, str]:
        """Parses the judge text response.

        Args:
            text: Raw string response from the judge runner.

        Returns:
            A tuple of (passed, accuracy, reasoning).

        Raises:
            ScorerError: If format parsing fails.
        """
        judge_output = text.strip()
        upper_output = judge_output.upper()

        if upper_output.startswith("YES"):
            passed = True
            accuracy = 1.0
            reasoning = judge_output[3:].lstrip(".,;:- ").strip()
        elif upper_output.startswith("NO"):
            passed = False
            accuracy = 0.0
            reasoning = judge_output[2:].lstrip(".,;:- ").strip()
        else:
            raise ScorerError(
                f"Invalid judge response format: Response must start with YES or NO. Got: '{judge_output}'"
            )

        if not reasoning:
            raise ScorerError(
                f"Invalid judge response format: Missing explaining sentence. Got: '{judge_output}'"
            )

        return passed, accuracy, reasoning
