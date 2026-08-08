"""Abstract base class for all runners."""

from abc import ABC, abstractmethod

from eval_core.cases.models import TestCase
from eval_core.runners.models import RunnerResponse


class BaseRunner(ABC):
    """Abstract base class for all model runners."""

    @abstractmethod
    def run(self, case: TestCase) -> RunnerResponse:
        """Runs the model on the given TestCase.

        Args:
            case: The evaluation TestCase object.

        Returns:
            A RunnerResponse object.
        """
        ...

    def run_prompt(self, prompt: str) -> RunnerResponse:
        """Convenience method for running a raw prompt string.

        This method creates a temporary TestCase and delegates to run().
        Used primarily by LLMJudgeScorer for evaluating responses.

        Args:
            prompt: The raw prompt string to send to the model.

        Returns:
            A RunnerResponse object.
        """
        temp_case = TestCase(
            id="_prompt",
            input=prompt,
            expected="",
        )
        return self.run(temp_case)
