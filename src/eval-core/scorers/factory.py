"""Scorer factory module."""

from eval_core.runners.base import BaseRunner
from eval_core.scorers.base import BaseScorer
from eval_core.scorers.exact import ExactScorer
from eval_core.scorers.exceptions import ScorerError
from eval_core.scorers.llm_judge import LLMJudgeScorer


def get_scorer(
    scorer_type: str,
    runner: BaseRunner | None = None,
) -> BaseScorer:
    """Returns a scorer instance based on the scorer type.

    Args:
        scorer_type: The type of scorer ('exact' or 'llm-judge').
        runner: Optional BaseRunner instance required for the 'llm-judge' scorer.

    Returns:
        A BaseScorer instance.

    Raises:
        ScorerError: If the scorer type is unknown, or if 'llm-judge' is requested
            without providing a runner.
    """
    normalized_type = scorer_type.lower().strip()
    if normalized_type == "exact":
        return ExactScorer()
    elif normalized_type == "llm-judge":
        if runner is None:
            raise ScorerError("runner must be provided for 'llm-judge' scorer")
        return LLMJudgeScorer(runner)
    else:
        raise ScorerError(f"Unknown scorer type: '{scorer_type}'")
