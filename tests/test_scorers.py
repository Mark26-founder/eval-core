import pytest

from eval_core.cases.models import TestCase
from eval_core.runners import BaseRunner, RunnerResponse
from eval_core.scorers import (
    Score,
    BaseScorer,
    ExactScorer,
    LLMJudgeScorer,
    ScorerError,
    get_scorer,
)


class MockRunner(BaseRunner):
    """Simple mock runner for testing LLMJudgeScorer."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_case = None

    def run(self, case: TestCase) -> RunnerResponse:
        self.last_case = case
        return RunnerResponse(
            output=self.response_text,
            latency_ms=150.0,
            input_tokens=10,
            output_tokens=5,
        )


def test_exact_scorer_match():
    scorer = ExactScorer()
    case = TestCase(id="case_1", input="hello", expected="world")
    response = RunnerResponse(
        output="world",
        latency_ms=100.0,
        input_tokens=5,
        output_tokens=5,
    )

    score = scorer.score("world", response, case)

    assert isinstance(score, Score)
    assert score.case_id == "case_1"
    assert score.passed is True
    assert score.accuracy == 1.0
    assert score.reasoning == "Exact match"
    assert score.latency_ms == 100.0
    assert score.input_tokens == 5
    assert score.output_tokens == 5


def test_exact_scorer_differ():
    scorer = ExactScorer()
    case = TestCase(id="case_2", input="hello", expected="world")
    response = RunnerResponse(
        output="different",
        latency_ms=100.0,
        input_tokens=5,
        output_tokens=5,
    )

    score = scorer.score("world", response, case)

    assert score.case_id == "case_2"
    assert score.passed is False
    assert score.accuracy == 0.0
    assert score.reasoning == "Outputs differ"


def test_exact_scorer_whitespace_trimming():
    scorer = ExactScorer()
    case = TestCase(id="case_3", input="hello", expected="  world  ")
    response = RunnerResponse(
        output="\nworld\t",
        latency_ms=50.0,
        input_tokens=4,
        output_tokens=4,
    )

    score = scorer.score("  world  ", response, case)

    assert score.passed is True
    assert score.accuracy == 1.0
    assert score.reasoning == "Exact match"


def test_factory_creation():
    exact_scorer = get_scorer("exact")
    assert isinstance(exact_scorer, ExactScorer)

    mock_runner = MockRunner("YES. Correct.")
    judge_scorer = get_scorer("llm-judge", runner=mock_runner)
    assert isinstance(judge_scorer, LLMJudgeScorer)


def test_factory_normalization():
    exact_scorer = get_scorer("  Exact ")
    assert isinstance(exact_scorer, ExactScorer)

    mock_runner = MockRunner("YES. Correct.")
    judge_scorer = get_scorer(" LLM-JUDGE ", runner=mock_runner)
    assert isinstance(judge_scorer, LLMJudgeScorer)


def test_factory_unknown_scorer():
    with pytest.raises(ScorerError, match="Unknown scorer type"):
        get_scorer("unknown")


def test_factory_missing_runner():
    with pytest.raises(ScorerError, match="runner must be provided"):
        get_scorer("llm-judge")


def test_llm_judge_scorer_yes_passed():
    runner = MockRunner("YES. The actual output is semantically matching the expected output.")
    scorer = LLMJudgeScorer(runner)
    case = TestCase(id="case_1", input="tell a joke", expected="joke")
    response = RunnerResponse(
        output="funny response",
        latency_ms=250.0,
        input_tokens=12,
        output_tokens=15,
    )

    score = scorer.score("joke", response, case)

    assert score.case_id == "case_1"
    assert score.passed is True
    assert score.accuracy == 1.0
    assert score.reasoning == "The actual output is semantically matching the expected output."
    assert score.latency_ms == 250.0
    assert score.input_tokens == 12
    assert score.output_tokens == 15
    assert runner.last_case is not None
    assert "Is the actual response correct" in runner.last_case.input


def test_llm_judge_scorer_delimiters():
    # Test verschiedene delimiter/Trennzeichen nach YES/NO
    cases_to_test = [
        ("YES - Correct response", "Correct response", True, 1.0),
        ("YES: Correct response", "Correct response", True, 1.0),
        ("YES Correct response", "Correct response", True, 1.0),
        ("NO - Incorrect response", "Incorrect response", False, 0.0),
        ("NO: Incorrect response", "Incorrect response", False, 0.0),
        ("NO Incorrect response", "Incorrect response", False, 0.0),
    ]
    for raw_judge, expected_reason, expected_passed, expected_acc in cases_to_test:
        runner = MockRunner(raw_judge)
        scorer = LLMJudgeScorer(runner)
        case = TestCase(id="case_x", input="in", expected="exp")
        response = RunnerResponse(output="out", latency_ms=100.0, input_tokens=5, output_tokens=5)
        score = scorer.score("exp", response, case)
        assert score.passed == expected_passed
        assert score.accuracy == expected_acc
        assert score.reasoning == expected_reason


def test_llm_judge_scorer_no_failed():
    runner = MockRunner("NO. The response is completely wrong.")
    scorer = LLMJudgeScorer(runner)
    case = TestCase(id="case_1", input="tell a joke", expected="joke")
    response = RunnerResponse(
        output="sad response",
        latency_ms=250.0,
        input_tokens=12,
        output_tokens=15,
    )

    score = scorer.score("joke", response, case)

    assert score.passed is False
    assert score.accuracy == 0.0
    assert score.reasoning == "The response is completely wrong."


def test_llm_judge_scorer_invalid_format_no_yes_no():
    runner = MockRunner("Maybe. I am not sure.")
    scorer = LLMJudgeScorer(runner)
    case = TestCase(id="case_1", input="input", expected="expected")
    response = RunnerResponse(output="output", latency_ms=100.0, input_tokens=5, output_tokens=5)

    with pytest.raises(ScorerError, match="Response must start with YES or NO"):
        scorer.score("expected", response, case)


def test_llm_judge_scorer_invalid_format_missing_sentence():
    runner = MockRunner("YES.")
    scorer = LLMJudgeScorer(runner)
    case = TestCase(id="case_1", input="input", expected="expected")
    response = RunnerResponse(output="output", latency_ms=100.0, input_tokens=5, output_tokens=5)

    with pytest.raises(ScorerError, match="Missing explaining sentence"):
        scorer.score("expected", response, case)


def test_llm_judge_scorer_runner_exception():
    class FailingRunner(BaseRunner):
        def run(self, case: TestCase) -> RunnerResponse:
            raise RuntimeError("API failure")

    scorer = LLMJudgeScorer(FailingRunner())
    case = TestCase(id="case_1", input="input", expected="expected")
    response = RunnerResponse(output="output", latency_ms=100.0, input_tokens=5, output_tokens=5)

    with pytest.raises(ScorerError, match="Runner failed during LLM Judge evaluation"):
        scorer.score("expected", response, case)


def test_score_validation():
    # Valid construction
    score = Score("c1", True, 1.0, "reason", 10.0, 5, 5)
    assert score.accuracy == 1.0

    # Invalid accuracy
    with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
        Score("c1", True, 1.5, "reason", 10.0, 5, 5)

    with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
        Score("c1", True, -0.1, "reason", 10.0, 5, 5)

    # Invalid latency
    with pytest.raises(ValueError, match="latency_ms must be non-negative"):
        Score("c1", True, 0.5, "reason", -10.0, 5, 5)

    # Invalid input tokens
    with pytest.raises(ValueError, match="input_tokens must be non-negative"):
        Score("c1", True, 0.5, "reason", 10.0, -1, 5)

    # Invalid output tokens
    with pytest.raises(ValueError, match="output_tokens must be non-negative"):
        Score("c1", True, 0.5, "reason", 10.0, 5, -5)

