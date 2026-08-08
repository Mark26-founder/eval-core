"""EvaluationEngine orchestration module."""

from pathlib import Path
from typing import Sequence

from eval_core.cases.models import TestCase
from eval_core.reports import Report, generate_report, load_report, save_report
from eval_core.runners.base import BaseRunner
from eval_core.scorers.base import BaseScorer
from eval_core.scorers.models import Score


class EvaluationEngine:
    """Orchestrates test case evaluation using runner, scorer, and report generator."""

    def __init__(self, runner: BaseRunner, scorer: BaseScorer) -> None:
        """Initializes EvaluationEngine.

        Args:
            runner: BaseRunner instance for generating model responses.
            scorer: BaseScorer instance for scoring responses.
        """
        self._runner = runner
        self._scorer = scorer

    def evaluate(
        self,
        cases: Sequence[TestCase],
        provider: str,
        model: str,
        previous_report_path: Path | str | None = None,
        output_report_path: Path | str | None = None,
    ) -> Report:
        """Executes full evaluation pipeline for a list of test cases.

        Pipeline:
        for each case
        ↓
        runner.run()
        ↓
        scorer.score()
        ↓
        collect Score objects
        ↓
        load previous report (if specified)
        ↓
        generate_report()
        ↓
        save report (if output path specified)
        ↓
        return Report

        Args:
            cases: Sequence of TestCase objects to evaluate.
            provider: Name of provider being evaluated.
            model: Name of model being evaluated.
            previous_report_path: Optional path to previous report JSON for comparison.
            output_report_path: Optional path to save the generated report JSON.

        Returns:
            The generated Report object.
        """
        scores: list[Score] = []
        for case in cases:
            response = self._runner.run(case)
            score = self._scorer.score(
                expected=case.expected,
                response=response,
                case=case,
            )
            scores.append(score)

        previous_report: Report | None = None
        if previous_report_path is not None:
            previous_report = load_report(previous_report_path)

        report = generate_report(
            cases=cases,
            scores=scores,
            provider=provider,
            model=model,
            previous_report=previous_report,
        )

        if output_report_path is not None:
            save_report(report, output_report_path)

        return report
