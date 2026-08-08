"""Typer CLI interface for EVAL-CORE."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import typer
from rich.console import Console

from eval_core.cases.loader import load_cases
from eval_core.cli.formatter import print_report
from eval_core.engine import EvaluationEngine
from eval_core.runners import get_runner
from eval_core.scorers import get_scorer

app = typer.Typer(
    name="eval-core",
    help="Lightweight LLM regression testing and quality assurance framework.",
    add_completion=False,
)

console = Console()


@app.callback(invoke_without_command=False)
def callback():
    """EVAL-CORE CLI - LLM regression testing and quality assurance framework."""
    pass


@app.command("run", help="Executes an evaluation test suite and prints/saves the evaluation report.")
def run(
    suite: Path = typer.Option(
        ...,
        "--suite",
        "-s",
        help="Path to evaluation test suite YAML file.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider (e.g. openrouter, gemini). Defaults to EVAL_CORE_PROVIDER env var.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model identifier. Defaults to EVAL_CORE_MODEL env var.",
    ),
    scorer: str = typer.Option(
        "exact",
        "--scorer",
        help="Scorer type ('exact' or 'llm-judge').",
    ),
    previous_report: Optional[Path] = typer.Option(
        None,
        "--previous-report",
        help="Path to previous report JSON for regression comparison.",
    ),
    output_report: Optional[Path] = typer.Option(
        None,
        "--output-report",
        help="Path where generated report JSON will be saved.",
    ),
) -> None:
    load_dotenv()

    if not suite.exists() or not suite.is_file():
        console.print(f"[bold red]Error:[/bold red] Test suite file not found: '{suite}'")
        raise typer.Exit(code=1)

    if previous_report is not None and (not previous_report.exists() or not previous_report.is_file()):
        console.print(f"[bold red]Error:[/bold red] Previous report file not found: '{previous_report}'")
        raise typer.Exit(code=1)

    resolved_provider = provider or os.getenv("EVAL_CORE_PROVIDER") or os.getenv("PROVIDER")
    resolved_model = model or os.getenv("EVAL_CORE_MODEL") or os.getenv("MODEL")

    if not resolved_provider:
        console.print(
            "[bold red]Error:[/bold red] Provider not specified. Use --provider or set EVAL_CORE_PROVIDER env var."
        )
        raise typer.Exit(code=1)

    if not resolved_model:
        console.print(
            "[bold red]Error:[/bold red] Model not specified. Use --model or set EVAL_CORE_MODEL env var."
        )
        raise typer.Exit(code=1)

    # Provider API key lookup
    api_key_env_var = f"{resolved_provider.upper()}_API_KEY"
    api_key = os.getenv(api_key_env_var) or os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print(
            f"[bold red]Error:[/bold red] Missing API key for provider '{resolved_provider}'. Set {api_key_env_var} environment variable."
        )
        raise typer.Exit(code=1)

    try:
        cases = load_cases(suite)
        runner_instance = get_runner(
            provider=resolved_provider,
            api_key=api_key,
            model=resolved_model,
        )
        scorer_instance = get_scorer(
            scorer_type=scorer,
            runner=runner_instance if scorer.lower().strip() == "llm-judge" else None,
        )

        engine = EvaluationEngine(runner=runner_instance, scorer=scorer_instance)
        report = engine.evaluate(
            cases=cases,
            provider=resolved_provider,
            model=resolved_model,
            previous_report_path=previous_report,
            output_report_path=output_report,
        )

        print_report(report, console=console)

        if report.regression_detected:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    except typer.Exit:
        raise
    except Exception as err:
        console.print(f"[bold red]Execution Error:[/bold red] {err}")
        raise typer.Exit(code=1)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
