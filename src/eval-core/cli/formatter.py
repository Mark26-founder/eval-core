"""Rich formatter helper module for CLI output."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval_core.reports.models import Report


def print_report(report: Report, console: Console | None = None) -> None:
    """Prints a styled report to terminal using Rich.

    Args:
        report: The Report object to format and display.
        console: Optional Console instance (defaults to standard Console).
    """
    if console is None:
        console = Console()

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Provider:", report.provider)
    table.add_row("Model:", report.model)
    table.add_row("Cases:", str(report.total_cases))
    table.add_row("Passed:", f"[green]{report.passed}[/green]")
    table.add_row("Failed:", f"[red]{report.failed}[/red]" if report.failed > 0 else "0")
    table.add_row("Accuracy:", f"{report.accuracy * 100:.1f}%")
    table.add_row("Average latency:", f"{report.average_latency_ms:.2f} ms")
    table.add_row("Input tokens:", str(report.total_input_tokens))
    table.add_row("Output tokens:", str(report.total_output_tokens))

    reg_status = (
        "[bold red]YES[/bold red]" if report.regression_detected else "[bold green]NO[/bold green]"
    )
    table.add_row("Regression detected:", reg_status)

    if report.regression_detected and report.regression_details:
        reg_cases = ", ".join(report.regression_details.cases_regressed)
        if reg_cases:
            table.add_row("Regressed cases:", f"[bold red]{reg_cases}[/bold red]")
        if report.regression_details.accuracy_change != 0:
            table.add_row(
                "Accuracy change:",
                f"{report.regression_details.accuracy_change * 100:+.1f}%",
            )

    panel = Panel(
        table,
        title="[bold yellow]EVAL-CORE REPORT[/bold yellow]",
        border_style="bright_blue",
        expand=False,
    )

    console.print()
    console.print(panel)
    console.print()
