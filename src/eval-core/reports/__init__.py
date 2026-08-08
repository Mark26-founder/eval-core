"""Reports package initialization.

Exposes only Report, RegressionDetails, generate_report, compare_reports,
save_report, load_report, and ReportError.
"""

from eval_core.reports.comparator import compare_reports
from eval_core.reports.exceptions import ReportError
from eval_core.reports.generator import generate_report
from eval_core.reports.models import RegressionDetails, Report
from eval_core.reports.persistence import load_report, save_report

__all__ = [
    "Report",
    "RegressionDetails",
    "generate_report",
    "compare_reports",
    "save_report",
    "load_report",
    "ReportError",
]
