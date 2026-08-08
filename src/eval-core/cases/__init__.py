"""Cases package for loading and validating evaluation test cases.

Exposes only TestCase, load_cases, and CaseLoadError.
"""

from eval_core.cases.exceptions import CaseLoadError
from eval_core.cases.loader import load_cases
from eval_core.cases.models import TestCase

__all__ = [
    "TestCase",
    "load_cases",
    "CaseLoadError",
]
