"""Loader module to parse and validate evaluation test cases from YAML files."""

from pathlib import Path
from typing import Any
import yaml

from eval_core.cases.exceptions import CaseLoadError
from eval_core.cases.models import TestCase
from eval_core.cases.validator import validate_case


def load_cases(path: Path | str) -> list[TestCase]:
    """Loads and validates test cases from a YAML file.

    Note: Empty strings are allowed for 'input' and 'expected' fields, as they may
    represent valid test scenarios. 'id' must be a non-empty, non-whitespace string.

    Args:
        path: Path to the YAML file.

    Returns:
        A list of validated TestCase objects.

    Raises:
        CaseLoadError: If the file does not exist, is invalid YAML, or contains
            invalid/duplicate test cases.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise CaseLoadError(f"File not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise CaseLoadError(f"Invalid YAML syntax in file {file_path}: {e}") from e
    except Exception as e:
        raise CaseLoadError(f"Error reading file {file_path}: {e}") from e

    if not isinstance(content, dict):
        raise CaseLoadError(f"YAML root must be a dictionary, got {type(content).__name__}")

    if "cases" not in content:
        raise CaseLoadError("Missing required key 'cases' at YAML root")

    raw_cases = content["cases"]
    if not isinstance(raw_cases, list):
        raise CaseLoadError(f"The 'cases' key must map to a list, got {type(raw_cases).__name__}")

    seen_ids: set[str] = set()
    validated_cases: list[TestCase] = []

    for index, case_data in enumerate(raw_cases):
        try:
            validate_case(case_data)
        except CaseLoadError as e:
            raise CaseLoadError(f"Validation failed for case at index {index}: {e}") from e

        case_id = case_data["id"]
        if case_id in seen_ids:
            raise CaseLoadError(f"Duplicate test case ID found: '{case_id}'")
        seen_ids.add(case_id)

        tags = list(case_data.get("tags", []))

        test_case = TestCase(
            id=case_id,
            input=case_data["input"],
            expected=case_data["expected"],
            description=case_data.get("description"),
            tags=tags,
        )
        validated_cases.append(test_case)

    return validated_cases
