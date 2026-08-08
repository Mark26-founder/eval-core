"""Validation logic for individual test case dictionaries."""

from typing import Any
from eval_core.cases.exceptions import CaseLoadError


def validate_case(data: Any) -> None:
    """Validates that a dictionary contains only valid and required fields for a TestCase.

    Note: Empty strings are allowed for 'input' and 'expected' fields, as they may
    represent valid test scenarios. 'id' must be a non-empty, non-whitespace string.

    Args:
        data: The raw data parsed from YAML representing a test case.

    Raises:
        CaseLoadError: If validation fails.
    """
    if not isinstance(data, dict):
        raise CaseLoadError(f"Test case must be a dictionary, got {type(data).__name__}")

    # Check for invalid fields
    allowed_keys = {"id", "input", "expected", "description", "tags"}
    extra_keys = set(data.keys()) - allowed_keys
    if extra_keys:
        raise CaseLoadError(
            f"Test case contains invalid fields: {', '.join(sorted(extra_keys))}"
        )

    # Check id
    if "id" not in data:
        raise CaseLoadError("Test case is missing the required field 'id'")
    case_id = data["id"]
    if not isinstance(case_id, str):
        raise CaseLoadError(f"Test case 'id' must be a string, got {type(case_id).__name__}")
    if not case_id.strip():
        raise CaseLoadError("Test case 'id' cannot be empty or only whitespace")

    # Check input
    if "input" not in data:
        raise CaseLoadError(f"Test case '{case_id}' is missing the required field 'input'")
    case_input = data["input"]
    if not isinstance(case_input, str):
        raise CaseLoadError(
            f"Test case '{case_id}' field 'input' must be a string, got {type(case_input).__name__}"
        )

    # Check expected
    if "expected" not in data:
        raise CaseLoadError(f"Test case '{case_id}' is missing the required field 'expected'")
    expected = data["expected"]
    if not isinstance(expected, str):
        raise CaseLoadError(
            f"Test case '{case_id}' field 'expected' must be a string, got {type(expected).__name__}"
        )

    # Check description (optional)
    if "description" in data:
        desc = data["description"]
        if desc is not None and not isinstance(desc, str):
            raise CaseLoadError(
                f"Test case '{case_id}' field 'description' must be a string or None, got {type(desc).__name__}"
            )

    # Check tags (optional)
    if "tags" in data:
        tags = data["tags"]
        if not isinstance(tags, list):
            raise CaseLoadError(
                f"Test case '{case_id}' field 'tags' must be a list of strings, got {type(tags).__name__}"
            )
        for i, tag in enumerate(tags):
            if not isinstance(tag, str):
                raise CaseLoadError(
                    f"Test case '{case_id}' field 'tags' must contain only strings, but element at index {i} is {type(tag).__name__}"
                )
