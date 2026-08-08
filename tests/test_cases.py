from pathlib import Path
import tempfile
import pytest
import yaml

from eval_core.cases import (
    load_cases,
    TestCase as EvaluationTestCase,
    CaseLoadError,
)


def create_temp_yaml(data: dict) -> Path:
    temp_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    yaml.dump(data, temp_file)
    temp_file.close()
    return Path(temp_file.name)


def test_load_cases_success():
    data = {
        "cases": [
            {
                "id": "case_1",
                "input": "hello",
                "expected": "world",
                "description": "test description",
                "tags": ["smoke", "fast"],
            },
            {
                "id": "case_2",
                "input": "ping",
                "expected": "pong",
            }
        ]
    }
    path = create_temp_yaml(data)
    try:
        cases = load_cases(path)
        assert len(cases) == 2
        assert isinstance(cases[0], EvaluationTestCase)
        assert isinstance(cases[1], EvaluationTestCase)
        assert cases[0].id == "case_1"
        assert cases[0].input == "hello"
        assert cases[0].expected == "world"
        assert cases[0].description == "test description"
        assert cases[0].tags == ["smoke", "fast"]

        assert cases[1].id == "case_2"
        assert cases[1].input == "ping"
        assert cases[1].expected == "pong"
        assert cases[1].description is None
        assert cases[1].tags == []
    finally:
        path.unlink()


def test_load_cases_missing_file():
    with pytest.raises(CaseLoadError, match="File not found"):
        load_cases("non_existent_file.yaml")


def test_load_cases_invalid_yaml():
    temp_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    temp_file.write("cases:\n  - id: 1\n  invalid yaml content here:")
    temp_file.close()
    path = Path(temp_file.name)
    try:
        with pytest.raises(CaseLoadError, match="Invalid YAML syntax"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_missing_cases_key():
    data = {"not_cases": []}
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="Missing required key 'cases'"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_non_list_cases():
    data = {"cases": "not a list"}
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="must map to a list"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_duplicate_ids():
    data = {
        "cases": [
            {"id": "dup", "input": "a", "expected": "b"},
            {"id": "dup", "input": "c", "expected": "d"},
        ]
    }
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="Duplicate test case ID"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_invalid_fields():
    data = {
        "cases": [
            {"id": "case_1", "input": "a", "expected": "b", "unknown_field": "error"}
        ]
    }
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="invalid fields: unknown_field"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_missing_required_fields():
    # Missing input
    data = {
        "cases": [
            {"id": "case_1", "expected": "b"}
        ]
    }
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="missing the required field 'input'"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_invalid_field_types():
    data = {
        "cases": [
            {"id": "case_1", "input": "a", "expected": "b", "tags": [1, 2]}
        ]
    }
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="must contain only strings"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_whitespace_only_id():
    data = {
        "cases": [
            {"id": "   ", "input": "hello", "expected": "world"}
        ]
    }
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="cannot be empty or only whitespace"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_non_string_input():
    data = {
        "cases": [
            {"id": "case_1", "input": 123, "expected": "abc"}
        ]
    }
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="field 'input' must be a string"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_non_string_expected():
    data = {
        "cases": [
            {"id": "case_1", "input": "hello", "expected": True}
        ]
    }
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="field 'expected' must be a string"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_yaml_root_is_list():
    data = [
        {"id": "case_1", "input": "hello", "expected": "world"}
    ]
    path = create_temp_yaml(data)
    try:
        with pytest.raises(CaseLoadError, match="YAML root must be a dictionary"):
            load_cases(path)
    finally:
        path.unlink()


def test_load_cases_allow_empty_strings():
    data = {
        "cases": [
            {"id": "case_1", "input": "", "expected": ""},
        ]
    }
    path = create_temp_yaml(data)
    try:
        cases = load_cases(path)
        assert len(cases) == 1
        assert isinstance(cases[0], EvaluationTestCase)
        assert cases[0].input == ""
        assert cases[0].expected == ""
    finally:
        path.unlink()

