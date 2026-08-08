"""Unit tests for src/eval_core/runners package."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from eval_core.cases.models import TestCase
from eval_core.runners import (
    BaseRunner,
    GeminiRunner,
    OpenRouterRunner,
    RunnerError,
    RunnerResponse,
    get_runner,
)


@pytest.fixture
def sample_test_case() -> TestCase:
    """Fixture providing a standard TestCase instance."""
    return TestCase(
        id="test-001",
        input="Hello LLM",
        expected="Hello User",
    )


# ============================================================================
# RunnerResponse Tests
# ============================================================================


def test_runner_response_creation_valid():
    """Tests successful creation of RunnerResponse with valid values."""
    res = RunnerResponse(
        output="Test output",
        latency_ms=123.45,
        input_tokens=10,
        output_tokens=20,
    )
    assert res.output == "Test output"
    assert res.latency_ms == 123.45
    assert res.input_tokens == 10
    assert res.output_tokens == 20


def test_runner_response_validation_negative_latency():
    """Tests validation error on negative latency_ms."""
    with pytest.raises(RunnerError, match="latency_ms must be non-negative"):
        RunnerResponse(
            output="Test",
            latency_ms=-1.0,
            input_tokens=10,
            output_tokens=20,
        )


def test_runner_response_validation_negative_input_tokens():
    """Tests validation error on negative input_tokens."""
    with pytest.raises(RunnerError, match="input_tokens must be non-negative"):
        RunnerResponse(
            output="Test",
            latency_ms=10.0,
            input_tokens=-5,
            output_tokens=20,
        )


def test_runner_response_validation_negative_output_tokens():
    """Tests validation error on negative output_tokens."""
    with pytest.raises(RunnerError, match="output_tokens must be non-negative"):
        RunnerResponse(
            output="Test",
            latency_ms=10.0,
            input_tokens=10,
            output_tokens=-1,
        )


def test_runner_response_immutability():
    """Tests that RunnerResponse is immutable (frozen dataclass)."""
    res = RunnerResponse(
        output="Test",
        latency_ms=10.0,
        input_tokens=5,
        output_tokens=5,
    )
    with pytest.raises(Exception):
        res.output = "Modified"  # type: ignore[misc]


# ============================================================================
# OpenRouterRunner Tests
# ============================================================================


def test_openrouter_missing_or_invalid_api_key():
    """Tests OpenRouterRunner constructor with missing or empty API key."""
    with pytest.raises(RunnerError, match="API key must be a non-empty string"):
        OpenRouterRunner(api_key="", model="openai/gpt-4o")

    with pytest.raises(RunnerError, match="API key must be a non-empty string"):
        OpenRouterRunner(api_key="   ", model="openai/gpt-4o")


def test_openrouter_missing_or_invalid_model():
    """Tests OpenRouterRunner constructor with missing or empty model."""
    with pytest.raises(RunnerError, match="Model must be a non-empty string"):
        OpenRouterRunner(api_key="sk-or-123", model="")


@patch("requests.post")
def test_openrouter_request_formatting_and_success(mock_post, sample_test_case):
    """Tests OpenRouter request formatting and parsing of a successful response."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "OpenRouter generated text",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
        },
    }
    mock_post.return_value = mock_response

    runner = OpenRouterRunner(api_key="test-key", model="anthropic/claude-3-5-sonnet")
    response = runner.run(sample_test_case)

    # Verify request call
    mock_post.assert_called_once_with(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        json={
            "model": "anthropic/claude-3-5-sonnet",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello LLM",
                }
            ],
        },
    )

    # Verify RunnerResponse
    assert response.output == "OpenRouter generated text"
    assert response.input_tokens == 15
    assert response.output_tokens == 25
    assert response.latency_ms >= 0


@patch("requests.post")
def test_openrouter_timeout(mock_post, sample_test_case):
    """Tests handling of network timeout in OpenRouterRunner."""
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
    runner = OpenRouterRunner(api_key="test-key", model="openai/gpt-4o")

    with pytest.raises(RunnerError, match="OpenRouter request failed"):
        runner.run(sample_test_case)


@patch("requests.post")
def test_openrouter_http_error(mock_post, sample_test_case):
    """Tests handling of HTTP errors in OpenRouterRunner."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
    mock_post.return_value = mock_response

    runner = OpenRouterRunner(api_key="test-key", model="openai/gpt-4o")

    with pytest.raises(RunnerError, match="OpenRouter request failed"):
        runner.run(sample_test_case)


@patch("requests.post")
def test_openrouter_invalid_json(mock_post, sample_test_case):
    """Tests handling of malformed JSON response in OpenRouterRunner."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_post.return_value = mock_response

    runner = OpenRouterRunner(api_key="test-key", model="openai/gpt-4o")

    with pytest.raises(RunnerError, match="Malformed JSON response from OpenRouter"):
        runner.run(sample_test_case)


@patch("requests.post")
def test_openrouter_missing_choices(mock_post, sample_test_case):
    """Tests handling of missing choices field in OpenRouter payload."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"usage": {"prompt_tokens": 5, "completion_tokens": 5}}
    mock_post.return_value = mock_response

    runner = OpenRouterRunner(api_key="test-key", model="openai/gpt-4o")

    with pytest.raises(RunnerError, match="Missing or invalid 'choices'"):
        runner.run(sample_test_case)


@patch("requests.post")
def test_openrouter_missing_usage_fields(mock_post, sample_test_case):
    """Tests handling of missing prompt_tokens/completion_tokens in OpenRouter payload."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hi"}}],
        "usage": {"prompt_tokens": 10},
    }
    mock_post.return_value = mock_response

    runner = OpenRouterRunner(api_key="test-key", model="openai/gpt-4o")

    with pytest.raises(RunnerError, match="Missing or invalid 'completion_tokens'"):
        runner.run(sample_test_case)


# ============================================================================
# GeminiRunner Tests
# ============================================================================


def test_gemini_missing_or_invalid_api_key():
    """Tests GeminiRunner constructor with missing or empty API key."""
    with pytest.raises(RunnerError, match="API key must be a non-empty string"):
        GeminiRunner(api_key="", model="gemini-1.5-flash")


def test_gemini_missing_or_invalid_model():
    """Tests GeminiRunner constructor with missing or empty model."""
    with pytest.raises(RunnerError, match="Model must be a non-empty string"):
        GeminiRunner(api_key="gem-key", model="")


@patch("requests.post")
def test_gemini_request_formatting_and_success(mock_post, sample_test_case):
    """Tests Gemini request formatting and parsing of a successful response."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Gemini generated text",
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 24,
        },
    }
    mock_post.return_value = mock_response

    runner = GeminiRunner(api_key="gemini-key", model="gemini-1.5-pro")
    response = runner.run(sample_test_case)

    # Verify request call
    mock_post.assert_called_once_with(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        headers={
            "x-goog-api-key": "gemini-key",
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Hello LLM",
                        }
                    ]
                }
            ]
        },
    )

    # Verify RunnerResponse
    assert response.output == "Gemini generated text"
    assert response.input_tokens == 12
    assert response.output_tokens == 24
    assert response.latency_ms >= 0


@patch("requests.post")
def test_gemini_timeout(mock_post, sample_test_case):
    """Tests handling of network timeout in GeminiRunner."""
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
    runner = GeminiRunner(api_key="gemini-key", model="gemini-1.5-flash")

    with pytest.raises(RunnerError, match="Gemini request failed"):
        runner.run(sample_test_case)


@patch("requests.post")
def test_gemini_http_error(mock_post, sample_test_case):
    """Tests handling of HTTP errors in GeminiRunner."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
    mock_post.return_value = mock_response

    runner = GeminiRunner(api_key="gemini-key", model="gemini-1.5-flash")

    with pytest.raises(RunnerError, match="Gemini request failed"):
        runner.run(sample_test_case)


@patch("requests.post")
def test_gemini_invalid_json(mock_post, sample_test_case):
    """Tests handling of malformed JSON response in GeminiRunner."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_post.return_value = mock_response

    runner = GeminiRunner(api_key="gemini-key", model="gemini-1.5-flash")

    with pytest.raises(RunnerError, match="Malformed JSON response from Gemini"):
        runner.run(sample_test_case)


@patch("requests.post")
def test_gemini_missing_candidates(mock_post, sample_test_case):
    """Tests handling of missing candidates field in Gemini response."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5}
    }
    mock_post.return_value = mock_response

    runner = GeminiRunner(api_key="gemini-key", model="gemini-1.5-flash")

    with pytest.raises(RunnerError, match="Missing or invalid 'candidates'"):
        runner.run(sample_test_case)


@patch("requests.post")
def test_gemini_missing_usage_metadata(mock_post, sample_test_case):
    """Tests handling of missing usageMetadata field in Gemini response."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello"}]
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    runner = GeminiRunner(api_key="gemini-key", model="gemini-1.5-flash")

    with pytest.raises(RunnerError, match="Missing or invalid 'usageMetadata'"):
        runner.run(sample_test_case)


# ============================================================================
# Factory Tests
# ============================================================================


def test_factory_get_runner_openrouter():
    """Tests factory creation of OpenRouterRunner with provider string normalization."""
    runner = get_runner(provider=" OpenRouter ", api_key="test-key", model="openai/gpt-4o")
    assert isinstance(runner, OpenRouterRunner)
    assert isinstance(runner, BaseRunner)


def test_factory_get_runner_gemini():
    """Tests factory creation of GeminiRunner with provider string normalization."""
    runner = get_runner(provider="GEMINI", api_key="test-key", model="gemini-1.5-pro")
    assert isinstance(runner, GeminiRunner)
    assert isinstance(runner, BaseRunner)


def test_factory_unknown_provider():
    """Tests factory raising RunnerError for unknown providers."""
    with pytest.raises(RunnerError, match="Unsupported provider: anthropic"):
        get_runner(provider="anthropic", api_key="test-key", model="claude-3-opus")


def test_factory_invalid_provider_type():
    """Tests factory raising RunnerError for non-string provider parameter."""
    with pytest.raises(RunnerError, match="Unsupported provider"):
        get_runner(provider=123, api_key="test-key", model="model")  # type: ignore[arg-type]
