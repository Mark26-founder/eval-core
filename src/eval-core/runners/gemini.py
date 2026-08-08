"""Gemini runner implementation."""

import time
from typing import Any

import requests

from eval_core.cases.models import TestCase
from eval_core.runners.base import BaseRunner
from eval_core.runners.exceptions import RunnerError
from eval_core.runners.models import RunnerResponse


class GeminiRunner(BaseRunner):
    """Runner implementation for Gemini REST API."""

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str) -> None:
        """Initializes GeminiRunner.

        Args:
            api_key: Gemini API key.
            model: Model name/identifier.

        Raises:
            RunnerError: If api_key or model is empty/invalid.
        """
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise RunnerError("API key must be a non-empty string")
        if not model or not isinstance(model, str) or not model.strip():
            raise RunnerError("Model must be a non-empty string")

        self._api_key = api_key
        self._model = model

    def run(self, case: TestCase) -> RunnerResponse:
        """Executes one TestCase using Gemini API.

        Args:
            case: The evaluation TestCase object.

        Returns:
            A RunnerResponse object.

        Raises:
            RunnerError: On request failure, timeout, HTTP error, or invalid payload.
        """
        url = f"{self._BASE_URL}/{self._model}:generateContent"
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": case.input,
                        }
                    ]
                }
            ]
        }

        start_time = time.perf_counter()
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as err:
            raise RunnerError(f"Gemini request failed: {err}") from err
        except ValueError as err:
            raise RunnerError(f"Malformed JSON response from Gemini: {err}") from err
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return self._parse_response(data, latency_ms)

    def _parse_response(self, data: Any, latency_ms: float) -> RunnerResponse:
        """Parses response data from Gemini API.

        Args:
            data: Parsed JSON response.
            latency_ms: Measured latency in milliseconds.

        Returns:
            A RunnerResponse object.

        Raises:
            RunnerError: If payload or expected fields are missing/malformed.
        """
        if not isinstance(data, dict):
            raise RunnerError("Response JSON must be an object")

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or len(candidates) == 0:
            raise RunnerError("Missing or invalid 'candidates' in response")

        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            raise RunnerError("Candidate item must be an object")

        content = first_candidate.get("content")
        if not isinstance(content, dict):
            raise RunnerError("Missing or invalid 'content' in candidate")

        parts = content.get("parts")
        if not isinstance(parts, list) or len(parts) == 0:
            raise RunnerError("Missing or invalid 'parts' in content")

        first_part = parts[0]
        if not isinstance(first_part, dict):
            raise RunnerError("Part item must be an object")

        output = first_part.get("text")
        if not isinstance(output, str):
            raise RunnerError("Missing or invalid 'text' in part")

        usage_metadata = data.get("usageMetadata")
        if not isinstance(usage_metadata, dict):
            raise RunnerError("Missing or invalid 'usageMetadata' in response")

        input_tokens = usage_metadata.get("promptTokenCount")
        output_tokens = usage_metadata.get("candidatesTokenCount")

        if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
            raise RunnerError("Missing or invalid 'promptTokenCount' in usageMetadata")
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
            raise RunnerError("Missing or invalid 'candidatesTokenCount' in usageMetadata")

        return RunnerResponse(
            output=output,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
