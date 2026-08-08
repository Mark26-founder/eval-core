"""OpenRouter runner implementation."""

import time
from typing import Any

import requests

from eval_core.cases.models import TestCase
from eval_core.runners.base import BaseRunner
from eval_core.runners.exceptions import RunnerError
from eval_core.runners.models import RunnerResponse


class OpenRouterRunner(BaseRunner):
    """Runner implementation for OpenRouter API."""

    _ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str) -> None:
        """Initializes OpenRouterRunner.

        Args:
            api_key: OpenRouter API key.
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
        """Executes one TestCase using OpenRouter API.

        Args:
            case: The evaluation TestCase object.

        Returns:
            A RunnerResponse object.

        Raises:
            RunnerError: On request failure, timeout, HTTP error, or invalid payload.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": case.input,
                }
            ],
        }

        start_time = time.perf_counter()
        try:
            response = requests.post(
                self._ENDPOINT,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as err:
            raise RunnerError(f"OpenRouter request failed: {err}") from err
        except ValueError as err:
            raise RunnerError(f"Malformed JSON response from OpenRouter: {err}") from err
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return self._parse_response(data, latency_ms)

    def _parse_response(self, data: Any, latency_ms: float) -> RunnerResponse:
        """Parses response data from OpenRouter API.

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

        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise RunnerError("Missing or invalid 'choices' in response")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RunnerError("Choice item must be an object")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RunnerError("Missing or invalid 'message' in choice")

        output = message.get("content")
        if not isinstance(output, str):
            raise RunnerError("Missing or invalid 'content' in message")

        usage = data.get("usage")
        if not isinstance(usage, dict):
            raise RunnerError("Missing or invalid 'usage' in response")

        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")

        if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
            raise RunnerError("Missing or invalid 'prompt_tokens' in usage")
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
            raise RunnerError("Missing or invalid 'completion_tokens' in usage")

        return RunnerResponse(
            output=output,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
