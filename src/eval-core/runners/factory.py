"""Runner factory implementation."""

from eval_core.runners.base import BaseRunner
from eval_core.runners.exceptions import RunnerError
from eval_core.runners.gemini import GeminiRunner
from eval_core.runners.openrouter import OpenRouterRunner


def get_runner(provider: str, api_key: str, model: str) -> BaseRunner:
    """Factory function to instantiate model runners.

    Args:
        provider: Provider identifier string (e.g. 'openrouter', 'gemini').
        api_key: Provider API key string.
        model: Model identifier string.

    Returns:
        An instance of BaseRunner subclass.

    Raises:
        RunnerError: If provider is unknown or invalid.
    """
    if not isinstance(provider, str):
        raise RunnerError(f"Unsupported provider: {provider}")

    normalized = provider.strip().lower()

    if normalized == "openrouter":
        return OpenRouterRunner(api_key=api_key, model=model)
    elif normalized == "gemini":
        return GeminiRunner(api_key=api_key, model=model)

    raise RunnerError(f"Unsupported provider: {provider}")
