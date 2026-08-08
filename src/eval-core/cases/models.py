"""Data models for evaluation test cases."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TestCase:
    """Immutable representation of an evaluation test case.

    Attributes:
        id: Unique identifier for the test case.
        input: The input prompt or string.
        expected: The expected reference output.
        description: Optional description of the test case.
        tags: List of tags associated with the test case.
    """

    id: str
    input: str
    expected: str
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
