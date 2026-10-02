"""
The repair loop: when extraction fails validation, don't just retry blindly.
Build a targeted correction prompt that tells the model EXACTLY what was
wrong, and ask it to fix only that.

In production, build this on top of the `instructor` library (which already
handles schema-constrained decoding + basic retries against a real LLM) --
add this module's targeted-repair-prompt logic on top of it. Here, extraction
is modeled as `extract_fn: Callable[[str], T]` so the loop is fully testable
with a fake/mock extractor and no network access.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TypeVar, Generic
from .schema import validate, ValidationResult

T = TypeVar("T")


@dataclass
class ExtractionResult(Generic[T]):
    value: T | None
    valid: bool
    attempts: int
    validation_history: list[ValidationResult]


class RepairLoop:
    """
    extract_fn(prompt: str) -> T                     -- your LLM extraction call
    Usage:
        loop = RepairLoop(extract_fn=my_llm_extract, max_attempts=3)
        result = loop.run(initial_prompt="Extract the event from: ...")
    """

    def __init__(self, extract_fn: Callable[[str], T], max_attempts: int = 3):
        self.extract_fn = extract_fn
        self.max_attempts = max_attempts

    def run(self, initial_prompt: str) -> ExtractionResult:
        prompt = initial_prompt
        history: list[ValidationResult] = []

        for attempt in range(1, self.max_attempts + 1):
            value = self.extract_fn(prompt)
            result = validate(value)
            history.append(result)

            if result.valid:
                return ExtractionResult(value=value, valid=True, attempts=attempt,
                                         validation_history=history)

            # Build a TARGETED correction prompt -- this is the key idea:
            # tell the model exactly what's wrong instead of a generic retry.
            error_lines = "\n".join(f"- {e.field_or_rule}: {e.message}" for e in result.errors)
            prompt = (
                f"{initial_prompt}\n\n"
                f"Your previous answer had these specific problems:\n{error_lines}\n"
                f"Please provide a corrected answer that fixes ONLY these issues."
            )

        return ExtractionResult(value=value, valid=False, attempts=self.max_attempts,
                                 validation_history=history)
