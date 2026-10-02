"""
Before/after evaluation for fine-tuned models, with PER-EXAMPLE regression
detection -- not just an aggregate score.

The actual fine-tuning (LoRA/QLoRA training) is delegated to a real backend
in production -- e.g. Unsloth or peft/transformers directly. This module
only concerns itself with the "did it actually get better, and where did it
get worse" question, which most fine-tuning tutorials skip. Model calls are
injected as plain callables so this is fully testable offline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal


@dataclass
class EvalCase:
    """One held-out example: an input plus a way to check correctness."""
    input: str
    expected_substring: str  # crude, dependency-free correctness check
    tags: list[str] = field(default_factory=list)  # e.g. ["edge_case", "common"]


@dataclass
class ExampleComparison:
    input: str
    base_output: str
    finetuned_output: str
    base_correct: bool
    finetuned_correct: bool
    verdict: Literal["improved", "regressed", "unchanged_correct", "unchanged_incorrect"]
    tags: list[str]


@dataclass
class ComparisonResult:
    comparisons: list[ExampleComparison]

    @property
    def base_accuracy(self) -> float:
        n = len(self.comparisons)
        return sum(c.base_correct for c in self.comparisons) / n if n else 0.0

    @property
    def finetuned_accuracy(self) -> float:
        n = len(self.comparisons)
        return sum(c.finetuned_correct for c in self.comparisons) / n if n else 0.0

    @property
    def regressions(self) -> list[ExampleComparison]:
        return [c for c in self.comparisons if c.verdict == "regressed"]

    @property
    def improvements(self) -> list[ExampleComparison]:
        return [c for c in self.comparisons if c.verdict == "improved"]

    def summary(self) -> str:
        return (
            f"Base accuracy: {self.base_accuracy:.0%}  ->  Fine-tuned accuracy: {self.finetuned_accuracy:.0%}\n"
            f"Improved: {len(self.improvements)}  |  Regressed: {len(self.regressions)}  |  "
            f"Total examples: {len(self.comparisons)}"
        )


class EvalHarness:
    """
    Usage:
        harness = EvalHarness(
            base_model_fn=lambda prompt: base_model.generate(prompt),
            finetuned_model_fn=lambda prompt: finetuned_model.generate(prompt),
        )
        result = harness.compare(eval_cases)
        print(result.summary())
        for r in result.regressions:
            print(r)  # inspect exactly what got worse
    """

    def __init__(self, base_model_fn: Callable[[str], str],
                 finetuned_model_fn: Callable[[str], str]):
        self.base_model_fn = base_model_fn
        self.finetuned_model_fn = finetuned_model_fn

    def compare(self, cases: list[EvalCase]) -> ComparisonResult:
        comparisons = []
        for case in cases:
            base_output = self.base_model_fn(case.input)
            finetuned_output = self.finetuned_model_fn(case.input)

            base_correct = case.expected_substring.lower() in base_output.lower()
            finetuned_correct = case.expected_substring.lower() in finetuned_output.lower()

            if not base_correct and finetuned_correct:
                verdict = "improved"
            elif base_correct and not finetuned_correct:
                verdict = "regressed"
            elif base_correct and finetuned_correct:
                verdict = "unchanged_correct"
            else:
                verdict = "unchanged_incorrect"

            comparisons.append(ExampleComparison(
                input=case.input, base_output=base_output, finetuned_output=finetuned_output,
                base_correct=base_correct, finetuned_correct=finetuned_correct,
                verdict=verdict, tags=case.tags,
            ))
        return ComparisonResult(comparisons=comparisons)
