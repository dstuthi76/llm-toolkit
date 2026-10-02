"""
Measure whether pruning actually helps, instead of assuming it does.

`run_benchmark` takes a list of (query, chunks, expected_answer_substring)
cases plus an `answer_fn(prompt) -> str` (your LLM call), and reports
accuracy with full context vs. with pruned context.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
from .pruner import ContextPruner, Chunk


@dataclass
class BenchmarkCase:
    query: str
    chunks: list[Chunk]
    expected_substring: str  # crude but dependency-free correctness check


@dataclass
class BenchmarkResult:
    full_context_accuracy: float
    pruned_context_accuracy: float
    avg_tokens_full: float
    avg_tokens_pruned: float
    n_cases: int


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    answer_fn: Callable[[str], str],
    token_budget: int,
    pruner: ContextPruner | None = None,
) -> BenchmarkResult:
    pruner = pruner or ContextPruner()
    full_correct = pruned_correct = 0
    full_tokens_total = pruned_tokens_total = 0

    for case in cases:
        full_text = "\n\n".join(c.text for c in case.chunks)
        full_tokens_total += sum(c.estimate_tokens() for c in case.chunks)
        full_prompt = f"Context:\n{full_text}\n\nQuestion: {case.query}"
        if case.expected_substring.lower() in answer_fn(full_prompt).lower():
            full_correct += 1

        pruned = pruner.prune(case.query, case.chunks, token_budget)
        pruned_tokens_total += pruned.total_tokens
        pruned_prompt = f"Context:\n{pruned.ordered_text}\n\nQuestion: {case.query}"
        if case.expected_substring.lower() in answer_fn(pruned_prompt).lower():
            pruned_correct += 1

    n = len(cases)
    return BenchmarkResult(
        full_context_accuracy=full_correct / n if n else 0.0,
        pruned_context_accuracy=pruned_correct / n if n else 0.0,
        avg_tokens_full=full_tokens_total / n if n else 0.0,
        avg_tokens_pruned=pruned_tokens_total / n if n else 0.0,
        n_cases=n,
    )
