"""
Core routing/cascade logic: try the cheap model first, escalate to the
expensive model only when a confidence check says the cheap answer is
likely insufficient.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
from .providers import LLMProvider, CompletionResult
from .cost_tracker import CostTracker, LogEntry


@dataclass
class RouteDecision:
    result: CompletionResult
    provider_used: str
    escalated: bool
    confidence: Optional[float]
    cost: float


ConfidenceFn = Callable[[str, CompletionResult], float]  # (prompt, cheap_result) -> 0..1


def word_count_confidence(prompt: str, result: CompletionResult) -> float:
    """
    Default, offline confidence heuristic: short, hedge-word-free answers to
    short questions are treated as confident; long/complex questions or
    hedging language ("I'm not sure", "it depends") lower confidence.
    This is a placeholder -- in production, swap in a cheap LLM call asking
    "rate your confidence 1-10" or a learned classifier.
    """
    hedges = ["i'm not sure", "i am not sure", "it depends", "i don't know",
              "cannot determine", "unclear", "might be", "not certain"]
    text = result.text.lower()
    if any(h in text for h in hedges):
        return 0.3
    if len(prompt) > 400:  # long/complex prompts get a lower default confidence
        return 0.5
    return 0.85


class Router:
    """
    Usage:
        router = Router(cheap=cheap_provider, expensive=expensive_provider,
                         confidence_threshold=0.6)
        decision = router.route("What's 15% of 80?")
    """

    def __init__(self, cheap: LLMProvider, expensive: LLMProvider,
                 confidence_threshold: float = 0.6,
                 confidence_fn: ConfidenceFn = word_count_confidence,
                 tracker: CostTracker | None = None):
        self.cheap = cheap
        self.expensive = expensive
        self.confidence_threshold = confidence_threshold
        self.confidence_fn = confidence_fn
        self.tracker = tracker or CostTracker()

    def route(self, prompt: str) -> RouteDecision:
        cheap_result = self.cheap.complete(prompt)
        confidence = self.confidence_fn(prompt, cheap_result)

        if confidence >= self.confidence_threshold:
            cost = self.cheap.cost_of(cheap_result)
            baseline_cost = self._baseline_cost(prompt)
            self.tracker.log(LogEntry(prompt[:60], self.cheap.name, False, cost, baseline_cost))
            return RouteDecision(cheap_result, self.cheap.name, False, confidence, cost)

        expensive_result = self.expensive.complete(prompt)
        cost = self.cheap.cost_of(cheap_result) + self.expensive.cost_of(expensive_result)
        baseline_cost = self._baseline_cost(prompt)
        self.tracker.log(LogEntry(prompt[:60], self.expensive.name, True, cost, baseline_cost))
        return RouteDecision(expensive_result, self.expensive.name, True, confidence, cost)

    def _baseline_cost(self, prompt: str) -> float:
        """What this request would have cost if we ALWAYS used the expensive model."""
        # Estimate using the same token counting the expensive provider would use.
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = input_tokens  # rough symmetric estimate for the baseline
        fake = type("R", (), {"input_tokens": input_tokens, "output_tokens": output_tokens})()
        return self.expensive.cost_of(fake)  # type: ignore[arg-type]
