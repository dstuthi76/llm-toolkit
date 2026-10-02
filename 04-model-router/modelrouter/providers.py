"""
Provider abstraction: one interface, many backends.

In production, use `litellm` under the hood for a single call signature
across OpenAI/Anthropic/local models -- `OpenAICompatibleProvider` here
shows the shape without requiring the dependency. `MockProvider` is fully
offline and deterministic, used for tests and demos.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMProvider(Protocol):
    name: str
    cost_per_1k_input: float
    cost_per_1k_output: float

    def complete(self, prompt: str) -> CompletionResult: ...

    def cost_of(self, result: CompletionResult) -> float:
        return (result.input_tokens / 1000) * self.cost_per_1k_input + \
               (result.output_tokens / 1000) * self.cost_per_1k_output


class MockProvider:
    """
    Deterministic offline provider for tests/demos. `behavior` lets tests
    control what this "model" returns and how confident it should sound,
    without any network access.
    """

    def __init__(self, name: str, cost_per_1k_input: float, cost_per_1k_output: float,
                 behavior=None):
        self.name = name
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        # behavior: Callable[[str], str] mapping prompt -> canned response
        self.behavior = behavior or (lambda prompt: f"[{name} mock answer for: {prompt[:40]}...]")

    def complete(self, prompt: str) -> CompletionResult:
        text = self.behavior(prompt)
        return CompletionResult(
            text=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            model=self.name,
        )

    def cost_of(self, result: CompletionResult) -> float:
        return (result.input_tokens / 1000) * self.cost_per_1k_input + \
               (result.output_tokens / 1000) * self.cost_per_1k_output


class OpenAICompatibleProvider:
    """Real provider hitting any OpenAI-compatible endpoint. Needs `requests` + network."""

    def __init__(self, name: str, model: str, api_key: str,
                 cost_per_1k_input: float, cost_per_1k_output: float,
                 base_url: str = "https://api.openai.com/v1"):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str) -> CompletionResult:
        import requests
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return CompletionResult(
            text=data["choices"][0]["message"]["content"],
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=self.model,
        )

    def cost_of(self, result: CompletionResult) -> float:
        return (result.input_tokens / 1000) * self.cost_per_1k_input + \
               (result.output_tokens / 1000) * self.cost_per_1k_output
