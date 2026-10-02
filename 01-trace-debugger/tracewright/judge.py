"""
The "judge" pass: take a completed Trace and ask an LLM to diagnose it.

This module defines a tiny LLM client interface (`LLMClient`) so tracewright
never depends on any specific provider SDK. Two implementations ship here:

- MockLLMClient: deterministic, offline, used in tests and for trying the
  tool out with zero API keys / zero network access.
- OpenAICompatibleClient: a thin wrapper around `requests` that talks to
  any OpenAI-compatible /chat/completions endpoint (OpenAI, local vLLM,
  Ollama's OpenAI-compat mode, etc). Bring your own API key.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Protocol
from .tracer import Trace, SpanType


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class MockLLMClient:
    """
    A fake LLM used for tests and offline demos.

    It doesn't call any API. Instead it applies a simple, transparent
    heuristic so the rest of the pipeline (report generation, CLI, etc.)
    can be fully exercised without a network connection:
      - the first span with an `error` set is the root cause.
      - if no span errored, the last tool_call span with an empty/falsy
        output is flagged as a likely soft-failure.
      - otherwise the run is judged successful.

    Swap this for OpenAICompatibleClient (or your own LLMClient) in
    production for real semantic diagnosis of runs that "succeeded" but
    gave a wrong or low-quality answer.
    """

    def complete(self, system: str, user: str) -> str:
        # `user` contains the serialized trace JSON, per Judge.diagnose()
        match = re.search(r"TRACE_JSON:\s*(\{.*\})\s*$", user, re.DOTALL)
        data = json.loads(match.group(1)) if match else {}
        spans = data.get("spans", [])

        for s in spans:
            if s.get("error"):
                return json.dumps({
                    "verdict": "failed",
                    "root_cause_step": s["step_number"],
                    "reason": f"Step {s['step_number']} ('{s['name']}') raised an error: {s['error']}",
                })

        for s in reversed(spans):
            if s.get("type") == "tool_call" and not s.get("output"):
                return json.dumps({
                    "verdict": "suspect",
                    "root_cause_step": s["step_number"],
                    "reason": f"Step {s['step_number']} ('{s['name']}') returned an empty result, "
                              f"which may have starved later steps of information.",
                })

        return json.dumps({
            "verdict": "success",
            "root_cause_step": None,
            "reason": "No errors or empty tool outputs detected across all spans.",
        })


class OpenAICompatibleClient:
    """
    Real client for any OpenAI-compatible /v1/chat/completions endpoint.
    Requires `requests` and network access at call time (not used in tests).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        import requests  # imported lazily so the mock path never needs it
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


JUDGE_SYSTEM_PROMPT = """You are debugging an AI agent's execution trace.
You will be given the agent's task and a JSON list of steps it took
(LLM calls and tool calls, each with input/output/error/duration).

Read the steps IN ORDER and determine:
1. verdict: "success", "suspect", or "failed"
2. root_cause_step: the step_number most responsible for any problem, or null
3. reason: one sentence explaining why

Respond with ONLY a JSON object with keys: verdict, root_cause_step, reason.
No other text."""


@dataclass
class Diagnosis:
    verdict: str
    root_cause_step: int | None
    reason: str


class Judge:
    """Runs the judge pass over a completed Trace."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or MockLLMClient()

    def diagnose(self, trace: Trace) -> Diagnosis:
        user_prompt = (
            f"TASK: {trace.task}\n\n"
            f"TRACE_JSON: {json.dumps(trace.to_dict())}"
        )
        raw = self.client.complete(JUDGE_SYSTEM_PROMPT, user_prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"verdict": "unknown", "root_cause_step": None,
                    "reason": f"Judge returned non-JSON output: {raw[:200]}"}
        return Diagnosis(
            verdict=data.get("verdict", "unknown"),
            root_cause_step=data.get("root_cause_step"),
            reason=data.get("reason", ""),
        )
