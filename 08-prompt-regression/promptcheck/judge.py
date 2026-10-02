"""
"Close enough" checking for LLM outputs. Since LLM outputs aren't
byte-identical between runs, exact string matching doesn't work for
prompt regression tests -- you need a judge that asks "does this output
satisfy the requirement?" rather than "does this output equal that string?".

MockJudge is a fully offline, deterministic stand-in (word-overlap based)
used in this package's own tests and for offline demos. SemanticJudge is
the production version: an LLM call that grades semantic equivalence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class JudgeVerdict:
    passed: bool
    reason: str


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class MockJudge:
    """
    Offline heuristic: passes if a meaningful fraction of the expected
    behavior's key words appear in the actual output. This is intentionally
    crude -- good enough for testing promptcheck itself without a network
    connection, NOT a substitute for a real semantic judge in production.
    """

    def __init__(self, overlap_threshold: float = 0.5):
        self.overlap_threshold = overlap_threshold

    def check(self, actual_output: str, expected_behavior: str) -> JudgeVerdict:
        expected_words = set(w.lower().strip(".,!?") for w in expected_behavior.split()
                              if len(w) > 3)  # skip tiny/stop-ish words
        actual_words = set(w.lower().strip(".,!?") for w in actual_output.split())
        if not expected_words:
            return JudgeVerdict(passed=True, reason="No substantive expected keywords to check.")
        overlap = len(expected_words & actual_words) / len(expected_words)
        passed = overlap >= self.overlap_threshold
        return JudgeVerdict(
            passed=passed,
            reason=f"Keyword overlap {overlap:.0%} ({'>=' if passed else '<'} {self.overlap_threshold:.0%} threshold)",
        )


class SemanticJudge:
    """
    Real judge: asks an LLM whether the actual output satisfies the expected
    behavior description. Requires an LLMClient implementation (e.g. an
    OpenAI-compatible client) with real network access -- not used in this
    package's own offline test suite.
    """

    SYSTEM_PROMPT = (
        "You are grading whether an AI system's output satisfies a described "
        "requirement. Respond with exactly 'PASS' or 'FAIL' on the first line, "
        "then one sentence explaining why."
    )

    def __init__(self, client: LLMClient):
        self.client = client

    def check(self, actual_output: str, expected_behavior: str) -> JudgeVerdict:
        user_prompt = (
            f"Requirement: {expected_behavior}\n\n"
            f"Actual output: {actual_output}\n\n"
            f"Does the actual output satisfy the requirement?"
        )
        raw = self.client.complete(self.SYSTEM_PROMPT, user_prompt)
        first_line = raw.strip().split("\n")[0].strip().upper()
        passed = first_line.startswith("PASS")
        reason = raw.strip().split("\n", 1)[1].strip() if "\n" in raw else raw
        return JudgeVerdict(passed=passed, reason=reason)


def assert_semantically_equivalent(actual_output: str, expected_behavior: str,
                                    judge: MockJudge | SemanticJudge | None = None) -> None:
    """Convenience assertion for use directly inside pytest/unittest test bodies."""
    judge = judge or MockJudge()
    verdict = judge.check(actual_output, expected_behavior)
    assert verdict.passed, f"Output did not satisfy expected behavior: {verdict.reason}\nGot: {actual_output!r}"
