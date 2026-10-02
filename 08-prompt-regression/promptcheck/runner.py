"""
Suite runner: define prompt test cases, run them against your LLM call,
grade with a judge, and diff against the last known-good snapshot.

This is the core logic behind the pytest plugin (pytest_plugin.py) but is
fully usable standalone -- e.g. from a plain script or a CI step that
doesn't use pytest at all.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .judge import MockJudge, JudgeVerdict
from .snapshot import SnapshotStore


@dataclass
class PromptCase:
    name: str
    prompt: str
    expected_behavior: str


@dataclass
class CaseResult:
    case: PromptCase
    output: str
    verdict: JudgeVerdict
    previous_snapshot_output: str | None
    changed_from_snapshot: bool


@dataclass
class SuiteResult:
    results: list[CaseResult]

    @property
    def passed(self) -> list[CaseResult]:
        return [r for r in self.results if r.verdict.passed]

    @property
    def failed(self) -> list[CaseResult]:
        return [r for r in self.results if not r.verdict.passed]

    @property
    def changed(self) -> list[CaseResult]:
        return [r for r in self.results if r.changed_from_snapshot]

    def summary(self) -> str:
        return (f"{len(self.passed)}/{len(self.results)} passed | "
                f"{len(self.changed)} changed since last snapshot")


def run_prompt_suite(cases: list[PromptCase], llm_fn: Callable[[str], str],
                      snapshot_store: SnapshotStore,
                      judge: MockJudge | None = None,
                      update_snapshots: bool = False) -> SuiteResult:
    judge = judge or MockJudge()
    results = []

    for case in cases:
        output = llm_fn(case.prompt)
        verdict = judge.check(output, case.expected_behavior)

        prev = snapshot_store.get(case.name)
        prev_output = prev.output if prev else None
        changed = prev_output is not None and prev_output != output

        if update_snapshots or prev is None:
            snapshot_store.save(case.name, case.prompt, output)

        results.append(CaseResult(
            case=case, output=output, verdict=verdict,
            previous_snapshot_output=prev_output, changed_from_snapshot=changed,
        ))

    return SuiteResult(results=results)
