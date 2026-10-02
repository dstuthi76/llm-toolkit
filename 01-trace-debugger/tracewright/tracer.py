"""
Core tracing primitives for tracewright.

A `Trace` is an ordered list of `Span` objects. A `Span` represents one
"step" an agent took: either an LLM call or a tool call. You wrap your
existing agent code with `Tracer` and it records everything for you.
"""
from __future__ import annotations
import time
import json
import functools
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional
from enum import Enum


class SpanType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"


@dataclass
class Span:
    """One recorded step in an agent's execution."""
    step_number: int
    type: SpanType
    name: str                      # e.g. "search_flights" or "gpt-4o"
    input: Any
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    start_ts: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class Trace:
    """A full run: the task description plus every span recorded during it."""
    task: str
    spans: list[Span] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"task": self.task, "spans": [s.to_dict() for s in self.spans]}

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Trace":
        with open(path) as f:
            data = json.load(f)
        spans = [Span(**{**s, "type": SpanType(s["type"])}) for s in data["spans"]]
        return cls(task=data["task"], spans=spans)


class Tracer:
    """
    Wrap your agent's LLM/tool calls with this to build a Trace automatically.

    Example:
        tracer = Tracer(task="Book a flight to Paris")

        @tracer.trace_tool("search_flights")
        def search_flights(destination):
            ...

        @tracer.trace_llm("gpt-4o")
        def call_llm(prompt):
            ...
    """

    def __init__(self, task: str):
        self.trace = Trace(task=task)
        self._counter = 0

    def _next_step(self) -> int:
        self._counter += 1
        return self._counter

    def _record(self, span_type: SpanType, name: str, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            step = self._next_step()
            start = time.time()
            span = Span(
                step_number=step,
                type=span_type,
                name=name,
                input={"args": _safe(args), "kwargs": _safe(kwargs)},
                start_ts=start,
            )
            try:
                result = fn(*args, **kwargs)
                span.output = _safe(result)
                return result
            except Exception as e:
                span.error = f"{type(e).__name__}: {e}"
                raise
            finally:
                span.duration_ms = (time.time() - start) * 1000
                self.trace.spans.append(span)
        return wrapper

    def trace_tool(self, name: str) -> Callable:
        return lambda fn: self._record(SpanType.TOOL_CALL, name, fn)

    def trace_llm(self, name: str) -> Callable:
        return lambda fn: self._record(SpanType.LLM_CALL, name, fn)

    def manual_span(self, span_type: SpanType, name: str, input: Any,
                     output: Any = None, error: Optional[str] = None,
                     duration_ms: float = 0.0) -> None:
        """Record a step directly, for frameworks you can't decorate (e.g. LangGraph callbacks)."""
        self.trace.spans.append(Span(
            step_number=self._next_step(), type=span_type, name=name,
            input=_safe(input), output=_safe(output), error=error,
            duration_ms=duration_ms, start_ts=time.time(),
        ))


def _safe(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-safe values."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
