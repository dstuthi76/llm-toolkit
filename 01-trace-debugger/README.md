# tracewright

**Stop scrolling through raw agent logs by hand. Trace every step your AI agent takes, then let an LLM tell you exactly which step caused a failure.**

When an agentic AI system fails, the error is rarely in the last step — it's usually three steps back, in a tool call that quietly returned garbage. `tracewright` records every LLM call and tool call your agent makes as a structured trace, then runs a "judge" pass that pinpoints the root-cause step and explains why, rendered as a browsable HTML report.

## Why

- Agent frameworks tell you "the run failed" — not *which* of the 15 steps caused it.
- Reading raw JSON logs by hand doesn't scale past a handful of runs.
- `tracewright` gives you a one-command diagnosis: `tracewright analyze run.json`.

## Install

```bash
pip install tracewright
# or, for real LLM-based judging instead of the offline heuristic judge:
pip install "tracewright[openai]"
```

## Quick start

Wrap the functions your agent calls with `@tracer.trace_tool(...)` or `@tracer.trace_llm(...)`:

```python
from tracewright import Tracer, Judge, generate_html_report

tracer = Tracer(task="Book a round-trip flight to Paris under $800")

@tracer.trace_tool("search_flights")
def search_flights(destination, date):
    return call_flight_api(destination, date)

@tracer.trace_llm("gpt-4o-mini")
def pick_best_flight(candidates):
    return call_llm(f"Pick the cheapest of: {candidates}")

# run your agent as normal
flights = search_flights("Paris", "2026-11-03")
answer = pick_best_flight(flights)

# diagnose what happened
diagnosis = Judge().diagnose(tracer.trace)
print(diagnosis.verdict, diagnosis.root_cause_step, diagnosis.reason)

with open("report.html", "w") as f:
    f.write(generate_html_report(tracer.trace, diagnosis))
```

Open `report.html` — failed or suspect steps are highlighted, with the judge's explanation at the top.

## Using a real LLM as the judge

By default, `Judge()` uses `MockLLMClient` — a deterministic, offline heuristic (flags the first errored step, or a tool call that returned an empty result) so you can try the tool and run its test suite with zero API keys. For real semantic diagnosis — e.g. catching a run that technically "succeeded" but gave a wrong answer — swap in a real client:

```python
from tracewright.judge import Judge, OpenAICompatibleClient

client = OpenAICompatibleClient(api_key="sk-...", model="gpt-4o-mini")
diagnosis = Judge(client).diagnose(tracer.trace)
```

`OpenAICompatibleClient` works with any OpenAI-compatible `/chat/completions` endpoint — OpenAI itself, a local vLLM server, or Ollama's OpenAI-compat mode — just change `base_url`.

## CLI

Save a trace to JSON (`tracer.trace.to_json("run.json")`) from a real run, then:

```bash
python -m tracewright.cli analyze run.json -o report.html
```

## Framework adapters

`Tracer.manual_span(...)` lets you feed spans in directly from any framework's callback/hook system (LangGraph's `on_tool_end`, CrewAI's step callbacks, etc.) without needing a decorator around the original function.

## How the judge works

The judge receives the full trace (task description + every step's input, output, error, and duration) and returns a JSON verdict: `success`, `suspect`, or `failed`, plus the specific `root_cause_step` and a one-sentence `reason`. This is the core idea: **use a second LLM call to review the first agent's execution**, the same way a senior engineer reviews a junior's work by reading their steps in order, rather than just looking at the final output.

## Development

```bash
git clone <this-repo>
cd tracewright
python -m unittest discover -s tests -v
```

## Roadmap

- [ ] Native LangGraph / CrewAI callback adapters
- [ ] Multi-run comparison (diff two traces of the same task)
- [ ] A `pytest` plugin: `assert_trace_succeeds(trace)`

## License

MIT
