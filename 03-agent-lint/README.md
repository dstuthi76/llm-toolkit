# agentlint

**A linter for multi-agent conversations. Catches loops, topic drift, and unresolved endings before you have to read 200 log lines by hand.**

Multi-agent frameworks (CrewAI, AutoGen, LangGraph) let several LLM "agents" pass messages back and forth to solve a task. When it goes wrong, it usually looks like: two agents repeating the same point forever, the conversation slowly wandering off-topic, or the agents giving up without ever answering the question. `agentlint` scans the message history and flags all three, in a format modeled on familiar code linters.

## Why

- Multi-agent orchestration frameworks manage message-passing, but not diagnosis.
- Reading a 40-message transcript to spot a subtle loop or drift is slow and error-prone.
- `agentlint` turns this into a one-line check you can even run in CI.

## Install

```bash
pip install agentlint
```

## Quick start

```python
from agentlint import ConversationAnalyzer, Message
from agentlint.report import format_findings

messages = [
    Message(0, "researcher", "Let's check if a bug caused the conversion drop."),
    Message(1, "critic", "Have you checked if a bug caused the conversion drop?"),
    # ...
]

analyzer = ConversationAnalyzer()
findings = analyzer.analyze(task="Find why conversion rate dropped", messages=messages)
print(format_findings(findings))
```

```
✗ [loop] messages 0,1: Message 0 (researcher) and message 1 (critic) are 69% similar -- possible repetition loop.
⚠ [drift] messages 4,5: Relevance to the original task dropped 100% ... the agents may have drifted off-topic.
⚠ [premature_termination] messages 5: Conversation ended ... with an unresolved-sounding statement.
```

## What it detects

| Finding | How |
|---|---|
| **Loop** | Any two messages within a sliding window score above a similarity threshold (TF-IDF cosine by default) |
| **Drift** | Average relevance-to-original-task of the last third of messages drops sharply vs. the first third |
| **Premature termination** | The final message matches known "giving up" phrasing (extend `UNRESOLVED_MARKERS` for your domain, or swap in an LLM-based classifier) |

## Wiring it into a real framework

`agentlint` is framework-agnostic by design — you just need to convert whatever message log your framework exposes (CrewAI's task outputs, AutoGen's `chat_history`, LangGraph's state messages) into a list of `Message(index, sender, content)` and pass it in.

## Wiring it into CI

Because findings are just a list of dataclasses, this plugs directly into a test:

```python
def test_no_infinite_loops():
    findings = analyzer.analyze(task, messages)
    loop_findings = [f for f in findings if f.kind == "loop"]
    assert len(loop_findings) == 0, format_findings(loop_findings)
```

## Tuning

- `loop_similarity_threshold` (default 0.85) — lower it to catch looser paraphrase-loops, raise it to reduce false positives.
- `drift_drop_threshold` (default 0.4) — the fraction drop in task-relevance that triggers a drift warning.
- `window` (default 4) — how many prior messages each new message is checked against for loops.

The default similarity backend is offline TF-IDF (see `contextprune`'s embeddings module for the same tradeoff) — swap in real sentence embeddings for paraphrase-level loop detection that catches semantically identical but lexically different repeats.

## Development

```bash
python -m unittest discover -s tests -v
```

## Roadmap

- [ ] Native adapters for CrewAI / AutoGen / LangGraph message formats
- [ ] LLM-based (not keyword-based) premature-termination classifier
- [ ] Conversation graph visualization (who talked to whom, via NetworkX)

## License

MIT
