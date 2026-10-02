# promptcheck

**Unit tests for prompts. Catch silent regressions before they ship, using the pytest workflow your team already knows.**

Someone tweaks a system prompt to fix one bug, and five other cases silently break — nothing catches it until a user complains. `promptcheck` lets you write prompt tests as normal pytest functions, checks output with a "close enough" semantic judge instead of exact string matching, and diffs against the last known-good snapshot so every PR shows exactly what changed.

## Why

- Prompts are code. Code gets tests. Prompts, historically, don't.
- LLM outputs vary between runs, so exact string matching doesn't work — you need semantic "does this satisfy the requirement" checking.
- A git-diffable snapshot means a reviewer sees the before/after in the PR, the same way they'd review a code diff.

## Install

```bash
pip install promptcheck
pip install "promptcheck[pytest]"   # if you don't already have pytest
```

## Quick start (pytest)

Register the plugin in your `conftest.py`, and override the `llm_case` fixture to call your real model:

```python
# conftest.py
import pytest

@pytest.fixture
def llm_case():
    return lambda prompt: my_real_llm_call(prompt)
```

```python
# test_prompts.py
from promptcheck import assert_semantically_equivalent

def test_refund_policy_answer(llm_case):
    output = llm_case("What's your refund policy?")
    assert_semantically_equivalent(output, "Refunds within 30 days, no questions asked")
```

```bash
pytest test_prompts.py -v
```

## Quick start (standalone, no pytest)

```python
from promptcheck import PromptCase, run_prompt_suite, SnapshotStore

cases = [
    PromptCase(name="refund_policy", prompt="What's your refund policy?",
               expected_behavior="Refunds within 30 days, no questions asked"),
]

store = SnapshotStore("snapshots.json")
result = run_prompt_suite(cases, llm_fn=my_llm_call, snapshot_store=store)

print(result.summary())          # "2/2 passed | 0 changed since last snapshot"
for r in result.failed:
    print(r.case.name, r.verdict.reason)
for r in result.changed:
    print(r.case.name, "changed:", r.previous_snapshot_output, "->", r.output)
```

## The judge: semantic, not exact-match

`MockJudge` (default) is an offline keyword-overlap heuristic — good for trying the tool with zero API keys. For real semantic grading in production:

```python
from promptcheck.judge import SemanticJudge

# any object with .complete(system, user) -> str, e.g. wrapping your OpenAI/Anthropic client
judge = SemanticJudge(client=my_llm_client)
result = run_prompt_suite(cases, llm_fn=my_llm_call, snapshot_store=store, judge=judge)
```

## Snapshots are git-diffable

`SnapshotStore` writes plain sorted JSON — commit `snapshots.json` to your repo, and a reviewer sees exactly what a prompt change did to each test case's output, directly in the PR diff, same as reviewing a code change.

## CI integration

A ready-to-use GitHub Actions workflow is included at `.github/workflows/promptcheck.yml` — it runs your prompt test suite on every PR that touches prompt-related files. Copy it into your repo and set `OPENAI_API_KEY` (or your provider's key) as a repo secret.

## Development

```bash
python -m unittest discover -s tests -v
python3 example.py
```

## Roadmap

- [ ] Multi-model test runs (catch regressions specific to one provider/version)
- [ ] Richer diff rendering (word-level diff, not just before/after strings)
- [ ] `pytest --llm-report` custom reporter for a readable CLI summary

## License

MIT
