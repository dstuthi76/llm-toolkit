# reliably

**Schema validation tells you the JSON is well-formed. `reliably` tells you if the *content* actually makes sense — and fixes it with a targeted correction instead of a blind retry.**

Structured-output tools (native JSON mode, `instructor`, `outlines`) guarantee your LLM's output matches a schema. They don't guarantee `end_date` is actually after `start_date`, or that an extracted ID isn't a hallucination that never appeared in the source document. `reliably` adds that layer, and when validation fails, builds a correction prompt describing exactly what was wrong — not a generic "try again."

## Why

- Valid JSON ≠ correct content. Models produce well-formed nonsense constantly.
- A generic retry often reproduces the same mistake. A targeted correction ("your end_date is before your start_date") fixes it far more reliably.
- Grounding checks catch hallucinated values that pass every structural check.

## Install

```bash
pip install reliably
```

## Quick start

```python
from dataclasses import dataclass
from reliably import field_validator, validate

@dataclass
class Event:
    name: str
    start_date: str
    end_date: str

@field_validator(Event)
def end_after_start(event: Event):
    if event.end_date < event.start_date:
        return f"end_date ({event.end_date}) is before start_date ({event.start_date})"
    return None  # None means valid

event = Event(name="Offsite", start_date="2026-09-12", end_date="2026-09-10")
result = validate(event)
print(result.valid)     # False
print(result.errors)    # [ValidationError(field_or_rule='end_after_start', message='end_date ...')]
```

## The repair loop: fixing it automatically

```python
from reliably import RepairLoop

def my_llm_extract(prompt: str) -> Event:
    return call_your_llm_and_parse(prompt)  # your existing extraction call

loop = RepairLoop(extract_fn=my_llm_extract, max_attempts=3)
result = loop.run("Extract the event from: The offsite runs Sept 10-12.")

print(result.valid, result.attempts, result.value)
```

When the first attempt fails validation, `RepairLoop` doesn't just resend the same prompt — it appends the *specific* validation errors and asks the model to fix only those, which measurably improves repair success rate over a blind retry.

## Grounding checks (catching hallucinations)

```python
from reliably import grounded_in

source = "Invoice INV-2024-001 issued by Acme Corp."
grounded_in("INV-2024-001", source)   # True
grounded_in("INV-9999-999", source)   # False -- flag as likely hallucinated
```

Wire this into a `@field_validator` to automatically reject extracted values that don't actually appear in the source material you gave the model.

## Relationship to `instructor` / `pydantic`

This package intentionally ships its own minimal, dependency-free validation core so it's easy to read and test standalone. In production, we recommend building the same *pattern* — semantic validators + targeted repair prompts — directly on top of [`instructor`](https://github.com/jxnl/instructor) and full `pydantic` models, which already handle schema-constrained decoding and basic retries against a real LLM. The `RepairLoop` logic here (build a targeted correction prompt from specific validation errors) is the part worth porting over; the schema layer is a drop-in for Pydantic's own validators.

## Development

```bash
python -m unittest discover -s tests -v
python3 example.py
```

## Roadmap

- [ ] Direct `pydantic`/`instructor` integration path (no reimplementation)
- [ ] Benchmark harness: repair-success-rate vs. plain-retry on a public function-calling dataset
- [ ] Multi-field cross-validation helpers (e.g. "sum of line items equals total")

## License

MIT
