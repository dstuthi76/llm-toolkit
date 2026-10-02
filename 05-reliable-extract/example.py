"""
Example: a fake "LLM" that gets a date range backwards on its first
attempt, then reads the targeted correction message and fixes it.

Run: python3 example.py
"""
from dataclasses import dataclass
from reliably.schema import field_validator, grounded_in
from reliably.repair import RepairLoop

SOURCE_TEXT = "The Q3 Planning Offsite runs from 2026-09-10 to 2026-09-12 in Austin."


@dataclass
class Event:
    name: str
    start_date: str
    end_date: str


@field_validator(Event)
def end_after_start(event: Event):
    if event.end_date < event.start_date:
        return f"end_date ({event.end_date}) is before start_date ({event.start_date})"
    return None


@field_validator(Event)
def name_is_grounded(event: Event):
    if not grounded_in(event.name, SOURCE_TEXT):
        return f"name '{event.name}' does not appear anywhere in the source text (possible hallucination)"
    return None


# --- simulate a flaky LLM extractor: gets dates backwards on attempt 1 ---
call_count = {"n": 0}


def fake_llm_extract(prompt: str) -> Event:
    call_count["n"] += 1
    print(f"--- LLM call #{call_count['n']} ---")
    print(prompt[-200:])
    print()
    if call_count["n"] == 1:
        # Simulate a common mistake: swapped start/end dates
        return Event(name="Q3 Planning Offsite", start_date="2026-09-12", end_date="2026-09-10")
    return Event(name="Q3 Planning Offsite", start_date="2026-09-10", end_date="2026-09-12")


loop = RepairLoop(extract_fn=fake_llm_extract, max_attempts=3)
result = loop.run(f"Extract the event name, start_date and end_date from:\n{SOURCE_TEXT}")

print(f"Valid: {result.valid} after {result.attempts} attempt(s)")
print(f"Final value: {result.value}")
