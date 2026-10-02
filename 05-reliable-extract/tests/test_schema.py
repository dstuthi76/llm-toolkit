import unittest
from dataclasses import dataclass
from reliably.schema import field_validator, validate, grounded_in


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


class TestSchema(unittest.TestCase):
    def test_valid_event_passes(self):
        event = Event(name="Conference", start_date="2026-01-01", end_date="2026-01-03")
        result = validate(event)
        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])

    def test_backwards_dates_fail_semantic_validator(self):
        event = Event(name="Conference", start_date="2026-01-05", end_date="2026-01-01")
        result = validate(event)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("end_date", result.errors[0].message)

    def test_wrong_type_fails_structural_check(self):
        event = Event(name="Conference", start_date=123, end_date="2026-01-03")
        result = validate(event)
        self.assertFalse(result.valid)
        self.assertTrue(any(e.field_or_rule == "start_date" for e in result.errors))

    def test_grounded_in_detects_hallucinated_value(self):
        source = "The invoice number is INV-2024-001, issued by Acme Corp."
        self.assertTrue(grounded_in("INV-2024-001", source))
        self.assertFalse(grounded_in("INV-9999-999", source))


if __name__ == "__main__":
    unittest.main()
