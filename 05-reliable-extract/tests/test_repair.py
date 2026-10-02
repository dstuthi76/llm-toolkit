import unittest
from dataclasses import dataclass
from reliably.schema import field_validator, validate
from reliably.repair import RepairLoop


@dataclass
class Event:
    name: str
    start_date: str
    end_date: str


@field_validator(Event)
def end_after_start_repair_test(event: Event):
    if event.end_date < event.start_date:
        return f"end_date ({event.end_date}) is before start_date ({event.start_date})"
    return None


class TestRepairLoop(unittest.TestCase):
    def test_succeeds_immediately_when_first_attempt_is_valid(self):
        def perfect_extractor(prompt):
            return Event(name="Conf", start_date="2026-01-01", end_date="2026-01-03")

        loop = RepairLoop(extract_fn=perfect_extractor, max_attempts=3)
        result = loop.run("extract event")
        self.assertTrue(result.valid)
        self.assertEqual(result.attempts, 1)

    def test_repairs_after_one_bad_attempt(self):
        call_count = {"n": 0}

        def flaky_extractor(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return Event(name="Conf", start_date="2026-01-05", end_date="2026-01-01")  # backwards
            # on retry, the "model" reads the correction prompt and fixes it
            assert "end_date" in prompt  # the correction was actually communicated
            return Event(name="Conf", start_date="2026-01-01", end_date="2026-01-05")

        loop = RepairLoop(extract_fn=flaky_extractor, max_attempts=3)
        result = loop.run("extract event")
        self.assertTrue(result.valid)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.validation_history), 2)

    def test_gives_up_after_max_attempts(self):
        def always_broken_extractor(prompt):
            return Event(name="Conf", start_date="2026-01-05", end_date="2026-01-01")

        loop = RepairLoop(extract_fn=always_broken_extractor, max_attempts=2)
        result = loop.run("extract event")
        self.assertFalse(result.valid)
        self.assertEqual(result.attempts, 2)


if __name__ == "__main__":
    unittest.main()
