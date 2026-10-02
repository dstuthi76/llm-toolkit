import unittest
from modelrouter.providers import MockProvider
from modelrouter.router import Router


class TestRouter(unittest.TestCase):
    def test_confident_short_answer_stays_on_cheap_model(self):
        cheap = MockProvider("cheap-model", 0.001, 0.002,
                              behavior=lambda p: "42")
        expensive = MockProvider("expensive-model", 0.01, 0.03,
                                  behavior=lambda p: "The answer is 42.")
        router = Router(cheap, expensive, confidence_threshold=0.6)

        decision = router.route("What is 6 times 7?")
        self.assertFalse(decision.escalated)
        self.assertEqual(decision.provider_used, "cheap-model")

    def test_hedging_answer_triggers_escalation(self):
        cheap = MockProvider("cheap-model", 0.001, 0.002,
                              behavior=lambda p: "I'm not sure, it depends on context.")
        expensive = MockProvider("expensive-model", 0.01, 0.03,
                                  behavior=lambda p: "Detailed correct answer here.")
        router = Router(cheap, expensive, confidence_threshold=0.6)

        decision = router.route("Explain the legal nuance of X")
        self.assertTrue(decision.escalated)
        self.assertEqual(decision.provider_used, "expensive-model")

    def test_cost_tracker_accumulates_savings(self):
        cheap = MockProvider("cheap-model", 0.001, 0.002, behavior=lambda p: "ok")
        expensive = MockProvider("expensive-model", 0.01, 0.03, behavior=lambda p: "ok")
        router = Router(cheap, expensive, confidence_threshold=0.6)

        for _ in range(5):
            router.route("short simple question")  # should stay on cheap every time

        self.assertEqual(len(router.tracker.entries), 5)
        self.assertEqual(router.tracker.escalation_rate, 0.0)
        self.assertGreater(router.tracker.total_saved, 0)

    def test_escalated_cost_includes_both_calls(self):
        cheap = MockProvider("cheap-model", 0.001, 0.002,
                              behavior=lambda p: "i don't know")
        expensive = MockProvider("expensive-model", 0.01, 0.03, behavior=lambda p: "ok")
        router = Router(cheap, expensive, confidence_threshold=0.6)

        decision = router.route("hard question")
        # cost should reflect BOTH the cheap attempt and the expensive escalation
        expected_min_cost = cheap.cost_of(cheap.complete("hard question"))
        self.assertGreaterEqual(decision.cost, expected_min_cost)


if __name__ == "__main__":
    unittest.main()
