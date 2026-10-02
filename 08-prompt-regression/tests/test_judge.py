import unittest
from promptcheck.judge import MockJudge, assert_semantically_equivalent


class TestMockJudge(unittest.TestCase):
    def test_passes_when_output_covers_expected_keywords(self):
        judge = MockJudge(overlap_threshold=0.5)
        verdict = judge.check(
            actual_output="Our refund policy allows returns within 30 days, no questions asked.",
            expected_behavior="Refunds within 30 days, no questions asked",
        )
        self.assertTrue(verdict.passed)

    def test_fails_when_output_misses_expected_keywords(self):
        judge = MockJudge(overlap_threshold=0.5)
        verdict = judge.check(
            actual_output="We are open Monday through Friday, 9 to 5.",
            expected_behavior="Refunds within 30 days, no questions asked",
        )
        self.assertFalse(verdict.passed)

    def test_assert_helper_raises_on_failure(self):
        with self.assertRaises(AssertionError):
            assert_semantically_equivalent(
                actual_output="completely unrelated text about weather",
                expected_behavior="Refunds within 30 days, no questions asked",
            )

    def test_assert_helper_passes_silently_on_success(self):
        # should not raise
        assert_semantically_equivalent(
            actual_output="Refunds are available within 30 days, no questions asked.",
            expected_behavior="Refunds within 30 days, no questions asked",
        )


if __name__ == "__main__":
    unittest.main()
