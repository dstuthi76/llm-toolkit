import unittest
from finetune_eval import EvalHarness, EvalCase


class TestEvalHarness(unittest.TestCase):
    def test_detects_improvement(self):
        harness = EvalHarness(
            base_model_fn=lambda p: "I don't know",
            finetuned_model_fn=lambda p: "The refund window is 30 days",
        )
        result = harness.compare([EvalCase(input="What's the refund window?", expected_substring="30 days")])
        self.assertEqual(result.comparisons[0].verdict, "improved")
        self.assertEqual(len(result.improvements), 1)
        self.assertEqual(len(result.regressions), 0)

    def test_detects_regression(self):
        harness = EvalHarness(
            base_model_fn=lambda p: "The refund window is 30 days",
            finetuned_model_fn=lambda p: "I don't know",
        )
        result = harness.compare([EvalCase(input="What's the refund window?", expected_substring="30 days")])
        self.assertEqual(result.comparisons[0].verdict, "regressed")
        self.assertEqual(len(result.regressions), 1)

    def test_mixed_results_report_correct_aggregate_accuracy(self):
        def base_fn(p):
            return "30 days" if "refund" in p else "wrong answer"

        def finetuned_fn(p):
            return "30 days" if "refund" in p else "still wrong"

        cases = [
            EvalCase(input="refund policy question", expected_substring="30 days"),
            EvalCase(input="unrelated question", expected_substring="specific fact"),
        ]
        result = EvalHarness(base_fn, finetuned_fn).compare(cases)
        self.assertEqual(result.base_accuracy, 0.5)
        self.assertEqual(result.finetuned_accuracy, 0.5)
        self.assertEqual(len(result.regressions), 0)
        self.assertEqual(len(result.improvements), 0)

    def test_unchanged_correct_and_incorrect_classified_properly(self):
        harness = EvalHarness(
            base_model_fn=lambda p: "correct answer here",
            finetuned_model_fn=lambda p: "correct answer here",
        )
        result = harness.compare([EvalCase(input="q", expected_substring="correct answer")])
        self.assertEqual(result.comparisons[0].verdict, "unchanged_correct")


if __name__ == "__main__":
    unittest.main()
