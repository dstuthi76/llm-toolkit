import unittest
import tempfile
import os
from promptcheck.runner import PromptCase, run_prompt_suite
from promptcheck.snapshot import SnapshotStore
from promptcheck.judge import MockJudge


class TestRunPromptSuite(unittest.TestCase):
    def test_first_run_saves_snapshot_and_reports_no_change(self):
        with tempfile.TemporaryDirectory() as d:
            store = SnapshotStore(os.path.join(d, "snap.json"))
            cases = [PromptCase(name="greet", prompt="Say hello", expected_behavior="hello greeting")]

            result = run_prompt_suite(cases, llm_fn=lambda p: "Hello there!", snapshot_store=store)
            self.assertEqual(len(result.results), 1)
            self.assertFalse(result.results[0].changed_from_snapshot)
            self.assertIsNotNone(store.get("greet"))

    def test_detects_output_changed_from_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            store = SnapshotStore(os.path.join(d, "snap.json"))
            cases = [PromptCase(name="greet", prompt="Say hello", expected_behavior="hello greeting")]

            run_prompt_suite(cases, llm_fn=lambda p: "Hello there!", snapshot_store=store)
            # second run: the prompt's behavior silently changed
            result = run_prompt_suite(cases, llm_fn=lambda p: "Goodbye forever.", snapshot_store=store)

            self.assertTrue(result.results[0].changed_from_snapshot)
            self.assertEqual(result.results[0].previous_snapshot_output, "Hello there!")

    def test_suite_summary_counts_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as d:
            store = SnapshotStore(os.path.join(d, "snap.json"))
            cases = [
                PromptCase(name="c1", prompt="p1", expected_behavior="refund 30 days"),
                PromptCase(name="c2", prompt="p2", expected_behavior="refund 30 days"),
            ]
            result = run_prompt_suite(
                cases,
                llm_fn=lambda p: "Refunds within 30 days." if p == "p1" else "totally unrelated",
                snapshot_store=store,
                judge=MockJudge(overlap_threshold=0.5),
            )
            self.assertEqual(len(result.passed), 1)
            self.assertEqual(len(result.failed), 1)


if __name__ == "__main__":
    unittest.main()
