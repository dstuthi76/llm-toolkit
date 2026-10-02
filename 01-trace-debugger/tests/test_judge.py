import unittest
from tracewright.tracer import Tracer
from tracewright.judge import Judge, MockLLMClient


class TestJudge(unittest.TestCase):
    def test_flags_errored_step_as_root_cause(self):
        tracer = Tracer(task="book a flight")

        @tracer.trace_tool("check_calendar")
        def check_calendar():
            return "free"

        @tracer.trace_tool("search_flights")
        def search_flights():
            raise ConnectionError("timeout contacting flight API")

        check_calendar()
        try:
            search_flights()
        except ConnectionError:
            pass

        diagnosis = Judge(MockLLMClient()).diagnose(tracer.trace)
        self.assertEqual(diagnosis.verdict, "failed")
        self.assertEqual(diagnosis.root_cause_step, 2)
        self.assertIn("ConnectionError", diagnosis.reason)

    def test_flags_empty_tool_output_as_suspect(self):
        tracer = Tracer(task="find hotel")

        @tracer.trace_tool("search_hotels")
        def search_hotels():
            return []

        search_hotels()
        diagnosis = Judge(MockLLMClient()).diagnose(tracer.trace)
        self.assertEqual(diagnosis.verdict, "suspect")
        self.assertEqual(diagnosis.root_cause_step, 1)

    def test_clean_run_is_success(self):
        tracer = Tracer(task="add numbers")

        @tracer.trace_tool("add")
        def add(a, b):
            return a + b

        add(2, 3)
        diagnosis = Judge(MockLLMClient()).diagnose(tracer.trace)
        self.assertEqual(diagnosis.verdict, "success")
        self.assertIsNone(diagnosis.root_cause_step)


if __name__ == "__main__":
    unittest.main()
