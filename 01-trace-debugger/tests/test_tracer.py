import unittest
from tracewright.tracer import Tracer, SpanType


class TestTracer(unittest.TestCase):
    def test_records_successful_tool_call(self):
        tracer = Tracer(task="test task")

        @tracer.trace_tool("add")
        def add(a, b):
            return a + b

        result = add(2, 3)
        self.assertEqual(result, 5)
        self.assertEqual(len(tracer.trace.spans), 1)
        span = tracer.trace.spans[0]
        self.assertEqual(span.type, SpanType.TOOL_CALL)
        self.assertEqual(span.name, "add")
        self.assertEqual(span.output, 5)
        self.assertIsNone(span.error)

    def test_records_error(self):
        tracer = Tracer(task="test task")

        @tracer.trace_tool("divide")
        def divide(a, b):
            return a / b

        with self.assertRaises(ZeroDivisionError):
            divide(1, 0)

        span = tracer.trace.spans[0]
        self.assertIn("ZeroDivisionError", span.error)

    def test_step_numbers_increment_across_calls(self):
        tracer = Tracer(task="test task")

        @tracer.trace_llm("model-a")
        def llm(prompt):
            return "ok"

        @tracer.trace_tool("tool-a")
        def tool(x):
            return x

        llm("hi")
        tool(1)
        llm("again")

        steps = [s.step_number for s in tracer.trace.spans]
        self.assertEqual(steps, [1, 2, 3])

    def test_manual_span(self):
        tracer = Tracer(task="test task")
        tracer.manual_span("tool_call", "search", input={"q": "paris"}, output=["r1"])
        self.assertEqual(len(tracer.trace.spans), 1)
        self.assertEqual(tracer.trace.spans[0].name, "search")

    def test_trace_json_roundtrip(self):
        import tempfile, os
        tracer = Tracer(task="round trip")

        @tracer.trace_tool("noop")
        def noop():
            return "done"

        noop()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.json")
            tracer.trace.to_json(path)
            from tracewright.tracer import Trace
            loaded = Trace.from_json(path)
            self.assertEqual(loaded.task, "round trip")
            self.assertEqual(len(loaded.spans), 1)
            self.assertEqual(loaded.spans[0].output, "done")


if __name__ == "__main__":
    unittest.main()
