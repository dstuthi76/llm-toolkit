"""Command-line entry point: `python -m tracewright.cli analyze trace.json`"""
import argparse
import sys
from .tracer import Trace
from .judge import Judge
from .report import generate_html_report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tracewright")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Diagnose a saved trace JSON file")
    analyze.add_argument("trace_path")
    analyze.add_argument("-o", "--output", default="tracewright_report.html")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        trace = Trace.from_json(args.trace_path)
        diagnosis = Judge().diagnose(trace)
        report_html = generate_html_report(trace, diagnosis)
        with open(args.output, "w") as f:
            f.write(report_html)
        print(f"Verdict: {diagnosis.verdict}")
        print(f"Reason:  {diagnosis.reason}")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
