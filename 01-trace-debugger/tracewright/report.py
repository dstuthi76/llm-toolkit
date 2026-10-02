"""Render a Trace + Diagnosis as a single self-contained HTML file."""
from __future__ import annotations
import html
from .tracer import Trace
from .judge import Diagnosis

_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>tracewright report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #0f1115; color: #e6e6e6; }}
  h1 {{ font-size: 20px; }}
  .verdict {{ display: inline-block; padding: 4px 12px; border-radius: 6px; font-weight: 600; }}
  .success {{ background: #1a3d2e; color: #6ee7a8; }}
  .suspect {{ background: #3d331a; color: #f5c563; }}
  .failed  {{ background: #3d1a1a; color: #f57070; }}
  .unknown {{ background: #2a2a2a; color: #aaa; }}
  .span {{ border: 1px solid #2a2d35; border-radius: 8px; padding: 12px 16px; margin: 10px 0; background: #171a21; }}
  .span.root {{ border-color: #f57070; box-shadow: 0 0 0 1px #f57070; }}
  .span.errored {{ border-left: 4px solid #f57070; }}
  .meta {{ color: #888; font-size: 12px; }}
  pre {{ white-space: pre-wrap; word-break: break-word; background: #0b0d11; padding: 8px; border-radius: 6px; font-size: 12px; }}
  .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 10px; background: #2a2d35; margin-right: 6px; }}
</style></head><body>
<h1>tracewright report</h1>
<p><b>Task:</b> {task}</p>
<p><span class="verdict {verdict_class}">{verdict}</span>
   {root_cause_note}</p>
<p class="meta">{reason}</p>
<h2>Steps ({n_spans})</h2>
{spans_html}
</body></html>"""

_SPAN_TEMPLATE = """<div class="span {classes}">
  <span class="badge">{type}</span> <b>{name}</b>
  <span class="meta">step {step} · {duration:.1f}ms{root_tag}</span>
  <pre>input:  {input}
output: {output}{error_line}</pre>
</div>"""


def generate_html_report(trace: Trace, diagnosis: Diagnosis) -> str:
    spans_html = []
    for s in trace.spans:
        classes = []
        root_tag = ""
        if s.error:
            classes.append("errored")
        if diagnosis.root_cause_step == s.step_number:
            classes.append("root")
            root_tag = " · ⚠ ROOT CAUSE"
        error_line = f"\nerror:  {html.escape(str(s.error))}" if s.error else ""
        spans_html.append(_SPAN_TEMPLATE.format(
            classes=" ".join(classes),
            type=s.type.value if hasattr(s.type, "value") else s.type,
            name=html.escape(s.name),
            step=s.step_number,
            duration=s.duration_ms,
            root_tag=root_tag,
            input=html.escape(str(s.input))[:500],
            output=html.escape(str(s.output))[:500],
            error_line=error_line,
        ))

    root_cause_note = (
        f"(root cause: step {diagnosis.root_cause_step})"
        if diagnosis.root_cause_step else ""
    )

    return _TEMPLATE.format(
        task=html.escape(trace.task),
        verdict=diagnosis.verdict,
        verdict_class=diagnosis.verdict if diagnosis.verdict in
            ("success", "suspect", "failed") else "unknown",
        root_cause_note=root_cause_note,
        reason=html.escape(diagnosis.reason),
        n_spans=len(trace.spans),
        spans_html="\n".join(spans_html),
    )
