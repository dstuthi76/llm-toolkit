"""Render a ComparisonResult as a color-coded HTML diff report."""
from __future__ import annotations
import html
from .eval_harness import ComparisonResult

_VERDICT_STYLE = {
    "improved": ("#1a3d2e", "#6ee7a8", "IMPROVED"),
    "regressed": ("#3d1a1a", "#f57070", "REGRESSED"),
    "unchanged_correct": ("#1a2a3d", "#7ab8f5", "still correct"),
    "unchanged_incorrect": ("#2a2a2a", "#999", "still incorrect"),
}

_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>finetune-eval report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 950px; margin: 40px auto; padding: 0 20px; background: #0f1115; color: #e6e6e6; }}
  .summary {{ font-size: 16px; background: #171a21; border: 1px solid #2a2d35; border-radius: 8px; padding: 16px; }}
  .row {{ border: 1px solid #2a2d35; border-radius: 8px; padding: 12px 16px; margin: 10px 0; }}
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; margin-bottom: 8px; }}
  .col {{ display: inline-block; width: 47%; vertical-align: top; }}
  pre {{ white-space: pre-wrap; word-break: break-word; background: #0b0d11; padding: 8px; border-radius: 6px; font-size: 12px; }}
  h3 {{ margin: 20px 0 4px; }}
</style></head><body>
<h1>finetune-eval report</h1>
<div class="summary"><pre>{summary}</pre></div>
<h3>Regressions ({n_regressions}) — look here first</h3>
{regressions_html}
<h3>All examples</h3>
{all_html}
</body></html>"""

_ROW_TEMPLATE = """<div class="row">
  <span class="tag" style="background:{bg};color:{fg}">{label}</span>
  <p><b>Input:</b> {input}</p>
  <div class="col"><b>Base model:</b><pre>{base}</pre></div>
  <div class="col"><b>Fine-tuned model:</b><pre>{finetuned}</pre></div>
</div>"""


def _row_html(c) -> str:
    bg, fg, label = _VERDICT_STYLE[c.verdict]
    return _ROW_TEMPLATE.format(
        bg=bg, fg=fg, label=label,
        input=html.escape(c.input),
        base=html.escape(c.base_output),
        finetuned=html.escape(c.finetuned_output),
    )


def generate_html_report(result: ComparisonResult) -> str:
    regressions_html = "\n".join(_row_html(c) for c in result.regressions) or "<p>None 🎉</p>"
    all_html = "\n".join(_row_html(c) for c in result.comparisons)
    return _TEMPLATE.format(
        summary=html.escape(result.summary()),
        n_regressions=len(result.regressions),
        regressions_html=regressions_html,
        all_html=all_html,
    )
