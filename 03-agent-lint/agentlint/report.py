"""Print findings in a linter-style format: `message 12: possible loop with message 7 (94%)`"""
from .analyzer import Finding

SEVERITY_ICON = {"error": "✗", "warning": "⚠"}


def format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "No issues found."
    lines = []
    for f in findings:
        icon = SEVERITY_ICON.get(f.severity, "-")
        loc = ",".join(str(i) for i in f.message_indices)
        lines.append(f"{icon} [{f.kind}] messages {loc}: {f.description}")
    return "\n".join(lines)
