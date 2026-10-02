"""
PII detection and redaction.

This module ships a regex-based redactor covering common, high-signal
patterns (emails, phone numbers, SSNs, credit-card-like numbers, and a
simple capitalized-name heuristic) so PII stripping works fully offline
with zero dependencies. In production, replace this with Microsoft's
`presidio` library (`pip install presidio-analyzer presidio-anonymizer`),
which has far more comprehensive, ML-backed entity recognition -- swap
`PIIRedactor` for a `PresidioRedactor` implementing the same interface.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class RedactionResult:
    original: str
    redacted: str
    entities_found: list[str]  # e.g. ["EMAIL", "PHONE", "SSN"]


_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "MEDICAL_RECORD_NUMBER": re.compile(r"\bMRN[-:\s]?\d{4,10}\b", re.IGNORECASE),
    # Simple heuristic: two consecutive capitalized words not at sentence
    # start, e.g. "John Smith" -- deliberately conservative; presidio's
    # NER-based name detection is far more accurate for production use.
    "NAME": re.compile(r"(?<!^)(?<!\. )\b[A-Z][a-z]+\s[A-Z][a-z]+\b"),
}


class PIIRedactor:
    def __init__(self, entity_types: list[str] | None = None):
        self.entity_types = entity_types or list(_PATTERNS.keys())

    def redact(self, text: str) -> RedactionResult:
        redacted = text
        found: list[str] = []
        for entity_type in self.entity_types:
            pattern = _PATTERNS[entity_type]
            if pattern.search(redacted):
                found.append(entity_type)
                redacted = pattern.sub(f"[{entity_type}]", redacted)
        return RedactionResult(original=text, redacted=redacted, entities_found=found)
