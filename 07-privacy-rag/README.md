# privacyrag

**RAG for sensitive documents. PII is redacted before it's ever embedded, stored, or sent anywhere — with a full, append-only audit trail of every query.**

RAG over medical, legal, or financial documents needs more than "send it to OpenAI and hope." `privacyrag` redacts names, emails, SSNs, phone numbers, and medical record numbers *before* documents are chunked or embedded, runs fully offline by default, and logs every query and what it retrieved to an immutable audit trail.

## Why

- Most RAG tutorials assume it's fine to send raw documents to a cloud API.
- Compliance-sensitive domains (healthcare, legal, finance) need PII stripped *before* it leaves the building, not after.
- Nobody easily gets an audit trail of "who asked what, and what was retrieved" out of a typical RAG stack.

## Install

```bash
pip install privacyrag
# for production-grade PII detection (recommended):
pip install "privacyrag[presidio]"
# for real local semantic embeddings instead of offline TF-IDF:
pip install "privacyrag[local-llm]"
```

## Quick start

```python
from privacyrag import PrivacyRAGPipeline, Document, AuditLog

audit_log = AuditLog(db_path="audit.db")  # ":memory:" for testing, a file path for persistence

pipeline = PrivacyRAGPipeline(
    generate_fn=lambda query, context: call_local_model(query, context),  # e.g. via Ollama
    audit_log=audit_log,
)

pipeline.ingest([
    Document(doc_id="patient_001", text="Patient Jane Doe (jane.doe@example.com) ..."),
])
# stored text is now: "Patient [NAME] ([EMAIL]) ..." -- redacted before storage

answer = pipeline.query("What was the follow-up period?", user="dr_alvarez")
```

## PII redaction: built-in regex, or Presidio for production

The bundled `PIIRedactor` covers emails, SSNs, phone numbers, medical record numbers, and a conservative capitalized-name heuristic — fully offline, zero dependencies, good for getting started and for tests. For production-grade, ML-backed entity recognition (many more entity types, far fewer false negatives), swap in Microsoft's [Presidio](https://microsoft.github.io/presidio/):

```python
# your own thin adapter implementing the same .redact(text) -> RedactionResult interface
pipeline = PrivacyRAGPipeline(generate_fn=..., redactor=PresidioRedactor())
```

## Running fully offline (local-first by design)

Point `generate_fn` at a local model via [Ollama](https://ollama.com) so nothing — not the documents, not the queries, not the answers — ever leaves the machine:

```python
import requests

def local_ollama_generate(query: str, context: str) -> str:
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3", "prompt": f"Context: {context}\n\nQuestion: {query}", "stream": False,
    })
    return resp.json()["response"]

pipeline = PrivacyRAGPipeline(generate_fn=local_ollama_generate)
```

Cloud models should be opt-in only, and should only ever receive the already-redacted context — never raw documents.

## Audit log

Every query is recorded with a timestamp, the (redacted) query text, which document IDs were retrieved, and which PII entity types were caught and stripped:

```python
for entry in audit_log.history_for_user("dr_alvarez"):
    print(entry.timestamp, entry.query, entry.retrieved_doc_ids, entry.redacted_entities)
```

The `AuditLog` class exposes no update or delete methods — only `record()` and read methods — so application code has no way to silently edit or remove a logged entry.

## Development

```bash
python -m unittest discover -s tests -v
python3 example.py
```

## Roadmap

- [ ] Native Presidio adapter shipped in this package
- [ ] Local sentence-transformers embedder as the default (instead of TF-IDF)
- [ ] Configurable redaction entity sets per document type (medical vs. legal vs. financial)
- [ ] Export audit log to a compliance-friendly report format

## License

MIT
