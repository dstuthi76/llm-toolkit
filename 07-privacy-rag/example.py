"""
Example: ingest a "medical record" containing PII, prove it's redacted
before storage, then query the pipeline and inspect the audit trail.

Run: python3 example.py
"""
from privacyrag import PrivacyRAGPipeline, Document, AuditLog

# A fake local "generation" function standing in for a real local model
# call (e.g. via Ollama). Note it only ever sees already-redacted context.
def fake_local_generate(query: str, context: str) -> str:
    return f"Based on the (PII-redacted) records: {context}"


audit_log = AuditLog()
pipeline = PrivacyRAGPipeline(generate_fn=fake_local_generate, audit_log=audit_log)

raw_records = [
    Document(
        doc_id="patient_001",
        text="Patient Jane Doe (jane.doe@example.com, SSN 123-45-6789) was prescribed "
             "medication for hypertension. Follow-up in 30 days.",
    ),
    Document(
        doc_id="patient_002",
        text="Patient John Smith, phone 555-123-4567, MRN-882910, reports improvement "
             "after 30 days of treatment.",
    ),
]

print("--- Ingesting documents (PII redacted before storage) ---")
entities_found = pipeline.ingest(raw_records)
print(f"Entity types redacted across all documents: {sorted(set(entities_found))}\n")

for doc in pipeline._docs:
    print(f"[{doc.doc_id}] stored as: {doc.text}")

print("\n--- Querying the pipeline ---")
answer = pipeline.query("What was the follow-up period after treatment?", user="dr_alvarez")
print(f"Answer: {answer}")

print("\n--- Audit trail for dr_alvarez ---")
for entry in audit_log.history_for_user("dr_alvarez"):
    print(f"[{entry.timestamp:.0f}] query={entry.query!r} retrieved={entry.retrieved_doc_ids} "
          f"redacted_in_query={entry.redacted_entities}")
