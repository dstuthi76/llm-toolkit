import unittest
from privacyrag.rag_pipeline import PrivacyRAGPipeline, Document
from privacyrag.audit_log import AuditLog


class TestPrivacyRAGPipeline(unittest.TestCase):
    def test_pii_is_redacted_before_storage(self):
        pipeline = PrivacyRAGPipeline(generate_fn=lambda q, c: f"Answer based on: {c}")
        entities = pipeline.ingest([
            Document(doc_id="d1", text="Patient John Smith (john.smith@example.com) has a refund request."),
        ])
        self.assertIn("EMAIL", entities)
        # the stored doc should NOT contain the raw email anymore
        self.assertNotIn("john.smith@example.com", pipeline._docs[0].text)

    def test_query_retrieves_relevant_redacted_context(self):
        # Note: default TfidfVectorizer does literal word matching, not
        # stemming -- "refund" and "refunds" are different tokens to it.
        # Production should swap in real semantic embeddings (see
        # contextprune's SentenceTransformerEmbedder for the same pattern).
        pipeline = PrivacyRAGPipeline(generate_fn=lambda q, c: c)
        pipeline.ingest([
            Document(doc_id="refund_policy", text="Our refund policy allows returns within 30 days of purchase."),
            Document(doc_id="office_info", text="The office has a rooftop garden."),
        ])
        answer = pipeline.query("What is the refund policy?", user="alice")
        self.assertIn("30 days", answer)

    def test_audit_log_records_every_query(self):
        audit_log = AuditLog()
        pipeline = PrivacyRAGPipeline(
            generate_fn=lambda q, c: "some answer",
            audit_log=audit_log,
        )
        pipeline.ingest([Document(doc_id="d1", text="Refunds within 30 days.")])
        pipeline.query("refund question", user="alice")
        pipeline.query("another refund question", user="alice")

        history = audit_log.history_for_user("alice")
        self.assertEqual(len(history), 2)

    def test_pii_in_query_itself_is_redacted_before_logging(self):
        audit_log = AuditLog()
        pipeline = PrivacyRAGPipeline(
            generate_fn=lambda q, c: "answer",
            audit_log=audit_log,
        )
        pipeline.ingest([Document(doc_id="d1", text="General company info.")])
        pipeline.query("My email is bob@example.com, what's the refund policy?", user="bob")

        history = audit_log.history_for_user("bob")
        self.assertNotIn("bob@example.com", history[0].query)
        self.assertIn("EMAIL", history[0].redacted_entities)


if __name__ == "__main__":
    unittest.main()
