import unittest
from privacyrag.audit_log import AuditLog


class TestAuditLog(unittest.TestCase):
    def test_records_and_retrieves_entry(self):
        log = AuditLog()
        entry = log.record(user="alice", query="what is the refund policy",
                            retrieved_doc_ids=["doc1", "doc2"], redacted_entities=["EMAIL"])
        self.assertIsNotNone(entry.row_id)

        history = log.history_for_user("alice")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].query, "what is the refund policy")
        self.assertEqual(history[0].retrieved_doc_ids, ["doc1", "doc2"])

    def test_isolates_history_per_user(self):
        log = AuditLog()
        log.record(user="alice", query="q1", retrieved_doc_ids=[], redacted_entities=[])
        log.record(user="bob", query="q2", retrieved_doc_ids=[], redacted_entities=[])

        self.assertEqual(len(log.history_for_user("alice")), 1)
        self.assertEqual(len(log.history_for_user("bob")), 1)
        self.assertEqual(len(log.all_entries()), 2)

    def test_no_update_or_delete_methods_exist(self):
        # Structural guarantee: this class exposes no way to mutate or
        # remove a row once written, only to record new ones and read.
        log = AuditLog()
        public_methods = [m for m in dir(log) if not m.startswith("_") and callable(getattr(log, m))]
        self.assertNotIn("update", public_methods)
        self.assertNotIn("delete", public_methods)


if __name__ == "__main__":
    unittest.main()
