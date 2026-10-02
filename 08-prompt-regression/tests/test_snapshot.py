import unittest
import tempfile
import os
from promptcheck.snapshot import SnapshotStore


class TestSnapshotStore(unittest.TestCase):
    def test_save_and_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snapshots.json")
            store = SnapshotStore(path)
            store.save("refund_case", prompt="What's the refund policy?", output="30 days")

            loaded = SnapshotStore(path)  # simulate a fresh process reading it back
            snap = loaded.get("refund_case")
            self.assertIsNotNone(snap)
            self.assertEqual(snap.output, "30 days")

    def test_get_missing_case_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            store = SnapshotStore(os.path.join(d, "snapshots.json"))
            self.assertIsNone(store.get("nonexistent"))

    def test_all_case_names(self):
        with tempfile.TemporaryDirectory() as d:
            store = SnapshotStore(os.path.join(d, "snapshots.json"))
            store.save("case_a", "prompt a", "output a")
            store.save("case_b", "prompt b", "output b")
            self.assertEqual(set(store.all_case_names()), {"case_a", "case_b"})


if __name__ == "__main__":
    unittest.main()
