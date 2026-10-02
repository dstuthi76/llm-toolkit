"""
Append-only audit log for compliance: every query, what was retrieved, and
who asked it. Uses SQLite (stdlib-adjacent, ships with Python) with a
design that only ever INSERTs -- no UPDATE/DELETE statements exist in this
module, so the log can't be silently edited by application code.
"""
from __future__ import annotations
import sqlite3
import time
import json
from dataclasses import dataclass, field


@dataclass
class AuditEntry:
    timestamp: float
    user: str
    query: str
    retrieved_doc_ids: list[str]
    redacted_entities: list[str]
    row_id: int | None = None


class AuditLog:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                user TEXT NOT NULL,
                query TEXT NOT NULL,
                retrieved_doc_ids TEXT NOT NULL,
                redacted_entities TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def record(self, user: str, query: str, retrieved_doc_ids: list[str],
               redacted_entities: list[str]) -> AuditEntry:
        ts = time.time()
        cursor = self.conn.execute(
            "INSERT INTO audit_log (timestamp, user, query, retrieved_doc_ids, redacted_entities) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, user, query, json.dumps(retrieved_doc_ids), json.dumps(redacted_entities)),
        )
        self.conn.commit()
        return AuditEntry(timestamp=ts, user=user, query=query,
                           retrieved_doc_ids=retrieved_doc_ids,
                           redacted_entities=redacted_entities, row_id=cursor.lastrowid)

    def history_for_user(self, user: str) -> list[AuditEntry]:
        rows = self.conn.execute(
            "SELECT id, timestamp, user, query, retrieved_doc_ids, redacted_entities "
            "FROM audit_log WHERE user = ? ORDER BY timestamp ASC", (user,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def all_entries(self) -> list[AuditEntry]:
        rows = self.conn.execute(
            "SELECT id, timestamp, user, query, retrieved_doc_ids, redacted_entities "
            "FROM audit_log ORDER BY timestamp ASC"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    @staticmethod
    def _row_to_entry(row) -> AuditEntry:
        row_id, ts, user, query, doc_ids_json, entities_json = row
        return AuditEntry(
            timestamp=ts, user=user, query=query,
            retrieved_doc_ids=json.loads(doc_ids_json),
            redacted_entities=json.loads(entities_json),
            row_id=row_id,
        )
