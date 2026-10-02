"""
The privacy-preserving RAG pipeline: redact PII before anything else
happens, embed and store locally (TF-IDF here, swap for local sentence-
transformers in production -- see contextprune's embeddings.py for that
same swap point), and log every query to an audit trail.

Generation is injected as a plain callable (`generate_fn`) so this runs
fully offline in tests. In production, point `generate_fn` at a fully
local model via Ollama so nothing ever leaves the machine, or -- only
with explicit opt-in -- at a cloud model, passing ONLY the already-
redacted context.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .pii import PIIRedactor
from .audit_log import AuditLog


@dataclass
class Document:
    doc_id: str
    text: str


class PrivacyRAGPipeline:
    def __init__(self, generate_fn: Callable[[str, str], str],
                 redactor: PIIRedactor | None = None,
                 audit_log: AuditLog | None = None,
                 top_k: int = 3):
        """
        generate_fn(query, context) -> answer_text
        """
        self.generate_fn = generate_fn
        self.redactor = redactor or PIIRedactor()
        self.audit_log = audit_log or AuditLog()
        self.top_k = top_k
        self._docs: list[Document] = []
        self._redacted_texts: list[str] = []

    def ingest(self, documents: list[Document]) -> list[str]:
        """Redact PII from each document BEFORE storing or embedding it. Returns entity types found per doc."""
        entity_report = []
        for doc in documents:
            result = self.redactor.redact(doc.text)
            self._docs.append(Document(doc_id=doc.doc_id, text=result.redacted))
            self._redacted_texts.append(result.redacted)
            entity_report.extend(result.entities_found)
        return entity_report

    def _retrieve(self, query: str) -> list[Document]:
        if not self._docs:
            return []
        texts = [query] + self._redacted_texts
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(texts).toarray()
        query_vec, doc_vecs = vectors[0:1], vectors[1:]
        sims = cosine_similarity(query_vec, doc_vecs)[0]
        ranked_idx = np.argsort(sims)[::-1][:self.top_k]
        return [self._docs[i] for i in ranked_idx if sims[i] > 0]

    def query(self, question: str, user: str = "anonymous") -> str:
        # Redact PII in the query itself too -- a user might paste a
        # customer's details directly into their question.
        redacted_query = self.redactor.redact(question)

        retrieved = self._retrieve(redacted_query.redacted)
        context = "\n\n".join(d.text for d in retrieved)

        answer = self.generate_fn(redacted_query.redacted, context)

        self.audit_log.record(
            user=user,
            query=redacted_query.redacted,
            retrieved_doc_ids=[d.doc_id for d in retrieved],
            redacted_entities=redacted_query.entities_found,
        )
        return answer
