"""
Core pruning logic: given a query and a list of candidate context chunks,
score each chunk's relevance, drop the low-value ones, and reorder the
survivors to counteract the "lost in the middle" effect (models pay less
attention to information buried in the center of a long prompt -- so the
most relevant chunks should sit at the very start and very end).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .embeddings import Embedder, TfidfEmbedder, cosine_sim_matrix


@dataclass
class Chunk:
    text: str
    source: str = ""
    token_count: int | None = None

    def estimate_tokens(self) -> int:
        if self.token_count is not None:
            return self.token_count
        # Rough heuristic: ~4 characters per token for English text.
        return max(1, len(self.text) // 4)


@dataclass
class ScoredChunk:
    chunk: Chunk
    relevance: float


@dataclass
class PrunedResult:
    kept: list[Chunk]
    dropped: list[Chunk]
    ordered_text: str          # ready to paste into a prompt
    total_tokens: int
    scores: list[ScoredChunk]  # every chunk's score, kept or not (for debugging)


class ContextPruner:
    """
    Usage:
        pruner = ContextPruner()  # defaults to offline TF-IDF embedder
        result = pruner.prune(query="What's our refund window?", chunks=chunks, token_budget=500)
        prompt = f"Context:\\n{result.ordered_text}\\n\\nQuestion: ...\"
    """

    def __init__(self, embedder: Embedder | None = None, relevance_floor: float = 0.05):
        self.embedder = embedder or TfidfEmbedder()
        self.relevance_floor = relevance_floor  # chunks scoring below this are always dropped

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[ScoredChunk]:
        if not chunks:
            return []
        texts = [query] + [c.text for c in chunks]
        vectors = self.embedder.embed(texts)
        query_vec, chunk_vecs = vectors[0:1], vectors[1:]
        sims = cosine_sim_matrix(query_vec, chunk_vecs)[0]
        return [ScoredChunk(chunk=c, relevance=float(s)) for c, s in zip(chunks, sims)]

    def prune(self, query: str, chunks: Sequence[Chunk], token_budget: int) -> PrunedResult:
        scored = self.score(query, chunks)
        # Highest relevance first for the greedy budget-fill pass.
        ranked = sorted(scored, key=lambda s: s.relevance, reverse=True)

        kept: list[ScoredChunk] = []
        used_tokens = 0
        for sc in ranked:
            if sc.relevance < self.relevance_floor:
                continue
            t = sc.chunk.estimate_tokens()
            if used_tokens + t > token_budget:
                continue
            kept.append(sc)
            used_tokens += t

        dropped = [sc.chunk for sc in scored if sc.chunk not in [k.chunk for k in kept]]

        ordered_text = self._order_for_attention(kept)

        return PrunedResult(
            kept=[k.chunk for k in kept],
            dropped=dropped,
            ordered_text=ordered_text,
            total_tokens=used_tokens,
            scores=scored,
        )

    @staticmethod
    def _order_for_attention(kept: list[ScoredChunk]) -> str:
        """
        Counteract 'lost in the middle': place the most relevant chunks at
        the START and END of the context, least relevant in the middle.
        """
        if not kept:
            return ""
        ranked = sorted(kept, key=lambda s: s.relevance, reverse=True)
        ordered: list[Chunk | None] = [None] * len(ranked)
        left, right = 0, len(ranked) - 1
        for i, sc in enumerate(ranked):
            if i % 2 == 0:
                ordered[left] = sc.chunk
                left += 1
            else:
                ordered[right] = sc.chunk
                right -= 1
        return "\n\n".join(c.text for c in ordered if c is not None)
