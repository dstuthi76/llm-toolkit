"""
Pluggable text -> vector embedding backends.

`TfidfEmbedder` runs fully offline using scikit-learn's TF-IDF vectorizer
(no model download, no network, no API cost) -- good enough to demonstrate
and test relevance scoring. Swap in `SentenceTransformerEmbedder` for real
semantic embeddings in production (it needs `pip install sentence-transformers`
and, on first use, a one-time model download).
"""
from __future__ import annotations
from typing import Protocol, Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n_texts, dim) array of vectors."""
        ...


class TfidfEmbedder:
    """
    Offline embedder using TF-IDF + cosine similarity.

    This is NOT a substitute for real semantic embeddings (it can't tell
    that "car" and "automobile" are related) -- it's a dependency-free
    stand-in so the relevance-scoring and pruning LOGIC can be built,
    tested, and demoed without a network connection or a multi-GB model
    download. Swap the embedder, keep everything else.
    """

    def __init__(self):
        # stop_words='english' matters more than it looks: without it, common
        # words like "the"/"is" inflate similarity for completely irrelevant
        # chunks that happen to share only stopwords with the query.
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        # Fit fresh each call so the vocabulary always matches the given
        # batch (query + chunks) -- fine for this tool's usage pattern,
        # where you always score one query against one set of chunks together.
        matrix = self._vectorizer.fit_transform(list(texts))
        return matrix.toarray()


class SentenceTransformerEmbedder:
    """
    Real semantic embeddings via `sentence-transformers`.
    Requires: pip install sentence-transformers
    Not used in tests (needs network access to download the model on first run).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(list(texts))


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cosine_similarity(a, b)
