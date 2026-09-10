"""
rag/embeddings.py — Local text embedding using sentence-transformers.

WHAT IS AN EMBEDDING?
---------------------
An embedding is a fixed-size vector (a list of numbers) that represents
the semantic meaning of a piece of text.  Two pieces of text with similar
meaning should produce vectors that are close together in vector space
(high cosine similarity).

This is used for RAG: we embed every document chunk, store the vectors,
then at query time embed the user's question and find the nearest chunks.

LOCAL MODEL
-----------
We use ``sentence-transformers`` with the 'all-MiniLM-L6-v2' model
(~22MB, 384-dimensional embeddings).  This runs 100% locally on CPU or GPU.
It is a pre-trained open-weight model used for the RAG embedding layer ONLY.

HONESTY NOTE: this embedding model is NOT our custom MiniLLM.
Our custom MiniLLM is a language model (next-token prediction).
The embedding model (sentence-transformers) is a separate pre-trained
model used to power the vector search component.
This distinction is documented in the honesty ledger.

FALLBACK
--------
If sentence-transformers is not installed, we fall back to a simple
TF-IDF bag-of-words embedding (pure Python, no dependencies).
The fallback is labeled clearly — it is less accurate than the neural model.
"""
from __future__ import annotations

import logging
import math
import re
import hashlib
from collections import Counter
from typing import Protocol

log = logging.getLogger(__name__)


class Embedder(Protocol):
    """Interface contract for any embedding backend."""
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ─────────────────────────────────────────────────────────────────────────────
# Neural embedder (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

class SentenceTransformerEmbedder:
    """
    Local neural embedding using sentence-transformers.
    Pre-trained open-weight model (all-MiniLM-L6-v2) — 384 dimensions.
    Install: pip install sentence-transformers
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            # RAG is designed to work in an air-gapped deployment. Do not
            # unexpectedly download model weights when the service starts;
            # use a cached local model or the stable hashing fallback.
            self._model = SentenceTransformer(self.MODEL_NAME, local_files_only=True)
            self._dim   = 384
            log.info("SentenceTransformerEmbedder loaded: %s  dim=%d",
                     self.MODEL_NAME, self._dim)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF fallback (no dependencies)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFEmbedder:
    """
    Bag-of-words TF-IDF embedding — pure Python, no ML dependencies.

    HONESTLY LABELLED: This is a statistical baseline, NOT a neural model.
    Semantic similarity is less accurate than sentence-transformers.
    Use only when sentence-transformers is unavailable.
    """

    def __init__(self, dim: int = 512) -> None:
        self._dim  = dim
        self._idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """Compute IDF weights from a corpus."""
        N   = len(corpus)
        df: Counter = Counter()
        for doc in corpus:
            for tok in set(self._tokenize(doc)):
                df[tok] += 1
        self._idf = {tok: math.log(N / (cnt + 1)) for tok, cnt in df.items()}
        self._fitted = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _tfidf_vec(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = [0.0] * self._dim
        for tok, count in tf.items():
            idf   = self._idf.get(tok, 1.0)
            score = (count / total) * idf
            # Python's built-in hash is randomized per process. A persistent
            # vector index built with it becomes invalid after a restart.
            idx = int.from_bytes(
                hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(),
                "big",
            ) % self._dim
            vec[idx] += score
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, text: str) -> list[float]:
        return self._tfidf_vec(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def get_embedder() -> "Embedder":
    """
    Return the best available embedder.

    Tries sentence-transformers first.  Falls back to TF-IDF with a warning.
    """
    try:
        embedder = SentenceTransformerEmbedder()
        return embedder
    except Exception as exc:
        log.warning(
            "Neural embedder unavailable (%s) — using stable hashing TF-IDF "
            "fallback. Install and cache sentence-transformers locally for "
            "better retrieval quality.", exc,
        )
        return TFIDFEmbedder()
