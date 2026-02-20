"""
vector_store.py
---------------
A minimal in-memory vector store using numpy.

No external database needed — just numpy arrays and cosine similarity.
In production you'd replace this with ChromaDB, Pinecone, Weaviate, etc.,
but the core idea is identical: store (chunk, embedding) pairs and retrieve
the top-k nearest neighbors for a query embedding.

HOW COSINE SIMILARITY WORKS:
  cos(θ) = (A · B) / (||A|| × ||B||)
  Range: [-1, 1]. Values closer to 1 = more similar.
  Why cosine and not euclidean? Because embeddings are high-dimensional and
  we care about *direction* (semantic meaning) not *magnitude* (length of text).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SearchResult:
    text: str
    doc_title: str
    doc_id: str
    chunk_index: int
    score: float       # cosine similarity [0, 1]
    strategy: str      # which chunking strategy produced this chunk


class VectorStore:
    """
    In-memory vector store backed by a numpy matrix.

    Storage:
      _embeddings: np.ndarray of shape (N, D)  — N chunks, D dimensions
      _chunks:     list of chunk metadata dicts — parallel to _embeddings

    Search:
      Computes cosine similarity between query and all stored embeddings.
      Returns top-k results by similarity score.
    """

    def __init__(self):
        self._chunks: List[dict] = []         # raw metadata
        self._embeddings: List[np.ndarray] = []  # 1-D arrays

    # ── Ingestion ────────────────────────────────────────────────
    def add(self, chunks, embeddings: List[List[float]]):
        """
        Add chunks and their embeddings to the store.

        chunks: list of Chunk objects (from chunking.py)
        embeddings: list of embedding vectors (one per chunk)
        """
        assert len(chunks) == len(embeddings), "Chunks and embeddings must be same length"
        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append({
                "text":        chunk.text,
                "doc_id":      chunk.doc_id,
                "doc_title":   chunk.doc_title,
                "chunk_index": chunk.chunk_index,
                "strategy":    chunk.strategy,
            })
            self._embeddings.append(np.array(emb, dtype=np.float32))

    # ── Retrieval ────────────────────────────────────────────────
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Retrieve top-k most similar chunks using cosine similarity.

        filter: optional dict of metadata conditions — ALL must match (AND logic).
          Examples:
            filter={"doc_title": "RSI - Relative Strength Index"}
            filter={"doc_id": "doc_2"}
            filter={"strategy": "recursive"}

        How metadata filtering works:
          1. Build a candidate list — only indices where every filter key matches
          2. Slice the embedding matrix to those candidates only
          3. Run cosine similarity on the reduced matrix (faster + scoped)
          4. Return top-k from the filtered candidates

        This is the same pre-filtering logic used by ChromaDB's `where` clause,
        Pinecone's `filter`, and Weaviate's `where` filter.
        """
        if not self._embeddings:
            return []

        # ── Step 1: apply metadata filter ────────────────────────
        if filter:
            candidate_idx = [
                i for i, c in enumerate(self._chunks)
                if all(c.get(key) == val for key, val in filter.items())
            ]
            if not candidate_idx:
                return []  # nothing matches — return empty rather than error
        else:
            candidate_idx = list(range(len(self._chunks)))

        # ── Step 2: build matrix from candidates only ─────────────
        q = np.array(query_embedding, dtype=np.float32)
        matrix = np.stack([self._embeddings[i] for i in candidate_idx])  # (M, D)

        # ── Step 3: normalised cosine similarity ──────────────────
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        m_norm = matrix / m_norms
        similarities = m_norm @ q_norm                  # shape: (M,)

        # ── Step 4: top-k from filtered candidates ────────────────
        actual_k = min(k, len(candidate_idx))
        top_local = np.argsort(similarities)[-actual_k:][::-1]

        results = []
        for local_i in top_local:
            global_i = candidate_idx[local_i]
            c = self._chunks[global_i]
            results.append(SearchResult(
                text=c["text"],
                doc_title=c["doc_title"],
                doc_id=c["doc_id"],
                chunk_index=c["chunk_index"],
                score=float(similarities[local_i]),
                strategy=c["strategy"],
            ))
        return results

    # ── Metadata ─────────────────────────────────────────────────
    def get_topics(self) -> List[str]:
        """Return all unique document titles in the store."""
        return sorted(set(c["doc_title"] for c in self._chunks))

    @property
    def size(self) -> int:
        return len(self._chunks)

    def __repr__(self):
        return f"VectorStore(chunks={self.size}, topics={len(self.get_topics())})"
