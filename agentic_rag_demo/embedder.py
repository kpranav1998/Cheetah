"""
embedder.py
-----------
Local open-source embeddings via sentence-transformers.

No API key or network calls after the first download.
The model weights (~80 MB) are cached in ~/.cache/huggingface/ after first use.

MODEL: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Fast inference (~14k sentences/sec on CPU)
  - Trained on 1B+ sentence pairs
  - MTEB score competitive with older OpenAI ada-002

Other drop-in alternatives (just change model_name):
  - "all-mpnet-base-v2"           → 768 dims, better quality, 3× slower
  - "BAAI/bge-small-en-v1.5"      → 384 dims, strong retrieval benchmark
  - "BAAI/bge-large-en-v1.5"      → 1024 dims, best quality, slow
  - "nomic-ai/nomic-embed-text-v1" → 768 dims, supports long context
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around a SentenceTransformer model with a simple interface.

    Lazy-loads the model on first use so import is instant.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"  Loading embedding model '{self.model_name}'... ", end="", flush=True)
            self._model = SentenceTransformer(self.model_name)
            print("done")
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns a plain Python list."""
        vec: np.ndarray = self.model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(
        self, texts: List[str], show_progress: bool = False, batch_size: int = 64
    ) -> List[List[float]]:
        """
        Embed a list of texts in one efficient pass.

        sentence-transformers handles batching internally — this is much
        faster than calling embed() in a loop.
        """
        vecs: np.ndarray = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.tolist()
