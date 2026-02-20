"""
chunking.py
-----------
Three chunking strategies demonstrated side-by-side.

WHY CHUNKING MATTERS:
  - Too large a chunk → noisy context, retrieval retrieves irrelevant parts
  - Too small a chunk → loses context, answer lacks surrounding information
  - The goal: chunks that are semantically coherent and retrievable

STRATEGIES:
  1. Fixed-Size   — naive, ignores sentence boundaries
  2. Sentence     — respects sentence structure
  3. Recursive    — hierarchical, tries paragraphs first then sentences
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    doc_id: str
    doc_title: str
    chunk_index: int
    strategy: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Chunk(doc='{self.doc_title}', idx={self.chunk_index}, len={len(self.text)}, strategy='{self.strategy}')"


# ─────────────────────────────────────────────────────────────
# Strategy 1: Fixed-Size Chunking
# ─────────────────────────────────────────────────────────────
def fixed_size_chunker(doc: dict, chunk_size: int = 400, overlap: int = 100) -> List[Chunk]:
    """
    Split text into fixed-size character windows with overlap.

    PROS: Simple, predictable chunk sizes, easy to implement.
    CONS: Breaks in the middle of sentences/words, ignores semantic structure.

    The overlap ensures that context at chunk boundaries isn't lost —
    a sentence split across two chunks will appear in both.
    """
    text = doc["content"]
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunks.append(Chunk(
            text=chunk_text,
            doc_id=doc["id"],
            doc_title=doc["title"],
            chunk_index=idx,
            strategy="fixed_size",
            metadata={"char_start": start, "char_end": end},
        ))

        start += chunk_size - overlap  # move forward by (chunk_size - overlap)
        idx += 1

    return chunks


# ─────────────────────────────────────────────────────────────
# Strategy 2: Sentence-Based Chunking
# ─────────────────────────────────────────────────────────────
def sentence_chunker(
    doc: dict, sentences_per_chunk: int = 3, overlap_sentences: int = 1
) -> List[Chunk]:
    """
    Split text into groups of N sentences, with M sentence overlap.

    PROS: Respects natural sentence boundaries, semantically cleaner.
    CONS: Chunk sizes vary widely (short vs. long sentences), still naive
          about paragraph/topic structure.

    The 1-sentence overlap helps preserve context at boundaries.
    """
    text = doc["content"]

    # Split on sentence-ending punctuation followed by whitespace
    # Handles: ". " "! " "? " and newlines between sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    step = max(1, sentences_per_chunk - overlap_sentences)

    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + sentences_per_chunk]
        if not chunk_sentences:
            continue

        chunk_text = " ".join(chunk_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            doc_id=doc["id"],
            doc_title=doc["title"],
            chunk_index=len(chunks),
            strategy="sentence",
            metadata={
                "sentence_start": i,
                "sentence_end": i + len(chunk_sentences),
                "n_sentences": len(chunk_sentences),
            },
        ))

    return chunks


# ─────────────────────────────────────────────────────────────
# Strategy 3: Recursive / Hierarchical Chunking
# ─────────────────────────────────────────────────────────────
def recursive_chunker(
    doc: dict, target_size: int = 400, min_size: int = 80
) -> List[Chunk]:
    """
    Split by paragraph → sentence → word, respecting a target chunk size.

    PROS: Best of both worlds — preserves semantic structure (paragraphs)
          while ensuring chunks don't exceed a size limit. Used by LangChain's
          RecursiveCharacterTextSplitter.
    CONS: Slightly more complex to implement.

    Algorithm:
      1. Try splitting on double-newline (paragraph breaks)
      2. If a part is still too large, split on single newline
      3. If still too large, split on ". " (sentence boundary)
      4. If still too large, split on " " (words)
      5. Merge adjacent tiny chunks until they reach target_size
    """
    text = doc["content"]
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, seps: List[str]) -> List[str]:
        """Recursively split text using the next separator if chunk is too large."""
        if not seps or len(text) <= target_size:
            return [text]

        sep = seps[0]
        parts = text.split(sep)
        result = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= target_size:
                result.append(part)
            else:
                # Part is still too big — go deeper with next separator
                result.extend(_split(part, seps[1:]))

        return result

    raw_parts = _split(text, separators)

    # Merge small adjacent chunks so we don't have tiny fragments
    merged: List[str] = []
    buffer = ""
    for part in raw_parts:
        if len(buffer) + len(part) + 1 <= target_size:
            buffer = (buffer + " " + part).strip() if buffer else part
        else:
            if buffer:
                merged.append(buffer)
            buffer = part
    if buffer:
        merged.append(buffer)

    return [
        Chunk(
            text=c,
            doc_id=doc["id"],
            doc_title=doc["title"],
            chunk_index=i,
            strategy="recursive",
            metadata={"length": len(c)},
        )
        for i, c in enumerate(merged)
        if len(c) >= min_size  # drop tiny leftover fragments
    ]


# ─────────────────────────────────────────────────────────────
# Comparison helper
# ─────────────────────────────────────────────────────────────
def compare_strategies(doc: dict) -> dict:
    """Return stats for all three strategies applied to the same document."""
    strategies = {
        "fixed_size": fixed_size_chunker(doc, chunk_size=400, overlap=100),
        "sentence":   sentence_chunker(doc, sentences_per_chunk=3, overlap_sentences=1),
        "recursive":  recursive_chunker(doc, target_size=400, min_size=80),
    }
    stats = {}
    for name, chunks in strategies.items():
        sizes = [len(c.text) for c in chunks]
        stats[name] = {
            "n_chunks": len(chunks),
            "avg_size": round(sum(sizes) / len(sizes)) if sizes else 0,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "chunks": chunks,
        }
    return stats
