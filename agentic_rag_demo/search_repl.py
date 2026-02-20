#!/usr/bin/env python3
"""
search_repl.py
--------------
Interactive search loop over the trading knowledge base.

No LLM required — just the local embedding model and vector store.
Type a query, get back the top-K most relevant chunks with scores.

Usage:
  python3 search_repl.py              # top 2 chunks (default)
  python3 search_repl.py --k 5        # top 5 chunks
  python3 search_repl.py --filter "RSI - Relative Strength Index"  # scoped search
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from knowledge_base import DOCUMENTS
from chunking import recursive_chunker
from embedder import Embedder
from vector_store import VectorStore


def build_store(embedder: Embedder) -> VectorStore:
    store = VectorStore()
    all_chunks = []
    for doc in DOCUMENTS:
        chunks = recursive_chunker(doc, target_size=400, min_size=80)
        all_chunks.extend(chunks)

    print(f"Embedding {len(all_chunks)} chunks with {embedder.model_name}...")
    embeddings = embedder.embed_batch([c.text for c in all_chunks])
    store.add(all_chunks, embeddings)
    print(f"Index ready: {store}\n")
    return store


def run_repl(store: VectorStore, embedder: Embedder, k: int, doc_filter: str):
    filter_dict = {"doc_title": doc_filter} if doc_filter else None

    if doc_filter:
        print(f"Filter active: only searching within '{doc_filter}'")
    print(f"Returning top {k} chunk(s) per query.")
    print("Topics available:", ", ".join(store.get_topics()))
    print("\nType a query and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        q_emb = embedder.embed(query)
        results = store.search(q_emb, k=k, filter=filter_dict)

        if not results:
            print("  No results found.\n")
            continue

        print()
        for i, r in enumerate(results, 1):
            bar = "█" * int(r.score * 20)
            print(f"  ── Result {i} ──────────────────────────────────────")
            print(f"  Doc   : {r.doc_title}")
            print(f"  Score : {r.score:.3f}  {bar}")
            print(f"  Chunk : #{r.chunk_index}")
            print()
            # Wrap text at 70 chars for readability
            words = r.text.split()
            line, lines = [], []
            for w in words:
                if sum(len(x) + 1 for x in line) + len(w) > 70:
                    lines.append(" ".join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines.append(" ".join(line))
            for l in lines:
                print(f"  {l}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Interactive RAG chunk search")
    parser.add_argument("--k", type=int, default=2,
                        help="Number of top chunks to return (default: 2)")
    parser.add_argument("--filter", type=str, default="",
                        help="Restrict search to a specific document title")
    args = parser.parse_args()

    embedder = Embedder("all-MiniLM-L6-v2")
    store = build_store(embedder)
    run_repl(store, embedder, k=args.k, doc_filter=args.filter)


if __name__ == "__main__":
    main()
