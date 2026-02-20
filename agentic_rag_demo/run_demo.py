#!/usr/bin/env python3
"""
run_demo.py
-----------
Full end-to-end Agentic RAG demo.

Sections:
  1. CHUNKING   — Compare 3 chunking strategies on the same document
  2. INDEXING   — Embed chunks and build the vector store
  3. NAIVE RAG  — One-shot retrieval baseline
  4. AGENTIC RAG — Multi-step agent with tool calling
  5. EVALUATION — 4 RAG metrics for both approaches

Run:
  python run_demo.py

Requirements:
  Set LLM_MODEL near the top of this file, then set the matching env var:
    ollama/*    → no key needed (ollama serve must be running)
    gpt-*       → export OPENAI_API_KEY=sk-...
    claude-*    → export ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import sys
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import print as rprint

# ── Import our modules ────────────────────────────────────────
from knowledge_base import DOCUMENTS
from chunking import (
    fixed_size_chunker,
    sentence_chunker,
    recursive_chunker,
    compare_strategies,
)
from vector_store import VectorStore
from embedder import Embedder
from rag_agent import NaiveRAG, AgenticRAG
from evaluator import RAGEvaluator


# ── Model config ──────────────────────────────────────────────
# Set via environment variable (recommended) or change the fallback below.
#
#   export LITELLM_MODEL=ollama/llama3.2        # local Ollama (default)
#   export LITELLM_MODEL=ollama/mistral
#   export LITELLM_MODEL=gpt-4o-mini            # OpenAI
#   export LITELLM_MODEL=claude-haiku-4-5-20251001  # Anthropic
#
# Ollama must be running:  ollama serve && ollama pull llama3.2
LLM_MODEL = os.environ.get("LITELLM_MODEL", "ollama/llama3.2")


console = Console()


# ─────────────────────────────────────────────────────────────
# TEST QUESTIONS
# ─────────────────────────────────────────────────────────────
# These three questions showcase different agentic RAG behaviors:
#
#   Q1 — Simple: Single document lookup. Both naive and agentic should do well.
#   Q2 — Multi-hop: Requires info from 2+ documents. Naive RAG struggles because
#          one query can't retrieve from both "momentum" and "RSI" docs equally well.
#          Agentic RAG makes multiple targeted searches.
#   Q3 — No retrieval: General knowledge. Agentic RAG smartly skips retrieval.
#          Naive RAG wastes tokens stuffing irrelevant chunks into the prompt.

TEST_QUESTIONS = [
    {
        "question": "What is the difference between SMA and EMA, and when should I use each?",
        "type":     "Simple — single document",
        "ground_truth": (
            "SMA (Simple Moving Average) gives equal weight to all prices in the window. "
            "EMA (Exponential Moving Average) weights recent prices more heavily using a "
            "smoothing factor k = 2/(n+1), making it faster to react to price changes. "
            "Use SMA for identifying long-term trends and support/resistance. "
            "Use EMA for short-term trading where responsiveness to recent prices matters."
        ),
    },
    {
        "question": (
            "I want to build a momentum strategy. How should I use RSI and MACD together "
            "to decide entry and exit points? What risk management rules should I apply?"
        ),
        "type": "Multi-hop — requires Momentum + RSI + MACD + Risk Management docs",
        "ground_truth": (
            "In a momentum strategy, use MACD for trend confirmation and RSI for "
            "overbought/oversold signals. Enter a long trade when price breaks out to "
            "new highs, MACD shows a bullish crossover, and RSI is above 50 but not yet "
            "overbought (above 70). Exit when RSI crosses above 70 and then reverses, or "
            "when MACD histogram bars start shrinking. For risk management: never risk more "
            "than 1-2% of capital per trade, use a stop loss below the entry-bar low or a "
            "key moving average, and target a minimum 2:1 risk-to-reward ratio."
        ),
    },
    {
        "question": "What is 15 multiplied by 8?",
        "type":     "No retrieval needed — pure arithmetic",
        "ground_truth": "120",
    },
]


# ─────────────────────────────────────────────────────────────
# Section 1: CHUNKING DEMO
# ─────────────────────────────────────────────────────────────
def demo_chunking():
    console.print(Rule("[bold blue]SECTION 1: CHUNKING STRATEGIES[/bold blue]"))
    console.print()
    console.print(
        "We chunk documents before embedding them. The chunk strategy affects "
        "retrieval quality:\n"
        "  • Too large  → noisy context, irrelevant info retrieved\n"
        "  • Too small  → loses surrounding context\n"
        "  • Best       → semantically coherent units\n"
    )

    doc = DOCUMENTS[0]  # Moving Averages — good for comparison
    console.print(f"[dim]Document: '{doc['title']}' ({len(doc['content'])} characters)[/dim]\n")

    stats = compare_strategies(doc)

    # Summary table
    table = Table(title="Chunking Strategy Comparison", show_lines=True)
    table.add_column("Strategy",   style="cyan",   min_width=30)
    table.add_column("# Chunks",   justify="right", style="green")
    table.add_column("Avg Size",   justify="right", style="yellow")
    table.add_column("Min Size",   justify="right", style="red")
    table.add_column("Max Size",   justify="right", style="red")

    labels = {
        "fixed_size": "Fixed-Size (400 chars, 100 overlap)",
        "sentence":   "Sentence-Based (3 sentences, 1 overlap)",
        "recursive":  "Recursive (target 400 chars)",
    }

    for key, label in labels.items():
        s = stats[key]
        table.add_row(label, str(s["n_chunks"]), str(s["avg_size"]), str(s["min_size"]), str(s["max_size"]))

    console.print(table)

    # Show example chunks side-by-side for the first 2 chunks
    console.print("\n[bold]Sample chunks (first 2) from each strategy:[/bold]\n")
    for key, label in labels.items():
        chunks = stats[key]["chunks"]
        console.print(Panel(
            f"[dim]{label}[/dim]\n\n" +
            "\n\n[dim]---[/dim]\n\n".join(
                f"[yellow]Chunk {c.chunk_index + 1} ({len(c.text)} chars):[/yellow]\n{c.text}"
                for c in chunks[:2]
            ),
            title=f"[cyan]{label}[/cyan]",
            expand=False,
        ))

    console.print()
    console.print(
        "[green]✓ Recursive chunking chosen for indexing[/green] — "
        "it preserves paragraph structure and avoids mid-sentence breaks.\n"
    )


# ─────────────────────────────────────────────────────────────
# Section 2: BUILD INDEX
# ─────────────────────────────────────────────────────────────
def build_index(embedder: Embedder) -> VectorStore:
    console.print(Rule("[bold blue]SECTION 2: INDEXING (Embed & Store)[/bold blue]"))
    console.print()

    store = VectorStore()

    # Chunk all documents using recursive strategy
    all_chunks = []
    for doc in DOCUMENTS:
        chunks = recursive_chunker(doc, target_size=400, min_size=80)
        all_chunks.extend(chunks)
        console.print(f"  [dim]{doc['title']}[/dim] → {len(chunks)} chunks")

    console.print(f"\n  Total chunks: [bold]{len(all_chunks)}[/bold] across {len(DOCUMENTS)} documents")

    # Batch embed locally — no API call, runs on CPU
    console.print(f"  Embedding all chunks via {embedder.model_name} (local)...")
    t0 = time.time()
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_batch(texts)
    elapsed = time.time() - t0
    console.print(f"  [green]✓ Embedded {len(embeddings)} chunks in {elapsed:.1f}s[/green]")
    console.print(f"  [dim]Embedding dimensions: {embedder.dimension}[/dim]\n")

    # Store in vector store
    store.add(all_chunks, embeddings)
    console.print(f"  [green]✓ Vector store ready: {store}[/green]\n")

    return store


# ─────────────────────────────────────────────────────────────
# Section 3 & 4: RAG QUERIES
# ─────────────────────────────────────────────────────────────
def run_queries(store: VectorStore, embedder: Embedder) -> tuple:
    """Run both naive and agentic RAG on all test questions."""
    console.print(Rule("[bold blue]SECTION 3 & 4: NAIVE RAG vs AGENTIC RAG[/bold blue]"))

    naive_rag   = NaiveRAG(store, embedder, k=4, model=LLM_MODEL)
    agentic_rag = AgenticRAG(store, embedder, max_iterations=6, model=LLM_MODEL)

    naive_results   = []
    agentic_results = []

    for i, q_data in enumerate(TEST_QUESTIONS):
        question = q_data["question"]
        q_type   = q_data["type"]

        console.print()
        console.print(Panel(
            f"[bold]Q{i+1}: {question}[/bold]\n[dim]{q_type}[/dim]",
            style="bold cyan",
        ))

        # ── Naive RAG ──
        console.print("\n  [yellow]NAIVE RAG[/yellow] (one-shot retrieve-then-read):")
        n_result = naive_rag.query(question)
        naive_results.append({**n_result, **q_data})
        console.print(f"  Retrieved [bold]{len(n_result['retrieved'])}[/bold] chunks with ONE query.")
        console.print(f"  [dim]Answer preview:[/dim] {n_result['answer'][:200]}...")

        console.print()

        # ── Agentic RAG ──
        console.print("  [green]AGENTIC RAG[/green] (agent decides when and how to search):")
        a_result = agentic_rag.query(question, verbose=True)
        agentic_results.append({**a_result, **q_data})
        console.print(f"  [dim]Answer preview:[/dim] {a_result['answer'][:200]}...")

    return naive_results, agentic_results


# ─────────────────────────────────────────────────────────────
# Section 5: EVALUATION
# ─────────────────────────────────────────────────────────────
def run_evaluation(naive_results: list, agentic_results: list, embedder: Embedder):
    console.print()
    console.print(Rule("[bold blue]SECTION 5: EVALUATION (LLM-as-Judge)[/bold blue]"))
    console.print()
    console.print(
        "Metrics:\n"
        "  [bold]Faithfulness[/bold]      — Is the answer grounded in the retrieved context? (no hallucinations)\n"
        "  [bold]Answer Relevance[/bold]  — Does the answer actually address the question?\n"
        "  [bold]Context Precision[/bold] — Are retrieved chunks actually useful for the question?\n"
        "  [bold]Context Recall[/bold]    — Did retrieval capture the info needed to answer?\n"
    )

    evaluator = RAGEvaluator(embedder, judge_model=LLM_MODEL)

    # Build results table
    table = Table(title="RAG Evaluation Results", show_lines=True)
    table.add_column("Q#", justify="center", style="bold", min_width=3)
    table.add_column("Mode", style="cyan", min_width=8)
    table.add_column("Retrievals", justify="center", min_width=10)
    table.add_column("Faithfulness", justify="center", min_width=12)
    table.add_column("Ans Relevance", justify="center", min_width=13)
    table.add_column("Ctx Precision", justify="center", min_width=13)
    table.add_column("Ctx Recall", justify="center", min_width=10)

    def score_fmt(s: float) -> str:
        color = "green" if s >= 0.75 else "yellow" if s >= 0.5 else "red"
        return f"[{color}]{s:.2f}[/{color}]"

    for i, (n_res, a_res) in enumerate(zip(naive_results, agentic_results)):
        gt = n_res.get("ground_truth")
        q_num = str(i + 1)

        # Evaluate naive
        console.print(f"\n  Evaluating Q{i+1} — Naive RAG:")
        n_scores = evaluator.evaluate(n_res, ground_truth=gt, verbose=True)

        # Evaluate agentic
        console.print(f"\n  Evaluating Q{i+1} — Agentic RAG:")
        a_scores = evaluator.evaluate(a_res, ground_truth=gt, verbose=True)

        # Helper to extract score safely
        def get(scores, key):
            return score_fmt(scores[key]["score"]) if key in scores else "[dim]N/A[/dim]"

        table.add_row(
            q_num, "Naive",
            str(n_res["n_retrievals"]),
            get(n_scores, "faithfulness"),
            get(n_scores, "answer_relevance"),
            get(n_scores, "context_precision"),
            get(n_scores, "context_recall"),
        )
        table.add_row(
            "", "Agentic",
            str(a_res["n_retrievals"]),
            get(a_scores, "faithfulness"),
            get(a_scores, "answer_relevance"),
            get(a_scores, "context_precision"),
            get(a_scores, "context_recall"),
        )

    console.print()
    console.print(table)

    # Print key takeaways
    console.print()
    console.print(Panel(
        "[bold]Key Takeaways:[/bold]\n\n"
        "1. [cyan]Q1 (Simple)[/cyan]: Both approaches perform similarly — "
           "a single query is enough.\n\n"
        "2. [cyan]Q2 (Multi-hop)[/cyan]: Agentic RAG makes multiple targeted searches "
           "(momentum + RSI + MACD + risk management) and achieves higher context recall. "
           "Naive RAG's single query can't cover all required topics equally.\n\n"
        "3. [cyan]Q3 (No retrieval)[/cyan]: Agentic RAG correctly skips retrieval "
           "(0 retrievals), avoiding irrelevant context. Naive RAG injects trading docs "
           "into the prompt for a math question — wasted tokens and potential confusion.\n\n"
        "4. [cyan]Context Precision[/cyan]: Agentic RAG tends to have higher precision "
           "because it formulates specific queries for each sub-problem instead of one "
           "broad query.",
        title="[bold green]Architecture Insights[/bold green]",
        style="green",
    ))


# ─────────────────────────────────────────────────────────────
# Architecture Overview
# ─────────────────────────────────────────────────────────────
def print_architecture():
    arch = """
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC RAG ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   USER QUERY                                                      │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────────────────────────────────────┐                     │
│  │           LLM AGENT (gpt-4o-mini)        │                     │
│  │                                          │                     │
│  │  Decision loop:                          │                     │
│  │  1. Do I need to retrieve anything?      │◄──── TOOL RESULTS  │
│  │  2. What query should I use?             │                     │
│  │  3. Are results sufficient?              │                     │
│  │  4. Should I reformulate & search again? │                     │
│  │  5. Synthesize final answer              │                     │
│  └────────────────┬────────────────────────┘                     │
│                   │ Tool Calls                                    │
│       ┌───────────┴────────────┐                                 │
│       │                        │                                  │
│       ▼                        ▼                                  │
│  search_knowledge_base()   list_available_topics()               │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────────────────────────────┐                        │
│  │        VECTOR STORE (numpy)           │                        │
│  │                                       │                        │
│  │  Documents → Chunks → Embeddings     │                        │
│  │  Cosine similarity search            │                        │
│  └──────────────────────────────────────┘                        │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│  CHUNKING                                                         │
│   Fixed-size │ Sentence-based │ Recursive (used here)            │
├─────────────────────────────────────────────────────────────────┤
│  EVALUATION (offline, LLM-as-judge)                               │
│   Faithfulness │ Answer Relevance │ Ctx Precision │ Ctx Recall   │
└─────────────────────────────────────────────────────────────────┘
"""
    console.print(Panel(arch, title="[bold magenta]Architecture[/bold magenta]", style="magenta"))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def _check_api_key(model: str) -> None:
    """Validate that the required API key is set for the chosen model."""
    if model.startswith("ollama/"):
        return  # Ollama runs locally — no API key needed
    if model.startswith(("gpt-", "openai/")):
        if not os.environ.get("OPENAI_API_KEY"):
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set.")
            console.print("  export OPENAI_API_KEY=sk-...")
            sys.exit(1)
    elif model.startswith(("claude-", "anthropic/")):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not set.")
            console.print("  export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)


def main():
    _check_api_key(LLM_MODEL)

    console.print()
    # Create one shared embedder — model loads once, reused everywhere
    embedder = Embedder("all-MiniLM-L6-v2")

    console.print(Panel(
        "[bold]Agentic RAG Demo[/bold]\n"
        "Knowledge Base: Trading Concepts (8 documents)\n"
        f"LLM: {LLM_MODEL}  │  Embeddings: {embedder.model_name} (local)  │  Vector DB: numpy",
        style="bold magenta",
    ))
    console.print()

    print_architecture()

    # Section 1: Chunking
    demo_chunking()

    # Section 2: Build index
    store = build_index(embedder)

    # Sections 3 & 4: Queries
    naive_results, agentic_results = run_queries(store, embedder)

    # Section 5: Evaluation
    run_evaluation(naive_results, agentic_results, embedder)

    console.print()
    console.print("[bold green]Demo complete![/bold green]")


if __name__ == "__main__":
    main()
