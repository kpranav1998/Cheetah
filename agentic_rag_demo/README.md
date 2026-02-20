# Agentic RAG Demo

**Problem:** Q&A over a trading concepts knowledge base (8 docs, ~400 words each).

## Run

```bash
export OPENAI_API_KEY=sk-...
cd agentic_rag_demo
python3 run_demo.py
```

No extra dependencies — uses `openai`, `numpy`, `rich` (already in requirements.txt).

---

## Architecture

```
USER QUERY
    │
    ▼
LLM AGENT (gpt-4o-mini) ──── Tool Calls ────► search_knowledge_base(query)
    │                                          list_available_topics()
    │  ◄── tool results                              │
    │                                               ▼
    │                                    VECTOR STORE (numpy)
    │                                    Cosine similarity search
    ▼
FINAL ANSWER (synthesized from multi-step retrieval)
```

### Naive RAG vs Agentic RAG

| | Naive RAG | Agentic RAG |
|--|--|--|
| Retrieval | One query, one shot | Agent decides when/how many times |
| Multi-hop | Single broad query misses topics | Multiple targeted sub-queries |
| No-retrieval | Injects irrelevant chunks | Skips retrieval entirely |
| Query quality | Raw user question | Agent reformulates as needed |

---

## Files

| File | What it teaches |
|------|----------------|
| `knowledge_base.py` | 8 documents about trading concepts |
| `chunking.py` | 3 chunking strategies compared |
| `vector_store.py` | numpy cosine similarity search |
| `rag_agent.py` | NaiveRAG + AgenticRAG with tool calling |
| `evaluator.py` | 4 evaluation metrics (LLM-as-judge) |
| `run_demo.py` | End-to-end demo runner |

---

## Chunking Strategies

```
Fixed-Size   │ Split every N chars with overlap
             │ PROS: simple  CONS: breaks mid-sentence
─────────────┼──────────────────────────────────────
Sentence     │ Group N sentences with M sentence overlap
             │ PROS: clean boundaries  CONS: variable chunk size
─────────────┼──────────────────────────────────────
Recursive    │ Try \\n\\n → \\n → ". " → " "  (used in demo)
             │ PROS: preserves structure  CONS: slightly complex
```

## Evaluation Metrics

```
Faithfulness      → supported_claims / total_claims  (hallucination check)
Answer Relevance  → cosine_sim(original_q, synthetic_questions_from_answer)
Context Precision → relevant_chunks / total_retrieved_chunks
Context Recall    → covered_gt_sentences / total_gt_sentences  (needs ground truth)
```

## Test Questions (showcase 3 behaviors)

1. **Simple** — single doc lookup. Both naive and agentic work well.
2. **Multi-hop** — needs RSI + MACD + Momentum + Risk Management. Agentic makes 3-4 targeted searches; naive makes one broad query and misses topics.
3. **No retrieval** — "What is 15 × 8?". Agentic correctly skips retrieval (0 API calls to vector store); naive wastes tokens stuffing trading docs into context.
