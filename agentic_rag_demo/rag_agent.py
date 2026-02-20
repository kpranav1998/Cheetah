"""
rag_agent.py
------------
Two RAG implementations to contrast:

  1. NaiveRAG   — one-shot: query → retrieve once → answer
  2. AgenticRAG — multi-step: agent decides WHEN and HOW MANY TIMES to retrieve

WHY AGENTIC RAG?
  Naive RAG fails on:
  - Multi-hop questions  (need to retrieve from 2+ topics)
  - Complex questions    (need to decompose into sub-queries)
  - No-retrieval cases   (LLM knows the answer; retrieval adds noise)
  - Bad initial queries  (agent can reformulate if first search misses)

  Agentic RAG uses OpenAI's tool calling so the LLM decides:
    • Whether to search at all
    • What query to use
    • Whether the results are sufficient or to search again with a different angle
    • How to synthesize multiple pieces of retrieved context
"""

import json
import os
from typing import List

from openai import OpenAI

from embedder import Embedder
from vector_store import VectorStore


CHAT_MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────
# 1. Naive RAG — simple baseline
# ─────────────────────────────────────────────────────────────
class NaiveRAG:
    """
    The simplest possible RAG pipeline:
      user query → embed → retrieve k chunks → stuff into prompt → answer

    No loops, no decisions, no reformulation.
    """

    def __init__(self, store: VectorStore, embedder: Embedder, k: int = 4):
        self.store = store
        self.embedder = embedder
        self.client = OpenAI()
        self.k = k

    def query(self, question: str) -> dict:
        # Step 1: embed the raw user question
        q_emb = self.embedder.embed(question)

        # Step 2: retrieve top-k chunks
        results = self.store.search(q_emb, k=self.k)

        # Step 3: build context string
        context = "\n\n---\n\n".join(
            f"[{r.doc_title}]\n{r.text}" for r in results
        )

        # Step 4: one-shot generation
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a trading assistant. Answer the user's question "
                        "using ONLY the provided context. If the context does not "
                        "contain enough information, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )

        return {
            "question":    question,
            "answer":      response.choices[0].message.content,
            "n_retrievals": 1,
            "retrieved":   [{"title": r.doc_title, "score": r.score, "text": r.text} for r in results],
            "mode":        "naive",
        }


# ─────────────────────────────────────────────────────────────
# 2. Agentic RAG — multi-step with tool calling
# ─────────────────────────────────────────────────────────────
class AgenticRAG:
    """
    RAG where the LLM is the decision-maker.

    The LLM is given two tools:
      search_knowledge_base(query, n_results)  — semantic search
      list_available_topics()                  — see what docs exist

    The agent can:
      • Call search multiple times with different queries
      • List topics first to understand what's available
      • Choose NOT to search (for general-knowledge questions)
      • Reformulate a query if the first results are poor
      • Decompose a multi-part question into separate searches
    """

    # Tools exposed to the LLM via OpenAI function calling
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": (
                    "Search the trading knowledge base for relevant information. "
                    "Call this multiple times with DIFFERENT, SPECIFIC queries "
                    "to cover all aspects of a complex question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Focused search query. Be specific.",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of chunks to retrieve (1-5). Default 3.",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_available_topics",
                "description": (
                    "List all document topics available in the knowledge base. "
                    "Use this first if you are unsure what information exists."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    SYSTEM_PROMPT = """You are an expert trading assistant with access to a structured knowledge base.

Your goal is to answer the user's question accurately using the knowledge base.

Guidelines:
1. THINK before searching — is this a general knowledge question (e.g. math) that doesn't need retrieval?
2. For trading-specific questions, ALWAYS search the knowledge base.
3. For COMPLEX questions with multiple parts, search MULTIPLE TIMES with different focused queries.
4. If your first search doesn't give enough information, REFORMULATE and search again.
5. Synthesize all retrieved information into a coherent, accurate answer.
6. Never make up facts — only use what's in the retrieved context.
"""

    def __init__(self, store: VectorStore, embedder: Embedder, max_iterations: int = 6):
        self.store = store
        self.embedder = embedder
        self.client = OpenAI()
        self.max_iterations = max_iterations

    def _do_search(self, query: str, n_results: int = 3) -> List[dict]:
        """Internal: embed query and search the store."""
        q_emb = self.embedder.embed(query)
        results = self.store.search(q_emb, k=n_results)
        return [
            {
                "title": r.doc_title,
                "score": round(r.score, 3),
                "text":  r.text,
            }
            for r in results
        ]

    def query(self, question: str, verbose: bool = True) -> dict:
        """
        Run the agentic RAG loop.

        The loop continues as long as the LLM calls tools.
        When the LLM produces a message WITHOUT tool calls, that is the final answer.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]

        retrieval_log = []   # track all searches made by the agent

        if verbose:
            print(f"\n  [Agent] Starting query: '{question}'")

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",  # LLM decides: call a tool OR answer directly
            )

            msg = response.choices[0].message

            # ── No tool calls → LLM produced the final answer ──────────────
            if not msg.tool_calls:
                if verbose:
                    print(f"  [Agent] Done after {len(retrieval_log)} retrieval(s). Generating answer.")
                return {
                    "question":     question,
                    "answer":       msg.content,
                    "n_retrievals": len(retrieval_log),
                    "retrieval_log": retrieval_log,
                    "messages":     messages,
                    "mode":         "agentic",
                }

            # ── Handle tool calls ───────────────────────────────────────────
            messages.append(msg)  # important: add assistant message before tool results

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name == "search_knowledge_base":
                    query_str = func_args["query"]
                    n = func_args.get("n_results", 3)
                    search_results = self._do_search(query_str, n_results=n)

                    retrieval_log.append({
                        "iteration": iteration + 1,
                        "query":     query_str,
                        "n_results": n,
                        "results":   search_results,
                    })

                    if verbose:
                        print(f"  [Search #{len(retrieval_log)}] '{query_str}' → {len(search_results)} results")
                        for r in search_results:
                            preview = r["text"][:70].replace("\n", " ")
                            print(f"      [{r['score']:.3f}] {r['title']}: {preview}...")

                    tool_result = json.dumps(search_results)

                elif func_name == "list_available_topics":
                    topics = self.store.get_topics()
                    tool_result = json.dumps(topics)
                    if verbose:
                        print(f"  [Topics] {topics}")

                else:
                    tool_result = json.dumps({"error": f"Unknown tool: {func_name}"})

                # Feed the tool result back into the conversation
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      tool_result,
                })

        # Fallback — shouldn't normally reach here
        return {
            "question":     question,
            "answer":       "Max iterations reached without a final answer.",
            "n_retrievals": len(retrieval_log),
            "retrieval_log": retrieval_log,
            "messages":     messages,
            "mode":         "agentic",
        }
