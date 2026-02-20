"""
rag_agent.py
------------
Two RAG implementations to contrast:

  1. NaiveRAG   — one-shot: query → retrieve once → answer
  2. AgenticRAG — multi-step: agent decides WHEN and HOW MANY TIMES to retrieve,
                  and CAN filter by document using metadata

LLM BACKEND — LiteLLM
  LiteLLM is a unified gateway: one API, any model.
  Switch models by changing the model string:

    "ollama/llama3.2"          → Ollama running locally (default)
    "ollama/mistral"           → Mistral via Ollama
    "gpt-4o-mini"              → OpenAI (needs OPENAI_API_KEY)
    "claude-haiku-4-5-20251001" → Anthropic (needs ANTHROPIC_API_KEY)

  For Ollama, start the server first:
    ollama serve
    ollama pull llama3.2

  Tool calling (function calling) is required for AgenticRAG.
  Ollama models with tool support: llama3.1, llama3.2, mistral, qwen2.5

METADATA FILTERING
  AgenticRAG now exposes doc_title as an optional filter in the search tool.
  The agent can:
    1. Call list_available_topics() to see all document titles
    2. Call search_knowledge_base(query, doc_title="RSI - Relative Strength Index")
       to restrict search to that one document's chunks — guaranteed precision.
"""

import json
from typing import List, Optional

import litellm

from embedder import Embedder
from vector_store import VectorStore


# ─────────────────────────────────────────────────────────────
# Default model — change this to switch backends
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "ollama/llama3.2"   # swap to "gpt-4o-mini" for Colab / OpenAI


# ─────────────────────────────────────────────────────────────
# 1. Naive RAG — simple baseline
# ─────────────────────────────────────────────────────────────
class NaiveRAG:
    """
    The simplest possible RAG pipeline:
      user query → embed → retrieve k chunks → stuff into prompt → answer

    No loops, no decisions, no reformulation.
    """

    def __init__(self, store: VectorStore, embedder: Embedder, k: int = 4,
                 model: str = DEFAULT_MODEL):
        self.store = store
        self.embedder = embedder
        self.model = model
        self.k = k

    def query(self, question: str) -> dict:
        # Step 1: embed the raw user question
        q_emb = self.embedder.embed(question)

        # Step 2: retrieve top-k chunks (no filter — searches everything)
        results = self.store.search(q_emb, k=self.k)

        # Step 3: build context string
        context = "\n\n---\n\n".join(
            f"[{r.doc_title}]\n{r.text}" for r in results
        )

        # Step 4: one-shot generation via LiteLLM
        response = litellm.completion(
            model=self.model,
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
            "question":     question,
            "answer":       response.choices[0].message.content,
            "n_retrievals": 1,
            "retrieved":    [{"title": r.doc_title, "score": r.score, "text": r.text}
                             for r in results],
            "mode":         "naive",
        }


# ─────────────────────────────────────────────────────────────
# 2. Agentic RAG — multi-step with tool calling + metadata filter
# ─────────────────────────────────────────────────────────────
class AgenticRAG:
    """
    RAG where the LLM is the decision-maker.

    Tools available to the agent:
      search_knowledge_base(query, n_results, doc_title?)
        — semantic search, optionally scoped to one document via metadata filter
      list_available_topics()
        — returns all document titles so the agent can pick the right one

    With metadata filtering, the agent workflow becomes:
      1. list_available_topics()            → sees ["RSI", "MACD", "Momentum", ...]
      2. search_knowledge_base(             → scoped to exactly the RSI document
             query="overbought signals",
             doc_title="RSI - Relative Strength Index")
      3. search_knowledge_base(             → scoped to MACD document
             query="bullish crossover",
             doc_title="MACD Indicator")
      4. Synthesise answer from both filtered results
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": (
                    "Search the trading knowledge base for relevant information. "
                    "Optionally restrict to a specific document with doc_title "
                    "(use list_available_topics first to get exact titles). "
                    "Call multiple times with different queries to cover a complex question."
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
                        "doc_title": {
                            "type": "string",
                            "description": (
                                "Optional. Restrict search to this document title only. "
                                "Must exactly match a title from list_available_topics. "
                                "Use when you know exactly which document contains the answer."
                            ),
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
                    "List all document titles in the knowledge base. "
                    "Call this first to understand what information is available, "
                    "then use doc_title in search_knowledge_base to filter precisely."
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
4. Use list_available_topics() to discover document titles, then use doc_title filter in
   search_knowledge_base() to restrict search to the most relevant document.
5. If your first search doesn't give enough information, REFORMULATE and search again.
6. Synthesize all retrieved information into a coherent, accurate answer.
7. Never make up facts — only use what's in the retrieved context.
"""

    def __init__(self, store: VectorStore, embedder: Embedder,
                 max_iterations: int = 6, model: str = DEFAULT_MODEL):
        self.store = store
        self.embedder = embedder
        self.model = model
        self.max_iterations = max_iterations

    def _do_search(self, query: str, n_results: int = 3,
                   doc_title: Optional[str] = None) -> List[dict]:
        """Embed query, optionally apply doc_title metadata filter, search."""
        q_emb = self.embedder.embed(query)
        filter_dict = {"doc_title": doc_title} if doc_title else None
        results = self.store.search(q_emb, k=n_results, filter=filter_dict)
        return [
            {
                "title": r.doc_title,
                "score": round(r.score, 3),
                "text":  r.text,
            }
            for r in results
        ]

    def _serialize_message(self, msg) -> dict:
        """Convert a LiteLLM/OpenAI message object to a plain dict for the next turn."""
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return entry

    def query(self, question: str, verbose: bool = True) -> dict:
        """
        Run the agentic RAG loop via LiteLLM.

        The loop continues as long as the LLM calls tools.
        When the LLM produces a message WITHOUT tool calls, that is the final answer.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]

        retrieval_log = []

        if verbose:
            print(f"\n  [Agent/{self.model}] '{question}'")

        for iteration in range(self.max_iterations):
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # ── No tool calls → LLM produced the final answer ──────
            if not msg.tool_calls:
                if verbose:
                    print(f"  [Agent] Done — {len(retrieval_log)} retrieval(s).")
                return {
                    "question":      question,
                    "answer":        msg.content,
                    "n_retrievals":  len(retrieval_log),
                    "retrieval_log": retrieval_log,
                    "messages":      messages,
                    "mode":          "agentic",
                }

            # Serialize assistant message as dict before appending
            messages.append(self._serialize_message(msg))

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name == "search_knowledge_base":
                    query_str = func_args["query"]
                    n         = func_args.get("n_results", 3)
                    doc_title = func_args.get("doc_title")    # may be None

                    search_results = self._do_search(query_str, n_results=n,
                                                     doc_title=doc_title)

                    retrieval_log.append({
                        "iteration": iteration + 1,
                        "query":     query_str,
                        "filter":    {"doc_title": doc_title} if doc_title else {},
                        "n_results": n,
                        "results":   search_results,
                    })

                    if verbose:
                        scope = f" [filter: {doc_title}]" if doc_title else ""
                        print(f"  [Search #{len(retrieval_log)}] '{query_str}'{scope}")
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

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      tool_result,
                })

        return {
            "question":      question,
            "answer":        "Max iterations reached without a final answer.",
            "n_retrievals":  len(retrieval_log),
            "retrieval_log": retrieval_log,
            "messages":      messages,
            "mode":          "agentic",
        }
