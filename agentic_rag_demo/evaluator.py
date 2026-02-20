"""
evaluator.py
------------
Four standard RAG evaluation metrics (inspired by RAGAS framework).

METRIC OVERVIEW:
  ┌──────────────────────┬─────────────────────────────────────────────────────┐
  │ Metric               │ Question it answers                                 │
  ├──────────────────────┼─────────────────────────────────────────────────────┤
  │ Faithfulness         │ Is the answer grounded in the retrieved context?    │
  │ Answer Relevance     │ Does the answer actually address the question?      │
  │ Context Precision    │ Are the retrieved chunks relevant to the question?  │
  │ Context Recall       │ Did retrieval capture what was needed to answer?    │
  └──────────────────────┴─────────────────────────────────────────────────────┘

  Faithfulness + Answer Relevance → quality of the GENERATOR
  Context Precision + Context Recall → quality of the RETRIEVER

Each metric uses an LLM-as-judge approach (gpt-4o-mini) for flexibility.
Answer Relevance also uses embedding similarity.
"""

import json
from typing import List, Optional

import numpy as np
from openai import OpenAI

from embedder import Embedder


JUDGE_MODEL = "gpt-4o-mini"


class RAGEvaluator:

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.client = OpenAI()

    # ── Helpers ──────────────────────────────────────────────────
    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.embedder.embed(text), dtype=np.float32)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _judge(self, prompt: str) -> dict:
        """Call LLM-as-judge with JSON output mode."""
        resp = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    # ── Metric 1: Faithfulness ────────────────────────────────────
    def faithfulness(self, question: str, answer: str, context: str) -> dict:
        """
        Are all claims in the answer supported by the retrieved context?

        Algorithm:
          1. Extract individual factual claims from the answer (LLM)
          2. For each claim, check if the context supports it (LLM)
          3. Score = supported_claims / total_claims

        A low faithfulness score means the LLM hallucinated — it stated
        things not found in the retrieved context.
        """
        prompt = f"""You are evaluating a RAG system's answer for faithfulness.

Question: {question}

Retrieved Context:
{context}

Answer to evaluate:
{answer}

TASK:
1. Extract each distinct factual claim from the Answer.
2. For each claim, determine if it is DIRECTLY SUPPORTED by the Context (not by general knowledge).

Return JSON:
{{
  "claims": [
    {{
      "claim": "exact claim text",
      "supported": true or false,
      "reason": "brief explanation"
    }}
  ]
}}"""

        result = self._judge(prompt)
        claims = result.get("claims", [])

        if not claims:
            return {"score": 1.0, "claims": [], "supported": 0, "total": 0}

        supported = sum(1 for c in claims if c.get("supported", False))
        score = supported / len(claims)

        return {
            "score":     round(score, 3),
            "claims":    claims,
            "supported": supported,
            "total":     len(claims),
        }

    # ── Metric 2: Answer Relevance ────────────────────────────────
    def answer_relevance(self, question: str, answer: str, n_synthetic: int = 3) -> dict:
        """
        Does the answer actually address the question asked?

        Algorithm (RAGAS-style):
          1. Generate N synthetic questions that the answer could be responding to
          2. Embed each synthetic question and the original question
          3. Score = average cosine similarity(original_question, synthetic_questions)

        High score → the answer is on-topic.
        Low score → the answer drifted off-topic (e.g., answered a different question).
        """
        prompt = f"""Given the following answer, generate {n_synthetic} different questions
that this answer could be a response to. Make them varied in phrasing.

Answer: {answer}

Return JSON: {{"questions": ["q1", "q2", ...]}}"""

        result = self._judge(prompt)
        synthetic_questions = result.get("questions", [])

        if not synthetic_questions:
            return {"score": 0.0, "synthetic_questions": [], "similarities": []}

        original_emb = self._embed(question)
        similarities = [
            self._cosine_sim(original_emb, self._embed(q))
            for q in synthetic_questions
        ]

        return {
            "score":               round(float(np.mean(similarities)), 3),
            "synthetic_questions": synthetic_questions,
            "similarities":        [round(s, 3) for s in similarities],
        }

    # ── Metric 3: Context Precision ───────────────────────────────
    def context_precision(self, question: str, retrieved_chunks: List[str]) -> dict:
        """
        What fraction of the retrieved chunks are actually relevant to the question?

        Algorithm:
          For each retrieved chunk, ask LLM: "Is this chunk useful for answering?"
          Score = relevant_chunks / total_retrieved_chunks

        Low precision → retriever is returning junk alongside good results.
        High precision → retriever is focused and accurate.
        """
        if not retrieved_chunks:
            return {"score": 0.0, "relevance": [], "relevant": 0, "total": 0}

        chunk_str = "\n\n".join(
            f"[Chunk {i+1}]:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
        )

        prompt = f"""You are evaluating which retrieved chunks are relevant for answering a question.

Question: {question}

Retrieved Chunks:
{chunk_str}

For each chunk, decide if it is USEFUL for answering the question.

Return JSON:
{{
  "relevance": [
    {{"chunk": 1, "relevant": true or false, "reason": "brief explanation"}}
  ]
}}"""

        result = self._judge(prompt)
        relevance = result.get("relevance", [])

        if not relevance:
            return {"score": 0.0, "relevance": [], "relevant": 0, "total": 0}

        relevant_count = sum(1 for r in relevance if r.get("relevant", False))
        score = relevant_count / len(relevance)

        return {
            "score":     round(score, 3),
            "relevance": relevance,
            "relevant":  relevant_count,
            "total":     len(relevance),
        }

    # ── Metric 4: Context Recall ──────────────────────────────────
    def context_recall(
        self, question: str, ground_truth: str, retrieved_chunks: List[str]
    ) -> dict:
        """
        How much of the ground truth answer is covered by the retrieved context?

        Algorithm:
          1. Break the ground truth into individual sentences/claims
          2. For each sentence, check if ANY retrieved chunk supports it
          3. Score = covered_sentences / total_sentences

        Low recall → retriever missed important documents needed to answer the question.
        High recall → retriever found everything needed.

        Note: requires a ground truth answer — use for offline evaluation.
        """
        if not retrieved_chunks:
            return {"score": 0.0, "sentences": [], "covered": 0, "total": 0}

        context = "\n\n---\n\n".join(retrieved_chunks)

        prompt = f"""You are evaluating how well retrieved context covers a ground truth answer.

Question: {question}

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

TASK: Break the ground truth answer into individual sentences/claims.
For each sentence, determine if it is SUPPORTED by the retrieved context.

Return JSON:
{{
  "sentences": [
    {{
      "sentence": "exact sentence from ground truth",
      "covered":  true or false,
      "reason":   "brief explanation"
    }}
  ]
}}"""

        result = self._judge(prompt)
        sentences = result.get("sentences", [])

        if not sentences:
            return {"score": 0.0, "sentences": [], "covered": 0, "total": 0}

        covered = sum(1 for s in sentences if s.get("covered", False))
        score = covered / len(sentences)

        return {
            "score":     round(score, 3),
            "sentences": sentences,
            "covered":   covered,
            "total":     len(sentences),
        }

    # ── Run all metrics ───────────────────────────────────────────
    def evaluate(
        self,
        result: dict,
        ground_truth: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Run all four metrics on an agent/naive RAG result dict.

        result must have keys: question, answer, retrieval_log (for agentic)
        or retrieved (for naive).
        """
        question = result["question"]
        answer   = result["answer"]

        # Collect all retrieved chunk texts
        if "retrieval_log" in result:
            # Agentic RAG format
            all_chunks = [
                r["text"]
                for log in result["retrieval_log"]
                for r in log["results"]
            ]
        else:
            # Naive RAG format
            all_chunks = [r["text"] for r in result.get("retrieved", [])]

        # Deduplicate while preserving order
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            if c not in seen:
                seen.add(c)
                unique_chunks.append(c)

        context = "\n\n---\n\n".join(unique_chunks)

        scores = {}

        if verbose:
            print("    → Faithfulness... ", end="", flush=True)
        scores["faithfulness"] = self.faithfulness(question, answer, context)
        if verbose:
            print(f"{scores['faithfulness']['score']:.2f}")

        if verbose:
            print("    → Answer Relevance... ", end="", flush=True)
        scores["answer_relevance"] = self.answer_relevance(question, answer)
        if verbose:
            print(f"{scores['answer_relevance']['score']:.2f}")

        if unique_chunks:
            if verbose:
                print("    → Context Precision... ", end="", flush=True)
            scores["context_precision"] = self.context_precision(
                question, unique_chunks[:5]
            )
            if verbose:
                print(f"{scores['context_precision']['score']:.2f}")

        if ground_truth and unique_chunks:
            if verbose:
                print("    → Context Recall... ", end="", flush=True)
            scores["context_recall"] = self.context_recall(
                question, ground_truth, unique_chunks[:5]
            )
            if verbose:
                print(f"{scores['context_recall']['score']:.2f}")

        return scores
