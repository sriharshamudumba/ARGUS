"""
app/reranker.py

Cohere reranking layer for ARGUS two-tier retrieval.

The existing pipeline does:
    Query -> Cohere embed -> FAISS similarity search -> top-K docs

This adds a second pass:
    ... -> FAISS top-K docs -> Cohere rerank -> re-sorted top-K docs


"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional

import cohere
from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    document:         Document
    faiss_rank:       int    # original FAISS rank (0 = most similar)
    rerank_score:     float  # Cohere relevance score
    rerank_rank:      int    # rank after reranking (0 = most relevant)
    faiss_similarity: float  # original cosine similarity score


class CohereReranker:
    """
    Wraps Cohere's rerank API to re-score FAISS candidates.

    Usage:
        reranker = CohereReranker(top_n=5, faiss_fetch_k=20)
        results = reranker.rerank(query, faiss_docs, faiss_scores)
    """

    def __init__(
        self,
        top_n:        int    = 5,
        faiss_fetch_k: int   = 20,
        model:        str    = "rerank-english-v3.0",
        api_key:      Optional[str] = None,
    ):
        self.top_n         = top_n
        self.faiss_fetch_k = faiss_fetch_k
        self.model         = model

        api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not set. "
                "Export it or pass api_key= to CohereReranker()."
            )
        self.client = cohere.Client(api_key)

        # Latency tracking
        self._rerank_times_ms: list[float] = []

    def rerank(
        self,
        query:        str,
        faiss_docs:   List[Document],
        faiss_scores: Optional[List[float]] = None,
    ) -> List[RankedResult]:
        """
        Re-score FAISS candidates with Cohere cross-encoder.

        Args:
            query:        User query string.
            faiss_docs:   Documents returned by FAISS (ordered by similarity).
            faiss_scores: Cosine similarity scores from FAISS (optional).

        Returns:
            List of RankedResult sorted by rerank_score descending,
            truncated to top_n.
        """
        if not faiss_docs:
            return []

        if faiss_scores is None:
            faiss_scores = [0.0] * len(faiss_docs)

        # Cohere rerank expects plain strings
        doc_texts = [doc.page_content for doc in faiss_docs]

        t0 = time.monotonic()
        try:
            response = self.client.rerank(
                query     = query,
                documents = doc_texts,
                model     = self.model,
                top_n     = min(self.top_n, len(faiss_docs)),
            )
        except Exception as e:
            logger.error(f"Cohere rerank API error: {e}. Falling back to FAISS order.")
            # Graceful fallback: return FAISS results as-is
            return [
                RankedResult(
                    document         = doc,
                    faiss_rank       = i,
                    rerank_score     = score,
                    rerank_rank      = i,
                    faiss_similarity = score,
                )
                for i, (doc, score) in enumerate(zip(faiss_docs[:self.top_n], faiss_scores))
            ]

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._rerank_times_ms.append(elapsed_ms)
        logger.debug(f"Cohere rerank: {len(faiss_docs)} docs -> {self.top_n} in {elapsed_ms:.0f}ms")

        # Build ranked results
        ranked = []
        for rerank_rank, result in enumerate(response.results):
            orig_idx = result.index
            ranked.append(RankedResult(
                document         = faiss_docs[orig_idx],
                faiss_rank       = orig_idx,
                rerank_score     = result.relevance_score,
                rerank_rank      = rerank_rank,
                faiss_similarity = faiss_scores[orig_idx] if faiss_scores else 0.0,
            ))

        return ranked

    def avg_latency_ms(self) -> float:
        if not self._rerank_times_ms:
            return 0.0
        return sum(self._rerank_times_ms) / len(self._rerank_times_ms)

    def p95_latency_ms(self) -> float:
        if not self._rerank_times_ms:
            return 0.0
        return sorted(self._rerank_times_ms)[int(0.95 * len(self._rerank_times_ms))]
