"""BM25 sparse retriever using rank_bm25."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters.

    Provides a simple, dependency-free tokeniser that is adequate for
    BM25 scoring.  For heavier corpora, swap in an NLTK / spaCy
    pipeline instead.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    """Okapi BM25 sparse retriever.

    Parameters
    ----------
    k1 : float
        Term-frequency saturation parameter (default 1.5).
    b : float
        Length-normalisation parameter (default 0.75).
    tokenizer : callable, optional
        Custom tokenisation function ``str -> list[str]``.
        Falls back to the built-in regex tokeniser.
    """

    name: str = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Any | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._tokenize = tokenizer or _tokenize

        # Populated by build_index
        self._index: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._documents: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Tokenize the corpus and build a BM25Okapi index.

        Parameters
        ----------
        doc_ids : list[str]
            Unique identifier for each document.
        documents : list[str]
            Raw document texts, same order as *doc_ids*.
        """
        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and documents ({len(documents)}) "
                "must have the same length."
            )

        logger.info("Building BM25 index over %d documents (k1=%.2f, b=%.2f) ...",
                     len(documents), self.k1, self.b)

        with Timer() as t:
            self._doc_ids = list(doc_ids)
            self._documents = list(documents)
            self._tokenized_corpus = [self._tokenize(doc) for doc in documents]
            self._index = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )

        logger.info("BM25 index built in %.2f s (%d documents).",
                     t.elapsed, len(documents))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Score every document against *query* and return the top-k results.

        Parameters
        ----------
        query : str
            Natural-language query string.
        top_k : int
            Number of documents to return.

        Returns
        -------
        list[RetrievedDoc]
            Results sorted by descending BM25 score.
        """
        self._ensure_index()

        tokenized_query = self._tokenize(query)
        scores: np.ndarray = self._index.get_scores(tokenized_query)

        # Partial sort for efficiency: grab the top-k indices.
        k = min(top_k, len(self._doc_ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort those k indices by descending score.
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[RetrievedDoc] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                RetrievedDoc(
                    doc_id=self._doc_ids[idx],
                    text=self._documents[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    method=self.name,
                )
            )
        return results

    def retrieve_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[RetrievedDoc]]:
        """Retrieve for a batch of queries sequentially.

        BM25 scoring is CPU-bound per query so there is limited benefit
        from batching, but the method is provided for interface
        consistency.

        Parameters
        ----------
        queries : list[str]
            Batch of query strings.
        top_k : int
            Number of results per query.

        Returns
        -------
        list[list[RetrievedDoc]]
            One result list per query.
        """
        self._ensure_index()
        logger.info("BM25 batch retrieval: %d queries, top_k=%d", len(queries), top_k)

        with Timer() as t:
            results = [self.retrieve(q, top_k) for q in queries]

        logger.info("BM25 batch retrieval finished in %.2f s (avg %.1f ms/query).",
                     t.elapsed, t.elapsed_ms / max(len(queries), 1))
        return results
