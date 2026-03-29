"""Reranker implementations: Cohere API, BGE local, ColBERT, FlashRank."""

from __future__ import annotations

from src.retrieval.base import BaseReranker
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)


class CohereReranker(BaseReranker):
    """Cohere Rerank API-based reranker."""

    name = "cohere_rerank"

    def __init__(self, model: str = "rerank-v3.5", top_n: int = 10):
        import cohere
        self.client = cohere.ClientV2()
        self.model = model
        self.top_n = top_n

    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        if not documents:
            return []

        texts = [d.text for d in documents]
        resp = self.client.rerank(
            query=query,
            documents=texts,
            model=self.model,
            top_n=min(top_k, len(documents)),
        )

        reranked = []
        for rank, result in enumerate(resp.results):
            orig = documents[result.index]
            reranked.append(RetrievedDoc(
                doc_id=orig.doc_id,
                text=orig.text,
                score=result.relevance_score,
                rank=rank,
                method=f"{orig.method}+cohere_rerank",
                metadata={**orig.metadata, "rerank_score": result.relevance_score},
            ))
        return reranked


class LocalCrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker (BGE, MiniLM, etc.)."""

    name = "cross_encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.name = model_name.split("/")[-1]

    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        if not documents:
            return []

        pairs = [[query, d.text] for d in documents]
        scores = self.model.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for rank, (doc, score) in enumerate(scored[:top_k]):
            reranked.append(RetrievedDoc(
                doc_id=doc.doc_id,
                text=doc.text,
                score=float(score),
                rank=rank,
                method=f"{doc.method}+{self.name}",
                metadata={**doc.metadata, "rerank_score": float(score)},
            ))
        return reranked


class FlashRankReranker(BaseReranker):
    """Lightweight FlashRank reranker (CPU-friendly)."""

    name = "flashrank"

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker
        self.ranker = Ranker(model_name=model_name)

    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        if not documents:
            return []

        passages = [{"id": d.doc_id, "text": d.text} for d in documents]
        results = self.ranker.rerank(
            query=query, passages=passages, top_k=min(top_k, len(documents))
        )

        doc_map = {d.doc_id: d for d in documents}
        reranked = []
        for rank, r in enumerate(results):
            orig = doc_map[r["id"]]
            reranked.append(RetrievedDoc(
                doc_id=orig.doc_id,
                text=orig.text,
                score=r["score"],
                rank=rank,
                method=f"{orig.method}+flashrank",
                metadata={**orig.metadata, "rerank_score": r["score"]},
            ))
        return reranked


class NoReranker(BaseReranker):
    """Pass-through (no reranking). Used as a baseline."""

    name = "none"

    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        return documents[:top_k]


def create_reranker(config: dict) -> BaseReranker:
    """Factory: create a reranker from config."""
    provider = config.get("provider", "none")
    if provider == "none":
        return NoReranker()
    elif provider == "cohere":
        return CohereReranker(config.get("model", "rerank-v3.5"), config.get("top_n", 10))
    elif provider == "local":
        return LocalCrossEncoderReranker(config.get("model", "BAAI/bge-reranker-v2-m3"))
    elif provider == "flashrank":
        return FlashRankReranker(config.get("model", "ms-marco-MiniLM-L-12-v2"))
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
