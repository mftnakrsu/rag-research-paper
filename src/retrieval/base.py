"""Abstract base classes for retrievers and embedders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.utils.common import RetrievedDoc, get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedder interface."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of documents. Returns (N, D) array."""
        ...

    @abstractmethod
    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed a list of queries. Returns (N, D) array."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    name: str = "base"

    @abstractmethod
    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Build the retrieval index from a corpus."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Retrieve top-k documents for a query."""
        ...

    def retrieve_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[RetrievedDoc]]:
        """Retrieve for a batch of queries. Default: sequential."""
        return [self.retrieve(q, top_k) for q in queries]


class BaseReranker(ABC):
    """Abstract reranker interface."""

    name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        """Rerank documents for a query."""
        ...


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API-based embedder."""

    def __init__(self, model: str = "text-embedding-3-large", dimensions: int = 3072):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self._dimension = dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(
                input=batch, model=self.model, dimensions=self._dimension
            )
            all_embeddings.extend([e.embedding for e in resp.data])
        return np.array(all_embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._embed(queries)


class CohereEmbedder(BaseEmbedder):
    """Cohere API-based embedder with query/document input types."""

    def __init__(self, model: str = "embed-v4.0", dimensions: int = 1024):
        import cohere
        self.client = cohere.ClientV2()
        self.model = model
        self._dimension = dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(
        self, texts: list[str], input_type: str, batch_size: int = 96
    ) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
                embedding_types=["float"],
            )
            all_embeddings.extend(resp.embeddings.float_)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, input_type="search_document")

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._embed(queries, input_type="search_query")


class VoyageEmbedder(BaseEmbedder):
    """Voyage AI API-based embedder."""

    def __init__(self, model: str = "voyage-3-large", dimensions: int = 1024):
        import voyageai
        self.client = voyageai.Client()
        self.model = model
        self._dimension = dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(
        self, texts: list[str], input_type: str, batch_size: int = 128
    ) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embed(batch, model=self.model, input_type=input_type)
            all_embeddings.extend(resp.embeddings)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, input_type="document")

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._embed(queries, input_type="query")


class LocalEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder (BGE-M3, E5, etc.)."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode(queries, show_progress_bar=True, normalize_embeddings=True)


def create_embedder(config: dict) -> BaseEmbedder:
    """Factory: create an embedder from config."""
    provider = config["provider"]
    if provider == "openai":
        return OpenAIEmbedder(config["model"], config.get("dimensions", 3072))
    elif provider == "cohere":
        return CohereEmbedder(config["model"], config.get("dimensions", 1024))
    elif provider == "voyage":
        return VoyageEmbedder(config["model"], config.get("dimensions", 1024))
    elif provider == "local":
        return LocalEmbedder(config["model"])
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
