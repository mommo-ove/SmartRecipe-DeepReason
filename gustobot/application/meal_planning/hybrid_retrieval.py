from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from .external_corpus import ExternalRecipeDocument
from .retrieval_baseline import ExternalBM25Retriever


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SearchAdapter(Protocol):
    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        *,
        top_k: int = 20,
    ) -> list[str]: ...


class PairReranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("encoder must return a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class DenseExternalRetriever:
    def __init__(
        self,
        documents: list[ExternalRecipeDocument],
        encoder: TextEncoder,
    ) -> None:
        self._documents = list(documents)
        self._encoder = encoder
        self._document_vectors = _normalize_rows(
            encoder.encode([document.search_text() for document in documents])
        )

    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        *,
        top_k: int = 20,
    ) -> list[str]:
        filters = metadata_filters or {}
        positions = [
            index
            for index, document in enumerate(self._documents)
            if _matches(document, filters)
        ]
        if not positions:
            return []
        query_vector = _normalize_rows(self._encoder.encode([query]))[0]
        scores = self._document_vectors[positions] @ query_vector
        ranked_positions = sorted(
            zip(positions, scores, strict=True),
            key=lambda item: (-float(item[1]), self._documents[item[0]].recipe_id),
        )
        return [
            self._documents[index].recipe_id
            for index, _ in ranked_positions[:top_k]
        ]


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
) -> list[str]:
    if k < 1:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings.values():
        seen: set[str] = set()
        for rank, recipe_id in enumerate(ranking, start=1):
            if recipe_id in seen:
                continue
            seen.add(recipe_id)
            scores[recipe_id] = scores.get(recipe_id, 0.0) + 1.0 / (k + rank)
            best_rank[recipe_id] = min(rank, best_rank.get(recipe_id, rank))
    return sorted(
        scores,
        key=lambda recipe_id: (
            -scores[recipe_id],
            best_rank[recipe_id],
            recipe_id,
        ),
    )


class RRFExternalRetriever:
    def __init__(
        self,
        retrievers: Mapping[str, SearchAdapter],
        *,
        rrf_k: int = 60,
        per_route_k: int = 30,
    ) -> None:
        self._retrievers = dict(retrievers)
        self._rrf_k = rrf_k
        self._per_route_k = per_route_k

    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        *,
        top_k: int = 20,
    ) -> list[str]:
        rankings = {
            name: retriever.search(
                query,
                metadata_filters,
                top_k=self._per_route_k,
            )
            for name, retriever in self._retrievers.items()
        }
        return rrf_fuse(rankings, k=self._rrf_k)[:top_k]


class RerankedExternalRetriever:
    def __init__(
        self,
        documents: list[ExternalRecipeDocument],
        candidate_retriever: SearchAdapter,
        reranker: PairReranker,
        *,
        candidate_k: int = 30,
    ) -> None:
        self._documents = {document.recipe_id: document for document in documents}
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        *,
        top_k: int = 20,
    ) -> list[str]:
        candidate_ids = self._candidate_retriever.search(
            query,
            metadata_filters,
            top_k=self._candidate_k,
        )
        candidate_ids = [
            recipe_id for recipe_id in candidate_ids if recipe_id in self._documents
        ]
        texts = [self._documents[recipe_id].search_text() for recipe_id in candidate_ids]
        scores = self._reranker.score(query, texts)
        if len(scores) != len(candidate_ids):
            raise ValueError("reranker score count does not match candidates")
        ranked = sorted(
            zip(candidate_ids, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [recipe_id for recipe_id, _ in ranked[:top_k]]


class BgeM3Encoder:
    def __init__(
        self,
        *,
        model: Any | None = None,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Install the retrieval-models optional dependencies to run BGE-M3"
                ) from error
            model = SentenceTransformer(model_name, device=device)
        self._model = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


class BgeReranker:
    def __init__(
        self,
        *,
        model: Any | None = None,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ) -> None:
        if model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as error:
                raise RuntimeError(
                    "Install the retrieval-models optional dependencies to run the BGE reranker"
                ) from error
            model = FlagReranker(model_name, use_fp16=use_fp16)
        self._model = model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        raw = self._model.compute_score(
            [[query, document] for document in documents],
            normalize=True,
        )
        if isinstance(raw, (int, float)):
            return [float(raw)]
        return [float(score) for score in raw]


def build_external_retriever(
    configuration: str,
    documents: list[ExternalRecipeDocument],
    *,
    encoder: TextEncoder | None = None,
    reranker: PairReranker | None = None,
    device: str | None = None,
) -> SearchAdapter:
    """Build one of the four retrievers used by the ablation benchmark."""

    allowed = {"bm25", "dense", "rrf", "rrf_rerank"}
    if configuration not in allowed:
        raise ValueError(f"unknown retrieval configuration: {configuration}")
    if configuration == "bm25":
        return ExternalBM25Retriever(documents)

    dense = DenseExternalRetriever(
        documents,
        encoder or BgeM3Encoder(device=device),
    )
    if configuration == "dense":
        return dense

    fused = RRFExternalRetriever(
        {
            "bm25": ExternalBM25Retriever(documents),
            "dense": dense,
        }
    )
    if configuration == "rrf":
        return fused
    if configuration == "rrf_rerank":
        return RerankedExternalRetriever(
            documents,
            fused,
            reranker or BgeReranker(),
        )
    raise AssertionError("validated retrieval configuration was not handled")


def _matches(
    document: ExternalRecipeDocument,
    filters: dict[str, Any],
) -> bool:
    meal_type = filters.get("meal_type")
    if meal_type is not None and str(meal_type) not in document.meal_types:
        return False
    max_minutes = filters.get("max_meal_minutes")
    if max_minutes is not None and document.total_minutes > int(max_minutes):
        return False
    min_protein = filters.get("min_protein_g")
    if min_protein is not None and document.protein_g < float(min_protein):
        return False
    return True
