from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .graph_retrieval import apply_recipe_id_gate
from .hybrid_retrieval import SearchAdapter, rrf_fuse
from .text2cypher.models import CypherRunStatus, Text2CypherResult


class RetrievalQueryKind(str, Enum):
    HARD_RELATION = "hard_relation"
    EXACT_KEYWORD = "exact_keyword"
    FUZZY_PREFERENCE = "fuzzy_preference"


class RetrievalRoute(str, Enum):
    GRAPH_ONLY = "graph_only"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    kind: RetrievalQueryKind
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=20, ge=1, le=100)


class RetrievalBoundaryError(RuntimeError):
    """Raised when a required graph boundary could not be established safely."""


class RetrievalRouter:
    def __init__(
        self,
        *,
        bm25: SearchAdapter,
        dense: SearchAdapter,
        rrf_k: int = 60,
        per_route_k: int = 30,
    ) -> None:
        if rrf_k < 1 or per_route_k < 1:
            raise ValueError("rrf_k and per_route_k must be positive")
        self._bm25 = bm25
        self._dense = dense
        self._rrf_k = rrf_k
        self._per_route_k = per_route_k

    @staticmethod
    def choose_route(request: RetrievalRequest) -> RetrievalRoute:
        if request.kind is RetrievalQueryKind.HARD_RELATION:
            return RetrievalRoute.GRAPH_ONLY
        if request.kind is RetrievalQueryKind.FUZZY_PREFERENCE:
            return RetrievalRoute.HYBRID
        return RetrievalRoute.BM25

    def search(
        self,
        request: RetrievalRequest,
        *,
        graph_result: Text2CypherResult | None = None,
    ) -> list[str]:
        route = self.choose_route(request)
        allowed_recipe_ids = self._graph_boundary(graph_result)
        if route is RetrievalRoute.GRAPH_ONLY and allowed_recipe_ids is None:
            raise RetrievalBoundaryError(
                "hard relation retrieval requires a Text2Cypher result"
            )
        if allowed_recipe_ids == []:
            return []
        if route is RetrievalRoute.GRAPH_ONLY:
            return apply_recipe_id_gate(
                allowed_recipe_ids or [],
                allowed_recipe_ids,
            )[: request.top_k]

        route_k = max(request.top_k, self._per_route_k)
        bm25_ranking = self._bm25.search(
            request.query,
            request.metadata_filters,
            top_k=route_k,
        )
        bm25_ranking = apply_recipe_id_gate(bm25_ranking, allowed_recipe_ids)
        if route is RetrievalRoute.BM25:
            return bm25_ranking[: request.top_k]

        dense_ranking = self._dense.search(
            request.query,
            request.metadata_filters,
            top_k=route_k,
        )
        dense_ranking = apply_recipe_id_gate(dense_ranking, allowed_recipe_ids)
        fused = rrf_fuse(
            {"bm25": bm25_ranking, "dense": dense_ranking},
            k=self._rrf_k,
        )
        return apply_recipe_id_gate(fused, allowed_recipe_ids)[: request.top_k]

    @staticmethod
    def _graph_boundary(
        result: Text2CypherResult | None,
    ) -> list[str] | None:
        if result is None:
            return None
        if result.status is CypherRunStatus.EMPTY:
            return []
        if result.status is not CypherRunStatus.SUCCESS:
            raise RetrievalBoundaryError(
                f"Text2Cypher boundary unavailable: {result.status.value}"
            )
        return list(result.selected_recipe_ids)

