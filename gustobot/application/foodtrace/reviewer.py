from __future__ import annotations

from typing import Literal

from .models import FoodTraceModel, RecallScope
from .repositories import ExposureLookupResult, GraphTraceResult


class ConsistencyReview(FoodTraceModel):
    status: Literal["consistent", "human_review"]
    scope: RecallScope
    reasons: tuple[str, ...] = ()


def review_consistency(
    graph_result: GraphTraceResult,
    exposure_result: ExposureLookupResult,
    *,
    max_batch_scope: int,
) -> ConsistencyReview:
    reasons: list[str] = []
    graph_batch_ids = set(graph_result.production_batch_ids)
    exposure_batch_ids = set(exposure_result.production_batch_ids)
    if graph_batch_ids != exposure_batch_ids:
        reasons.append("graph_sql_batch_scope_mismatch")
    if len(graph_batch_ids) > max_batch_scope:
        reasons.append(f"batch_scope_expanded:{len(graph_batch_ids)}>{max_batch_scope}")

    scope = RecallScope(
        impacted_product_ids=graph_result.product_ids,
        impacted_batch_ids=graph_result.production_batch_ids,
        impacted_store_ids={
            row.store_id
            for row in (*exposure_result.inventories, *exposure_result.orders)
        },
        impacted_order_ids={row.order_id for row in exposure_result.orders},
    )
    return ConsistencyReview(
        status="human_review" if reasons else "consistent",
        scope=scope,
        reasons=tuple(sorted(reasons)),
    )
