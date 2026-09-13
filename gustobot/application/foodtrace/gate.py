from __future__ import annotations

from typing import Literal

from pydantic import Field

from .evidence import EvidenceItem
from .models import FoodTraceModel


GateState = Literal["ALLOW_REPORT", "HUMAN_REVIEW", "BLOCKED"]
REQUIRED_REPORT_EVIDENCE = frozenset(
    {"graph_scope", "order_exposure", "consistency_review"}
)
CLAIM_EVIDENCE_KINDS = {
    "impacted_product_count": "graph_scope",
    "impacted_batch_count": "graph_scope",
    "impacted_store_count": "order_exposure",
    "impacted_order_count": "order_exposure",
}


class ImpactClaim(FoodTraceModel):
    metric: str = Field(min_length=1)
    value: int = Field(ge=0)
    evidence_ids: tuple[str, ...] = ()


class GateInput(FoodTraceModel):
    evidence: tuple[EvidenceItem, ...]
    claims: tuple[ImpactClaim, ...]
    dependency_failures: tuple[str, ...] = ()
    safety_reasons: tuple[str, ...] = ()


class GateDecision(FoodTraceModel):
    state: GateState
    reasons: tuple[str, ...] = ()


def evaluate_gate(gate_input: GateInput) -> GateDecision:
    if gate_input.dependency_failures:
        return GateDecision(
            state="BLOCKED",
            reasons=tuple(sorted(set(gate_input.dependency_failures))),
        )

    reasons = set(gate_input.safety_reasons)
    evidence_kinds = {item.kind for item in gate_input.evidence}
    for missing_kind in sorted(REQUIRED_REPORT_EVIDENCE - evidence_kinds):
        reasons.add(f"missing_evidence:{missing_kind}")

    evidence_by_id = {item.evidence_id: item for item in gate_input.evidence}
    for claim in gate_input.claims:
        if not claim_is_grounded(claim, tuple(evidence_by_id.values())):
            reasons.add(f"ungrounded_claim:{claim.metric}")

    if reasons:
        return GateDecision(state="HUMAN_REVIEW", reasons=tuple(sorted(reasons)))
    return GateDecision(state="ALLOW_REPORT")


def claim_is_grounded(
    claim: ImpactClaim,
    evidence: tuple[EvidenceItem, ...] | list[EvidenceItem],
) -> bool:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    cited_evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in claim.evidence_ids
        if evidence_id in evidence_by_id
    ]
    required_kind = CLAIM_EVIDENCE_KINDS.get(claim.metric)
    return bool(
        claim.evidence_ids
        and len(cited_evidence) == len(set(claim.evidence_ids))
        and required_kind is not None
        and any(
            item.kind == required_kind
            and _supported_claim_value(item, claim.metric) == claim.value
            for item in cited_evidence
        )
    )


def _supported_claim_value(item: EvidenceItem, metric: str) -> int | None:
    try:
        payload = item.payload()
    except (TypeError, ValueError):
        return None

    if metric == "impacted_product_count":
        return _collection_size(payload, "product_ids", "impacted_product_ids")
    if metric == "impacted_batch_count":
        return _collection_size(
            payload,
            "production_batch_ids",
            "batch_ids",
            "impacted_batch_ids",
        )
    if metric == "impacted_order_count":
        return _collection_size(payload, "orders", "order_ids", "impacted_order_ids")
    if metric == "impacted_store_count":
        explicit_count = _collection_size(
            payload,
            "store_ids",
            "impacted_store_ids",
        )
        if explicit_count is not None:
            return explicit_count
        stores = {
            row.get("store_id")
            for key in ("inventories", "orders")
            for row in payload.get(key, ())
            if isinstance(row, dict) and isinstance(row.get("store_id"), str)
        }
        return len(stores)
    return None


def _collection_size(payload: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            return len(set(map(str, value)))
    return None
