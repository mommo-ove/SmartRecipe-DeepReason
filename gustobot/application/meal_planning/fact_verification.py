from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from gustobot.application.deepreason.models import EvidenceItem


class FactOperator(str, Enum):
    EQUALS = "equals"
    EXISTS = "exists"
    IN_SELECTED = "in_selected"
    EXCLUDES = "excludes"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    NOT_VERIFIABLE = "not_verifiable"


class FactClaim(BaseModel):
    claim_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    field: str | None = None
    operator: FactOperator
    expected_value: Any = None
    evidence_ids: list[str] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    claim_id: str
    status: ClaimStatus
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str


class FactVerificationResult(BaseModel):
    all_supported: bool
    checks: list[ClaimCheck]


def verify_fact_claims(
    claims: list[FactClaim],
    evidence: list[EvidenceItem],
    *,
    selected_recipe_ids: list[str] | None = None,
) -> FactVerificationResult:
    ledger = {item.evidence_id: item for item in evidence}
    selected = set(selected_recipe_ids or [])
    checks = [_verify_claim(claim, ledger, selected) for claim in claims]
    return FactVerificationResult(
        all_supported=all(check.status is ClaimStatus.SUPPORTED for check in checks),
        checks=checks,
    )


def _verify_claim(
    claim: FactClaim,
    ledger: dict[str, EvidenceItem],
    selected_recipe_ids: set[str],
) -> ClaimCheck:
    if not claim.evidence_ids:
        return _check(claim, ClaimStatus.UNSUPPORTED, "claim has no evidence link")
    if any(evidence_id not in ledger for evidence_id in claim.evidence_ids):
        return _check(
            claim,
            ClaimStatus.NOT_VERIFIABLE,
            "one or more referenced evidence items do not exist",
        )

    referenced = [ledger[evidence_id] for evidence_id in claim.evidence_ids]
    entity_facts = [
        item
        for item in referenced
        if str(item.metadata.get("entity_id")) == claim.entity_id
    ]
    if not entity_facts:
        return _check(
            claim,
            ClaimStatus.NOT_VERIFIABLE,
            "referenced evidence belongs to another entity",
        )

    if claim.operator is FactOperator.IN_SELECTED:
        status = (
            ClaimStatus.SUPPORTED
            if claim.entity_id in selected_recipe_ids
            else ClaimStatus.CONTRADICTED
        )
        return _check(claim, status, "checked against selected recipe IDs")

    if claim.operator is FactOperator.EXISTS:
        return _check(claim, ClaimStatus.SUPPORTED, "entity evidence exists")

    field_facts = [
        item for item in entity_facts if item.metadata.get("field") == claim.field
    ]
    if not field_facts:
        return _check(
            claim,
            ClaimStatus.NOT_VERIFIABLE,
            "referenced evidence does not contain the claimed field",
        )

    if claim.operator is FactOperator.EQUALS:
        supported = any(
            _values_equal(item.metadata.get("value"), claim.expected_value)
            for item in field_facts
        )
        return _check(
            claim,
            ClaimStatus.SUPPORTED if supported else ClaimStatus.CONTRADICTED,
            "compared claimed value with atomic evidence",
        )

    excluded = _normalized_set(claim.expected_value)
    actual = set().union(
        *(_normalized_set(item.metadata.get("value")) for item in field_facts)
    )
    if actual & excluded:
        return _check(
            claim,
            ClaimStatus.CONTRADICTED,
            "ingredient evidence contains an excluded value",
        )
    complete_facts = [
        item for item in field_facts if item.metadata.get("complete") is True
    ]
    if not complete_facts:
        return _check(
            claim,
            ClaimStatus.NOT_VERIFIABLE,
            "absence requires a complete ingredient set",
        )
    return _check(
        claim,
        ClaimStatus.SUPPORTED,
        "checked ingredient evidence against excluded values",
    )


def _check(
    claim: FactClaim,
    status: ClaimStatus,
    reason: str,
) -> ClaimCheck:
    return ClaimCheck(
        claim_id=claim.claim_id,
        status=status,
        evidence_ids=claim.evidence_ids,
        reason=reason,
    )


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    try:
        return Decimal(str(actual)) == Decimal(str(expected))
    except (InvalidOperation, TypeError, ValueError):
        return actual == expected


def _normalized_set(value: Any) -> set[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {
        str(item).strip().casefold()
        for item in values
        if item is not None and str(item).strip()
    }
