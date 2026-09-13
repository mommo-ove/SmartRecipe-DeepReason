from __future__ import annotations

from collections import Counter
from enum import Enum
from math import ceil
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from gustobot.application.deepreason.models import EvidenceItem

from .capabilities import assess_request_capabilities
from .models import MealPlanConstraints, MealSlot, RecipeCandidate
from .text2cypher.models import CypherRunStatus, Text2CypherResult


class CandidateBoundaryStatus(str, Enum):
    VERIFIED = "verified"
    EMPTY = "empty"
    FAILED = "failed"


class CandidateBoundary(BaseModel):
    status: CandidateBoundaryStatus
    selected_recipe_ids: list[str] = Field(default_factory=list)
    evidence_ids_by_recipe: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("selected_recipe_ids")
    @classmethod
    def normalize_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

    @model_validator(mode="after")
    def validate_evidence_coverage(self) -> "CandidateBoundary":
        if self.status is CandidateBoundaryStatus.EMPTY and self.selected_recipe_ids:
            raise ValueError("empty boundary cannot contain recipe IDs")
        if self.status is CandidateBoundaryStatus.VERIFIED:
            missing = [
                recipe_id
                for recipe_id in self.selected_recipe_ids
                if not self.evidence_ids_by_recipe.get(recipe_id)
            ]
            if missing:
                raise ValueError(
                    "verified boundary requires evidence for every recipe ID: "
                    + ", ".join(missing)
                )
        return self


def build_candidate_boundary(
    result: Text2CypherResult,
    evidence: list[EvidenceItem],
) -> CandidateBoundary:
    if result.status is CypherRunStatus.EMPTY:
        return CandidateBoundary(status=CandidateBoundaryStatus.EMPTY)
    if result.status is not CypherRunStatus.SUCCESS:
        return CandidateBoundary(status=CandidateBoundaryStatus.FAILED)

    selected = set(result.selected_recipe_ids)
    evidence_by_recipe: dict[str, list[str]] = {
        recipe_id: [] for recipe_id in result.selected_recipe_ids
    }
    for item in evidence:
        entity_id = str(item.metadata.get("entity_id", ""))
        if entity_id in selected:
            evidence_by_recipe[entity_id].append(item.evidence_id)
    return CandidateBoundary(
        status=CandidateBoundaryStatus.VERIFIED,
        selected_recipe_ids=result.selected_recipe_ids,
        evidence_ids_by_recipe=evidence_by_recipe,
    )


def apply_candidate_boundary(
    recipes: list[RecipeCandidate],
    boundary: CandidateBoundary | None,
) -> list[RecipeCandidate]:
    if boundary is None:
        allowed = None
    elif boundary.status is CandidateBoundaryStatus.EMPTY:
        return []
    elif boundary.status is not CandidateBoundaryStatus.VERIFIED:
        raise ValueError("candidate boundary must be verified before retrieval")
    else:
        allowed = set(boundary.selected_recipe_ids)

    seen: set[str] = set()
    filtered: list[RecipeCandidate] = []
    for recipe in recipes:
        if recipe.recipe_id in seen or allowed is not None and recipe.recipe_id not in allowed:
            continue
        seen.add(recipe.recipe_id)
        filtered.append(recipe)
    return filtered


class SlotShortage(BaseModel):
    slot: MealSlot
    available: int = Field(ge=0)
    required: int = Field(ge=1)


class CandidateBackfillRequest(BaseModel):
    slot: MealSlot
    needed_candidates: int = Field(ge=1)
    semantic_query: str
    metadata_filters: dict[str, Any]


class SlotRetrievalRequest(BaseModel):
    slot: MealSlot
    minimum_distinct_candidates: int = Field(ge=1)
    top_k: int = Field(ge=1)
    semantic_query: str
    metadata_filters: dict[str, Any]


class CandidatePoolReport(BaseModel):
    ready: bool
    required_distinct_per_slot: int = Field(ge=1)
    eligible_recipe_ids: list[str]
    rejected_counts: dict[str, int]
    shortages: list[SlotShortage]
    backfill_requests: list[CandidateBackfillRequest]


def build_slot_retrieval_requests(
    constraints: MealPlanConstraints,
    *,
    semantic_query: str,
    safety_buffer: int = 2,
) -> list[SlotRetrievalRequest]:
    if safety_buffer < 0:
        raise ValueError("safety_buffer cannot be negative")
    minimum = ceil(constraints.days / constraints.max_recipe_repeats)
    return [
        SlotRetrievalRequest(
            slot=slot,
            minimum_distinct_candidates=minimum,
            top_k=minimum + safety_buffer,
            semantic_query=" ".join(
                part for part in (semantic_query.strip(), slot.value) if part
            ),
            metadata_filters={
                "planning_eligible": True,
                "meal_type": slot.value,
                "excluded_allergens": sorted(constraints.excluded_allergens),
                "max_meal_minutes": constraints.max_meal_minutes,
            },
        )
        for slot in MealSlot
    ]


def build_backfill_retrieval_requests(
    report: CandidatePoolReport,
    *,
    safety_buffer: int = 2,
) -> list[SlotRetrievalRequest]:
    if safety_buffer < 0:
        raise ValueError("safety_buffer cannot be negative")
    return [
        SlotRetrievalRequest(
            slot=request.slot,
            minimum_distinct_candidates=report.required_distinct_per_slot,
            top_k=request.needed_candidates + safety_buffer,
            semantic_query=request.semantic_query,
            metadata_filters=request.metadata_filters,
        )
        for request in report.backfill_requests
    ]


def assess_candidate_pool(
    constraints: MealPlanConstraints,
    recipes: list[RecipeCandidate],
    *,
    semantic_query: str,
) -> CandidatePoolReport:
    """Check necessary candidate coverage before invoking the solver.

    This gate cannot prove that the full plan is feasible. It prevents an
    obviously incomplete retrieval result from being sent to CP-SAT and emits
    targeted requests that a retrieval adapter can use to backfill one slot.
    """

    rejected: Counter[str] = Counter()
    eligible: list[RecipeCandidate] = []
    capability_report = assess_request_capabilities(constraints, recipes)
    capability_ready = set(capability_report.ready_recipe_ids)
    for recipe in recipes:
        if recipe.recipe_id not in capability_ready:
            for capability in capability_report.rejected[recipe.recipe_id]:
                rejected[f"missing_{capability.value}_capability"] += 1
            continue
        if recipe.allergens & constraints.excluded_allergens:
            rejected["allergen_conflict"] += 1
            continue
        if (
            constraints.max_meal_minutes is not None
            and recipe.total_minutes is not None
            and recipe.total_minutes > constraints.max_meal_minutes
        ):
            rejected["too_slow"] += 1
            continue
        eligible.append(recipe)

    required = ceil(constraints.days / constraints.max_recipe_repeats)
    counts = {
        slot: sum(slot in recipe.meal_types for recipe in eligible)
        for slot in MealSlot
    }
    shortages = [
        SlotShortage(slot=slot, available=counts[slot], required=required)
        for slot in MealSlot
        if counts[slot] < required
    ]
    backfills = [
        CandidateBackfillRequest(
            slot=item.slot,
            needed_candidates=item.required - item.available,
            semantic_query=" ".join(
                part for part in (semantic_query.strip(), item.slot.value) if part
            ),
            metadata_filters={
                "planning_eligible": True,
                "meal_type": item.slot.value,
                "excluded_allergens": sorted(constraints.excluded_allergens),
                "max_meal_minutes": constraints.max_meal_minutes,
            },
        )
        for item in shortages
    ]
    return CandidatePoolReport(
        ready=not shortages,
        required_distinct_per_slot=required,
        eligible_recipe_ids=[recipe.recipe_id for recipe in eligible],
        rejected_counts=dict(rejected),
        shortages=shortages,
        backfill_requests=backfills,
    )
