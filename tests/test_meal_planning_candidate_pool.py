from __future__ import annotations

from decimal import Decimal
from math import ceil
from pathlib import Path

import pytest

from gustobot.application.deepreason.models import EvidenceItem
from gustobot.application.meal_planning.candidate_pool import (
    CandidateBoundary,
    CandidateBoundaryStatus,
    apply_candidate_boundary,
    assess_candidate_pool,
    build_candidate_boundary,
    build_backfill_retrieval_requests,
    build_slot_retrieval_requests,
)
from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.models import MealPlanConstraints, MealSlot
from gustobot.application.meal_planning.text2cypher.models import (
    CypherRunStatus,
    Text2CypherResult,
)


DATA_DIR = Path(__file__).parents[1] / "gustobot" / "data" / "meal_planning"


def constraints(**updates) -> MealPlanConstraints:
    values = {
        "days": 7,
        "daily_calories_min": 1600,
        "daily_calories_max": 1800,
        "daily_protein_min_g": Decimal("100"),
        "excluded_allergens": {"peanut"},
        "preferred_tags": {"light", "high_protein"},
        "max_meal_minutes": 30,
        "weekly_budget_cents": 30000,
        "max_recipe_repeats": 2,
    }
    values.update(updates)
    return MealPlanConstraints(**values)


def recipes():
    return load_seed_corpus(
        DATA_DIR / "recipes.v1.json",
        DATA_DIR / "manifest.v1.json",
    ).recipes


def test_complete_seed_pool_has_enough_distinct_candidates_per_slot():
    result = assess_candidate_pool(
        constraints(),
        recipes(),
        semantic_query="light high protein chicken",
    )

    assert result.ready is True
    assert result.required_distinct_per_slot == ceil(7 / 2)
    assert result.shortages == []
    assert result.backfill_requests == []


def test_incomplete_recipe_is_not_solver_eligible_and_triggers_slot_backfill():
    candidates = recipes()
    breakfast = next(
        recipe for recipe in candidates if MealSlot.BREAKFAST in recipe.meal_types
    )
    broken = breakfast.model_copy(update={"calories_kcal": None})
    candidates = [broken if recipe.recipe_id == broken.recipe_id else recipe for recipe in candidates]

    result = assess_candidate_pool(
        constraints(),
        candidates,
        semantic_query="light high protein chicken",
    )

    shortage = next(item for item in result.shortages if item.slot is MealSlot.BREAKFAST)
    request = next(
        item for item in result.backfill_requests if item.slot is MealSlot.BREAKFAST
    )
    assert breakfast.recipe_id not in result.eligible_recipe_ids
    assert shortage.available == 3
    assert shortage.required == 4
    assert request.needed_candidates == 1
    assert request.semantic_query == "light high protein chicken breakfast"
    assert request.metadata_filters == {
        "planning_eligible": True,
        "meal_type": "breakfast",
        "excluded_allergens": ["peanut"],
        "max_meal_minutes": 30,
    }


def test_allergen_and_time_filters_are_applied_before_slot_coverage_check():
    result = assess_candidate_pool(
        constraints(excluded_allergens={"egg", "milk"}, max_meal_minutes=10),
        recipes(),
        semantic_query="quick meal",
    )

    assert result.ready is False
    assert result.rejected_counts["allergen_conflict"] > 0
    assert result.rejected_counts["too_slow"] > 0
    assert {item.slot for item in result.shortages} == set(MealSlot)


def test_initial_retrieval_is_partitioned_by_meal_slot_with_safety_buffer():
    requests = build_slot_retrieval_requests(
        constraints(days=7, max_recipe_repeats=2),
        semantic_query="light high protein",
        safety_buffer=2,
    )

    assert {request.slot for request in requests} == set(MealSlot)
    assert all(request.minimum_distinct_candidates == 4 for request in requests)
    assert all(request.top_k == 6 for request in requests)
    assert all(
        request.metadata_filters["meal_type"] == request.slot.value
        for request in requests
    )


def test_slot_candidate_budget_grows_with_plan_days():
    requests = build_slot_retrieval_requests(
        constraints(days=14, max_recipe_repeats=2),
        semantic_query="balanced meals",
        safety_buffer=2,
    )

    assert all(request.minimum_distinct_candidates == 7 for request in requests)
    assert all(request.top_k == 9 for request in requests)


def test_candidate_gate_uses_request_capabilities_instead_of_global_eligibility():
    candidate = recipes()[0].model_copy(
        update={
            "estimated_cost_cents": None,
            "allergen_status_verified": False,
        }
    )

    without_budget_or_allergen = assess_candidate_pool(
        constraints(
            days=1,
            weekly_budget_cents=None,
            excluded_allergens=set(),
            max_recipe_repeats=1,
        ),
        [candidate],
        semantic_query="quick meal",
    )
    with_budget = assess_candidate_pool(
        constraints(
            days=1,
            weekly_budget_cents=3000,
            excluded_allergens=set(),
            max_recipe_repeats=1,
        ),
        [candidate],
        semantic_query="quick meal",
    )

    assert candidate.recipe_id in without_budget_or_allergen.eligible_recipe_ids
    assert candidate.recipe_id not in with_budget.eligible_recipe_ids
    assert with_budget.rejected_counts["missing_budget_capability"] == 1


def test_verified_boundary_filters_candidates_and_requires_evidence_per_id():
    candidates = recipes()
    boundary = CandidateBoundary(
        status=CandidateBoundaryStatus.VERIFIED,
        selected_recipe_ids=["r101", "r201", "r101"],
        evidence_ids_by_recipe={
            "r101": ["ev_r101"],
            "r201": ["ev_r201"],
        },
    )

    filtered = apply_candidate_boundary(candidates, boundary)

    assert [recipe.recipe_id for recipe in filtered] == ["r101", "r201"]


def test_failed_or_unproven_boundary_cannot_reach_candidate_pool():
    failed = CandidateBoundary(status=CandidateBoundaryStatus.FAILED)

    try:
        apply_candidate_boundary(recipes(), failed)
    except ValueError as error:
        assert "verified" in str(error)
    else:
        raise AssertionError("failed boundary must be rejected")


def test_backfill_requests_become_slot_scoped_retrieval_requests():
    report = assess_candidate_pool(
        constraints(days=1),
        [recipe for recipe in recipes() if MealSlot.BREAKFAST not in recipe.meal_types],
        semantic_query="light meal",
    )

    requests = build_backfill_retrieval_requests(report, safety_buffer=2)

    breakfast = next(item for item in requests if item.slot is MealSlot.BREAKFAST)
    assert breakfast.minimum_distinct_candidates == 1
    assert breakfast.top_k == 3
    assert breakfast.metadata_filters["meal_type"] == "breakfast"


def test_text2cypher_result_and_ledger_build_a_verified_candidate_boundary():
    result = Text2CypherResult(
        question="safe recipes",
        status=CypherRunStatus.SUCCESS,
        selected_recipe_ids=["r101", "r201"],
    )
    evidence = [
        EvidenceItem.create_fact(
            task_id="recipe-1",
            source_type="fact_atom",
            source="neo4j_readonly",
            entity_type="recipe",
            entity_id=recipe_id,
            field="exists",
            value=True,
        )
        for recipe_id in result.selected_recipe_ids
    ]

    boundary = build_candidate_boundary(result, evidence)

    assert boundary.status is CandidateBoundaryStatus.VERIFIED
    assert boundary.selected_recipe_ids == ["r101", "r201"]
    assert all(boundary.evidence_ids_by_recipe.values())


def test_candidate_boundary_refuses_text2cypher_id_without_ledger_evidence():
    result = Text2CypherResult(
        question="safe recipes",
        status=CypherRunStatus.SUCCESS,
        selected_recipe_ids=["r101"],
    )

    with pytest.raises(ValueError, match="evidence"):
        build_candidate_boundary(result, [])
