from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from gustobot.application.meal_planning.diagnostics import diagnose_infeasibility
from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.models import (
    MealPlanConstraints,
    RecipeCandidate,
    SolveStatus,
)
from gustobot.application.meal_planning.solver import MealPlanSolveResult, MealPlanSolver
from gustobot.application.meal_planning.verification import (
    VerificationDecision,
    verify_solved_plan,
)


DATA_DIR = Path(__file__).parents[1] / "gustobot" / "data" / "meal_planning"


def seed_recipes() -> list[RecipeCandidate]:
    return load_seed_corpus(
        DATA_DIR / "recipes.v1.json",
        DATA_DIR / "manifest.v1.json",
    ).recipes


def constraints(**overrides) -> MealPlanConstraints:
    values = {
        "daily_calories_min": 1600,
        "daily_calories_max": 1800,
        "daily_protein_min_g": Decimal("100"),
        "excluded_allergens": {"peanut"},
        "max_meal_minutes": 30,
        "weekly_budget_cents": 30000,
        "max_recipe_repeats": 2,
        "max_main_ingredient_repeats": 7,
        "days": 7,
    }
    values.update(overrides)
    return MealPlanConstraints(**values)


def solved_plan():
    recipes = seed_recipes()
    result = MealPlanSolver().solve(constraints(), recipes)
    assert result.plan is not None
    return recipes, result


def test_feasible_plan_passes_deterministic_verification():
    recipes, result = solved_plan()

    verified = verify_solved_plan(constraints(), recipes, result)

    assert verified.decision is VerificationDecision.ALLOW
    assert verified.findings == []


def test_selected_peanut_recipe_is_denied_even_if_plan_text_claims_safe():
    recipes, result = solved_plan()
    selected_id = result.plan.days[0].assignments[0].recipe_id
    poisoned = [
        recipe.model_copy(update={"allergens": {"peanut"}})
        if recipe.recipe_id == selected_id
        else recipe
        for recipe in recipes
    ]

    verified = verify_solved_plan(constraints(), poisoned, result)

    assert verified.decision is VerificationDecision.DENY
    assert any(item.code == "excluded_allergen_selected" for item in verified.findings)


def test_tampered_assignment_calories_trigger_replan():
    recipes, result = solved_plan()
    assignment = result.plan.days[0].assignments[0]
    assignment.calories_kcal += Decimal("100")

    verified = verify_solved_plan(constraints(), recipes, result)

    assert verified.decision is VerificationDecision.REPLAN
    assert any(item.code == "assignment_fact_mismatch" for item in verified.findings)


def test_impossible_request_returns_sufficient_conflict_constraints():
    report = diagnose_infeasibility(
        constraints(
            days=1,
            daily_calories_min=5000,
            daily_calories_max=5200,
            daily_protein_min_g=Decimal("500"),
            weekly_budget_cents=100,
        ),
        seed_recipes(),
    )

    assert report.is_infeasible is True
    assert "daily_calories_min" in report.conflict_constraints
    assert "daily_protein_min_g" in report.conflict_constraints
    assert "weekly_budget_cents" in report.conflict_constraints


def test_unknown_solver_status_requests_retry_instead_of_fabricating_plan():
    unknown = MealPlanSolveResult(status=SolveStatus.UNKNOWN)

    verified = verify_solved_plan(constraints(), seed_recipes(), unknown)

    assert verified.decision is VerificationDecision.RETRY
    assert verified.plan_releasable is False
