from collections import Counter
from decimal import Decimal
from pathlib import Path

from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.models import (
    MealPlanConstraints,
    MealSlot,
    RecipeCandidate,
    SolveStatus,
)
from gustobot.application.meal_planning.solver import MealPlanSolver


DATA_DIR = Path(__file__).parents[1] / "gustobot" / "data" / "meal_planning"


def seed_recipes() -> list[RecipeCandidate]:
    return load_seed_corpus(
        DATA_DIR / "recipes.v1.json",
        DATA_DIR / "manifest.v1.json",
    ).recipes


def feasible_constraints(**overrides) -> MealPlanConstraints:
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


def test_solver_builds_seven_day_plan_satisfying_hard_constraints():
    constraints = feasible_constraints()

    result = MealPlanSolver().solve(constraints, seed_recipes())

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert result.plan is not None
    assert len(result.plan.days) == 7
    assert all(len(day.assignments) == 3 for day in result.plan.days)
    assert all(
        {assignment.slot for assignment in day.assignments}
        == {MealSlot.BREAKFAST, MealSlot.LUNCH, MealSlot.DINNER}
        for day in result.plan.days
    )
    assert all(
        constraints.daily_calories_min
        <= day.calories_kcal
        <= constraints.daily_calories_max
        for day in result.plan.days
    )
    assert all(
        day.protein_g >= constraints.daily_protein_min_g for day in result.plan.days
    )
    assert result.plan.total_cost_cents <= constraints.weekly_budget_cents


def test_solver_respects_recipe_repetition_limit():
    result = MealPlanSolver().solve(feasible_constraints(), seed_recipes())

    selected = Counter(
        assignment.recipe_id
        for day in result.plan.days
        for assignment in day.assignments
    )
    assert max(selected.values()) <= 2


def test_solver_never_selects_an_excluded_allergen():
    constraints = feasible_constraints(
        excluded_allergens={"shellfish"},
        max_recipe_repeats=3,
    )
    recipes = seed_recipes()
    recipes_by_id = {recipe.recipe_id: recipe for recipe in recipes}

    result = MealPlanSolver().solve(constraints, recipes)

    assert result.plan is not None
    assert all(
        not recipes_by_id[assignment.recipe_id].allergens
        & constraints.excluded_allergens
        for day in result.plan.days
        for assignment in day.assignments
    )


def test_solver_excludes_recipe_that_fails_quality_gate():
    recipes = seed_recipes()
    incomplete = recipes[0].model_copy(
        update={"recipe_id": "r999", "calories_kcal": None}
    )

    result = MealPlanSolver().solve(feasible_constraints(), [*recipes, incomplete])

    assert result.plan is not None
    assert all(
        assignment.recipe_id != "r999"
        for day in result.plan.days
        for assignment in day.assignments
    )


def test_solver_returns_infeasible_when_no_breakfast_survives_safety_filter():
    constraints = feasible_constraints(
        excluded_allergens={"milk", "egg", "gluten", "soy"},
        max_recipe_repeats=7,
    )

    result = MealPlanSolver().solve(constraints, seed_recipes())

    assert result.status is SolveStatus.INFEASIBLE
    assert result.plan is None


def test_solver_handles_decimal_nutrition_by_integer_scaling():
    recipes = seed_recipes()
    recipes[0] = recipes[0].model_copy(
        update={"calories_kcal": Decimal("450.5"), "protein_g": Decimal("28.5")}
    )

    result = MealPlanSolver(nutrient_scale=10).solve(
        feasible_constraints(),
        recipes,
    )

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}


def test_solver_accepts_missing_cost_when_request_has_no_budget():
    without_cost = [
        recipe.model_copy(update={"estimated_cost_cents": None})
        for recipe in seed_recipes()
    ]
    constraints = feasible_constraints().model_copy(
        update={"weekly_budget_cents": None}
    )

    result = MealPlanSolver().solve(constraints, without_cost)

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert result.plan is not None
    assert result.plan.total_cost_cents is None
    assert "total_cost_cents" not in result.objective_values
