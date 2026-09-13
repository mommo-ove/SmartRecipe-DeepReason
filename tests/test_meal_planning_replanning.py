from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.models import (
    MealPlanConstraints,
    MealSlot,
    RecipeCandidate,
    SolveStatus,
)
from gustobot.application.meal_planning.replanning import MealPlanReplanner
from gustobot.application.meal_planning.solver import MealPlanSolver


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
        "preferred_tags": {"vegetarian"},
        "max_meal_minutes": 30,
        "weekly_budget_cents": 30000,
        "max_recipe_repeats": 2,
        "max_main_ingredient_repeats": 7,
        "days": 7,
    }
    values.update(overrides)
    return MealPlanConstraints(**values)


def assignment_map(plan):
    return {
        (assignment.day, assignment.slot): assignment.recipe_id
        for day in plan.days
        for assignment in day.assignments
    }


def test_hard_feasibility_wins_over_preference():
    recipes = seed_recipes()
    unsafe_preferred = recipes[4].model_copy(
        update={
            "recipe_id": "unsafe-favorite",
            "allergens": {"peanut"},
            "preference_tags": {"vegetarian"},
        }
    )

    result = MealPlanSolver().solve(constraints(), [*recipes, unsafe_preferred])

    assert result.plan is not None
    assert all(
        assignment.recipe_id != "unsafe-favorite"
        for day in result.plan.days
        for assignment in day.assignments
    )


def test_preference_is_optimized_before_cost():
    recipes = seed_recipes()
    preferred_expensive = recipes[4].model_copy(
        update={
            "recipe_id": "preferred-lunch",
            "preference_tags": {"vegetarian"},
            "estimated_cost_cents": 1499,
        }
    )
    cheap_unpreferred = recipes[4].model_copy(
        update={
            "recipe_id": "cheap-lunch",
            "preference_tags": set(),
            "estimated_cost_cents": 1,
        }
    )

    result = MealPlanSolver().solve(
        constraints(days=1, max_recipe_repeats=1),
        [*recipes, preferred_expensive, cheap_unpreferred],
    )

    assert result.plan is not None
    lunch = next(
        item
        for item in result.plan.days[0].assignments
        if item.slot is MealSlot.LUNCH
    )
    assert lunch.recipe_id == "preferred-lunch"
    assert result.objective_values["preference_matches"] >= 1


def test_replacing_tuesday_dinner_locks_every_other_assignment():
    recipes = seed_recipes()
    base = MealPlanSolver().solve(constraints(), recipes)
    assert base.plan is not None
    before = assignment_map(base.plan)
    old_tuesday_dinner = before[(2, MealSlot.DINNER)]
    old_recipe = next(
        recipe for recipe in recipes if recipe.recipe_id == old_tuesday_dinner
    )
    equivalent_replacement = old_recipe.model_copy(
        update={
            "recipe_id": "tuesday-dinner-replacement",
            "name": "Tuesday dinner replacement",
        }
    )

    replanned = MealPlanReplanner(MealPlanSolver()).replace_meal(
        constraints=constraints(),
        recipes=[*recipes, equivalent_replacement],
        current_plan=base.plan,
        day=2,
        slot=MealSlot.DINNER,
        forbidden_recipe_ids={old_tuesday_dinner},
    )

    assert replanned.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert replanned.plan is not None
    after = assignment_map(replanned.plan)
    assert after[(2, MealSlot.DINNER)] != old_tuesday_dinner
    assert {
        key: recipe_id
        for key, recipe_id in after.items()
        if key != (2, MealSlot.DINNER)
    } == {
        key: recipe_id
        for key, recipe_id in before.items()
        if key != (2, MealSlot.DINNER)
    }
    assert replanned.changed_assignments == 1


def test_impossible_locked_replan_returns_conflict_without_rewriting_week():
    recipes = seed_recipes()
    base = MealPlanSolver().solve(constraints(), recipes)
    assert base.plan is not None
    all_dinner_ids = {
        recipe.recipe_id
        for recipe in recipes
        if MealSlot.DINNER in recipe.meal_types
    }

    replanned = MealPlanReplanner(MealPlanSolver()).replace_meal(
        constraints=constraints(),
        recipes=recipes,
        current_plan=base.plan,
        day=2,
        slot=MealSlot.DINNER,
        forbidden_recipe_ids=all_dinner_ids,
    )

    assert replanned.status is SolveStatus.INFEASIBLE
    assert replanned.plan is None
    assert replanned.changed_assignments == 0
    assert "target_meal_has_no_allowed_candidate" in replanned.conflict_constraints
    assert "unaffected_meals_locked" in replanned.conflict_constraints
