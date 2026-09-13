from decimal import Decimal

import pytest
from pydantic import ValidationError

from gustobot.application.meal_planning.models import (
    DailyPlan,
    IngredientAmount,
    MealAssignment,
    MealPlan,
    MealPlanConstraints,
    MealSlot,
    RecipeCandidate,
    SolveStatus,
)


def test_constraints_reject_inverted_calorie_range():
    with pytest.raises(ValidationError, match="daily_calories_max"):
        MealPlanConstraints(
            daily_calories_min=1800,
            daily_calories_max=1500,
            daily_protein_min_g=100,
        )


def test_constraints_normalize_allergen_names():
    constraints = MealPlanConstraints(
        daily_calories_min=1600,
        daily_calories_max=1800,
        daily_protein_min_g=100,
        excluded_allergens=[" 花生 ", "PEANUT", "peanut"],
    )

    assert constraints.excluded_allergens == {"花生", "peanut"}


def test_recipe_is_not_eligible_without_normalized_nutrition():
    recipe = RecipeCandidate(
        recipe_id="r1",
        name="番茄炒蛋",
        meal_types={MealSlot.LUNCH, MealSlot.DINNER},
        calories_kcal=None,
        protein_g=Decimal("20"),
        total_minutes=15,
        estimated_cost_cents=1200,
        allergens=set(),
        allergen_status_verified=True,
        source_refs=["recipe:r1"],
    )

    assert recipe.planning_eligible is False


def test_unknown_allergen_status_is_not_equivalent_to_no_allergens():
    recipe = RecipeCandidate(
        recipe_id="r2",
        name="蔬菜汤",
        meal_types={MealSlot.DINNER},
        calories_kcal=Decimal("220"),
        protein_g=Decimal("8"),
        total_minutes=20,
        estimated_cost_cents=900,
        allergens=set(),
        allergen_status_verified=False,
        source_refs=["recipe:r2", "nutrition:source:r2"],
    )

    assert recipe.planning_eligible is False


def test_complete_recipe_is_planning_eligible():
    recipe = RecipeCandidate(
        recipe_id="r3",
        name="西兰花鸡胸肉",
        meal_types={MealSlot.LUNCH, MealSlot.DINNER},
        calories_kcal=Decimal("430.5"),
        protein_g=Decimal("42.2"),
        total_minutes=25,
        estimated_cost_cents=1600,
        allergens=set(),
        allergen_status_verified=True,
        ingredients_normalized=[
            IngredientAmount(ingredient_id="egg", amount=Decimal("100"), unit="g")
        ],
        source_refs=["recipe:r3", "nutrition:fdc:171077"],
    )

    assert recipe.planning_eligible is True


def test_meal_plan_rejects_duplicate_day_numbers():
    assignment = MealAssignment(
        day=1,
        slot=MealSlot.BREAKFAST,
        recipe_id="r4",
        recipe_name="燕麦牛奶",
        calories_kcal=Decimal("350"),
        protein_g=Decimal("15"),
        total_minutes=10,
        estimated_cost_cents=800,
    )
    day = DailyPlan(day=1, assignments=[assignment])

    with pytest.raises(ValidationError, match="day numbers"):
        MealPlan(status=SolveStatus.FEASIBLE, days=[day, day])


def test_money_is_rejected_when_not_an_integer_number_of_cents():
    with pytest.raises(ValidationError):
        MealPlanConstraints(
            daily_calories_min=1600,
            daily_calories_max=1800,
            daily_protein_min_g=100,
            weekly_budget_cents=299.9,
        )
