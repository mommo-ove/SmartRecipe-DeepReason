from decimal import Decimal

from gustobot.application.meal_planning.capabilities import (
    PlanningCapability,
    assess_request_capabilities,
    required_capabilities,
)
from gustobot.application.meal_planning.models import (
    IngredientAmount,
    MealPlanConstraints,
    MealSlot,
    RecipeCandidate,
)


def constraints(**updates) -> MealPlanConstraints:
    values = {
        "days": 3,
        "daily_calories_min": 1500,
        "daily_calories_max": 1900,
        "daily_protein_min_g": Decimal("90"),
        "excluded_allergens": set(),
        "max_meal_minutes": 30,
        "weekly_budget_cents": None,
    }
    values.update(updates)
    return MealPlanConstraints(**values)


def partial_recipe(**updates) -> RecipeCandidate:
    values = {
        "recipe_id": "r1",
        "name": "鸡胸肉西兰花",
        "meal_types": {MealSlot.LUNCH, MealSlot.DINNER},
        "calories_kcal": Decimal("520"),
        "protein_g": Decimal("42"),
        "total_minutes": 25,
        "estimated_cost_cents": None,
        "allergens": set(),
        "allergen_status_verified": False,
        "ingredients_normalized": [
            IngredientAmount(ingredient_id="chicken", amount=200, unit="g")
        ],
        "source_refs": ["source:r1"],
    }
    values.update(updates)
    return RecipeCandidate(**values)


def test_request_without_budget_or_allergen_does_not_require_those_capabilities():
    required = required_capabilities(constraints())

    assert required == {
        PlanningCapability.NUTRITION,
        PlanningCapability.TIME,
        PlanningCapability.INGREDIENTS,
        PlanningCapability.LINEAGE,
    }
    report = assess_request_capabilities(constraints(), [partial_recipe()])
    assert report.ready_recipe_ids == ["r1"]
    assert report.rejected == {}


def test_allergen_request_rejects_unverified_allergen_facts():
    report = assess_request_capabilities(
        constraints(excluded_allergens={"peanut"}),
        [partial_recipe()],
    )

    assert report.ready_recipe_ids == []
    assert report.rejected["r1"] == [PlanningCapability.ALLERGEN]


def test_budget_request_rejects_missing_cost_facts():
    report = assess_request_capabilities(
        constraints(weekly_budget_cents=20000),
        [partial_recipe()],
    )

    assert report.ready_recipe_ids == []
    assert report.rejected["r1"] == [PlanningCapability.BUDGET]
