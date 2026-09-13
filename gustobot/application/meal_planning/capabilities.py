from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .models import MealPlanConstraints, RecipeCandidate


class PlanningCapability(str, Enum):
    NUTRITION = "nutrition"
    TIME = "time"
    INGREDIENTS = "ingredients"
    LINEAGE = "lineage"
    ALLERGEN = "allergen"
    BUDGET = "budget"


class RequestCapabilityReport(BaseModel):
    required: set[PlanningCapability]
    ready_recipe_ids: list[str]
    rejected: dict[str, list[PlanningCapability]] = Field(default_factory=dict)


def required_capabilities(
    constraints: MealPlanConstraints,
) -> set[PlanningCapability]:
    required = {
        PlanningCapability.NUTRITION,
        PlanningCapability.INGREDIENTS,
        PlanningCapability.LINEAGE,
    }
    if constraints.max_meal_minutes is not None:
        required.add(PlanningCapability.TIME)
    if constraints.excluded_allergens:
        required.add(PlanningCapability.ALLERGEN)
    if constraints.weekly_budget_cents is not None:
        required.add(PlanningCapability.BUDGET)
    return required


def assess_request_capabilities(
    constraints: MealPlanConstraints,
    recipes: list[RecipeCandidate],
) -> RequestCapabilityReport:
    required = required_capabilities(constraints)
    ready: list[str] = []
    rejected: dict[str, list[PlanningCapability]] = {}
    for recipe in recipes:
        missing = [
            capability
            for capability in sorted(required, key=lambda item: item.value)
            if not _supports(recipe, capability)
        ]
        if missing:
            rejected[recipe.recipe_id] = missing
        else:
            ready.append(recipe.recipe_id)
    return RequestCapabilityReport(
        required=required,
        ready_recipe_ids=ready,
        rejected=rejected,
    )


def _supports(
    recipe: RecipeCandidate,
    capability: PlanningCapability,
) -> bool:
    checks = {
        PlanningCapability.NUTRITION: (
            recipe.calories_kcal is not None and recipe.protein_g is not None
        ),
        PlanningCapability.TIME: recipe.total_minutes is not None,
        PlanningCapability.INGREDIENTS: bool(recipe.ingredients_normalized),
        PlanningCapability.LINEAGE: bool(recipe.source_refs),
        PlanningCapability.ALLERGEN: recipe.allergen_status_verified,
        PlanningCapability.BUDGET: recipe.estimated_cost_cents is not None,
    }
    return checks[capability]
