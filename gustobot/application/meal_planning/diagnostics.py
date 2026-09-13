from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from pydantic import BaseModel, Field

from .models import MealPlanConstraints, MealSlot, RecipeCandidate
from .quality import partition_recipes


class InfeasibilityReport(BaseModel):
    is_infeasible: bool
    conflict_constraints: list[str] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)


def diagnose_infeasibility(
    constraints: MealPlanConstraints,
    recipes: list[RecipeCandidate],
) -> InfeasibilityReport:
    eligible = [
        recipe
        for recipe in partition_recipes(recipes).planning_eligible
        if not recipe.allergens & constraints.excluded_allergens
        and (
            constraints.max_meal_minutes is None
            or recipe.total_minutes <= constraints.max_meal_minutes
        )
    ]
    by_slot: dict[MealSlot, list[RecipeCandidate]] = defaultdict(list)
    for recipe in eligible:
        for slot in recipe.meal_types:
            by_slot[slot].append(recipe)

    conflicts: list[str] = []
    explanations: list[str] = []
    for slot in MealSlot:
        if not by_slot[slot]:
            conflicts.append(f"meal_slot:{slot.value}")
            explanations.append(f"{slot.value} has no candidate after safety filters")

    if all(by_slot[slot] for slot in MealSlot):
        maximum_calories = sum(
            max(recipe.calories_kcal for recipe in by_slot[slot])
            for slot in MealSlot
        )
        minimum_calories = sum(
            min(recipe.calories_kcal for recipe in by_slot[slot])
            for slot in MealSlot
        )
        maximum_protein = sum(
            max(recipe.protein_g for recipe in by_slot[slot])
            for slot in MealSlot
        )
        if maximum_calories < constraints.daily_calories_min:
            conflicts.append("daily_calories_min")
            explanations.append(
                f"daily maximum reachable calories {maximum_calories} is below "
                f"requested minimum {constraints.daily_calories_min}"
            )
        if minimum_calories > constraints.daily_calories_max:
            conflicts.append("daily_calories_max")
            explanations.append(
                f"daily minimum reachable calories {minimum_calories} exceeds "
                f"requested maximum {constraints.daily_calories_max}"
            )
        if maximum_protein < constraints.daily_protein_min_g:
            conflicts.append("daily_protein_min_g")
            explanations.append(
                f"daily maximum reachable protein {maximum_protein}g is below "
                f"requested minimum {constraints.daily_protein_min_g}g"
            )
        if constraints.weekly_budget_cents is not None:
            minimum_daily_cost = sum(
                min(recipe.estimated_cost_cents for recipe in by_slot[slot])
                for slot in MealSlot
            )
            minimum_total_cost = minimum_daily_cost * constraints.days
            if minimum_total_cost > constraints.weekly_budget_cents:
                conflicts.append("weekly_budget_cents")
                explanations.append(
                    f"minimum reachable cost {minimum_total_cost} cents exceeds "
                    f"budget {constraints.weekly_budget_cents} cents"
                )

    return InfeasibilityReport(
        is_infeasible=bool(conflicts),
        conflict_constraints=list(dict.fromkeys(conflicts)),
        explanations=explanations,
    )
