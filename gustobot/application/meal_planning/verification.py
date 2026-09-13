from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field

from .models import MealPlan, MealPlanConstraints, MealSlot, RecipeCandidate, SolveStatus
from .solver import MealPlanSolveResult


class VerificationDecision(str, Enum):
    ALLOW = "allow"
    RETRY = "retry"
    REPLAN = "replan"
    CLARIFY = "clarify"
    DENY = "deny"


class VerificationFinding(BaseModel):
    code: str
    message: str
    recipe_id: str | None = None
    day: int | None = None
    slot: MealSlot | None = None


class PlanVerificationResult(BaseModel):
    decision: VerificationDecision
    plan_releasable: bool = False
    findings: list[VerificationFinding] = Field(default_factory=list)


def verify_solved_plan(
    constraints: MealPlanConstraints,
    recipes: list[RecipeCandidate],
    result: MealPlanSolveResult,
) -> PlanVerificationResult:
    if result.status is SolveStatus.UNKNOWN:
        return PlanVerificationResult(
            decision=VerificationDecision.RETRY,
            findings=[VerificationFinding(code="solver_unknown", message="solver did not produce a conclusive result")],
        )
    if result.plan is None:
        return PlanVerificationResult(
            decision=VerificationDecision.REPLAN,
            findings=[VerificationFinding(code="plan_missing", message="no candidate plan is available for verification")],
        )

    facts = {recipe.recipe_id: recipe for recipe in recipes}
    findings: list[VerificationFinding] = []
    safety_findings: list[VerificationFinding] = []
    recipe_counts: Counter[str] = Counter()
    ingredient_counts: Counter[str] = Counter()
    total_cost = 0
    cost_complete = True

    for daily_plan in result.plan.days:
        if {item.slot for item in daily_plan.assignments} != set(MealSlot):
            findings.append(VerificationFinding(code="meal_slot_count_mismatch", message="day does not contain exactly one assignment per slot", day=daily_plan.day))
        day_calories = 0
        day_protein = 0
        for assignment in daily_plan.assignments:
            recipe = facts.get(assignment.recipe_id)
            if recipe is None or not recipe.planning_eligible:
                findings.append(VerificationFinding(code="recipe_fact_missing", message="selected recipe has no eligible planning facts", recipe_id=assignment.recipe_id, day=assignment.day, slot=assignment.slot))
                continue
            excluded = recipe.allergens & constraints.excluded_allergens
            if excluded:
                safety_findings.append(VerificationFinding(code="excluded_allergen_selected", message=f"selected recipe contains excluded allergens: {sorted(excluded)}", recipe_id=recipe.recipe_id, day=assignment.day, slot=assignment.slot))
            if constraints.max_meal_minutes is not None and recipe.total_minutes > constraints.max_meal_minutes:
                findings.append(VerificationFinding(code="meal_time_exceeded", message="selected recipe exceeds the meal time limit", recipe_id=recipe.recipe_id, day=assignment.day, slot=assignment.slot))
            if (
                assignment.calories_kcal != recipe.calories_kcal
                or assignment.protein_g != recipe.protein_g
                or assignment.total_minutes != recipe.total_minutes
                or assignment.estimated_cost_cents != recipe.estimated_cost_cents
            ):
                findings.append(VerificationFinding(code="assignment_fact_mismatch", message="assignment values do not match hydrated recipe facts", recipe_id=recipe.recipe_id, day=assignment.day, slot=assignment.slot))
            day_calories += recipe.calories_kcal
            day_protein += recipe.protein_g
            if recipe.estimated_cost_cents is None:
                cost_complete = False
            else:
                total_cost += recipe.estimated_cost_cents
            recipe_counts[recipe.recipe_id] += 1
            ingredient_counts[recipe.ingredients_normalized[0].ingredient_id] += 1
        if not constraints.daily_calories_min <= day_calories <= constraints.daily_calories_max:
            findings.append(VerificationFinding(code="daily_calories_out_of_range", message=f"recalculated daily calories {day_calories} are outside the requested range", day=daily_plan.day))
        if day_protein < constraints.daily_protein_min_g:
            findings.append(VerificationFinding(code="daily_protein_below_minimum", message=f"recalculated daily protein {day_protein}g is below the requested minimum", day=daily_plan.day))

    if len(result.plan.days) != constraints.days:
        findings.append(VerificationFinding(code="day_count_mismatch", message="plan day count does not match the request"))
    if (
        constraints.weekly_budget_cents is not None
        and (not cost_complete or total_cost > constraints.weekly_budget_cents)
    ):
        findings.append(VerificationFinding(code="weekly_budget_exceeded", message="recalculated plan cost exceeds the weekly budget"))
    if recipe_counts and max(recipe_counts.values()) > constraints.max_recipe_repeats:
        findings.append(VerificationFinding(code="recipe_repeat_limit_exceeded", message="a recipe exceeds the repetition limit"))
    if ingredient_counts and max(ingredient_counts.values()) > constraints.max_main_ingredient_repeats:
        findings.append(VerificationFinding(code="ingredient_repeat_limit_exceeded", message="a main ingredient exceeds the repetition limit"))

    if safety_findings:
        return PlanVerificationResult(decision=VerificationDecision.DENY, findings=[*safety_findings, *findings])
    if findings:
        return PlanVerificationResult(decision=VerificationDecision.REPLAN, findings=findings)
    return PlanVerificationResult(decision=VerificationDecision.ALLOW, plan_releasable=True)
