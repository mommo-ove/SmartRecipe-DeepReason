from __future__ import annotations

from pydantic import BaseModel, Field

from .models import MealPlan, MealPlanConstraints, MealSlot, RecipeCandidate, SolveStatus
from .solver import MealPlanSolver


class MealPlanReplanResult(BaseModel):
    status: SolveStatus
    plan: MealPlan | None = None
    changed_assignments: int = Field(default=0, ge=0)
    conflict_constraints: list[str] = Field(default_factory=list)


class MealPlanReplanner:
    def __init__(self, solver: MealPlanSolver) -> None:
        self._solver = solver

    def replace_meal(
        self,
        *,
        constraints: MealPlanConstraints,
        recipes: list[RecipeCandidate],
        current_plan: MealPlan,
        day: int,
        slot: MealSlot,
        forbidden_recipe_ids: set[str] | None = None,
    ) -> MealPlanReplanResult:
        forbidden = forbidden_recipe_ids or set()
        allowed_target_candidates = [
            recipe
            for recipe in recipes
            if slot in recipe.meal_types
            and recipe.recipe_id not in forbidden
            and recipe.planning_eligible
            and not recipe.allergens & constraints.excluded_allergens
            and (
                constraints.max_meal_minutes is None
                or recipe.total_minutes <= constraints.max_meal_minutes
            )
        ]
        if not allowed_target_candidates:
            return MealPlanReplanResult(
                status=SolveStatus.INFEASIBLE,
                conflict_constraints=[
                    "target_meal_has_no_allowed_candidate",
                    "unaffected_meals_locked",
                ],
            )

        before = {
            (assignment.day, assignment.slot): assignment.recipe_id
            for daily_plan in current_plan.days
            for assignment in daily_plan.assignments
        }
        locked = {
            key: recipe_id
            for key, recipe_id in before.items()
            if key != (day, slot)
        }
        solved = self._solver.solve(
            constraints,
            recipes,
            locked_assignments=locked,
            forbidden_assignments={(day, slot): forbidden},
        )
        if solved.plan is None:
            return MealPlanReplanResult(
                status=solved.status,
                conflict_constraints=[
                    "locked_plan_conflicts_with_requested_change",
                    "unaffected_meals_locked",
                ],
            )
        after = {
            (assignment.day, assignment.slot): assignment.recipe_id
            for daily_plan in solved.plan.days
            for assignment in daily_plan.assignments
        }
        changed = sum(before.get(key) != recipe_id for key, recipe_id in after.items())
        return MealPlanReplanResult(
            status=solved.status,
            plan=solved.plan,
            changed_assignments=changed,
        )
