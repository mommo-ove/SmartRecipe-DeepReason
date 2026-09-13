from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Mapping

from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

from .capabilities import assess_request_capabilities
from .models import (
    DailyPlan,
    MealAssignment,
    MealPlan,
    MealPlanConstraints,
    MealSlot,
    RecipeCandidate,
    SolveStatus,
)


class MealPlanSolveResult(BaseModel):
    status: SolveStatus
    plan: MealPlan | None = None
    solver_wall_time_ms: float = Field(default=0, ge=0)
    objective_values: dict[str, int] = Field(default_factory=dict)


class MealPlanSolver:
    def __init__(
        self,
        *,
        nutrient_scale: int = 10,
        max_solver_seconds: float = 10.0,
    ) -> None:
        if nutrient_scale < 1:
            raise ValueError("nutrient_scale must be positive")
        self._nutrient_scale = nutrient_scale
        self._max_solver_seconds = max_solver_seconds

    def solve(
        self,
        constraints: MealPlanConstraints,
        recipes: list[RecipeCandidate],
        *,
        locked_assignments: Mapping[tuple[int, MealSlot], str] | None = None,
        forbidden_recipe_ids: set[str] | None = None,
        forbidden_assignments: Mapping[tuple[int, MealSlot], set[str]] | None = None,
    ) -> MealPlanSolveResult:
        capability_report = assess_request_capabilities(constraints, recipes)
        capability_ready = set(capability_report.ready_recipe_ids)
        eligible = [
            recipe for recipe in recipes if recipe.recipe_id in capability_ready
        ]
        candidates = [
            recipe
            for recipe in eligible
            if self._passes_candidate_filters(recipe, constraints)
            and recipe.recipe_id not in (forbidden_recipe_ids or set())
        ]
        model = cp_model.CpModel()
        variables: dict[tuple[int, MealSlot, str], cp_model.IntVar] = {}

        for day in range(1, constraints.days + 1):
            for slot in MealSlot:
                slot_variables = []
                for recipe in candidates:
                    if slot not in recipe.meal_types:
                        continue
                    if recipe.recipe_id in (forbidden_assignments or {}).get(
                        (day, slot), set()
                    ):
                        continue
                    variable = model.NewBoolVar(
                        f"d{day}_{slot.value}_{recipe.recipe_id}"
                    )
                    variables[(day, slot, recipe.recipe_id)] = variable
                    slot_variables.append(variable)
                model.Add(sum(slot_variables) == 1)

        self._add_daily_nutrition_constraints(
            model,
            variables,
            candidates,
            constraints,
        )
        self._add_weekly_constraints(model, variables, candidates, constraints)

        for (day, slot), recipe_id in (locked_assignments or {}).items():
            locked = variables.get((day, slot, recipe_id))
            if locked is None:
                model.Add(False)
            else:
                model.Add(locked == 1)

        preference_score = sum(
            len(recipe.preference_tags & constraints.preferred_tags)
            * variables[(day, slot, recipe.recipe_id)]
            for (day, slot, recipe_id), variable in variables.items()
            for recipe in candidates
            if recipe.recipe_id == recipe_id
        )
        recipe_repeat_penalty, ingredient_repeat_penalty = (
            self._build_repeat_penalties(model, variables, candidates, constraints)
        )
        total_repeat_penalty = recipe_repeat_penalty + ingredient_repeat_penalty
        recipes_by_id = {recipe.recipe_id: recipe for recipe in candidates}
        total_cost = None
        if constraints.weekly_budget_cents is not None:
            total_cost = sum(
                recipes_by_id[recipe_id].estimated_cost_cents * variable
                for (_, _, recipe_id), variable in variables.items()
            )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._max_solver_seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        raw_status = solver.Solve(model)
        status = self._map_status(raw_status)
        total_wall_time_ms = solver.WallTime() * 1000
        if status not in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
            return MealPlanSolveResult(
                status=status,
                solver_wall_time_ms=total_wall_time_ms,
            )

        if constraints.preferred_tags:
            model.Maximize(preference_score)
            raw_status = solver.Solve(model)
            total_wall_time_ms += solver.WallTime() * 1000
            status = self._map_status(raw_status)
            if status not in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
                return MealPlanSolveResult(status=status, solver_wall_time_ms=total_wall_time_ms)
            best_preference = int(solver.Value(preference_score))
            model.Add(preference_score == best_preference)
        else:
            best_preference = 0

        model.Minimize(total_repeat_penalty)
        raw_status = solver.Solve(model)
        total_wall_time_ms += solver.WallTime() * 1000
        status = self._map_status(raw_status)
        best_repeat_penalty = int(solver.Value(total_repeat_penalty))
        model.Add(total_repeat_penalty == best_repeat_penalty)

        if total_cost is not None:
            model.Minimize(total_cost)
            raw_status = solver.Solve(model)
            total_wall_time_ms += solver.WallTime() * 1000
            status = self._map_status(raw_status)
        wall_time_ms = total_wall_time_ms

        days = self._build_days(
            solver,
            variables,
            recipes_by_id,
            constraints.days,
        )
        plan = MealPlan(
            status=status,
            days=days,
            solver_wall_time_ms=wall_time_ms,
        )
        objective_values = {
            "preference_matches": best_preference,
            "repeat_penalty": best_repeat_penalty,
        }
        if total_cost is not None:
            objective_values["total_cost_cents"] = int(solver.Value(total_cost))
        return MealPlanSolveResult(
            status=status,
            plan=plan,
            solver_wall_time_ms=wall_time_ms,
            objective_values=objective_values,
        )

    def _build_repeat_penalties(
        self,
        model: cp_model.CpModel,
        variables: dict[tuple[int, MealSlot, str], cp_model.IntVar],
        recipes: list[RecipeCandidate],
        constraints: MealPlanConstraints,
    ):
        variables_by_recipe: dict[str, list[cp_model.IntVar]] = defaultdict(list)
        for (_, _, recipe_id), variable in variables.items():
            variables_by_recipe[recipe_id].append(variable)

        recipe_excess = []
        for recipe_id, recipe_variables in variables_by_recipe.items():
            excess = model.NewIntVar(
                0,
                max(0, constraints.max_recipe_repeats - 1),
                f"recipe_repeat_excess_{recipe_id}",
            )
            model.Add(excess >= sum(recipe_variables) - 1)
            recipe_excess.append(excess)

        variables_by_ingredient: dict[str, list[cp_model.IntVar]] = defaultdict(list)
        for recipe in recipes:
            variables_by_ingredient[recipe.ingredients_normalized[0].ingredient_id].extend(
                variables_by_recipe[recipe.recipe_id]
            )
        ingredient_excess = []
        for ingredient_id, ingredient_variables in variables_by_ingredient.items():
            excess = model.NewIntVar(
                0,
                max(0, constraints.max_main_ingredient_repeats - 1),
                f"ingredient_repeat_excess_{ingredient_id}",
            )
            model.Add(excess >= sum(ingredient_variables) - 1)
            ingredient_excess.append(excess)
        return sum(recipe_excess), sum(ingredient_excess)

    def _passes_candidate_filters(
        self,
        recipe: RecipeCandidate,
        constraints: MealPlanConstraints,
    ) -> bool:
        if recipe.allergens & constraints.excluded_allergens:
            return False
        if (
            constraints.max_meal_minutes is not None
            and recipe.total_minutes > constraints.max_meal_minutes
        ):
            return False
        return True

    def _add_daily_nutrition_constraints(
        self,
        model: cp_model.CpModel,
        variables: dict[tuple[int, MealSlot, str], cp_model.IntVar],
        recipes: list[RecipeCandidate],
        constraints: MealPlanConstraints,
    ) -> None:
        recipes_by_id = {recipe.recipe_id: recipe for recipe in recipes}
        for day in range(1, constraints.days + 1):
            day_variables = [
                (variable, recipes_by_id[recipe_id])
                for (variable_day, _, recipe_id), variable in variables.items()
                if variable_day == day
            ]
            calories = sum(
                self._scale(recipe.calories_kcal) * variable
                for variable, recipe in day_variables
            )
            protein = sum(
                self._scale(recipe.protein_g) * variable
                for variable, recipe in day_variables
            )
            model.Add(calories >= constraints.daily_calories_min * self._nutrient_scale)
            model.Add(calories <= constraints.daily_calories_max * self._nutrient_scale)
            model.Add(protein >= self._scale(constraints.daily_protein_min_g))

    def _add_weekly_constraints(
        self,
        model: cp_model.CpModel,
        variables: dict[tuple[int, MealSlot, str], cp_model.IntVar],
        recipes: list[RecipeCandidate],
        constraints: MealPlanConstraints,
    ) -> None:
        variables_by_recipe: dict[str, list[cp_model.IntVar]] = defaultdict(list)
        for (_, _, recipe_id), variable in variables.items():
            variables_by_recipe[recipe_id].append(variable)
        for recipe_variables in variables_by_recipe.values():
            model.Add(sum(recipe_variables) <= constraints.max_recipe_repeats)

        variables_by_main_ingredient: dict[str, list[cp_model.IntVar]] = defaultdict(
            list
        )
        for recipe in recipes:
            main_ingredient = recipe.ingredients_normalized[0].ingredient_id
            variables_by_main_ingredient[main_ingredient].extend(
                variables_by_recipe[recipe.recipe_id]
            )
        for ingredient_variables in variables_by_main_ingredient.values():
            model.Add(
                sum(ingredient_variables) <= constraints.max_main_ingredient_repeats
            )

        if constraints.weekly_budget_cents is not None:
            recipes_by_id = {recipe.recipe_id: recipe for recipe in recipes}
            total_cost = sum(
                recipes_by_id[recipe_id].estimated_cost_cents * variable
                for (_, _, recipe_id), variable in variables.items()
            )
            model.Add(total_cost <= constraints.weekly_budget_cents)

    def _build_days(
        self,
        solver: cp_model.CpSolver,
        variables: dict[tuple[int, MealSlot, str], cp_model.IntVar],
        recipes_by_id: dict[str, RecipeCandidate],
        day_count: int,
    ) -> list[DailyPlan]:
        days: list[DailyPlan] = []
        for day in range(1, day_count + 1):
            assignments: list[MealAssignment] = []
            for slot in MealSlot:
                selected_id = next(
                    recipe_id
                    for (
                        candidate_day,
                        candidate_slot,
                        recipe_id,
                    ), variable in variables.items()
                    if candidate_day == day
                    and candidate_slot is slot
                    and solver.BooleanValue(variable)
                )
                recipe = recipes_by_id[selected_id]
                assignments.append(
                    MealAssignment(
                        day=day,
                        slot=slot,
                        recipe_id=recipe.recipe_id,
                        recipe_name=recipe.name,
                        calories_kcal=recipe.calories_kcal,
                        protein_g=recipe.protein_g,
                        total_minutes=recipe.total_minutes,
                        estimated_cost_cents=recipe.estimated_cost_cents,
                    )
                )
            days.append(DailyPlan(day=day, assignments=assignments))
        return days

    def _scale(self, value: Decimal) -> int:
        return int(
            (value * self._nutrient_scale).to_integral_value(rounding=ROUND_HALF_UP)
        )

    @staticmethod
    def _map_status(status: int) -> SolveStatus:
        return {
            cp_model.OPTIMAL: SolveStatus.OPTIMAL,
            cp_model.FEASIBLE: SolveStatus.FEASIBLE,
            cp_model.INFEASIBLE: SolveStatus.INFEASIBLE,
        }.get(status, SolveStatus.UNKNOWN)
