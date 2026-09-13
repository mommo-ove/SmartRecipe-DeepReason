from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import Protocol

from gustobot.application.deepreason.models import (
    AgentTask,
    Domain,
    EvidenceItem,
    Handoff,
    TaskStatus,
)

from .candidate_pool import (
    CandidateBoundary,
    CandidateBoundaryStatus,
    SlotRetrievalRequest,
    apply_candidate_boundary,
    assess_candidate_pool,
    build_backfill_retrieval_requests,
    build_slot_retrieval_requests,
)
from .models import MealPlanConstraints, RecipeCandidate
from .solver import MealPlanSolver
from .verification import VerificationDecision, verify_solved_plan


class MealCandidateSource(Protocol):
    def search(self, request: SlotRetrievalRequest) -> list[RecipeCandidate]: ...


class InMemoryMealCandidateSource:
    """Deterministic adapter used by the versioned local planning corpus."""

    def __init__(self, recipes: list[RecipeCandidate]) -> None:
        self._recipes = list(recipes)

    def search(self, request: SlotRetrievalRequest) -> list[RecipeCandidate]:
        allergens = set(request.metadata_filters.get("excluded_allergens", []))
        max_minutes = request.metadata_filters.get("max_meal_minutes")
        terms = set(request.semantic_query.casefold().replace("_", " ").split())
        candidates = [
            recipe
            for recipe in self._recipes
            if request.slot in recipe.meal_types
            and not recipe.allergens & allergens
            and (
                max_minutes is None
                or recipe.total_minutes is not None
                and recipe.total_minutes <= max_minutes
            )
        ]

        def score(recipe: RecipeCandidate) -> tuple[int, str]:
            searchable = " ".join(
                [
                    recipe.name,
                    *recipe.preference_tags,
                    *(item.ingredient_id for item in recipe.ingredients_normalized),
                ]
            ).casefold().replace("_", " ")
            return (-sum(term in searchable for term in terms), recipe.recipe_id)

        return sorted(candidates, key=score)[: request.top_k]


class MealPlanningAgent:
    name = "meal_planning_agent"
    domain = Domain.MEAL_PLANNING

    def __init__(
        self,
        candidate_source: MealCandidateSource,
        *,
        solver: MealPlanSolver | None = None,
    ) -> None:
        self._candidate_source = candidate_source
        self._solver = solver or MealPlanSolver()

    async def execute(self, task: AgentTask, context: dict) -> Handoff:
        started = perf_counter()
        request = context.get("meal_planning_request") or {}
        try:
            constraints = MealPlanConstraints.model_validate(request["constraints"])
        except Exception:
            return self._failed(
                task,
                started,
                status="invalid_constraints",
                error="validated meal planning constraints are required",
            )

        semantic_query = str(request.get("retrieval_query") or "").strip()
        try:
            boundary_payload = request.get("candidate_boundary")
            boundary = (
                CandidateBoundary.model_validate(boundary_payload)
                if boundary_payload is not None
                else None
            )
            if request.get("graph_constraints_required") and boundary is None:
                raise ValueError("required graph boundary is missing")
            if boundary is not None and boundary.status is CandidateBoundaryStatus.FAILED:
                raise ValueError("graph boundary is not verified")
        except Exception as error:
            return self._failed(
                task,
                started,
                status="graph_boundary_invalid",
                error=str(error),
            )

        retrieval_requests = build_slot_retrieval_requests(
            constraints,
            semantic_query=semantic_query,
        )
        recipes_by_id: dict[str, RecipeCandidate] = {}

        def retrieve(retrieval_request: SlotRetrievalRequest) -> None:
            if boundary is not None and boundary.status is CandidateBoundaryStatus.EMPTY:
                return
            retrieved = self._candidate_source.search(retrieval_request)
            for recipe in apply_candidate_boundary(retrieved, boundary):
                recipes_by_id.setdefault(recipe.recipe_id, recipe)

        for retrieval_request in retrieval_requests:
            retrieve(retrieval_request)
        recipes = list(recipes_by_id.values())

        pool = assess_candidate_pool(
            constraints,
            recipes,
            semantic_query=semantic_query,
        )
        backfill_attempts = build_backfill_retrieval_requests(pool)
        if not pool.ready:
            for backfill_request in backfill_attempts:
                retrieve(backfill_request)
            recipes = list(recipes_by_id.values())
            pool = assess_candidate_pool(
                constraints,
                recipes,
                semantic_query=semantic_query,
            )
        if not pool.ready:
            return self._failed(
                task,
                started,
                status="candidate_shortage",
                error="candidate pool does not cover every meal slot",
                output={
                    "shortages": [item.model_dump(mode="json") for item in pool.shortages],
                    "backfill_requests": [
                        item.model_dump(mode="json") for item in pool.backfill_requests
                    ],
                    "rejected_counts": pool.rejected_counts,
                    "backfill_attempts": [
                        item.model_dump(mode="json") for item in backfill_attempts
                    ],
                },
            )

        eligible = [
            recipe
            for recipe in recipes
            if recipe.recipe_id in set(pool.eligible_recipe_ids)
        ]
        solved = await asyncio.to_thread(self._solver.solve, constraints, eligible)
        if solved.plan is not None and boundary is not None:
            for day in solved.plan.days:
                for assignment in day.assignments:
                    assignment.evidence_ids = list(
                        boundary.evidence_ids_by_recipe.get(assignment.recipe_id, [])
                    )
        verified = verify_solved_plan(constraints, eligible, solved)
        if verified.decision is not VerificationDecision.ALLOW or solved.plan is None:
            return self._failed(
                task,
                started,
                status="verification_blocked",
                error=f"meal plan verifier returned {verified.decision.value}",
                output={
                    "solve_status": solved.status.value,
                    "verification": verified.model_dump(mode="json"),
                },
            )

        selected_ids = list(
            dict.fromkeys(
                assignment.recipe_id
                for day in solved.plan.days
                for assignment in day.assignments
            )
        )
        evidence = self._build_evidence(task, eligible, selected_ids)
        output = {
            "status": "verified",
            "meal_plan": solved.plan.model_dump(mode="json"),
            "selected_recipe_ids": selected_ids,
            "candidate_pool": pool.model_dump(mode="json"),
            "retrieval_requests": [
                item.model_dump(mode="json") for item in retrieval_requests
            ],
            "backfill_attempts": [
                item.model_dump(mode="json") for item in backfill_attempts
            ],
            "candidate_boundary": (
                boundary.model_dump(mode="json") if boundary is not None else None
            ),
            "solver": {
                "status": solved.status.value,
                "wall_time_ms": solved.solver_wall_time_ms,
                "objective_values": solved.objective_values,
            },
            "verification": verified.model_dump(mode="json"),
        }
        return Handoff(
            task_id=task.task_id,
            agent=self.name,
            domain=self.domain,
            status=TaskStatus.SUCCESS,
            summary=self._summary(solved.plan.model_dump(mode="json")),
            output=output,
            evidence=evidence,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )

    def _failed(
        self,
        task: AgentTask,
        started: float,
        *,
        status: str,
        error: str,
        output: dict | None = None,
    ) -> Handoff:
        return Handoff(
            task_id=task.task_id,
            agent=self.name,
            domain=self.domain,
            status=TaskStatus.FAILED,
            output={"status": status, **(output or {})},
            error=error,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )

    @staticmethod
    def _build_evidence(
        task: AgentTask,
        recipes: list[RecipeCandidate],
        selected_ids: list[str],
    ) -> list[EvidenceItem]:
        recipes_by_id = {recipe.recipe_id: recipe for recipe in recipes}
        return [
            EvidenceItem.create(
                task_id=task.task_id,
                source_type="meal_plan_recipe_fact",
                source=recipes_by_id[recipe_id].source_refs[0],
                content=json.dumps(
                    recipes_by_id[recipe_id].model_dump(mode="json"),
                    ensure_ascii=False,
                ),
                metadata={"recipe_id": recipe_id},
            )
            for recipe_id in selected_ids
        ]

    @staticmethod
    def _summary(plan: dict) -> str:
        lines = []
        for day in plan["days"]:
            meals = "、".join(
                f"{item['slot']}：{item['recipe_name']}"
                for item in day["assignments"]
            )
            lines.append(f"第{day['day']}天：{meals}")
        return "已生成并验证餐单。\n" + "\n".join(lines)
