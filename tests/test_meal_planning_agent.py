from decimal import Decimal
from pathlib import Path

import pytest

from gustobot.application.deepreason.models import AgentTask, Domain, TaskStatus
from gustobot.application.deepreason.domain_agents import (
    build_demo_registry,
    build_gustobot_registry,
)
from gustobot.application.meal_planning.agent import (
    InMemoryMealCandidateSource,
    MealPlanningAgent,
)
from gustobot.application.meal_planning.candidate_pool import (
    CandidateBoundary,
    CandidateBoundaryStatus,
)
from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.models import MealPlanConstraints, MealSlot


DATA_DIR = Path(__file__).parents[1] / "gustobot" / "data" / "meal_planning"


def seed_recipes():
    return load_seed_corpus(
        DATA_DIR / "recipes.v1.json",
        DATA_DIR / "manifest.v1.json",
    ).recipes


def constraints(**updates):
    values = {
        "days": 3,
        "daily_calories_min": 1600,
        "daily_calories_max": 1800,
        "daily_protein_min_g": Decimal("100"),
        "excluded_allergens": {"peanut"},
        "preferred_tags": {"light", "high_protein"},
        "max_meal_minutes": 30,
        "weekly_budget_cents": 30000,
        "max_recipe_repeats": 2,
        "max_main_ingredient_repeats": 7,
    }
    values.update(updates)
    return MealPlanConstraints(**values)


def task():
    return AgentTask(
        task_id="meal-planning-1",
        domain=Domain.MEAL_PLANNING,
        instruction="plan three days of light high protein meals",
    )


def verified_boundary(recipe_ids=None):
    recipe_ids = recipe_ids or [recipe.recipe_id for recipe in seed_recipes()]
    return CandidateBoundary(
        status=CandidateBoundaryStatus.VERIFIED,
        selected_recipe_ids=recipe_ids,
        evidence_ids_by_recipe={
            recipe_id: [f"ev_{recipe_id}"] for recipe_id in recipe_ids
        },
    )


@pytest.mark.asyncio
async def test_meal_planning_agent_retrieves_solves_and_verifies_a_plan():
    agent = MealPlanningAgent(InMemoryMealCandidateSource(seed_recipes()))

    handoff = await agent.execute(
        task(),
        {
            "meal_planning_request": {
                "constraints": constraints().model_dump(mode="json"),
                "retrieval_query": "light high_protein",
            }
        },
    )

    assert handoff.status is TaskStatus.SUCCESS
    assert handoff.output["status"] == "verified"
    assert handoff.output["verification"]["decision"] == "allow"
    assert len(handoff.output["meal_plan"]["days"]) == 3
    assert all(
        len(day["assignments"]) == len(MealSlot)
        for day in handoff.output["meal_plan"]["days"]
    )
    assert handoff.output["selected_recipe_ids"]
    assert handoff.evidence


@pytest.mark.asyncio
async def test_meal_planning_agent_blocks_when_slot_candidates_are_insufficient():
    breakfast_only = [
        recipe
        for recipe in seed_recipes()
        if recipe.meal_types == {MealSlot.BREAKFAST}
    ][:1]
    agent = MealPlanningAgent(InMemoryMealCandidateSource(breakfast_only))

    handoff = await agent.execute(
        task(),
        {
            "meal_planning_request": {
                "constraints": constraints().model_dump(mode="json"),
                "retrieval_query": "light high_protein",
            }
        },
    )

    assert handoff.status is TaskStatus.FAILED
    assert handoff.output["status"] == "candidate_shortage"
    assert {item["slot"] for item in handoff.output["shortages"]} >= {
        "lunch",
        "dinner",
    }
    assert handoff.output["backfill_requests"]


@pytest.mark.asyncio
async def test_meal_planning_agent_requires_and_enforces_verified_graph_boundary():
    class CapturingSolver:
        def __init__(self):
            from gustobot.application.meal_planning.solver import MealPlanSolver

            self.delegate = MealPlanSolver()
            self.recipe_ids = []

        def solve(self, plan_constraints, candidates):
            self.recipe_ids = [recipe.recipe_id for recipe in candidates]
            return self.delegate.solve(plan_constraints, candidates)

    solver = CapturingSolver()
    agent = MealPlanningAgent(
        InMemoryMealCandidateSource(seed_recipes()),
        solver=solver,
    )
    allowed = [recipe.recipe_id for recipe in seed_recipes() if recipe.recipe_id != "r104"]

    handoff = await agent.execute(
        task(),
        {
            "meal_planning_request": {
                "constraints": constraints().model_dump(mode="json"),
                "retrieval_query": "light high_protein",
                "graph_constraints_required": True,
                "candidate_boundary": verified_boundary(allowed).model_dump(mode="json"),
            }
        },
    )

    assert handoff.status is TaskStatus.SUCCESS
    assert "r104" not in solver.recipe_ids
    assert set(solver.recipe_ids) <= set(allowed)
    assert all(
        assignment["evidence_ids"] == [f"ev_{assignment['recipe_id']}"]
        for day in handoff.output["meal_plan"]["days"]
        for assignment in day["assignments"]
    )


@pytest.mark.asyncio
async def test_missing_required_graph_boundary_fails_before_retrieval():
    class NoCallSource:
        def search(self, request):
            raise AssertionError("retrieval must not run without a required boundary")

    handoff = await MealPlanningAgent(NoCallSource()).execute(
        task(),
        {
            "meal_planning_request": {
                "constraints": constraints().model_dump(mode="json"),
                "retrieval_query": "without peanut",
                "graph_constraints_required": True,
            }
        },
    )

    assert handoff.status is TaskStatus.FAILED
    assert handoff.output["status"] == "graph_boundary_invalid"


@pytest.mark.asyncio
async def test_agent_performs_one_targeted_backfill_before_solving():
    all_recipes = seed_recipes()

    class StagedSource:
        def __init__(self):
            self.calls = []
            self.slot_calls = {slot: 0 for slot in MealSlot}

        def search(self, request):
            self.calls.append(request)
            self.slot_calls[request.slot] += 1
            matches = [
                recipe for recipe in all_recipes if request.slot in recipe.meal_types
            ]
            if request.slot is MealSlot.BREAKFAST and self.slot_calls[request.slot] == 1:
                return []
            return matches[: request.top_k]

    source = StagedSource()
    agent = MealPlanningAgent(source)
    one_day = constraints(days=1, max_recipe_repeats=1)
    boundary = verified_boundary()

    handoff = await agent.execute(
        task(),
        {
            "meal_planning_request": {
                "constraints": one_day.model_dump(mode="json"),
                "retrieval_query": "light high_protein",
                "candidate_boundary": boundary.model_dump(mode="json"),
            }
        },
    )

    assert handoff.status is TaskStatus.SUCCESS
    assert source.slot_calls[MealSlot.BREAKFAST] == 2
    assert handoff.output["backfill_attempts"]


@pytest.mark.asyncio
async def test_infeasible_solver_result_is_not_released():
    agent = MealPlanningAgent(InMemoryMealCandidateSource(seed_recipes()))
    impossible = constraints(
        days=1,
        daily_calories_min=1,
        daily_calories_max=2,
        daily_protein_min_g=Decimal("0"),
        max_recipe_repeats=1,
    )

    handoff = await agent.execute(
        task(),
        {
            "meal_planning_request": {
                "constraints": impossible.model_dump(mode="json"),
                "retrieval_query": "meal",
            }
        },
    )

    assert handoff.status is TaskStatus.FAILED
    assert handoff.output["status"] == "verification_blocked"
    assert handoff.output["solve_status"] == "infeasible"


def test_production_registry_registers_the_real_meal_planning_agent():
    registry = build_gustobot_registry()

    agent = registry.get(Domain.MEAL_PLANNING)

    assert isinstance(agent, MealPlanningAgent)


def test_demo_registry_also_registers_the_real_local_meal_planning_agent():
    registry = build_demo_registry()

    agent = registry.get(Domain.MEAL_PLANNING)

    assert isinstance(agent, MealPlanningAgent)
