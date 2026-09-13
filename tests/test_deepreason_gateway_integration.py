import pytest

from gustobot.application.deepreason.domain_agents import (
    DomainAgentRegistry,
    FunctionDomainAgent,
)
from gustobot.application.deepreason.models import Domain
from gustobot.application.deepreason.orchestrator import DeepReasonOrchestrator
from gustobot.application.meal_planning.gateway import MealGatewayDecision
from gustobot.application.meal_planning.models import MealPlanConstraints, MealPlanDraft
from gustobot.application.meal_planning.planning_entry import (
    MealPlanningEntryResult,
    MealPlanningEntryStatus,
)
from gustobot.application.meal_planning.routing import (
    BusinessIntent,
    ExecutionPolicy,
    MealRouteDraft,
)


class StubGateway:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def decide(self, query):
        self.calls.append(query)
        return self.decision


def decision(intent, policy, *, target_domain=None, questions=None):
    return MealGatewayDecision(
        route=MealRouteDraft(business_intent=intent, confidence=0.9),
        policy=policy,
        reason_code="test_decision",
        target_domain=target_domain,
        questions=questions or [],
    )


def ready_meal_plan_decision():
    constraints = MealPlanConstraints(
        days=3,
        daily_calories_min=1600,
        daily_calories_max=1800,
        daily_protein_min_g=100,
        excluded_allergens={"peanut"},
        max_meal_minutes=30,
    )
    return MealGatewayDecision(
        route=MealRouteDraft(
            business_intent=BusinessIntent.MEAL_PLAN,
            confidence=0.93,
        ),
        policy=ExecutionPolicy.WORKFLOW,
        reason_code="meal_plan_constraints_ready",
        planning_result=MealPlanningEntryResult(
            status=MealPlanningEntryStatus.READY,
            draft=MealPlanDraft(**constraints.model_dump()),
            constraints=constraints,
            retrieval_query="light high_protein",
        ),
    )


@pytest.mark.asyncio
async def test_clarify_short_circuits_before_coordinator_and_agents(tmp_path):
    registry = DomainAgentRegistry()
    gateway = StubGateway(
        decision(
            BusinessIntent.UNKNOWN,
            ExecutionPolicy.CLARIFY,
            questions=["Please specify whether to search or plan meals."],
        )
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        ledger_path=tmp_path / "ledger.jsonl",
        request_gateway=gateway,
    )

    result = await orchestrator.run("ambiguous", session_id="s1")

    assert result.answer == "Please specify whether to search or plan meals."
    assert result.plan.tasks == []
    assert result.handoffs == []
    assert [event["kind"] for event in result.events] == [
        "run_started",
        "route_decided",
        "clarification_requested",
        "run_completed",
    ]


@pytest.mark.asyncio
async def test_direct_recipe_route_skips_coordinator_but_keeps_evidence_gate(tmp_path):
    calls = []

    async def recipe_handler(task, context):
        calls.append(task)
        return {
            "answer": "recipe answer",
            "documents": [
                {
                    "content": "recipe evidence",
                    "metadata": {"source": "recipe-1", "recipe_id": "r1"},
                }
            ],
        }

    registry = DomainAgentRegistry()
    registry.register(
        FunctionDomainAgent("recipe_agent", Domain.RECIPE, recipe_handler)
    )
    gateway = StubGateway(
        decision(
            BusinessIntent.RECIPE_LOOKUP,
            ExecutionPolicy.DIRECT,
            target_domain="recipe",
        )
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        ledger_path=tmp_path / "ledger.jsonl",
        request_gateway=gateway,
    )

    result = await orchestrator.run("how to cook it", session_id="s1")

    assert result.answer == "recipe answer"
    assert len(calls) == 1
    assert result.plan.review_status == "bypassed_for_direct_route"
    assert result.gate.status.value == "allow"
    assert all(event["kind"] != "coordinator_completed" for event in result.events)


@pytest.mark.asyncio
async def test_workflow_route_preserves_existing_deepreason_graph(tmp_path):
    async def recipe_handler(task, context):
        return {"answer": "ok", "documents": [{"content": "evidence"}]}

    registry = DomainAgentRegistry()
    registry.register(
        FunctionDomainAgent("recipe_agent", Domain.RECIPE, recipe_handler)
    )
    gateway = StubGateway(
        decision(BusinessIntent.MEAL_PLAN, ExecutionPolicy.WORKFLOW)
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        ledger_path=tmp_path / "ledger.jsonl",
        request_gateway=gateway,
    )

    result = await orchestrator.run("plan a weekly recipe menu", session_id="s1")

    assert any(event["kind"] == "coordinator_completed" for event in result.events)
    assert result.events[1]["kind"] == "route_decided"


@pytest.mark.asyncio
async def test_ready_meal_plan_is_converted_to_a_meal_planning_agent_task(tmp_path):
    captured_context = {}

    async def planning_handler(task, context):
        captured_context.update(context)
        return {"answer": "verified meal plan", "documents": [{"content": "plan evidence"}]}

    registry = DomainAgentRegistry()
    registry.register(
        FunctionDomainAgent(
            "meal_planning_agent",
            Domain.MEAL_PLANNING,
            planning_handler,
        )
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        ledger_path=tmp_path / "ledger.jsonl",
        request_gateway=StubGateway(ready_meal_plan_decision()),
    )

    result = await orchestrator.run("plan meals", session_id="s1")

    assert [item.domain for item in result.plan.tasks] == [Domain.MEAL_PLANNING]
    assert captured_context["meal_planning_request"]["constraints"]["days"] == 3
    assert result.handoffs[0].agent == "meal_planning_agent"
