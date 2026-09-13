from decimal import Decimal

from gustobot.application.meal_planning.models import MealPlanDraft
from gustobot.application.meal_planning.planning_entry import (
    MealPlanningEntryResult,
    MealPlanningEntryStatus,
)
from gustobot.application.meal_planning.routing import (
    BusinessIntent,
    ExecutionPolicy,
    MealRouteDraft,
)
from gustobot.application.meal_planning.gateway import MealRequestGateway


class StubRouter:
    def __init__(self, route):
        self.route_result = route
        self.calls = []

    def route(self, query):
        self.calls.append(query)
        return self.route_result


class StubPlanningEntry:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def handle(self, query, *, previous_draft=None):
        self.calls.append((query, previous_draft))
        return self.result


def complete_draft():
    return MealPlanDraft(
        days=3,
        daily_calories_min=1600,
        daily_calories_max=1800,
        daily_protein_min_g=Decimal("100"),
        excluded_allergens={"peanut"},
        max_meal_minutes=30,
    )


def test_direct_recipe_lookup_does_not_call_constraint_extractor():
    router = StubRouter(
        MealRouteDraft(
            business_intent=BusinessIntent.RECIPE_LOOKUP,
            confidence=0.94,
            requested_capabilities={"recipe_retrieval"},
        )
    )
    entry = StubPlanningEntry(AssertionError("must not be called"))

    decision = MealRequestGateway(router, entry).decide("番茄炒蛋怎么做？")

    assert decision.policy is ExecutionPolicy.DIRECT
    assert decision.target_domain == "recipe"
    assert entry.calls == []


def test_multiple_capabilities_enter_workflow_instead_of_direct_dispatch():
    router = StubRouter(
        MealRouteDraft(
            business_intent=BusinessIntent.RECIPE_LOOKUP,
            confidence=0.91,
            requested_capabilities={"recipe_retrieval", "nutrition_query"},
        )
    )

    decision = MealRequestGateway(router, None).decide("推荐两道菜并计算总热量")

    assert decision.policy is ExecutionPolicy.WORKFLOW
    assert decision.reason_code == "multiple_capabilities_require_coordination"


def test_meal_plan_alone_invokes_constraint_extraction_and_clarifies_missing_fields():
    router = StubRouter(
        MealRouteDraft(
            business_intent=BusinessIntent.MEAL_PLAN,
            confidence=0.93,
        )
    )
    entry = StubPlanningEntry(
        MealPlanningEntryResult(
            status=MealPlanningEntryStatus.CLARIFY,
            draft=MealPlanDraft(days=7),
            missing_fields=["daily_calories_range", "allergen_status"],
            questions=["每日目标热量是多少？", "需要排除哪些过敏食材？"],
        )
    )

    decision = MealRequestGateway(router, entry).decide("帮我规划一周餐单")

    assert decision.policy is ExecutionPolicy.CLARIFY
    assert decision.questions == ["每日目标热量是多少？", "需要排除哪些过敏食材？"]
    assert len(entry.calls) == 1


def test_meal_plan_capabilities_do_not_bypass_constraint_extraction():
    router = StubRouter(
        MealRouteDraft(
            business_intent=BusinessIntent.MEAL_PLAN,
            confidence=0.93,
            requested_capabilities={
                "recipe_retrieval",
                "nutrition_query",
                "constraint_solving",
            },
        )
    )
    entry = StubPlanningEntry(
        MealPlanningEntryResult(
            status=MealPlanningEntryStatus.CLARIFY,
            draft=MealPlanDraft(days=7),
            missing_fields=["allergen_status"],
            questions=["需要排除哪些过敏食材？"],
        )
    )

    decision = MealRequestGateway(router, entry).decide("规划一周餐单")

    assert decision.policy is ExecutionPolicy.CLARIFY
    assert len(entry.calls) == 1


def test_explicit_fallback_constraints_can_run_without_an_llm_extractor():
    draft = complete_draft()
    router = StubRouter(
        MealRouteDraft(
            business_intent=BusinessIntent.MEAL_PLAN,
            confidence=0.82,
            constraints=draft,
        )
    )

    decision = MealRequestGateway(router, None).decide("explicit complete plan")

    assert decision.policy is ExecutionPolicy.WORKFLOW
    assert decision.planning_result is not None
    assert decision.planning_result.constraints is not None
    assert decision.planning_result.constraints.days == 3
    assert decision.reason_code == "meal_plan_constraints_ready_from_fallback"


def test_complete_meal_plan_constraints_enter_workflow():
    draft = complete_draft()
    router = StubRouter(
        MealRouteDraft(
            business_intent=BusinessIntent.MEAL_PLAN,
            confidence=0.93,
        )
    )
    entry = StubPlanningEntry(
        MealPlanningEntryResult(
            status=MealPlanningEntryStatus.READY,
            draft=draft,
            constraints={
                **draft.model_dump(),
                "max_recipe_repeats": 2,
                "max_main_ingredient_repeats": 4,
            },
            retrieval_query="high_protein light",
        )
    )

    decision = MealRequestGateway(router, entry).decide("完整约束的三天餐单")

    assert decision.policy is ExecutionPolicy.WORKFLOW
    assert decision.planning_result is not None
    assert decision.reason_code == "meal_plan_constraints_ready"


def test_low_confidence_route_clarifies_without_calling_more_models():
    router = StubRouter(
        MealRouteDraft(
            business_intent=BusinessIntent.RECIPE_LOOKUP,
            confidence=0.42,
        )
    )
    entry = StubPlanningEntry(AssertionError("must not be called"))

    decision = MealRequestGateway(router, entry).decide("这个怎么弄？")

    assert decision.policy is ExecutionPolicy.CLARIFY
    assert decision.reason_code == "low_confidence_or_unknown_intent"
    assert entry.calls == []
