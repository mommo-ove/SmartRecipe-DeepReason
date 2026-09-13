from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from .planning_entry import (
    MealPlanningEntryResult,
    MealPlanningEntryStatus,
)
from .clarification import build_clarification_questions, find_missing_fields
from .models import MealPlanConstraints
from .semantic_validation import validate_draft_semantics
from .routing import (
    BusinessIntent,
    ExecutionPolicy,
    MealRouteDraft,
    decide_execution_policy,
)


class IntentRouter(Protocol):
    def route(self, query: str) -> MealRouteDraft: ...


class PlanningEntry(Protocol):
    def handle(self, user_query: str, *, previous_draft=None) -> MealPlanningEntryResult: ...


class MealGatewayDecision(BaseModel):
    route: MealRouteDraft
    policy: ExecutionPolicy
    reason_code: str
    target_domain: str | None = None
    questions: list[str] = Field(default_factory=list)
    planning_result: MealPlanningEntryResult | None = None


class MealRequestGateway:
    """Separate language understanding from deterministic execution control."""

    def __init__(
        self,
        router: IntentRouter,
        planning_entry: PlanningEntry | None,
    ) -> None:
        self._router = router
        self._planning_entry = planning_entry

    def decide(self, query: str) -> MealGatewayDecision:
        route = self._router.route(query)

        if route.confidence < 0.7 or route.business_intent is BusinessIntent.UNKNOWN:
            return MealGatewayDecision(
                route=route,
                policy=ExecutionPolicy.CLARIFY,
                reason_code="low_confidence_or_unknown_intent",
                questions=["你希望查询一道菜、生成餐单，还是修改已有餐单？"],
            )

        if route.business_intent is BusinessIntent.MEAL_PLAN:
            if self._planning_entry is None:
                return self._from_explicit_fallback(route)
            planning_result = self._planning_entry.handle(query)
            if planning_result.status is MealPlanningEntryStatus.CLARIFY:
                return MealGatewayDecision(
                    route=route,
                    policy=ExecutionPolicy.CLARIFY,
                    reason_code="meal_plan_constraints_incomplete",
                    questions=planning_result.questions,
                    planning_result=planning_result,
                )
            if planning_result.status is MealPlanningEntryStatus.RETRY:
                return MealGatewayDecision(
                    route=route,
                    policy=ExecutionPolicy.CLARIFY,
                    reason_code="meal_plan_constraint_extraction_failed",
                    questions=["我没有可靠识别餐单约束，请重新描述天数、热量和过敏信息。"],
                    planning_result=planning_result,
                )
            return MealGatewayDecision(
                route=route,
                policy=ExecutionPolicy.WORKFLOW,
                reason_code="meal_plan_constraints_ready",
                planning_result=planning_result,
            )

        if len(route.requested_capabilities) > 1:
            return MealGatewayDecision(
                route=route,
                policy=ExecutionPolicy.WORKFLOW,
                reason_code="multiple_capabilities_require_coordination",
            )

        policy = decide_execution_policy(route)
        domains = {
            BusinessIntent.RECIPE_LOOKUP: "recipe",
            BusinessIntent.DATA_QUERY: "analytics",
            BusinessIntent.PLAN_EDIT: None,
            BusinessIntent.GENERAL: "general",
        }
        return MealGatewayDecision(
            route=route,
            policy=policy,
            reason_code=(
                "single_capability_direct_dispatch"
                if policy is ExecutionPolicy.DIRECT
                else "stateful_plan_edit_requires_workflow"
            ),
            target_domain=domains.get(route.business_intent),
        )

    @staticmethod
    def _from_explicit_fallback(route: MealRouteDraft) -> MealGatewayDecision:
        missing = find_missing_fields(route.constraints)
        semantic_issues = validate_draft_semantics(route.constraints)
        if missing or semantic_issues:
            return MealGatewayDecision(
                route=route,
                policy=ExecutionPolicy.CLARIFY,
                reason_code="meal_plan_constraints_incomplete",
                questions=[
                    *build_clarification_questions(missing),
                    *(issue.question for issue in semantic_issues),
                ],
            )
        draft = route.constraints
        constraints = MealPlanConstraints(
            days=draft.days,
            daily_calories_min=draft.daily_calories_min,
            daily_calories_max=draft.daily_calories_max,
            daily_protein_min_g=draft.daily_protein_min_g,
            excluded_allergens=draft.excluded_allergens,
            max_meal_minutes=draft.max_meal_minutes,
            weekly_budget_cents=draft.weekly_budget_cents,
            preferred_tags=draft.preferred_tags,
        )
        query_terms = sorted(draft.requested_ingredients | draft.preferred_tags)
        planning_result = MealPlanningEntryResult(
            status=MealPlanningEntryStatus.READY,
            draft=draft,
            constraints=constraints,
            retrieval_query=" ".join(query_terms),
        )
        return MealGatewayDecision(
            route=route,
            policy=ExecutionPolicy.WORKFLOW,
            reason_code="meal_plan_constraints_ready_from_fallback",
            planning_result=planning_result,
        )
