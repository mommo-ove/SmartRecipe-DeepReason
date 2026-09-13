from decimal import Decimal

from gustobot.application.meal_planning.models import MealPlanDraft
from gustobot.application.meal_planning.routing import (
    BusinessIntent,
    ExecutionPolicy,
    IntentRouteOutput,
    MealIntentRouter,
    MealRouteDraft,
    RecipeResponseMode,
    decide_recipe_response_mode,
    decide_execution_policy,
)


def complete_constraints() -> MealPlanDraft:
    return MealPlanDraft(
        days=7,
        daily_calories_min=1600,
        daily_calories_max=1800,
        daily_protein_min_g=Decimal("100"),
        excluded_allergens={"peanut"},
        max_meal_minutes=30,
    )


def test_recipe_lookup_uses_direct_short_path():
    route = MealRouteDraft(
        business_intent=BusinessIntent.RECIPE_LOOKUP,
        confidence=0.92,
    )

    assert decide_execution_policy(route) is ExecutionPolicy.DIRECT


def test_complete_multi_day_plan_uses_workflow():
    route = MealRouteDraft(
        business_intent=BusinessIntent.MEAL_PLAN,
        confidence=0.9,
        constraints=complete_constraints(),
    )

    assert decide_execution_policy(route) is ExecutionPolicy.WORKFLOW


def test_recipe_request_requiring_nutrition_uses_workflow():
    route = MealRouteDraft(
        business_intent=BusinessIntent.RECIPE_LOOKUP,
        confidence=0.92,
        requested_capabilities={"recipe_retrieval", "nutrition_query"},
    )

    assert decide_execution_policy(route) is ExecutionPolicy.WORKFLOW


def test_missing_critical_meal_plan_fields_use_clarification():
    route = MealRouteDraft(
        business_intent=BusinessIntent.MEAL_PLAN,
        confidence=0.91,
        constraints=MealPlanDraft(days=7),
    )

    assert decide_execution_policy(route) is ExecutionPolicy.CLARIFY


def test_low_confidence_never_forces_a_domain_route():
    route = MealRouteDraft(
        business_intent=BusinessIntent.RECIPE_LOOKUP,
        confidence=0.45,
    )

    assert decide_execution_policy(route) is ExecutionPolicy.CLARIFY


def test_plan_edit_uses_workflow_because_it_requires_checkpoint_state():
    route = MealRouteDraft(
        business_intent=BusinessIntent.PLAN_EDIT,
        confidence=0.88,
    )

    assert decide_execution_policy(route) is ExecutionPolicy.WORKFLOW


class FailingChain:
    def invoke(self, messages):
        raise RuntimeError("provider timeout")


class FailingModel:
    def with_structured_output(self, schema):
        assert schema is IntentRouteOutput
        return FailingChain()


def test_invalid_llm_route_falls_back_to_keyword_intent_without_image_route():
    route = MealIntentRouter(FailingModel()).route(
        "把周二晚餐换成别的，其他不变"
    )

    assert route.business_intent is BusinessIntent.PLAN_EDIT
    assert decide_execution_policy(route) is ExecutionPolicy.WORKFLOW


def test_heuristic_fallback_merges_recipe_qa_and_search_into_recipe_lookup():
    router = MealIntentRouter(model=None)

    how_to = router.route("西红柿炒蛋怎么做？")
    search = router.route("推荐一道鸡胸肉菜谱")

    assert how_to.business_intent is BusinessIntent.RECIPE_LOOKUP
    assert search.business_intent is BusinessIntent.RECIPE_LOOKUP


def test_recipe_lookup_uses_detail_mode_for_how_to_questions():
    strategy = decide_recipe_response_mode("西红柿炒蛋需要哪些食材，怎么做？")

    assert strategy.mode is RecipeResponseMode.DETAIL
    assert strategy.result_limit == 1


def test_recipe_lookup_uses_list_mode_and_extracts_requested_count():
    strategy = decide_recipe_response_mode("推荐三道高蛋白鸡胸肉菜谱")

    assert strategy.mode is RecipeResponseMode.LIST
    assert strategy.result_limit == 3


def test_fallback_distinguishes_general_chat_from_ambiguous_food_request():
    router = MealIntentRouter(model=None)

    assert router.route("你好，你能做什么？").business_intent is BusinessIntent.GENERAL
    ambiguous = router.route("我应该怎么吃？")

    assert ambiguous.business_intent is BusinessIntent.UNKNOWN
    assert decide_execution_policy(ambiguous) is ExecutionPolicy.CLARIFY


def test_fallback_recognizes_recipe_recommendation_synonyms():
    route = MealIntentRouter(model=None).route("推荐三道高蛋白的鸡胸肉菜")

    assert route.business_intent is BusinessIntent.RECIPE_LOOKUP
    assert decide_execution_policy(route) is ExecutionPolicy.DIRECT


def test_fallback_extracts_only_explicit_complete_planning_constraints():
    route = MealIntentRouter(model=None).route(
        "规划3天餐单，每天1600到1800千卡、蛋白质100克以上、"
        "不含花生、每餐30分钟内"
    )

    assert route.business_intent is BusinessIntent.MEAL_PLAN
    assert route.constraints.days == 3
    assert route.constraints.daily_calories_min == 1600
    assert route.constraints.daily_calories_max == 1800
    assert route.constraints.daily_protein_min_g == 100
    assert route.constraints.excluded_allergens == {"peanut"}
    assert route.constraints.max_meal_minutes == 30
    assert decide_execution_policy(route) is ExecutionPolicy.WORKFLOW


class ReturningChain:
    def invoke(self, messages):
        return {
            "business_intent": "recipe_lookup",
            "confidence": 0.91,
            "normalized_query": "recommend two dishes and calculate calories",
            "requested_capabilities": ["recipe_retrieval", "nutrition_query"],
        }


class RecordingModel:
    def __init__(self):
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return ReturningChain()


def test_llm_router_only_outputs_intent_and_capabilities_not_meal_constraints():
    model = RecordingModel()

    route = MealIntentRouter(model).route("recommend two dishes and calculate calories")

    assert model.schema is IntentRouteOutput
    assert route.requested_capabilities == {"recipe_retrieval", "nutrition_query"}
    assert route.constraints == MealPlanDraft()
    assert decide_execution_policy(route) is ExecutionPolicy.WORKFLOW


class RawMessage:
    usage_metadata = {"input_tokens": 123, "output_tokens": 17}


class RawReturningChain:
    def invoke(self, messages):
        return {
            "parsed": IntentRouteOutput(
                business_intent=BusinessIntent.DATA_QUERY,
                confidence=0.9,
                requested_capabilities={"nutrition_query"},
            ),
            "raw": RawMessage(),
            "parsing_error": None,
        }


class RawRecordingModel:
    def with_structured_output(self, schema, *, include_raw=False):
        assert schema is IntentRouteOutput
        assert include_raw is True
        return RawReturningChain()


def test_llm_router_records_raw_token_usage_for_experiments():
    router = MealIntentRouter(RawRecordingModel())

    route = router.route("calculate average calories")

    assert route.business_intent is BusinessIntent.DATA_QUERY
    assert router.last_fallback_used is False
    assert router.last_input_tokens == 123
    assert router.last_output_tokens == 17
