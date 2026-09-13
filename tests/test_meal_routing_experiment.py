import pytest

from gustobot.application.meal_planning.routing import (
    BusinessIntent,
    ExecutionPolicy,
    RouterPromptVersion,
    build_router_system_prompt,
)
from gustobot.application.meal_planning.routing_benchmark import RouteBenchmarkCase
from gustobot.application.meal_planning.routing_experiment import (
    RouteObservation,
    summarize_route_experiment,
)


def cases():
    return [
        RouteBenchmarkCase(
            case_id="direct",
            query="how to cook tofu",
            expected_intent=BusinessIntent.RECIPE_LOOKUP,
            expected_policy=ExecutionPolicy.DIRECT,
            split="frozen",
            review_status="reviewed",
        ),
        RouteBenchmarkCase(
            case_id="clarify",
            query="what should I eat",
            expected_intent=BusinessIntent.UNKNOWN,
            expected_policy=ExecutionPolicy.CLARIFY,
            split="frozen",
            review_status="reviewed",
        ),
        RouteBenchmarkCase(
            case_id="workflow",
            query="recommend two meals and total calories",
            expected_intent=BusinessIntent.RECIPE_LOOKUP,
            expected_policy=ExecutionPolicy.WORKFLOW,
            split="frozen",
            review_status="reviewed",
        ),
    ]


def test_prompt_rounds_are_versioned_and_r3_requires_mined_errors():
    zero_shot = build_router_system_prompt(RouterPromptVersion.R0_ZERO_SHOT)
    round_one = build_router_system_prompt(RouterPromptVersion.R1_FEW_SHOT)
    round_two = build_router_system_prompt(RouterPromptVersion.R2_HARD_NEGATIVE)

    assert "Examples" not in zero_shot
    assert "Examples" in round_one
    assert "推荐三道鸡胸肉菜" in round_two
    assert "制定三天餐单" in round_two

    try:
        build_router_system_prompt(RouterPromptVersion.R3_MINED_ERRORS)
    except ValueError as error:
        assert "mined hard-negative examples" in str(error)
    else:
        raise AssertionError("R3 must not run without previous-round errors")


def test_experiment_reports_quality_latency_fallback_tokens_and_cost():
    observations = [
        RouteObservation(
            case_id="direct",
            predicted_intent=BusinessIntent.RECIPE_LOOKUP,
            predicted_policy=ExecutionPolicy.DIRECT,
            confidence=0.91,
            latency_ms=100,
            fallback_used=False,
            input_tokens=100,
            output_tokens=20,
        ),
        RouteObservation(
            case_id="clarify",
            predicted_intent=BusinessIntent.GENERAL,
            predicted_policy=ExecutionPolicy.DIRECT,
            confidence=0.95,
            latency_ms=300,
            fallback_used=True,
            input_tokens=200,
            output_tokens=30,
        ),
        RouteObservation(
            case_id="workflow",
            predicted_intent=BusinessIntent.RECIPE_LOOKUP,
            predicted_policy=ExecutionPolicy.DIRECT,
            confidence=0.88,
            latency_ms=200,
            fallback_used=False,
            input_tokens=300,
            output_tokens=50,
        ),
    ]

    result = summarize_route_experiment(
        cases(),
        observations,
        router_name="llm",
        prompt_version="r0_zero_shot",
        model_name="qwen-plus",
        input_cost_per_million=0.8,
        output_cost_per_million=2.0,
    )

    assert result.intent_accuracy == 2 / 3
    assert result.policy_error_rate == pytest.approx(2 / 3)
    assert result.unsafe_route_rate == pytest.approx(2 / 3)
    assert result.clarification_rate == 0
    assert result.fallback_rate == pytest.approx(1 / 3)
    assert result.average_latency_ms == 200
    assert result.p95_latency_ms == 300
    assert result.input_tokens == 600
    assert result.output_tokens == 100
    assert result.estimated_cost == 0.00068
