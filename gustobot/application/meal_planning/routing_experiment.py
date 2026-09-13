from __future__ import annotations

from math import ceil

from pydantic import BaseModel, Field

from .routing import BusinessIntent, ExecutionPolicy
from .routing_benchmark import (
    RouteBenchmarkCase,
    evaluate_route_predictions,
)


class RouteObservation(BaseModel):
    case_id: str
    predicted_intent: BusinessIntent
    predicted_policy: ExecutionPolicy
    confidence: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    fallback_used: bool = False
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class RouteExperimentResult(BaseModel):
    router_name: str
    prompt_version: str
    model_name: str
    total: int
    intent_accuracy: float
    intent_macro_f1: float
    policy_accuracy: float
    policy_error_rate: float
    unsafe_route_rate: float
    clarification_rate: float
    fallback_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    errors: list[dict]


def summarize_route_experiment(
    cases: list[RouteBenchmarkCase],
    observations: list[RouteObservation],
    *,
    router_name: str,
    prompt_version: str,
    model_name: str,
    input_cost_per_million: float = 0,
    output_cost_per_million: float = 0,
) -> RouteExperimentResult:
    by_id = {observation.case_id: observation for observation in observations}
    predictions = {
        case_id: (
            observation.predicted_intent,
            observation.predicted_policy,
            observation.confidence,
        )
        for case_id, observation in by_id.items()
    }
    quality = evaluate_route_predictions(cases, predictions)
    latencies = sorted(observation.latency_ms for observation in observations)
    unsafe = sum(
        by_id[case.case_id].predicted_policy is ExecutionPolicy.DIRECT
        and case.expected_policy is not ExecutionPolicy.DIRECT
        for case in cases
    )
    input_tokens = sum(item.input_tokens for item in observations)
    output_tokens = sum(item.output_tokens for item in observations)
    cost = (
        input_tokens * input_cost_per_million
        + output_tokens * output_cost_per_million
    ) / 1_000_000
    return RouteExperimentResult(
        router_name=router_name,
        prompt_version=prompt_version,
        model_name=model_name,
        total=len(cases),
        intent_accuracy=quality.intent_accuracy,
        intent_macro_f1=quality.intent_macro_f1,
        policy_accuracy=quality.policy_accuracy,
        policy_error_rate=1 - quality.policy_accuracy,
        unsafe_route_rate=unsafe / len(cases),
        clarification_rate=sum(
            item.predicted_policy is ExecutionPolicy.CLARIFY
            for item in observations
        )
        / len(cases),
        fallback_rate=sum(item.fallback_used for item in observations) / len(cases),
        average_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=latencies[max(0, ceil(0.95 * len(latencies)) - 1)],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=round(cost, 10),
        errors=[
            {
                "case_id": error.case_id,
                "query": error.query,
                "expected_intent": error.expected_intent.value,
                "predicted_intent": error.predicted_intent.value,
                "expected_policy": error.expected_policy.value,
                "predicted_policy": error.predicted_policy.value,
                "confidence": error.confidence,
            }
            for error in quality.errors
        ],
    )
