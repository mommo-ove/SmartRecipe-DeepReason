from pathlib import Path

from gustobot.application.meal_planning.routing import BusinessIntent, ExecutionPolicy
from gustobot.application.meal_planning.routing_benchmark import (
    load_frozen_route_benchmark,
    RouteBenchmarkCase,
    evaluate_route_predictions,
    load_route_benchmark,
)


ROUTE_CASES = (
    Path(__file__).parents[1]
    / "benchmark"
    / "meal_planning"
    / "routing"
    / "cases.v1.jsonl"
)
ROUTE_CASES_V2 = ROUTE_CASES.with_name("cases.v2.jsonl")


def test_route_benchmark_reports_accuracy_and_confusion_counts():
    cases = [
        RouteBenchmarkCase(
            case_id="1",
            query="怎么做番茄炒蛋？",
            expected_intent=BusinessIntent.RECIPE_LOOKUP,
            expected_policy=ExecutionPolicy.DIRECT,
        ),
        RouteBenchmarkCase(
            case_id="2",
            query="帮我做一周餐单",
            expected_intent=BusinessIntent.MEAL_PLAN,
            expected_policy=ExecutionPolicy.CLARIFY,
        ),
    ]
    predictions = {
        "1": (BusinessIntent.RECIPE_LOOKUP, ExecutionPolicy.DIRECT),
        "2": (BusinessIntent.RECIPE_LOOKUP, ExecutionPolicy.DIRECT),
    }

    result = evaluate_route_predictions(cases, predictions)

    assert result.intent_accuracy == 0.5
    assert result.policy_accuracy == 0.5
    assert result.intent_confusion["meal_plan"]["recipe_lookup"] == 1
    assert result.policy_confusion["clarify"]["direct"] == 1


def test_versioned_route_benchmark_contains_real_chinese_queries():
    cases = load_route_benchmark(ROUTE_CASES)

    assert len(cases) == 16
    assert {case.expected_intent for case in cases} == {
        BusinessIntent.RECIPE_LOOKUP,
        BusinessIntent.MEAL_PLAN,
        BusinessIntent.PLAN_EDIT,
        BusinessIntent.GENERAL,
    }
    assert any("西红柿炒蛋" in case.query for case in cases)


def test_route_benchmark_reports_macro_f1_and_high_confidence_errors():
    cases = [
        RouteBenchmarkCase(
            case_id="lookup",
            query="番茄炒蛋怎么做？",
            expected_intent=BusinessIntent.RECIPE_LOOKUP,
            expected_policy=ExecutionPolicy.DIRECT,
            split="test",
        ),
        RouteBenchmarkCase(
            case_id="plan",
            query="帮我安排一周餐单",
            expected_intent=BusinessIntent.MEAL_PLAN,
            expected_policy=ExecutionPolicy.CLARIFY,
            split="test",
        ),
    ]
    predictions = {
        "lookup": (BusinessIntent.RECIPE_LOOKUP, ExecutionPolicy.DIRECT, 0.91),
        "plan": (BusinessIntent.RECIPE_LOOKUP, ExecutionPolicy.DIRECT, 0.95),
    }

    result = evaluate_route_predictions(cases, predictions)

    assert result.intent_macro_f1 == 1 / 3
    assert result.intent_per_class["meal_plan"].recall == 0
    assert result.errors[0].case_id == "plan"
    assert result.errors[0].confidence == 0.95
    assert [error.case_id for error in result.high_confidence_errors] == ["plan"]


def test_route_benchmark_can_load_only_the_independent_test_split(tmp_path):
    benchmark = tmp_path / "routes.jsonl"
    benchmark.write_text(
        "\n".join(
            [
                '{"case_id":"dev-1","query":"怎么做饭","expected_intent":"recipe_lookup",'
                '"expected_policy":"direct","split":"development","review_status":"draft"}',
                '{"case_id":"test-1","query":"安排一周","expected_intent":"meal_plan",'
                '"expected_policy":"clarify","split":"test","review_status":"reviewed"}',
            ]
        ),
        encoding="utf-8",
    )

    cases = load_route_benchmark(benchmark, split="test")

    assert [case.case_id for case in cases] == ["test-1"]
    assert cases[0].review_status == "reviewed"


def test_v2_route_benchmark_separates_development_and_test_cases():
    development = load_route_benchmark(ROUTE_CASES_V2, split="development")
    test = load_route_benchmark(ROUTE_CASES_V2, split="test")

    assert len(development) == 17
    assert len(test) == 26
    assert not ({case.case_id for case in development} & {case.case_id for case in test})
    assert {case.expected_intent for case in test} == set(BusinessIntent)
    assert any(
        case.expected_policy is ExecutionPolicy.WORKFLOW
        and case.expected_intent is BusinessIntent.MEAL_PLAN
        for case in test
    )


def test_frozen_router_benchmark_is_hash_pinned_and_balanced():
    root = Path(__file__).parents[1] / "benchmark" / "meal_planning" / "routing" / "frozen"

    benchmark = load_frozen_route_benchmark(
        root / "cases.v1.jsonl",
        root / "manifest.v1.json",
    )

    assert benchmark.manifest.status == "frozen_pending_human_review"
    assert benchmark.manifest.resume_metric_eligible is False
    assert benchmark.manifest.record_count == 60
    assert len(benchmark.cases) == 60
    supports = {
        intent: sum(case.expected_intent is intent for case in benchmark.cases)
        for intent in BusinessIntent
    }
    assert min(supports.values()) >= 6
    assert all(case.split == "frozen" for case in benchmark.cases)


def test_iteration_queries_do_not_overlap_with_frozen_queries():
    iteration_cases = load_route_benchmark(ROUTE_CASES_V2)
    frozen_root = (
        Path(__file__).parents[1]
        / "benchmark"
        / "meal_planning"
        / "routing"
        / "frozen"
    )
    frozen = load_frozen_route_benchmark(
        frozen_root / "cases.v1.jsonl",
        frozen_root / "manifest.v1.json",
    )

    iteration_queries = {_normalize_query(case.query) for case in iteration_cases}
    frozen_queries = {_normalize_query(case.query) for case in frozen.cases}

    assert iteration_queries.isdisjoint(frozen_queries)


def _normalize_query(query: str) -> str:
    return "".join(query.lower().split()).rstrip("，。？！?!")
