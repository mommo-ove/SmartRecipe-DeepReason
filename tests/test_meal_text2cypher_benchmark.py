from __future__ import annotations

import json
from pathlib import Path

import pytest

from gustobot.application.meal_planning.text2cypher.benchmark import (
    Text2CypherBenchmarkCase,
    Text2CypherBenchmarkResult,
    Text2CypherSecurityCase,
    build_few_shot_examples,
    evaluate_security_cases,
    evaluate_text2cypher_results,
    load_jsonl_models,
)
from gustobot.application.meal_planning.text2cypher.models import (
    CypherCandidate,
    CypherExecutionResult,
    CypherRunStatus,
    CypherSource,
    Text2CypherResult,
)


DATASET_HASH = "a" * 64
ROOT = Path(__file__).parents[1]


def benchmark_case(case_id: str, *, split: str) -> Text2CypherBenchmarkCase:
    return Text2CypherBenchmarkCase(
        case_id=case_id,
        split=split,
        question="Find egg recipes without peanut.",
        gold_recipe_ids=["r1"],
        expected_route="template",
        dataset_sha256=DATASET_HASH,
        parameters={"ingredient_ids": ["egg"], "excluded_ingredient_ids": ["peanut"]},
        template_request={
            "operation": "search_recipes",
            "dataset_version": "v1",
            "required_ingredient_ids": ["egg"],
            "excluded_ingredient_ids": ["peanut"],
            "top_k": 20,
        },
    )


def successful_result(*, recipe_ids: list[str], source: CypherSource) -> Text2CypherResult:
    candidate = CypherCandidate(
        statement="MATCH (r:Recipe) RETURN r.recipe_id AS recipe_id LIMIT 20",
        source=source,
    )
    execution = CypherExecutionResult(
        status=CypherRunStatus.SUCCESS,
        records=[{"recipe_id": recipe_id} for recipe_id in recipe_ids],
        selected_recipe_ids=recipe_ids,
        latency_ms=3.0,
    )
    return Text2CypherResult(
        question="Find egg recipes without peanut.",
        status=CypherRunStatus.SUCCESS,
        candidate=candidate,
        execution=execution,
        selected_recipe_ids=recipe_ids,
        trace=[
            "template_route:hit:0.1ms",
            "validate:safety:ok:0.1ms",
            "validate:schema:ok:0.1ms",
            "explain:ok:0.2ms",
            "execute:success:3.0ms",
        ],
    )


def test_benchmark_models_require_stable_ids_gold_route_and_dataset_hash():
    case = benchmark_case("text2cypher-001", split="test")

    assert case.gold_recipe_ids == ["r1"]
    with pytest.raises(ValueError):
        benchmark_case("mutable id", split="test")
    with pytest.raises(ValueError):
        Text2CypherBenchmarkCase(
            case_id="text2cypher-002",
            split="test",
            question="query",
            gold_recipe_ids=[],
            expected_route="template",
            dataset_sha256="not-a-hash",
            parameters={},
            template_request={},
        )


def test_jsonl_loader_rejects_duplicate_case_ids(tmp_path):
    case = benchmark_case("text2cypher-001", split="test").model_dump(mode="json")
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join((json.dumps(case), json.dumps(case))), "utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_jsonl_models(path, Text2CypherBenchmarkCase)


def test_few_shot_builder_never_uses_frozen_test_cases():
    cases = [
        benchmark_case("text2cypher-001", split="development").model_copy(
            update={"question": "Development-only egg query."}
        ),
        benchmark_case("text2cypher-002", split="test").model_copy(
            update={"question": "Frozen test query must stay hidden."}
        ),
    ]

    examples = build_few_shot_examples(cases)

    assert len(examples) == 1
    assert cases[0].question in examples[0][0]
    assert cases[1].question not in "\n".join(text for pair in examples for text in pair)


def test_metrics_report_exact_match_stages_repairs_and_latency():
    cases = [
        benchmark_case("text2cypher-001", split="test"),
        benchmark_case("text2cypher-002", split="test"),
    ]
    first = successful_result(recipe_ids=["r1"], source=CypherSource.TEMPLATE)
    second = successful_result(recipe_ids=["wrong"], source=CypherSource.DYNAMIC)
    second.candidate.attempt = 1
    second.trace.insert(-1, "repair:attempt_1:4.0ms")

    report = evaluate_text2cypher_results(
        configuration="template_first_validated",
        cases=cases,
        results={
            "text2cypher-001": Text2CypherBenchmarkResult(result=first, latency_ms=10),
            "text2cypher-002": Text2CypherBenchmarkResult(result=second, latency_ms=30),
        },
    )

    assert report.case_count == 2
    assert report.logical_exact_match == pytest.approx(0.5)
    assert report.template_route_accuracy == pytest.approx(0.5)
    assert report.generation_format_rate == pytest.approx(1.0)
    assert report.schema_validation_pass_rate == pytest.approx(1.0)
    assert report.explain_pass_rate == pytest.approx(1.0)
    assert report.execution_success_rate == pytest.approx(1.0)
    assert report.repair_success_rate == pytest.approx(1.0)
    assert report.average_repair_count == pytest.approx(0.5)
    assert report.latency_ms.p50 == pytest.approx(20.0)
    assert report.latency_ms.p95 == pytest.approx(29.0)


def test_security_metric_requires_expected_issue_code_and_no_execution():
    cases = [
        Text2CypherSecurityCase(
            case_id="text2cypher-security-001",
            split="test",
            statement="MATCH (r:Recipe) DELETE r RETURN r.recipe_id AS recipe_id LIMIT 1",
            expected_issue_codes=["UNSAFE_CLAUSE"],
        ),
        Text2CypherSecurityCase(
            case_id="text2cypher-security-002",
            split="test",
            statement="MATCH (r:Recipe) RETURN r.recipe_id AS recipe_id",
            expected_issue_codes=["MISSING_LIMIT"],
        ),
    ]

    report = evaluate_security_cases(cases)

    assert report.case_count == 2
    assert report.unsafe_query_rejection_rate == pytest.approx(1.0)
    assert report.issue_code_match_rate == pytest.approx(1.0)


def test_frozen_repository_benchmark_has_40_relation_cases_and_separate_security_set():
    benchmark_dir = ROOT / "benchmark" / "meal_planning" / "text2cypher"
    relation_cases = load_jsonl_models(
        benchmark_dir / "cases.v1.jsonl", Text2CypherBenchmarkCase
    )
    security_cases = load_jsonl_models(
        benchmark_dir / "security_cases.v1.jsonl", Text2CypherSecurityCase
    )
    corpus_manifest = json.loads(
        (
            ROOT
            / "gustobot"
            / "data"
            / "meal_planning"
            / "external"
            / "manifest.v1.json"
        ).read_text("utf-8")
    )

    assert len(relation_cases) == 40
    assert sum(case.split == "development" for case in relation_cases) == 10
    assert sum(case.split == "test" for case in relation_cases) == 30
    assert {case.dataset_sha256 for case in relation_cases} == {
        corpus_manifest["corpus_sha256"]
    }
    assert len(security_cases) >= 12
    assert all(case.split == "test" for case in security_cases)
