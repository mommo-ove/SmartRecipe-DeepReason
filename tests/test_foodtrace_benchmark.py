import json
from pathlib import Path

from gustobot.application.foodtrace.benchmark import (
    FOODTRACE_METRIC_NAMES,
    load_foodtrace_benchmark_cases,
    run_foodtrace_benchmark,
    write_benchmark_report,
)


def test_benchmark_cases_cover_recall_and_fail_closed_scenarios():
    cases = load_foodtrace_benchmark_cases()
    tags = {tag for case in cases for tag in case.scenario_tags}

    assert 10 <= len(cases) <= 30
    assert {
        "happy_path",
        "single_path",
        "multi_recipe",
        "unaffected_negative",
        "missing_batch",
        "missing_edge",
        "inconsistency",
        "sql_dependency_failure",
        "sop_missing",
        "unsafe_sql",
        "ungrounded_number",
    } <= tags
    assert len({case.case_id for case in cases}) == len(cases)


def test_benchmark_reports_all_fixed_metrics_and_raw_case_results():
    report = run_foodtrace_benchmark(load_foodtrace_benchmark_cases())

    assert report.total_cases == 11
    assert set(report.metrics) == set(FOODTRACE_METRIC_NAMES)
    assert report.metrics["impacted_batch_precision"] == 1.0
    assert report.metrics["impacted_batch_recall"] == 0.4643
    assert report.metrics["impacted_order_recall"] == 0.3571
    assert report.metrics["false_negative_rate"] == 0.5357
    assert report.metrics["evidence_coverage"] == 0.9524
    assert report.metrics["inconsistency_gate_recall"] == 1.0
    assert report.metrics["unsafe_sql_block_rate"] == 1.0
    assert report.metrics["end_to_end_success_rate"] == 1.0
    assert report.metrics["p50_runtime_ms"] >= 0
    assert report.metrics["p95_runtime_ms"] >= report.metrics["p50_runtime_ms"]
    assert len(report.cases) == report.total_cases
    assert all(case.passed for case in report.cases)
    assert all(case.evidence_ids for case in report.cases)
    assert report.case_set_version == "foodtrace-benchmark-v1"
    assert len(report.case_set_sha256) == 64
    assert report.metric_counts["impacted_batch_recall"].numerator == 13
    assert report.metric_counts["impacted_batch_recall"].denominator == 28
    assert report.metric_counts["evidence_coverage"].numerator == 20
    assert report.metric_counts["evidence_coverage"].denominator == 21
    assert report.metric_counts["unsafe_sql_block_rate"].numerator == 1
    assert report.metric_counts["unsafe_sql_block_rate"].denominator == 1
    by_case = {case.case_id: case for case in report.cases}
    assert by_case["graph_sql_mismatch"].gate_reasons == (
        "graph_sql_batch_scope_mismatch",
    )
    assert by_case["sql_dependency_failure"].gate_reasons == (
        "order_exposure:dependency_unavailable",
    )
    assert by_case["ungrounded_number"].gate_reasons == (
        "ungrounded_claim:impacted_order_count",
    )
    assert by_case["missing_batch"].actual_gate == "HUMAN_REVIEW"
    assert by_case["missing_graph_edge"].actual_gate == "BLOCKED"
    assert by_case["unsafe_sql_scope_overflow"].gate_reasons == (
        "order_exposure:dependency_unavailable",
    )
    assert by_case["ungrounded_number"].claims[-1].evidence_ids == (
        by_case["ungrounded_number"].evidence[0].evidence_id,
    )
    assert by_case["ungrounded_number"].evidence[0].kind == "incident_notice"


def test_benchmark_report_writer_preserves_raw_json(tmp_path):
    report = run_foodtrace_benchmark(load_foodtrace_benchmark_cases())
    target = tmp_path / "foodtrace-results.json"

    write_benchmark_report(report, target)

    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["total_cases"] == 11
    assert len(raw["cases"]) == 11
    assert raw["metrics"]["end_to_end_success_rate"] == 1.0
    assert raw["metric_counts"]["end_to_end_success_rate"] == {
        "numerator": 11,
        "denominator": 11,
        "definition": "cases matching expected gate and report scope",
    }
    assert raw["case_set_version"] == "foodtrace-benchmark-v1"
    assert len(raw["case_set_sha256"]) == 64
    assert Path(raw["cases"][0]["case_id"]).is_absolute() is False
