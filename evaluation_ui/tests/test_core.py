from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrecipe_eval_ui.core import (
    build_backend_command,
    build_text2cypher_backend_command,
    inspect_case,
    inspect_text2cypher_case,
    load_interview_questions,
    load_saved_reports,
    load_text2cypher_report,
    metric_rows,
    text2cypher_metric_rows,
    text2cypher_report_case_ids,
    text2cypher_report_configurations,
)


def write_report(path: Path, configuration: str, *, created_at: str) -> None:
    payload = {
        "created_at": created_at,
        "split": "test",
        "corpus": {"corpus_sha256": "corpus-hash", "record_count": 300},
        "benchmark": {"cases_sha256": "case-hash", "case_count": 40},
        "results": {
            configuration: {
                "configuration": configuration,
                "case_count": 30,
                "metrics": {
                    "recall@5": 0.4,
                    "recall@20": 0.9,
                    "mrr": 0.3,
                    "ndcg@20": 0.5,
                },
                "ingredient_exclusion_violation@20": 0.2,
                "latency_ms": {"mean": 10.0, "p50": 9.0, "p95": 15.0},
            }
        },
        "rankings": {configuration: {"case-1": ["unsafe", "safe"]}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_saved_reports_merge_configurations_only_when_dataset_identity_matches(
    tmp_path,
):
    write_report(tmp_path / "bm25.json", "bm25", created_at="2026-01-01T00:00:00Z")
    write_report(
        tmp_path / "graph.json", "graph_gate", created_at="2026-01-02T00:00:00Z"
    )

    snapshot = load_saved_reports(tmp_path)
    rows = metric_rows(snapshot)

    assert snapshot.split == "test"
    assert {row["configuration"] for row in rows} == {"bm25", "graph_gate"}
    assert rows[0]["recall@20"] == pytest.approx(90.0)


def test_saved_reports_reject_mixed_benchmark_hashes(tmp_path):
    write_report(tmp_path / "bm25.json", "bm25", created_at="2026-01-01T00:00:00Z")
    write_report(
        tmp_path / "graph.json", "graph_gate", created_at="2026-01-02T00:00:00Z"
    )
    payload = json.loads((tmp_path / "graph.json").read_text("utf-8"))
    payload["benchmark"]["cases_sha256"] = "different"
    (tmp_path / "graph.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark identity"):
        load_saved_reports(tmp_path)


def test_case_inspector_marks_hits_and_forbidden_ingredient_violations(tmp_path):
    write_report(tmp_path / "bm25.json", "bm25", created_at="2026-01-01T00:00:00Z")
    snapshot = load_saved_reports(tmp_path)
    case = {
        "case_id": "case-1",
        "query": "egg tomato without peanut",
        "relevant_recipe_ids": ["safe"],
        "metadata_filters": {
            "graph_ingredient_ids": ["egg", "tomato"],
            "excluded_graph_ingredient_ids": ["peanut"],
        },
    }
    documents = {
        "safe": {"ingredients": ["egg", "tomato"]},
        "unsafe": {"ingredients": ["egg", "tomato", "peanut"]},
    }

    detail = inspect_case(snapshot, "bm25", case, documents)

    assert detail["ranking_rows"][0]["recipe_id"] == "unsafe"
    assert detail["ranking_rows"][0]["violates_exclusion"] is True
    assert detail["ranking_rows"][1]["is_gold"] is True


def test_backend_command_is_argument_safe_and_selects_real_benchmark_script(tmp_path):
    command = build_backend_command(
        project_root=tmp_path,
        configurations=["bm25", "rrf_with_graph_gate"],
        model_path=tmp_path / "model",
        neo4j_url="bolt://localhost:17687",
        split="test",
        output_path=tmp_path / "report.json",
    )

    assert command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_graph_retrieval_benchmark.py",
    ]
    assert "rrf_with_graph_gate" in command
    assert command[-2:] == ["--output", str(tmp_path / "report.json")]


def test_interview_bank_contains_exactly_100_numbered_questions(tmp_path):
    source = tmp_path / "questions.md"
    source.write_text(
        "# Bank\n\n"
        + "\n\n".join(
            f"## {index:03d}. Question {index}\n\nAnswer {index}."
            for index in range(1, 101)
        ),
        encoding="utf-8",
    )

    questions = load_interview_questions(source)

    assert len(questions) == 100
    assert questions[0]["number"] == 1
    assert questions[-1]["number"] == 100


def test_text2cypher_report_exposes_ablation_metrics_and_raw_trace(tmp_path):
    report_path = tmp_path / "text2cypher.json"
    report_path.write_text(
        json.dumps(
            {
                "split": "test",
                "model": "deepseek-v4-flash",
                "prompt_version": "meal-text2cypher-v1",
                "benchmark": {"cases_sha256": "case-hash", "test_count": 30},
                "configurations": {
                    "template_first_validated": {
                        "case_count": 30,
                        "logical_exact_match": 1.0,
                        "execution_success_rate": 1.0,
                        "schema_validation_pass_rate": 1.0,
                        "explain_pass_rate": 1.0,
                        "repair_success_rate": 0.0,
                        "average_repair_count": 0.0,
                        "latency_ms": {"mean": 20.0, "p50": 18.0, "p95": 31.0},
                    }
                },
                "security": {"unsafe_query_rejection_rate": 1.0},
                "raw_runs": {
                    "template_first_validated": {
                        "text2cypher-001": {
                            "question": "query",
                            "gold_recipe_ids": ["r1"],
                            "result": {
                                "selected_recipe_ids": ["r1"],
                                "trace": ["validate:safety:ok:0.1ms", "execute:success:2ms"],
                                "validation_reports": [{"issues": []}],
                            },
                            "latency_ms": 20.0,
                            "error": None,
                        }
                    }
                },
            }
        ),
        "utf-8",
    )

    report = load_text2cypher_report(report_path)
    rows = text2cypher_metric_rows(report)
    detail = inspect_text2cypher_case(
        report, "template_first_validated", "text2cypher-001"
    )

    assert rows[0]["logical_exact_match"] == pytest.approx(100.0)
    assert rows[0]["unsafe_rejection"] == pytest.approx(100.0)
    assert rows[0]["repair_success"] == "N/A"
    assert detail["selected_recipe_ids"] == ["r1"]
    assert detail["trace"][0].startswith("validate:safety")
    assert text2cypher_report_case_ids(report) == ["text2cypher-001"]
    assert text2cypher_report_configurations(report) == ["template_first_validated"]


def test_text2cypher_backend_command_selects_specialized_runner(tmp_path):
    command = build_text2cypher_backend_command(
        project_root=tmp_path,
        configurations=["template_first_validated"],
        neo4j_url="bolt://localhost:17687",
        split="test",
        output_path=tmp_path / "report.json",
    )

    assert command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_text2cypher_benchmark.py",
    ]
    assert command[-2:] == ["--output", str(tmp_path / "report.json")]
