from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class EvaluationSnapshot:
    split: str
    corpus_hash: str
    benchmark_hash: str
    corpus_count: int
    benchmark_count: int
    results: dict[str, dict[str, Any]]
    rankings: dict[str, dict[str, list[str]]]


def load_saved_reports(run_dir: Path) -> EvaluationSnapshot:
    paths = sorted(run_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no evaluation reports found in {run_dir}")

    identity: tuple[str, str, str] | None = None
    corpus_count = 0
    benchmark_count = 0
    latest: dict[str, tuple[str, dict[str, Any], dict[str, list[str]]]] = {}
    for path in paths:
        report = json.loads(path.read_text("utf-8"))
        current_identity = (
            str(report["split"]),
            str(report["corpus"]["corpus_sha256"]),
            str(report["benchmark"]["cases_sha256"]),
        )
        if identity is None:
            identity = current_identity
            corpus_count = int(report["corpus"]["record_count"])
            benchmark_count = int(report["benchmark"]["case_count"])
        elif current_identity != identity:
            raise ValueError("saved reports do not share the same benchmark identity")
        created_at = str(report.get("created_at") or "")
        for configuration, result in report.get("results", {}).items():
            previous = latest.get(configuration)
            if previous is None or created_at >= previous[0]:
                latest[configuration] = (
                    created_at,
                    result,
                    report.get("rankings", {}).get(configuration, {}),
                )

    assert identity is not None
    return EvaluationSnapshot(
        split=identity[0],
        corpus_hash=identity[1],
        benchmark_hash=identity[2],
        corpus_count=corpus_count,
        benchmark_count=benchmark_count,
        results={name: value[1] for name, value in latest.items()},
        rankings={name: value[2] for name, value in latest.items()},
    )


def metric_rows(snapshot: EvaluationSnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for configuration in sorted(snapshot.results):
        result = snapshot.results[configuration]
        metrics = result["metrics"]
        latency = result["latency_ms"]
        rows.append(
            {
                "configuration": configuration,
                "Recall@5": round(float(metrics["recall@5"]) * 100, 2),
                "recall@20": round(float(metrics["recall@20"]) * 100, 2),
                "MRR": round(float(metrics["mrr"]) * 100, 2),
                "NDCG@20": round(float(metrics["ndcg@20"]) * 100, 2),
                "violation@20": round(
                    float(result["ingredient_exclusion_violation@20"]) * 100, 2
                ),
                "mean_ms": round(float(latency["mean"]), 2),
                "p95_ms": round(float(latency["p95"]), 2),
            }
        )
    return rows


def inspect_case(
    snapshot: EvaluationSnapshot,
    configuration: str,
    case: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    ranking = snapshot.rankings.get(configuration, {}).get(case_id, [])
    gold = set(case["relevant_recipe_ids"])
    filters = case.get("metadata_filters", {})
    excluded = set(filters.get("excluded_graph_ingredient_ids", []))
    ranking_rows = []
    for rank, recipe_id in enumerate(ranking, start=1):
        ingredients = set(documents.get(recipe_id, {}).get("ingredients", []))
        ranking_rows.append(
            {
                "rank": rank,
                "recipe_id": recipe_id,
                "is_gold": recipe_id in gold,
                "violates_exclusion": bool(ingredients & excluded),
                "matched_forbidden": ", ".join(sorted(ingredients & excluded)),
            }
        )
    return {
        "case_id": case_id,
        "query": case["query"],
        "required_ingredients": filters.get("graph_ingredient_ids", []),
        "excluded_ingredients": sorted(excluded),
        "gold_recipe_ids": sorted(gold),
        "ranking_rows": ranking_rows,
    }


def build_backend_command(
    *,
    project_root: Path,
    configurations: list[str],
    model_path: Path,
    neo4j_url: str,
    split: str,
    output_path: Path,
) -> list[str]:
    if not configurations:
        raise ValueError("select at least one evaluation configuration")
    if split not in {"development", "test"}:
        raise ValueError("split must be development or test")
    return [
        "uv",
        "run",
        "python",
        "scripts/run_graph_retrieval_benchmark.py",
        "--configurations",
        *configurations,
        "--model",
        str(model_path),
        "--neo4j-url",
        neo4j_url,
        "--split",
        split,
        "--output",
        str(output_path),
    ]


def load_text2cypher_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text("utf-8"))
    required = {"split", "benchmark", "configurations", "security", "raw_runs"}
    missing = sorted(required - set(report))
    if missing:
        raise ValueError("invalid Text2Cypher report, missing: " + ", ".join(missing))
    return report


def text2cypher_metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    unsafe_rejection = float(
        report.get("security", {}).get("unsafe_query_rejection_rate", 0)
    )
    rows: list[dict[str, Any]] = []
    for name, metrics in report.get("configurations", {}).items():
        latency = metrics.get("latency_ms", {})
        average_repairs = float(metrics["average_repair_count"])
        rows.append(
            {
                "configuration": name,
                "case_count": int(metrics["case_count"]),
                "logical_exact_match": round(
                    float(metrics["logical_exact_match"]) * 100, 2
                ),
                "execution_success": round(
                    float(metrics["execution_success_rate"]) * 100, 2
                ),
                "schema_pass": round(
                    float(metrics["schema_validation_pass_rate"]) * 100, 2
                ),
                "explain_pass": round(float(metrics["explain_pass_rate"]) * 100, 2),
                "repair_success": (
                    "N/A"
                    if average_repairs == 0
                    else round(float(metrics["repair_success_rate"]) * 100, 2)
                ),
                "average_repairs": round(average_repairs, 3),
                "unsafe_rejection": round(unsafe_rejection * 100, 2),
                "p50_ms": round(float(latency.get("p50", 0)), 2),
                "p95_ms": round(float(latency.get("p95", 0)), 2),
            }
        )
    return rows


def inspect_text2cypher_case(
    report: dict[str, Any],
    configuration: str,
    case_id: str,
) -> dict[str, Any]:
    run = report.get("raw_runs", {}).get(configuration, {}).get(case_id)
    if run is None:
        raise KeyError(f"no raw Text2Cypher run for {configuration}/{case_id}")
    result = run.get("result") or {}
    return {
        "case_id": case_id,
        "configuration": configuration,
        "question": run.get("question"),
        "gold_recipe_ids": run.get("gold_recipe_ids", []),
        "selected_recipe_ids": result.get("selected_recipe_ids", []),
        "trace": result.get("trace", []),
        "validation_reports": result.get("validation_reports", []),
        "candidate": result.get("candidate"),
        "latency_ms": run.get("latency_ms"),
        "error": run.get("error"),
    }


def text2cypher_report_case_ids(report: dict[str, Any]) -> list[str]:
    case_ids = {
        str(case_id)
        for runs in report.get("raw_runs", {}).values()
        for case_id in runs
    }
    return sorted(case_ids)


def text2cypher_report_configurations(report: dict[str, Any]) -> list[str]:
    return sorted(str(name) for name in report.get("configurations", {}))


def build_text2cypher_backend_command(
    *,
    project_root: Path,
    configurations: list[str],
    neo4j_url: str,
    split: str,
    output_path: Path,
) -> list[str]:
    if not configurations:
        raise ValueError("select at least one Text2Cypher configuration")
    if split not in {"development", "test"}:
        raise ValueError("split must be development or test")
    return [
        "uv",
        "run",
        "python",
        "scripts/run_text2cypher_benchmark.py",
        "--configurations",
        *configurations,
        "--neo4j-url",
        neo4j_url,
        "--split",
        split,
        "--output",
        str(output_path),
    ]


_QUESTION_HEADING = re.compile(r"^##\s+(\d{3})\.\s+(.+?)\s*$")
_CATEGORY_HEADING = re.compile(r"^#\s+([^#].+?)\s*$")


def load_interview_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    category = "未分类"
    current: dict[str, Any] | None = None
    answer_lines: list[str] = []
    for line in path.read_text("utf-8").splitlines():
        question_match = _QUESTION_HEADING.match(line)
        category_match = _CATEGORY_HEADING.match(line)
        if question_match:
            if current is not None:
                current["answer"] = "\n".join(answer_lines).strip()
                questions.append(current)
            current = {
                "number": int(question_match.group(1)),
                "category": category,
                "question": question_match.group(2),
            }
            answer_lines = []
        elif category_match and current is None:
            category = category_match.group(1)
        elif category_match:
            current["answer"] = "\n".join(answer_lines).strip()
            questions.append(current)
            current = None
            answer_lines = []
            category = category_match.group(1)
        elif current is not None:
            answer_lines.append(line)
    if current is not None:
        current["answer"] = "\n".join(answer_lines).strip()
        questions.append(current)
    return questions
