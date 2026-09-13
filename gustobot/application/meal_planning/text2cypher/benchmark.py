from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from pydantic import BaseModel, Field

from .models import CypherRunStatus, CypherSource, Text2CypherResult
from .safety import validate_cypher_safety
from .templates import TemplateRequest, build_template_candidate


Split = Literal["development", "test"]
ExpectedRoute = Literal["template", "dynamic"]


class Text2CypherBenchmarkCase(BaseModel):
    case_id: str = Field(pattern=r"^text2cypher-\d{3}$")
    split: Split
    question: str = Field(min_length=1)
    gold_recipe_ids: list[str] = Field(min_length=1)
    expected_route: ExpectedRoute
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: dict[str, object]
    template_request: dict[str, object]
    source_case_id: str | None = None


class Text2CypherSecurityCase(BaseModel):
    case_id: str = Field(pattern=r"^text2cypher-security-\d{3}$")
    split: Split
    statement: str = Field(min_length=1)
    expected_issue_codes: list[str] = Field(min_length=1)


class Text2CypherBenchmarkResult(BaseModel):
    result: Text2CypherResult | None = None
    latency_ms: float = Field(ge=0)
    error: str | None = None


class LatencySummary(BaseModel):
    mean: float = Field(ge=0)
    p50: float = Field(ge=0)
    p95: float = Field(ge=0)


class Text2CypherMetrics(BaseModel):
    configuration: str
    case_count: int = Field(ge=1)
    template_route_accuracy: float = Field(ge=0, le=1)
    generation_format_rate: float = Field(ge=0, le=1)
    schema_validation_pass_rate: float = Field(ge=0, le=1)
    explain_pass_rate: float = Field(ge=0, le=1)
    execution_success_rate: float = Field(ge=0, le=1)
    logical_exact_match: float = Field(ge=0, le=1)
    repair_success_rate: float = Field(ge=0, le=1)
    average_repair_count: float = Field(ge=0)
    latency_ms: LatencySummary


class Text2CypherSecurityMetrics(BaseModel):
    case_count: int = Field(ge=1)
    unsafe_query_rejection_rate: float = Field(ge=0, le=1)
    issue_code_match_rate: float = Field(ge=0, le=1)
    cases: list[dict[str, object]]


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_jsonl_models(path: Path, model: type[ModelT]) -> list[ModelT]:
    items = [
        model.model_validate_json(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    for item in items:
        case_id = str(getattr(item, "case_id"))
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
    return items


def build_few_shot_examples(
    cases: list[Text2CypherBenchmarkCase],
) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for case in cases:
        if case.split != "development":
            continue
        candidate = build_template_candidate(
            TemplateRequest.model_validate(case.template_request)
        )
        if candidate is not None:
            examples.append((case.question, candidate.statement))
    return examples


def _trace_passed(result: Text2CypherResult | None, prefix: str) -> bool:
    return bool(
        result
        and any(
            entry.startswith(prefix) and ":ok:" in entry
            for entry in result.trace
        )
    )


def _repair_count(result: Text2CypherResult | None) -> int:
    if result is None:
        return 0
    return sum(entry.startswith("repair:") for entry in result.trace)


def _latency_summary(values: list[float]) -> LatencySummary:
    matrix = np.asarray(values, dtype=np.float64)
    return LatencySummary(
        mean=float(np.mean(matrix)),
        p50=float(np.percentile(matrix, 50)),
        p95=float(np.percentile(matrix, 95)),
    )


def evaluate_text2cypher_results(
    *,
    configuration: str,
    cases: list[Text2CypherBenchmarkCase],
    results: dict[str, Text2CypherBenchmarkResult],
) -> Text2CypherMetrics:
    if not cases:
        raise ValueError("at least one benchmark case is required")
    missing = [case.case_id for case in cases if case.case_id not in results]
    if missing:
        raise ValueError("missing benchmark results: " + ", ".join(missing))

    runs = [results[case.case_id] for case in cases]
    service_results = [run.result for run in runs]
    route_matches = 0
    exact_matches = 0
    successful_executions = 0
    repaired_cases = 0
    repaired_successes = 0
    repair_counts: list[int] = []
    for case, result in zip(cases, service_results, strict=True):
        if result is not None and result.candidate is not None:
            expected_source = (
                CypherSource.TEMPLATE
                if case.expected_route == "template"
                else CypherSource.DYNAMIC
            )
            route_matches += result.candidate.source is expected_source
            exact_matches += set(result.selected_recipe_ids) == set(
                case.gold_recipe_ids
            )
            successful_executions += result.status in {
                CypherRunStatus.SUCCESS,
                CypherRunStatus.EMPTY,
            }
        repairs = _repair_count(result)
        repair_counts.append(repairs)
        if repairs:
            repaired_cases += 1
            repaired_successes += bool(
                result
                and result.status
                in {CypherRunStatus.SUCCESS, CypherRunStatus.EMPTY}
            )

    count = len(cases)
    return Text2CypherMetrics(
        configuration=configuration,
        case_count=count,
        template_route_accuracy=route_matches / count,
        generation_format_rate=sum(
            result is not None and result.candidate is not None
            for result in service_results
        )
        / count,
        schema_validation_pass_rate=sum(
            _trace_passed(result, "validate:schema") for result in service_results
        )
        / count,
        explain_pass_rate=sum(
            _trace_passed(result, "explain") for result in service_results
        )
        / count,
        execution_success_rate=successful_executions / count,
        logical_exact_match=exact_matches / count,
        repair_success_rate=(
            repaired_successes / repaired_cases if repaired_cases else 0.0
        ),
        average_repair_count=sum(repair_counts) / count,
        latency_ms=_latency_summary([run.latency_ms for run in runs]),
    )


def evaluate_security_cases(
    cases: list[Text2CypherSecurityCase],
) -> Text2CypherSecurityMetrics:
    if not cases:
        raise ValueError("at least one security case is required")
    rejected = 0
    issue_matches = 0
    details: list[dict[str, object]] = []
    for case in cases:
        report = validate_cypher_safety(case.statement)
        actual_codes = sorted({issue.code for issue in report.issues})
        expected_codes = set(case.expected_issue_codes)
        rejected += not report.valid
        issue_matches += expected_codes.issubset(actual_codes)
        details.append(
            {
                "case_id": case.case_id,
                "expected_issue_codes": sorted(expected_codes),
                "actual_issue_codes": actual_codes,
                "rejected": not report.valid,
            }
        )
    count = len(cases)
    return Text2CypherSecurityMetrics(
        case_count=count,
        unsafe_query_rejection_rate=rejected / count,
        issue_code_match_rate=issue_matches / count,
        cases=details,
    )
