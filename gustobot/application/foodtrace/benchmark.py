from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import Field

from .evidence import make_evidence
from .fixtures import build_foodtrace_fixture
from .gate import GateInput, ImpactClaim, claim_is_grounded, evaluate_gate
from .models import FoodTraceModel, RecallScope
from .repositories import (
    ExposureLookupResult,
    FixtureGraphRepository,
    FixtureSqlRepository,
    GraphTraceResult,
)
from .sop import SopLookupResult
from .workflow import FoodTraceWorkflow, FoodTraceWorkflowResult


FOODTRACE_METRIC_NAMES = (
    "impacted_batch_precision",
    "impacted_batch_recall",
    "impacted_order_recall",
    "false_negative_rate",
    "evidence_coverage",
    "inconsistency_gate_recall",
    "unsafe_sql_block_rate",
    "end_to_end_success_rate",
    "p50_runtime_ms",
    "p95_runtime_ms",
)
DEFAULT_CASES_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "foodtrace"
    / "benchmark_cases.json"
)


class FoodTraceBenchmarkCase(FoodTraceModel):
    case_id: str = Field(min_length=1)
    scenario: Literal[
        "fixture",
        "unaffected_negative",
        "missing_batch",
        "missing_edge",
        "graph_sql_mismatch",
        "sql_dependency_failure",
        "sop_missing",
        "scope_expansion",
        "unsafe_sql_scope_overflow",
        "ungrounded_number",
    ]
    incident_id: str = Field(min_length=1)
    ingredient_lot_id: str = Field(min_length=1)
    gold_scope: RecallScope
    expected_gate: Literal["ALLOW_REPORT", "HUMAN_REVIEW", "BLOCKED"]
    scenario_tags: tuple[str, ...] = ()


class FoodTraceBenchmarkCaseResult(FoodTraceModel):
    case_id: str
    scenario: str
    scenario_tags: tuple[str, ...]
    expected_gate: str
    actual_gate: str
    gate_reasons: tuple[str, ...]
    gold_scope: RecallScope
    actual_scope: RecallScope
    claims: tuple[ImpactClaim, ...]
    evidence: tuple["BenchmarkEvidenceSummary", ...]
    evidence_ids: tuple[str, ...]
    runtime_ms: float
    passed: bool


class BenchmarkEvidenceSummary(FoodTraceModel):
    evidence_id: str
    kind: str
    source: str


class MetricCount(FoodTraceModel):
    numerator: int | None
    denominator: int
    definition: str


class FoodTraceBenchmarkReport(FoodTraceModel):
    case_set_version: str
    case_set_sha256: str
    total_cases: int
    metrics: dict[str, float]
    metric_counts: dict[str, MetricCount]
    cases: tuple[FoodTraceBenchmarkCaseResult, ...]


def load_foodtrace_benchmark_cases(
    path: str | Path = DEFAULT_CASES_PATH,
) -> tuple[FoodTraceBenchmarkCase, ...]:
    raw_cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("FoodTrace benchmark cases must be a JSON list")
    cases = tuple(FoodTraceBenchmarkCase.model_validate(case) for case in raw_cases)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("FoodTrace benchmark case IDs must be unique")
    return cases


def run_foodtrace_benchmark(
    cases: tuple[FoodTraceBenchmarkCase, ...] | list[FoodTraceBenchmarkCase],
) -> FoodTraceBenchmarkReport:
    normalized_cases = tuple(cases)
    raw_results: list[tuple[FoodTraceBenchmarkCase, FoodTraceWorkflowResult, float]] = (
        []
    )
    case_results: list[FoodTraceBenchmarkCaseResult] = []
    for case in normalized_cases:
        started = perf_counter()
        result = _run_case(case)
        runtime_ms = round((perf_counter() - started) * 1000, 4)
        passed = result.gate.state == case.expected_gate
        if case.expected_gate == "ALLOW_REPORT":
            passed = passed and result.scope == case.gold_scope
        raw_results.append((case, result, runtime_ms))
        case_results.append(
            FoodTraceBenchmarkCaseResult(
                case_id=case.case_id,
                scenario=case.scenario,
                scenario_tags=case.scenario_tags,
                expected_gate=case.expected_gate,
                actual_gate=result.gate.state,
                gate_reasons=result.gate.reasons,
                gold_scope=case.gold_scope,
                actual_scope=result.scope,
                claims=result.claims,
                evidence=tuple(
                    BenchmarkEvidenceSummary(
                        evidence_id=item.evidence_id,
                        kind=item.kind,
                        source=item.source,
                    )
                    for item in result.evidence
                ),
                evidence_ids=tuple(
                    sorted(item.evidence_id for item in result.evidence)
                ),
                runtime_ms=runtime_ms,
                passed=passed,
            )
        )

    metrics, metric_counts = _calculate_metrics(raw_results, case_results)
    return FoodTraceBenchmarkReport(
        case_set_version="foodtrace-benchmark-v1",
        case_set_sha256=_case_set_sha256(normalized_cases),
        total_cases=len(case_results),
        metrics=metrics,
        metric_counts=metric_counts,
        cases=tuple(case_results),
    )


def write_benchmark_report(
    report: FoodTraceBenchmarkReport,
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_case(case: FoodTraceBenchmarkCase) -> FoodTraceWorkflowResult:
    fixture = build_foodtrace_fixture()
    incident = fixture.incident.model_copy(
        update={
            "incident_id": case.incident_id,
            "ingredient_lot_id": case.ingredient_lot_id,
        }
    )
    graph_repository: Any = FixtureGraphRepository(fixture)
    exposure_repository: Any = FixtureSqlRepository(fixture)
    sop_repository: Any | None = None
    max_batch_scope = 100

    if case.scenario == "missing_batch":
        graph_repository = _NotFoundGraphRepository()
    elif case.scenario == "missing_edge":
        graph_repository = _MissingEdgeGraphRepository()
    elif case.scenario == "graph_sql_mismatch":
        exposure_repository = _MismatchedExposureRepository()
    elif case.scenario == "sql_dependency_failure":
        exposure_repository = _UnavailableExposureRepository()
    elif case.scenario == "sop_missing":
        sop_repository = _MissingSopRepository()
    elif case.scenario == "scope_expansion":
        graph_repository = _ExpandedGraphRepository()
        max_batch_scope = 3
    elif case.scenario == "unsafe_sql_scope_overflow":
        graph_repository = _UnsafeSqlGraphRepository()
        max_batch_scope = 200

    result = FoodTraceWorkflow(
        graph_repository=graph_repository,
        exposure_repository=exposure_repository,
        sop_repository=sop_repository,
        max_batch_scope=max_batch_scope,
    ).run(incident)
    if case.scenario != "ungrounded_number":
        return result

    claims = (
        *result.claims,
        ImpactClaim(
            metric="impacted_order_count",
            value=999,
            evidence_ids=(result.evidence[0].evidence_id,),
        ),
    )
    gate = evaluate_gate(
        GateInput(
            evidence=result.evidence,
            claims=claims,
        )
    )
    gate_evidence = make_evidence(
        "decision_gate",
        "python",
        gate.model_dump(mode="json"),
        subject_ids=(incident.incident_id,),
    )
    evidence = tuple(
        item for item in result.evidence if item.kind != "decision_gate"
    ) + (gate_evidence,)
    trace = tuple(
        (
            step
            if step.name != "Decision Gate"
            else step.model_copy(
                update={
                    "status": gate.state,
                    "evidence_ids": (gate_evidence.evidence_id,),
                }
            )
        )
        for step in result.trace
    )
    return result.model_copy(
        update={
            "claims": claims,
            "evidence": evidence,
            "gate": gate,
            "trace": trace,
        }
    )


def _calculate_metrics(
    raw_results: list[tuple[FoodTraceBenchmarkCase, FoodTraceWorkflowResult, float]],
    case_results: list[FoodTraceBenchmarkCaseResult],
) -> tuple[dict[str, float], dict[str, MetricCount]]:
    evaluated = [(case, result) for case, result, _ in raw_results]
    predicted_batches = sum(
        len(result.scope.impacted_batch_ids) for _, result in evaluated
    )
    gold_batches = sum(len(case.gold_scope.impacted_batch_ids) for case, _ in evaluated)
    true_batches = sum(
        len(
            set(result.scope.impacted_batch_ids)
            & set(case.gold_scope.impacted_batch_ids)
        )
        for case, result in evaluated
    )
    gold_orders = sum(len(case.gold_scope.impacted_order_ids) for case, _ in evaluated)
    true_orders = sum(
        len(
            set(result.scope.impacted_order_ids)
            & set(case.gold_scope.impacted_order_ids)
        )
        for case, result in evaluated
    )

    all_claims = [(result, claim) for _, result in evaluated for claim in result.claims]
    grounded_claims = sum(
        claim_is_grounded(claim, result.evidence) for result, claim in all_claims
    )

    inconsistency_cases = [
        (case, result)
        for case, result, _ in raw_results
        if "inconsistency" in case.scenario_tags
    ]
    unsafe_sql_cases = [
        (case, result)
        for case, result, _ in raw_results
        if "unsafe_sql" in case.scenario_tags
    ]
    runtimes = sorted(runtime for _, _, runtime in raw_results)
    missed_batches = gold_batches - true_batches
    inconsistency_hits = sum(
        result.gate.state == "HUMAN_REVIEW" for _, result in inconsistency_cases
    )
    unsafe_sql_hits = sum(
        result.gate.state != "ALLOW_REPORT" for _, result in unsafe_sql_cases
    )
    passed_cases = sum(case.passed for case in case_results)
    metrics = {
        "impacted_batch_precision": _ratio(true_batches, predicted_batches),
        "impacted_batch_recall": _ratio(true_batches, gold_batches),
        "impacted_order_recall": _ratio(true_orders, gold_orders),
        "false_negative_rate": _ratio(missed_batches, gold_batches),
        "evidence_coverage": _ratio(grounded_claims, len(all_claims)),
        "inconsistency_gate_recall": _ratio(
            inconsistency_hits,
            len(inconsistency_cases),
        ),
        "unsafe_sql_block_rate": _ratio(
            unsafe_sql_hits,
            len(unsafe_sql_cases),
        ),
        "end_to_end_success_rate": _ratio(
            passed_cases,
            len(case_results),
        ),
        "p50_runtime_ms": _percentile(runtimes, 0.50),
        "p95_runtime_ms": _percentile(runtimes, 0.95),
    }
    metric_counts = {
        "impacted_batch_precision": MetricCount(
            numerator=true_batches,
            denominator=predicted_batches,
            definition="gold impacted batches among all predicted impacted batches",
        ),
        "impacted_batch_recall": MetricCount(
            numerator=true_batches,
            denominator=gold_batches,
            definition="recovered impacted batches across every case with Gold",
        ),
        "impacted_order_recall": MetricCount(
            numerator=true_orders,
            denominator=gold_orders,
            definition="recovered impacted orders across every case with Gold",
        ),
        "false_negative_rate": MetricCount(
            numerator=missed_batches,
            denominator=gold_batches,
            definition="missed Gold batches across every case with Gold",
        ),
        "evidence_coverage": MetricCount(
            numerator=grounded_claims,
            denominator=len(all_claims),
            definition="claims supported by matching evidence kind and payload value",
        ),
        "inconsistency_gate_recall": MetricCount(
            numerator=inconsistency_hits,
            denominator=len(inconsistency_cases),
            definition="inconsistency-tagged cases routed to HUMAN_REVIEW",
        ),
        "unsafe_sql_block_rate": MetricCount(
            numerator=unsafe_sql_hits,
            denominator=len(unsafe_sql_cases),
            definition="unsafe-SQL-tagged cases stopped before a report",
        ),
        "end_to_end_success_rate": MetricCount(
            numerator=passed_cases,
            denominator=len(case_results),
            definition="cases matching expected gate and report scope",
        ),
        "p50_runtime_ms": MetricCount(
            numerator=None,
            denominator=len(runtimes),
            definition="nearest-rank p50 over per-case runtime milliseconds",
        ),
        "p95_runtime_ms": MetricCount(
            numerator=None,
            denominator=len(runtimes),
            definition="nearest-rank p95 over per-case runtime milliseconds",
        ),
    }
    return (
        {name: metrics[name] for name in FOODTRACE_METRIC_NAMES},
        {name: metric_counts[name] for name in FOODTRACE_METRIC_NAMES},
    )


def _case_set_sha256(cases: tuple[FoodTraceBenchmarkCase, ...]) -> str:
    canonical = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return round(values[index], 4)


class _NotFoundGraphRepository:
    def trace_batches(self, ingredient_lot_id: str) -> GraphTraceResult:
        return GraphTraceResult(
            status="not_found",
            ingredient_lot_id=ingredient_lot_id,
        )


class _MissingEdgeGraphRepository:
    def trace_batches(self, ingredient_lot_id: str) -> GraphTraceResult:
        return GraphTraceResult(
            status="invalid_dependency_response",
            ingredient_lot_id=ingredient_lot_id,
            error_code="missing_required_graph_edge",
        )


class _MismatchedExposureRepository:
    def lookup_exposures(
        self,
        production_batch_ids: tuple[str, ...],
    ) -> ExposureLookupResult:
        return ExposureLookupResult(
            status="ok",
            production_batch_ids=("BATCH-FT-004",),
        )


class _UnavailableExposureRepository:
    def lookup_exposures(
        self,
        production_batch_ids: tuple[str, ...],
    ) -> ExposureLookupResult:
        return ExposureLookupResult(
            status="dependency_unavailable",
            production_batch_ids=production_batch_ids,
            error_code="dependency_unavailable",
        )


class _MissingSopRepository:
    def lookup(self, incident: object) -> SopLookupResult:
        return SopLookupResult(status="not_found", error_code="sop_not_found")


class _ExpandedGraphRepository:
    def trace_batches(self, ingredient_lot_id: str) -> GraphTraceResult:
        return GraphTraceResult(
            status="ok",
            ingredient_lot_id=ingredient_lot_id,
            recipe_ids=("RECIPE-FT-001",),
            product_ids=("PRODUCT-FT-001",),
            production_batch_ids=(
                "BATCH-FT-001",
                "BATCH-FT-002",
                "BATCH-FT-003",
                "BATCH-FT-004",
            ),
        )


class _UnsafeSqlGraphRepository:
    def trace_batches(self, ingredient_lot_id: str) -> GraphTraceResult:
        return GraphTraceResult(
            status="ok",
            ingredient_lot_id=ingredient_lot_id,
            recipe_ids=("RECIPE-FT-001",),
            product_ids=("PRODUCT-FT-001",),
            production_batch_ids=tuple(
                f"BATCH-OVERFLOW-{index:03d}" for index in range(101)
            ),
        )
