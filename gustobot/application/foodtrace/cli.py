from __future__ import annotations

from .evidence import EvidenceItem
from .fixtures import build_foodtrace_fixture
from .gate import GateDecision
from .models import FoodTraceModel, IncidentNotice, RecallScope
from .repositories import FixtureGraphRepository, FixtureSqlRepository
from .workflow import FoodTraceWorkflow, TraceStep


class RecallReport(FoodTraceModel):
    incident_id: str
    impacted_product_ids: tuple[str, ...]
    impacted_batch_ids: tuple[str, ...]
    impacted_store_ids: tuple[str, ...]
    impacted_order_ids: tuple[str, ...]


class FoodTraceDemoResult(FoodTraceModel):
    case_id: str
    incident: IncidentNotice
    scope: RecallScope
    evidence: tuple[EvidenceItem, ...]
    gate: GateDecision
    trace: tuple[TraceStep, ...]
    report: RecallReport | None = None


def run_foodtrace_case(case_id: str) -> FoodTraceDemoResult:
    fixture = build_foodtrace_fixture()
    matching_case = next(
        (case for case in fixture.gold_cases if case.case_id == case_id),
        None,
    )
    if matching_case is None:
        raise ValueError(f"unknown FoodTrace case: {case_id}")
    return run_foodtrace_incident(matching_case.incident, case_id=matching_case.case_id)


def run_foodtrace_incident(
    incident: IncidentNotice,
    *,
    case_id: str = "ad_hoc_incident",
) -> FoodTraceDemoResult:
    fixture = build_foodtrace_fixture()
    workflow_result = FoodTraceWorkflow(
        graph_repository=FixtureGraphRepository(fixture),
        exposure_repository=FixtureSqlRepository(fixture),
    ).run(incident)

    trace = list(workflow_result.trace)
    report: RecallReport | None = None
    if workflow_result.gate.state == "ALLOW_REPORT":
        report = RecallReport(
            incident_id=incident.incident_id,
            impacted_product_ids=workflow_result.scope.impacted_product_ids,
            impacted_batch_ids=workflow_result.scope.impacted_batch_ids,
            impacted_store_ids=workflow_result.scope.impacted_store_ids,
            impacted_order_ids=workflow_result.scope.impacted_order_ids,
        )
        report_evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for claim in workflow_result.claims
                    for evidence_id in claim.evidence_ids
                }
            )
        )
        trace.append(
            TraceStep(
                name="Recall Report",
                status="ok",
                evidence_ids=report_evidence_ids,
            )
        )
    else:
        trace.append(
            TraceStep(
                name="Recall Report",
                status="skipped",
                detail="gate_not_allowed",
            )
        )

    return FoodTraceDemoResult(
        case_id=case_id,
        incident=incident,
        scope=workflow_result.scope,
        evidence=workflow_result.evidence,
        gate=workflow_result.gate,
        trace=tuple(trace),
        report=report,
    )


def render_demo_trace(demo: FoodTraceDemoResult) -> str:
    lines = [
        "FoodTrace undeclared-allergen incident demo",
        f"Case: {demo.case_id}",
        f"Incident: {demo.incident.incident_id}",
        "",
        "Trace:",
    ]
    for index, step in enumerate(demo.trace, start=1):
        evidence_suffix = (
            f" | evidence={','.join(step.evidence_ids)}" if step.evidence_ids else ""
        )
        lines.append(f"  {index}. {step.name}: {step.status}{evidence_suffix}")

    lines.extend(["", f"Gate: {demo.gate.state}"])
    if demo.gate.reasons:
        lines.append("Reasons: " + ", ".join(demo.gate.reasons))

    if demo.report is not None:
        lines.extend(
            [
                "",
                "Recall scope:",
                "  Products: " + ", ".join(demo.report.impacted_product_ids),
                "  Batches: " + ", ".join(demo.report.impacted_batch_ids),
                "  Stores: " + ", ".join(demo.report.impacted_store_ids),
                "  Orders: " + ", ".join(demo.report.impacted_order_ids),
            ]
        )

    lines.extend(["", "Evidence:"])
    for item in demo.evidence:
        lines.append(f"  {item.evidence_id} | {item.kind} | {item.source}")
    return "\n".join(lines)
