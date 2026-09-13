from __future__ import annotations

from typing import Any

from pydantic import Field

from .evidence import EvidenceItem, make_evidence
from .gate import GateDecision, GateInput, ImpactClaim, evaluate_gate
from .models import FoodTraceModel, IncidentNotice, RecallScope
from .reviewer import review_consistency
from .sop import LocalAllergenSopRepository


class TraceStep(FoodTraceModel):
    name: str
    status: str
    evidence_ids: tuple[str, ...] = ()
    detail: str | None = None


class FoodTraceWorkflowResult(FoodTraceModel):
    incident: IncidentNotice
    scope: RecallScope = Field(default_factory=RecallScope)
    claims: tuple[ImpactClaim, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    gate: GateDecision
    trace: tuple[TraceStep, ...]


class FoodTraceWorkflow:
    def __init__(
        self,
        *,
        graph_repository: Any,
        exposure_repository: Any,
        sop_repository: Any | None = None,
        max_batch_scope: int = 100,
    ) -> None:
        if max_batch_scope < 1:
            raise ValueError("max_batch_scope must be at least 1")
        self.graph_repository = graph_repository
        self.exposure_repository = exposure_repository
        self.sop_repository = sop_repository or LocalAllergenSopRepository()
        self.max_batch_scope = max_batch_scope

    def run(self, incident: IncidentNotice) -> FoodTraceWorkflowResult:
        evidence: list[EvidenceItem] = []
        trace: list[TraceStep] = []
        safety_reasons: list[str] = []

        incident_evidence = make_evidence(
            "incident_notice",
            "incident_parser",
            incident.model_dump(mode="json"),
            subject_ids=(incident.incident_id, incident.ingredient_lot_id),
        )
        evidence.append(incident_evidence)
        trace.append(_trace("Incident Parser", "ok", incident_evidence))

        try:
            sop_result = self.sop_repository.lookup(incident)
        except Exception:
            trace.append(TraceStep(name="SOP Lookup", status="dependency_unavailable"))
            return self._finish_blocked(
                incident,
                evidence,
                trace,
                dependency_failure="sop_lookup:dependency_unavailable",
            )
        if sop_result.status == "ok":
            sop_evidence = make_evidence(
                "sop_procedure",
                "sop_repository",
                sop_result.model_dump(mode="json"),
                subject_ids=(sop_result.procedure_id,),
            )
            evidence.append(sop_evidence)
            trace.append(_trace("SOP Lookup", "ok", sop_evidence))
        else:
            reason = f"sop_lookup:{sop_result.status}"
            trace.append(TraceStep(name="SOP Lookup", status=sop_result.status))
            if sop_result.status == "dependency_unavailable":
                return self._finish_blocked(
                    incident,
                    evidence,
                    trace,
                    dependency_failure=reason,
                )
            safety_reasons.append(reason)

        try:
            graph_result = self.graph_repository.trace_batches(
                incident.ingredient_lot_id
            )
        except Exception:
            trace.append(TraceStep(name="Graph Scope", status="dependency_unavailable"))
            return self._finish_blocked(
                incident,
                evidence,
                trace,
                dependency_failure="graph_scope:dependency_unavailable",
            )
        if graph_result.status != "ok":
            reason = f"graph_scope:{graph_result.status}"
            trace.append(TraceStep(name="Graph Scope", status=graph_result.status))
            if graph_result.status in {
                "dependency_unavailable",
                "invalid_dependency_response",
            }:
                return self._finish_blocked(
                    incident,
                    evidence,
                    trace,
                    dependency_failure=reason,
                )
            safety_reasons.append(reason)
            return self._finish_review(
                incident,
                evidence,
                trace,
                safety_reasons=safety_reasons,
            )

        graph_evidence = make_evidence(
            "graph_scope",
            "graph_repository",
            graph_result.model_dump(mode="json"),
            subject_ids=(
                *graph_result.product_ids,
                *graph_result.production_batch_ids,
            ),
        )
        evidence.append(graph_evidence)
        trace.append(_trace("Graph Scope", "ok", graph_evidence))

        batch_scope_size = len(graph_result.production_batch_ids)
        if batch_scope_size > self.max_batch_scope:
            safety_reasons.append(
                f"batch_scope_expanded:{batch_scope_size}>{self.max_batch_scope}"
            )
            return self._finish_review(
                incident,
                evidence,
                trace,
                safety_reasons=safety_reasons,
            )

        try:
            exposure_result = self.exposure_repository.lookup_exposures(
                graph_result.production_batch_ids
            )
        except Exception:
            trace.append(
                TraceStep(name="Order Exposure", status="dependency_unavailable")
            )
            return self._finish_blocked(
                incident,
                evidence,
                trace,
                dependency_failure="order_exposure:dependency_unavailable",
            )
        if exposure_result.status != "ok":
            reason = f"order_exposure:{exposure_result.status}"
            trace.append(
                TraceStep(name="Order Exposure", status=exposure_result.status)
            )
            if exposure_result.status in {
                "dependency_unavailable",
                "invalid_dependency_response",
            }:
                return self._finish_blocked(
                    incident,
                    evidence,
                    trace,
                    dependency_failure=reason,
                )
            safety_reasons.append(reason)
            return self._finish_review(
                incident,
                evidence,
                trace,
                safety_reasons=safety_reasons,
            )

        exposure_evidence = make_evidence(
            "order_exposure",
            "exposure_repository",
            exposure_result.model_dump(mode="json"),
            subject_ids=(
                *(row.inventory_id for row in exposure_result.inventories),
                *(row.order_id for row in exposure_result.orders),
            ),
        )
        evidence.append(exposure_evidence)
        trace.append(_trace("Order Exposure", "ok", exposure_evidence))

        review = review_consistency(
            graph_result,
            exposure_result,
            max_batch_scope=self.max_batch_scope,
        )
        safety_reasons.extend(review.reasons)
        review_evidence = make_evidence(
            "consistency_review",
            "python",
            review.model_dump(mode="json"),
            subject_ids=review.scope.impacted_batch_ids,
        )
        evidence.append(review_evidence)
        trace.append(_trace("Consistency Reviewer", review.status, review_evidence))

        claims = (
            ImpactClaim(
                metric="impacted_product_count",
                value=len(review.scope.impacted_product_ids),
                evidence_ids=(graph_evidence.evidence_id,),
            ),
            ImpactClaim(
                metric="impacted_batch_count",
                value=len(review.scope.impacted_batch_ids),
                evidence_ids=(graph_evidence.evidence_id,),
            ),
            ImpactClaim(
                metric="impacted_store_count",
                value=len(review.scope.impacted_store_ids),
                evidence_ids=(exposure_evidence.evidence_id,),
            ),
            ImpactClaim(
                metric="impacted_order_count",
                value=len(review.scope.impacted_order_ids),
                evidence_ids=(exposure_evidence.evidence_id,),
            ),
        )
        return self._finish(
            incident,
            review.scope,
            claims,
            evidence,
            trace,
            safety_reasons=safety_reasons,
        )

    def _finish_blocked(
        self,
        incident: IncidentNotice,
        evidence: list[EvidenceItem],
        trace: list[TraceStep],
        *,
        dependency_failure: str,
    ) -> FoodTraceWorkflowResult:
        return self._finish(
            incident,
            RecallScope(),
            (),
            evidence,
            trace,
            dependency_failures=(dependency_failure,),
        )

    def _finish_review(
        self,
        incident: IncidentNotice,
        evidence: list[EvidenceItem],
        trace: list[TraceStep],
        *,
        safety_reasons: list[str],
    ) -> FoodTraceWorkflowResult:
        return self._finish(
            incident,
            RecallScope(),
            (),
            evidence,
            trace,
            safety_reasons=safety_reasons,
        )

    @staticmethod
    def _finish(
        incident: IncidentNotice,
        scope: RecallScope,
        claims: tuple[ImpactClaim, ...],
        evidence: list[EvidenceItem],
        trace: list[TraceStep],
        *,
        dependency_failures: tuple[str, ...] = (),
        safety_reasons: list[str] | tuple[str, ...] = (),
    ) -> FoodTraceWorkflowResult:
        gate = evaluate_gate(
            GateInput(
                evidence=tuple(evidence),
                claims=claims,
                dependency_failures=dependency_failures,
                safety_reasons=tuple(safety_reasons),
            )
        )
        gate_evidence = make_evidence(
            "decision_gate",
            "python",
            gate.model_dump(mode="json"),
            subject_ids=(incident.incident_id,),
        )
        evidence.append(gate_evidence)
        trace.append(_trace("Decision Gate", gate.state, gate_evidence))
        return FoodTraceWorkflowResult(
            incident=incident,
            scope=scope,
            claims=claims,
            evidence=tuple(evidence),
            gate=gate,
            trace=tuple(trace),
        )


def _trace(name: str, status: str, evidence: EvidenceItem) -> TraceStep:
    return TraceStep(
        name=name,
        status=status,
        evidence_ids=(evidence.evidence_id,),
    )
