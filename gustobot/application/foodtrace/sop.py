from __future__ import annotations

from typing import Literal

from .models import FoodTraceModel, IncidentNotice


class SopLookupResult(FoodTraceModel):
    status: Literal["ok", "not_found", "dependency_unavailable"]
    procedure_id: str | None = None
    title: str | None = None
    error_code: str | None = None


class LocalAllergenSopRepository:
    """Deterministic MVP SOP source; replaceable by an enterprise repository."""

    def lookup(self, incident: IncidentNotice) -> SopLookupResult:
        if incident.allergen != "peanut" or incident.declared_on_label:
            return SopLookupResult(status="not_found", error_code="sop_not_found")
        return SopLookupResult(
            status="ok",
            procedure_id="SOP-ALLERGEN-RECALL-001",
            title="Undeclared allergen containment and recall",
        )
