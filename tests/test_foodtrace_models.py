from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from gustobot.application.foodtrace.models import (
    GoldRecallCase,
    IncidentNotice,
    RecallScope,
)


def test_incident_contract_accepts_only_unlabeled_peanut_notice():
    incident = IncidentNotice(
        incident_id="INC-PEANUT-001",
        supplier_id="SUP-001",
        ingredient_lot_id="LOT-PEANUT-001",
        allergen="peanut",
        declared_on_label=False,
        received_at=datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc),
    )

    assert incident.allergen == "peanut"
    assert incident.declared_on_label is False

    with pytest.raises(ValidationError):
        IncidentNotice(
            incident_id="INC-MILK-001",
            supplier_id="SUP-001",
            ingredient_lot_id="LOT-MILK-001",
            allergen="milk",
            declared_on_label=False,
            received_at=datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc),
        )


def test_recall_scope_normalizes_ids_for_reproducible_comparison():
    scope = RecallScope(
        impacted_product_ids=["PROD-002", "PROD-001", "PROD-001"],
        impacted_batch_ids=["BATCH-002", "BATCH-001"],
        impacted_store_ids=["STORE-002", "STORE-001"],
        impacted_order_ids=["ORDER-002", "ORDER-001", "ORDER-001"],
    )

    assert scope.impacted_product_ids == ("PROD-001", "PROD-002")
    assert scope.impacted_batch_ids == ("BATCH-001", "BATCH-002")
    assert scope.impacted_store_ids == ("STORE-001", "STORE-002")
    assert scope.impacted_order_ids == ("ORDER-001", "ORDER-002")


@pytest.mark.parametrize(
    "malformed_ids",
    ["PRODUCT-FT-001", b"PRODUCT-FT-001", ["PRODUCT-FT-001", " "]],
)
def test_recall_scope_rejects_scalar_or_blank_id_containers(malformed_ids):
    with pytest.raises(ValidationError):
        RecallScope(impacted_product_ids=malformed_ids)


def test_gold_case_round_trips_without_losing_expected_ids():
    case = GoldRecallCase(
        case_id="allergen_peanut_001",
        incident=IncidentNotice(
            incident_id="INC-PEANUT-001",
            supplier_id="SUP-001",
            ingredient_lot_id="LOT-PEANUT-001",
            allergen="peanut",
            declared_on_label=False,
            received_at=datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc),
        ),
        gold_scope=RecallScope(
            impacted_product_ids=["PROD-001"],
            impacted_batch_ids=["BATCH-001"],
            impacted_store_ids=["STORE-001"],
            impacted_order_ids=["ORDER-001"],
        ),
        expected_gate="ALLOW_REPORT",
        scenario_tags=["happy_path", "unlabeled_peanut"],
    )

    restored = GoldRecallCase.model_validate_json(case.model_dump_json())

    assert restored == case
