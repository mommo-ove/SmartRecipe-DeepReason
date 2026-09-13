import pytest

from gustobot.application.foodtrace.evidence import make_evidence
from gustobot.application.foodtrace.gate import (
    GateInput,
    ImpactClaim,
    evaluate_gate,
)


def test_gate_allows_complete_evidence_bound_review():
    evidence = (
        make_evidence("graph_scope", "neo4j", {"batch_ids": ["B1"]}),
        make_evidence("order_exposure", "mysql", {"order_ids": ["O1"]}),
        make_evidence("consistency_review", "python", {"status": "consistent"}),
    )
    decision = evaluate_gate(
        GateInput(
            evidence=evidence,
            claims=(
                ImpactClaim(
                    metric="impacted_batch_count",
                    value=1,
                    evidence_ids=(evidence[0].evidence_id,),
                ),
            ),
        )
    )

    assert decision.state == "ALLOW_REPORT"
    assert decision.reasons == ()


def test_evidence_id_canonical_encoding_does_not_collapse_subject_boundaries():
    first = make_evidence(
        "graph_scope",
        "neo4j",
        {"batch_ids": ["B1"]},
        subject_ids=("a,b", "c"),
    )
    second = make_evidence(
        "graph_scope",
        "neo4j",
        {"batch_ids": ["B1"]},
        subject_ids=("a", "b,c"),
    )

    assert first.evidence_id != second.evidence_id


def test_evidence_rejects_scalar_subject_id_collection():
    with pytest.raises(ValueError, match="subject_ids"):
        make_evidence(
            "graph_scope",
            "neo4j",
            {"batch_ids": ["B1"]},
            subject_ids="BATCH-FT-001",
        )


def test_gate_routes_missing_gold_evidence_to_human_review():
    graph_evidence = make_evidence(
        "graph_scope",
        "neo4j",
        {"batch_ids": ["B1"]},
    )
    decision = evaluate_gate(
        GateInput(
            evidence=(graph_evidence,),
            claims=(
                ImpactClaim(
                    metric="impacted_batch_count",
                    value=1,
                    evidence_ids=(graph_evidence.evidence_id,),
                ),
            ),
        )
    )

    assert decision.state == "HUMAN_REVIEW"
    assert set(decision.reasons) == {
        "missing_evidence:consistency_review",
        "missing_evidence:order_exposure",
    }


def test_gate_rejects_impacted_number_without_known_evidence_id():
    evidence = (
        make_evidence("graph_scope", "neo4j", {"batch_ids": ["B1"]}),
        make_evidence("order_exposure", "mysql", {"order_ids": ["O1"]}),
        make_evidence("consistency_review", "python", {"status": "consistent"}),
    )
    decision = evaluate_gate(
        GateInput(
            evidence=evidence,
            claims=(
                ImpactClaim(
                    metric="impacted_order_count",
                    value=7,
                    evidence_ids=("EV-DOES-NOT-EXIST",),
                ),
            ),
        )
    )

    assert decision.state == "HUMAN_REVIEW"
    assert decision.reasons == ("ungrounded_claim:impacted_order_count",)


def test_gate_rejects_claim_bound_to_wrong_evidence_kind():
    graph_evidence = make_evidence(
        "graph_scope",
        "neo4j",
        {"batch_ids": ["B1"]},
    )
    evidence = (
        graph_evidence,
        make_evidence("order_exposure", "mysql", {"order_ids": ["O1"]}),
        make_evidence("consistency_review", "python", {"status": "consistent"}),
    )
    decision = evaluate_gate(
        GateInput(
            evidence=evidence,
            claims=(
                ImpactClaim(
                    metric="impacted_order_count",
                    value=999,
                    evidence_ids=(graph_evidence.evidence_id,),
                ),
            ),
        )
    )

    assert decision.state == "HUMAN_REVIEW"
    assert decision.reasons == ("ungrounded_claim:impacted_order_count",)


def test_gate_rejects_claim_value_not_supported_by_evidence_payload():
    evidence = (
        make_evidence("graph_scope", "neo4j", {"batch_ids": ["B1"]}),
        make_evidence("order_exposure", "mysql", {"order_ids": ["O1"]}),
        make_evidence("consistency_review", "python", {"status": "consistent"}),
    )
    decision = evaluate_gate(
        GateInput(
            evidence=evidence,
            claims=(
                ImpactClaim(
                    metric="impacted_order_count",
                    value=2,
                    evidence_ids=(evidence[1].evidence_id,),
                ),
            ),
        )
    )

    assert decision.state == "HUMAN_REVIEW"
    assert decision.reasons == ("ungrounded_claim:impacted_order_count",)


def test_gate_blocks_dependency_failure_even_when_old_evidence_exists():
    evidence = (
        make_evidence("graph_scope", "neo4j", {"batch_ids": ["B1"]}),
        make_evidence("order_exposure", "mysql", {"order_ids": ["O1"]}),
        make_evidence("consistency_review", "python", {"status": "consistent"}),
    )
    decision = evaluate_gate(
        GateInput(
            evidence=evidence,
            claims=(),
            dependency_failures=("graph_scope:dependency_unavailable",),
        )
    )

    assert decision.state == "BLOCKED"
    assert decision.reasons == ("graph_scope:dependency_unavailable",)
