from gustobot.application.deepreason.models import EvidenceItem
from gustobot.application.meal_planning.fact_verification import (
    ClaimStatus,
    FactClaim,
    FactOperator,
    verify_fact_claims,
)


def fact(entity_id, field, value, *, complete=None):
    return EvidenceItem.create_fact(
        task_id="recipe-1",
        source_type="recipe_fact",
        source="neo4j_readonly",
        entity_type="recipe",
        entity_id=entity_id,
        field=field,
        value=value,
        provenance={} if complete is None else {"complete": complete},
    )


def test_atomic_fact_evidence_keeps_value_and_provenance():
    evidence = fact("r101", "total_minutes", 15)

    assert evidence.source == "neo4j_readonly"
    assert evidence.metadata == {
        "entity_type": "recipe",
        "entity_id": "r101",
        "field": "total_minutes",
        "value": 15,
    }
    assert '"total_minutes"' in evidence.content


def test_numeric_claim_is_supported_or_contradicted_deterministically():
    evidence = fact("r101", "calories_kcal", 280)
    supported = FactClaim(
        claim_id="c1",
        entity_id="r101",
        field="calories_kcal",
        operator=FactOperator.EQUALS,
        expected_value=280,
        evidence_ids=[evidence.evidence_id],
    )
    contradicted = supported.model_copy(
        update={"claim_id": "c2", "expected_value": 380}
    )

    result = verify_fact_claims([supported, contradicted], [evidence])

    assert result.checks[0].status is ClaimStatus.SUPPORTED
    assert result.checks[1].status is ClaimStatus.CONTRADICTED
    assert result.all_supported is False


def test_selected_membership_and_excluded_ingredients_are_hard_checks():
    identity = fact("r101", "exists", True)
    ingredients = fact(
        "r101",
        "ingredients",
        ["egg", "tomato"],
        complete=True,
    )
    membership = FactClaim(
        claim_id="membership",
        entity_id="r101",
        operator=FactOperator.IN_SELECTED,
        evidence_ids=[identity.evidence_id],
    )
    allergen_safe = FactClaim(
        claim_id="allergen",
        entity_id="r101",
        field="ingredients",
        operator=FactOperator.EXCLUDES,
        expected_value=["peanut"],
        evidence_ids=[ingredients.evidence_id],
    )

    allowed = verify_fact_claims(
        [membership, allergen_safe],
        [identity, ingredients],
        selected_recipe_ids=["r101"],
    )
    denied = verify_fact_claims(
        [membership],
        [identity],
        selected_recipe_ids=["r205"],
    )

    assert allowed.all_supported is True
    assert denied.checks[0].status is ClaimStatus.CONTRADICTED


def test_excluded_ingredient_absence_requires_a_complete_ingredient_set():
    partial = fact("r101", "ingredients", ["egg", "tomato"], complete=False)
    claim = FactClaim(
        claim_id="allergen",
        entity_id="r101",
        field="ingredients",
        operator=FactOperator.EXCLUDES,
        expected_value=["peanut"],
        evidence_ids=[partial.evidence_id],
    )

    result = verify_fact_claims([claim], [partial])

    assert result.checks[0].status is ClaimStatus.NOT_VERIFIABLE


def test_missing_claim_links_and_unknown_evidence_are_distinguished():
    unsupported = FactClaim(
        claim_id="unsupported",
        entity_id="r101",
        field="suitable_for_weight_loss",
        operator=FactOperator.EQUALS,
        expected_value=True,
    )
    unknown = unsupported.model_copy(
        update={"claim_id": "unknown", "evidence_ids": ["ev_missing"]}
    )

    result = verify_fact_claims([unsupported, unknown], [])

    assert result.checks[0].status is ClaimStatus.UNSUPPORTED
    assert result.checks[1].status is ClaimStatus.NOT_VERIFIABLE
