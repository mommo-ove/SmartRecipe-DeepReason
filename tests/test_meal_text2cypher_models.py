import pytest
from pydantic import ValidationError

from gustobot.application.meal_planning.text2cypher.models import (
    CypherCandidate,
    CypherExecutionResult,
    CypherRunStatus,
    CypherSource,
    Text2CypherResult,
    ValidationIssue,
    ValidationReport,
)


def test_candidate_requires_a_non_empty_statement_and_bounds_repair_attempts():
    with pytest.raises(ValidationError):
        CypherCandidate(statement=" ", source=CypherSource.DYNAMIC)

    with pytest.raises(ValidationError):
        CypherCandidate(
            statement="MATCH (recipe:Recipe) RETURN recipe.recipe_id AS recipe_id",
            source=CypherSource.DYNAMIC,
            attempt=3,
        )


def test_validation_report_exposes_structured_issues():
    issue = ValidationIssue(
        code="WRITE_CLAUSE",
        stage="safety",
        message="CREATE is not allowed",
    )

    report = ValidationReport(issues=[issue])

    assert report.valid is False
    assert report.issues[0].code == "WRITE_CLAUSE"


def test_execution_result_normalizes_recipe_ids_without_guessing_from_names():
    execution = CypherExecutionResult(
        status=CypherRunStatus.SUCCESS,
        records=[
            {"recipe_id": "r101", "recipe_name": "A"},
            {"recipe_id": "r101", "recipe_name": "A"},
            {"recipe_id": "r205", "recipe_name": "B"},
            {"recipe_name": "No stable id"},
        ],
        selected_recipe_ids=["r101", "r101", "r205", ""],
        latency_ms=1.2,
    )

    assert execution.selected_recipe_ids == ["r101", "r205"]


def test_text2cypher_result_keeps_failure_status_and_trace_explicit():
    result = Text2CypherResult(
        question="Find recipes with celery",
        status=CypherRunStatus.VALIDATION_FAILED,
        trace=["template_route", "generate", "validate"],
        validation_reports=[
            ValidationReport(
                issues=[
                    ValidationIssue(
                        code="UNKNOWN_LABEL",
                        stage="schema",
                        message="Unknown label Dish",
                    )
                ]
            )
        ],
    )

    assert result.status is CypherRunStatus.VALIDATION_FAILED
    assert result.selected_recipe_ids == []
    assert result.trace[-1] == "validate"
