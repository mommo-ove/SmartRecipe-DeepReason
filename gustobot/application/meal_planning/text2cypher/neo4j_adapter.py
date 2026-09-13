from __future__ import annotations

from time import perf_counter
from typing import Any

from .models import (
    CypherCandidate,
    CypherExecutionResult,
    CypherRunStatus,
    ValidationIssue,
    ValidationReport,
)


def _explain_issue(error: Exception) -> ValidationIssue:
    neo4j_code = str(getattr(error, "code", ""))
    if neo4j_code.endswith("SyntaxError"):
        code = "CYPHER_SYNTAX_ERROR"
    elif neo4j_code.endswith("ParameterMissing"):
        code = "MISSING_PARAMETER"
    else:
        code = "NEO4J_EXPLAIN_ERROR"
    return ValidationIssue(code=code, stage="explain", message=str(error))


class Neo4jCypherAdapter:
    def __init__(self, driver: Any, *, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    def explain(self, candidate: CypherCandidate) -> ValidationReport:
        try:
            self._driver.execute_query(
                f"EXPLAIN {candidate.statement}",
                parameters_=candidate.parameters,
                database_=self._database,
            )
        except Exception as error:
            return ValidationReport(issues=[_explain_issue(error)])
        return ValidationReport()

    def execute(self, candidate: CypherCandidate) -> CypherExecutionResult:
        started = perf_counter()
        try:
            raw_records, _, _ = self._driver.execute_query(
                candidate.statement,
                parameters_=candidate.parameters,
                database_=self._database,
            )
            records = [dict(record) for record in raw_records]
            recipe_ids = [
                str(record["recipe_id"])
                for record in records
                if record.get("recipe_id") not in {None, ""}
            ]
            return CypherExecutionResult(
                status=(CypherRunStatus.SUCCESS if records else CypherRunStatus.EMPTY),
                records=records,
                selected_recipe_ids=recipe_ids,
                latency_ms=(perf_counter() - started) * 1000,
            )
        except Exception as error:
            return CypherExecutionResult(
                status=CypherRunStatus.EXECUTION_FAILED,
                latency_ms=(perf_counter() - started) * 1000,
                error=str(error),
            )
