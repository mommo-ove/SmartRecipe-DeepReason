from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol, Sequence

from .generator import Text2CypherGenerator
from .models import (
    CypherCandidate,
    CypherRunStatus,
    Text2CypherResult,
    ValidationIssue,
    ValidationReport,
)
from .neo4j_adapter import Neo4jCypherAdapter
from .safety import validate_cypher_safety
from .schema import GraphSchemaSnapshot, Neo4jSchemaLoader, validate_cypher_schema
from .templates import TemplateRequest, build_template_candidate


_NON_REPAIRABLE_SAFETY_CODES = {
    "INVALID_START",
    "MULTIPLE_STATEMENTS",
    "UNSAFE_CLAUSE",
}


class SchemaLoader(Protocol):
    def load(self) -> GraphSchemaSnapshot: ...


def _trace(stage: str, status: str, started: float) -> str:
    duration_ms = (perf_counter() - started) * 1000
    return f"{stage}:{status}:{duration_ms:.3f}ms"


class Text2CypherService:
    """Deterministically controls generation, validation, repair and execution."""

    def __init__(
        self,
        *,
        generator: Text2CypherGenerator,
        adapter: Neo4jCypherAdapter,
        schema_loader: SchemaLoader | Neo4jSchemaLoader,
    ) -> None:
        self._generator = generator
        self._adapter = adapter
        self._schema_loader = schema_loader

    async def run(
        self,
        *,
        question: str,
        template_request: TemplateRequest | None = None,
        parameters: dict[str, Any] | None = None,
        examples: Sequence[tuple[str, str]] = (),
        require_recipe_ids: bool = True,
    ) -> Text2CypherResult:
        trace: list[str] = []
        reports: list[ValidationReport] = []
        schema = self._schema_loader.load()

        started = perf_counter()
        candidate = (
            build_template_candidate(template_request)
            if template_request is not None
            else None
        )
        trace.append(
            _trace("template_route", "hit" if candidate is not None else "miss", started)
        )

        if candidate is None:
            started = perf_counter()
            candidate = await self._generator.generate(
                question=question,
                schema=schema,
                examples=examples,
                parameters=parameters or {},
            )
            trace.append(_trace("generate", "ok", started))

        while True:
            safety = self._validate_safety(
                candidate,
                require_recipe_ids=require_recipe_ids,
                trace=trace,
            )
            reports.append(safety)
            if not safety.valid:
                if any(
                    issue.code in _NON_REPAIRABLE_SAFETY_CODES
                    for issue in safety.issues
                ):
                    return self._validation_failure(
                        question=question,
                        candidate=candidate,
                        reports=reports,
                        trace=trace,
                    )
                repaired = await self._repair(
                    question=question,
                    candidate=candidate,
                    issues=safety.issues,
                    schema=schema,
                    trace=trace,
                )
                if repaired is None:
                    return self._validation_failure(
                        question=question,
                        candidate=candidate,
                        reports=reports,
                        trace=trace,
                    )
                candidate = repaired
                continue

            schema_report = self._validate_schema(candidate, schema, trace)
            reports.append(schema_report)
            if not schema_report.valid:
                repaired = await self._repair(
                    question=question,
                    candidate=candidate,
                    issues=schema_report.issues,
                    schema=schema,
                    trace=trace,
                )
                if repaired is None:
                    return self._validation_failure(
                        question=question,
                        candidate=candidate,
                        reports=reports,
                        trace=trace,
                    )
                candidate = repaired
                continue

            started = perf_counter()
            explain_report = self._adapter.explain(candidate)
            trace.append(
                _trace("explain", "ok" if explain_report.valid else "failed", started)
            )
            reports.append(explain_report)
            if not explain_report.valid:
                repaired = await self._repair(
                    question=question,
                    candidate=candidate,
                    issues=explain_report.issues,
                    schema=schema,
                    trace=trace,
                )
                if repaired is None:
                    return self._validation_failure(
                        question=question,
                        candidate=candidate,
                        reports=reports,
                        trace=trace,
                    )
                candidate = repaired
                continue

            started = perf_counter()
            execution = self._adapter.execute(candidate)
            trace.append(_trace("execute", execution.status.value, started))
            return Text2CypherResult(
                question=question,
                status=execution.status,
                candidate=candidate,
                validation_reports=reports,
                execution=execution,
                selected_recipe_ids=execution.selected_recipe_ids,
                trace=trace,
            )

    @staticmethod
    def _validate_safety(
        candidate: CypherCandidate,
        *,
        require_recipe_ids: bool,
        trace: list[str],
    ) -> ValidationReport:
        started = perf_counter()
        report = validate_cypher_safety(
            candidate.statement,
            require_recipe_ids=require_recipe_ids,
        )
        trace.append(
            _trace("validate:safety", "ok" if report.valid else "failed", started)
        )
        return report

    @staticmethod
    def _validate_schema(
        candidate: CypherCandidate,
        schema: GraphSchemaSnapshot,
        trace: list[str],
    ) -> ValidationReport:
        started = perf_counter()
        report = validate_cypher_schema(candidate.statement, schema)
        trace.append(
            _trace("validate:schema", "ok" if report.valid else "failed", started)
        )
        return report

    async def _repair(
        self,
        *,
        question: str,
        candidate: CypherCandidate,
        issues: Sequence[ValidationIssue],
        schema: GraphSchemaSnapshot,
        trace: list[str],
    ) -> CypherCandidate | None:
        if candidate.attempt >= 2:
            return None
        started = perf_counter()
        repaired = await self._generator.repair(
            question=question,
            candidate=candidate,
            issues=issues,
            schema=schema,
        )
        trace.append(_trace("repair", f"attempt_{repaired.attempt}", started))
        return repaired

    @staticmethod
    def _validation_failure(
        *,
        question: str,
        candidate: CypherCandidate,
        reports: list[ValidationReport],
        trace: list[str],
    ) -> Text2CypherResult:
        return Text2CypherResult(
            question=question,
            status=CypherRunStatus.VALIDATION_FAILED,
            candidate=candidate,
            validation_reports=reports,
            trace=trace,
        )
