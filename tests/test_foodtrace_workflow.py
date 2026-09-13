from gustobot.application.foodtrace.fixtures import build_foodtrace_fixture
from gustobot.application.foodtrace.repositories import (
    ExposureLookupResult,
    FixtureGraphRepository,
    FixtureSqlRepository,
    GraphTraceResult,
)
from gustobot.application.foodtrace.workflow import FoodTraceWorkflow


def test_workflow_reproduces_gold_scope_with_evidence_bound_claims():
    fixture = build_foodtrace_fixture()
    result = FoodTraceWorkflow(
        graph_repository=FixtureGraphRepository(fixture),
        exposure_repository=FixtureSqlRepository(fixture),
    ).run(fixture.incident)

    assert result.gate.state == "ALLOW_REPORT"
    assert result.scope == fixture.gold_cases[0].gold_scope
    assert [step.name for step in result.trace] == [
        "Incident Parser",
        "SOP Lookup",
        "Graph Scope",
        "Order Exposure",
        "Consistency Reviewer",
        "Decision Gate",
    ]
    evidence_ids = {item.evidence_id for item in result.evidence}
    assert evidence_ids
    assert all(claim.evidence_ids for claim in result.claims)
    assert all(set(claim.evidence_ids) <= evidence_ids for claim in result.claims)


def test_order_exposure_never_runs_when_graph_scope_fails():
    class FailingGraphRepository:
        def trace_batches(self, ingredient_lot_id):
            return GraphTraceResult(
                status="dependency_unavailable",
                ingredient_lot_id=ingredient_lot_id,
                error_code="dependency_unavailable",
            )

    class RecordingExposureRepository:
        def __init__(self):
            self.calls = []

        def lookup_exposures(self, production_batch_ids):
            self.calls.append(production_batch_ids)
            raise AssertionError("order exposure must wait for graph scope")

    exposure_repository = RecordingExposureRepository()
    result = FoodTraceWorkflow(
        graph_repository=FailingGraphRepository(),
        exposure_repository=exposure_repository,
    ).run(build_foodtrace_fixture().incident)

    assert exposure_repository.calls == []
    assert result.gate.state == "BLOCKED"
    assert "graph_scope:dependency_unavailable" in result.gate.reasons
    assert [step.name for step in result.trace] == [
        "Incident Parser",
        "SOP Lookup",
        "Graph Scope",
        "Decision Gate",
    ]


def test_workflow_routes_graph_sql_scope_mismatch_to_human_review():
    fixture = build_foodtrace_fixture()

    class MismatchedExposureRepository:
        def lookup_exposures(self, production_batch_ids):
            return ExposureLookupResult(
                status="ok",
                production_batch_ids=("BATCH-FT-004",),
            )

    result = FoodTraceWorkflow(
        graph_repository=FixtureGraphRepository(fixture),
        exposure_repository=MismatchedExposureRepository(),
    ).run(fixture.incident)

    assert result.gate.state == "HUMAN_REVIEW"
    assert "graph_sql_batch_scope_mismatch" in result.gate.reasons


def test_workflow_routes_abnormally_expanded_batch_scope_to_human_review():
    fixture = build_foodtrace_fixture()

    class ExpandedGraphRepository:
        def trace_batches(self, ingredient_lot_id):
            return GraphTraceResult(
                status="ok",
                ingredient_lot_id=ingredient_lot_id,
                recipe_ids=("R1",),
                product_ids=("P1",),
                production_batch_ids=("B1", "B2", "B3", "B4"),
            )

    class EmptyExposureRepository:
        def __init__(self):
            self.calls = []

        def lookup_exposures(self, production_batch_ids):
            self.calls.append(production_batch_ids)
            return ExposureLookupResult(
                status="ok",
                production_batch_ids=production_batch_ids,
            )

    exposure_repository = EmptyExposureRepository()
    result = FoodTraceWorkflow(
        graph_repository=ExpandedGraphRepository(),
        exposure_repository=exposure_repository,
        max_batch_scope=3,
    ).run(fixture.incident)

    assert exposure_repository.calls == []
    assert result.gate.state == "HUMAN_REVIEW"
    assert "batch_scope_expanded:4>3" in result.gate.reasons


def test_unexpected_dependency_exception_is_sanitized_and_blocked():
    fixture = build_foodtrace_fixture()

    class ExplodingGraphRepository:
        def trace_batches(self, ingredient_lot_id):
            raise RuntimeError("secret connection details")

    class RecordingExposureRepository:
        def __init__(self):
            self.calls = []

        def lookup_exposures(self, production_batch_ids):
            self.calls.append(production_batch_ids)

    exposure_repository = RecordingExposureRepository()
    result = FoodTraceWorkflow(
        graph_repository=ExplodingGraphRepository(),
        exposure_repository=exposure_repository,
    ).run(fixture.incident)

    assert exposure_repository.calls == []
    assert result.gate.state == "BLOCKED"
    assert result.gate.reasons == ("graph_scope:dependency_unavailable",)
    serialized_trace = result.model_dump_json()
    assert "secret connection details" not in serialized_trace


def test_unexpected_sop_dependency_exception_is_sanitized_and_blocked():
    fixture = build_foodtrace_fixture()

    class ExplodingSopRepository:
        def lookup(self, incident):
            raise RuntimeError("sop-api-token")

    result = FoodTraceWorkflow(
        graph_repository=FixtureGraphRepository(fixture),
        exposure_repository=FixtureSqlRepository(fixture),
        sop_repository=ExplodingSopRepository(),
    ).run(fixture.incident)

    assert result.gate.state == "BLOCKED"
    assert result.gate.reasons == ("sop_lookup:dependency_unavailable",)
    assert "sop-api-token" not in result.model_dump_json()


def test_unexpected_order_dependency_exception_is_sanitized_and_blocked():
    fixture = build_foodtrace_fixture()

    class ExplodingExposureRepository:
        def lookup_exposures(self, production_batch_ids):
            raise RuntimeError("password=mysql-secret")

    result = FoodTraceWorkflow(
        graph_repository=FixtureGraphRepository(fixture),
        exposure_repository=ExplodingExposureRepository(),
    ).run(fixture.incident)

    assert result.gate.state == "BLOCKED"
    assert result.gate.reasons == ("order_exposure:dependency_unavailable",)
    assert "mysql-secret" not in result.model_dump_json()
