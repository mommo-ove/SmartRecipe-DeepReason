from sqlalchemy import create_engine, text
import pytest

from gustobot.application.foodtrace.fixtures import build_foodtrace_fixture
from gustobot.application.foodtrace.repositories import (
    FixtureGraphRepository,
    FixtureSqlRepository,
    Neo4jGraphRepository,
    SqlAlchemyExposureRepository,
)


def test_fixture_repositories_reproduce_gold_scope_without_crossing_negative_path():
    fixture = build_foodtrace_fixture()
    graph_repository = FixtureGraphRepository(fixture)
    sql_repository = FixtureSqlRepository(fixture)

    graph_result = graph_repository.trace_batches(fixture.incident.ingredient_lot_id)
    exposure_result = sql_repository.lookup_exposures(graph_result.production_batch_ids)
    gold = fixture.gold_cases[0].gold_scope

    assert graph_result.status == "ok"
    assert set(graph_result.product_ids) == set(gold.impacted_product_ids)
    assert set(graph_result.production_batch_ids) == set(gold.impacted_batch_ids)
    assert "PRODUCT-FT-003" not in graph_result.product_ids
    assert "BATCH-FT-004" not in graph_result.production_batch_ids
    assert exposure_result.status == "ok"
    assert {row.store_id for row in exposure_result.inventories} == set(
        gold.impacted_store_ids
    )
    assert {row.order_id for row in exposure_result.orders} == set(
        gold.impacted_order_ids
    )
    assert "ORDER-FT-004" not in {row.order_id for row in exposure_result.orders}


def test_fixture_graph_trace_stays_on_an_unaffected_lot_path():
    result = FixtureGraphRepository(build_foodtrace_fixture()).trace_batches(
        "LOT-TOMATO-001"
    )

    assert result.status == "ok"
    assert result.recipe_ids == ("RECIPE-FT-003",)
    assert result.product_ids == ("PRODUCT-FT-003",)
    assert result.production_batch_ids == ("BATCH-FT-004",)


def test_unknown_lot_is_not_found_instead_of_fabricating_success():
    result = FixtureGraphRepository(build_foodtrace_fixture()).trace_batches(
        "LOT-DOES-NOT-EXIST"
    )

    assert result.status == "not_found"
    assert result.production_batch_ids == ()


def test_real_graph_adapter_uses_fixed_query_and_normalizes_rows():
    class RecordingGraph:
        def __init__(self):
            self.calls = []

        def query(self, statement, params=None):
            self.calls.append((statement, params))
            return [
                {
                    "recipe_id": "RECIPE-FT-001",
                    "product_id": "PRODUCT-FT-001",
                    "production_batch_id": "BATCH-FT-002",
                },
                {
                    "recipe_id": "RECIPE-FT-001",
                    "product_id": "PRODUCT-FT-001",
                    "production_batch_id": "BATCH-FT-001",
                },
            ]

    graph = RecordingGraph()
    result = Neo4jGraphRepository(graph).trace_batches("LOT-PEANUT-001")

    assert result.status == "ok"
    assert result.production_batch_ids == ("BATCH-FT-001", "BATCH-FT-002")
    assert graph.calls[0][1] == {"ingredient_lot_id": "LOT-PEANUT-001"}


@pytest.mark.parametrize(
    "malformed_rows",
    [
        [{"recipe_id": "RECIPE-FT-001", "product_id": "PRODUCT-FT-001"}],
        [
            {
                "recipe_id": " ",
                "product_id": "PRODUCT-FT-001",
                "production_batch_id": "B1",
            }
        ],
        ["not-a-mapping"],
    ],
)
def test_graph_adapter_rejects_malformed_dependency_rows(malformed_rows):
    class MalformedGraph:
        def query(self, statement, params=None):
            return malformed_rows

    result = Neo4jGraphRepository(MalformedGraph()).trace_batches("LOT-PEANUT-001")

    assert result.status == "invalid_dependency_response"
    assert result.recipe_ids == ()
    assert result.product_ids == ()
    assert result.production_batch_ids == ()
    assert result.error_code == "invalid_dependency_response"


def test_sqlalchemy_adapter_executes_parameterized_queries_against_real_engine():
    fixture = build_foodtrace_fixture()
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE foodtrace_store_inventory ("
                "inventory_id TEXT PRIMARY KEY, store_id TEXT, "
                "production_batch_id TEXT, quantity INTEGER)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE foodtrace_order_exposures ("
                "order_id TEXT PRIMARY KEY, store_id TEXT, "
                "production_batch_id TEXT, quantity INTEGER, "
                "customer_ref TEXT, ordered_at TEXT)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO foodtrace_store_inventory VALUES "
                "('INV-TEST', 'STORE-TEST', 'BATCH-FT-001', 4)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO foodtrace_order_exposures VALUES "
                "('ORDER-TEST', 'STORE-TEST', 'BATCH-FT-001', 1, "
                "'CUSTOMER-HASH-TEST', '2026-07-25 12:00:00')"
            )
        )

    result = SqlAlchemyExposureRepository(engine).lookup_exposures(
        fixture.gold_cases[0].gold_scope.impacted_batch_ids
    )

    assert result.status == "ok"
    assert [row.inventory_id for row in result.inventories] == ["INV-TEST"]
    assert [row.order_id for row in result.orders] == ["ORDER-TEST"]


@pytest.mark.parametrize(
    ("mapping", "error_code"),
    [
        ({"BATCH-FT-001": "SQL-BATCH-001"}, "incomplete_batch_mapping"),
        (
            {
                "BATCH-FT-001": "SQL-BATCH-001",
                "BATCH-FT-002": "SQL-BATCH-001",
            },
            "non_unique_batch_mapping",
        ),
        (
            {
                "BATCH-FT-001": "SQL-BATCH-001",
                "BATCH-FT-002": "SQL-BATCH-002",
                "UNREQUESTED-BATCH": "SQL-BATCH-001",
            },
            "non_unique_batch_mapping",
        ),
        (
            {
                "BATCH-FT-001": None,
                "BATCH-FT-002": "SQL-BATCH-002",
            },
            "invalid_batch_mapping",
        ),
        (
            {
                "BATCH-FT-001": "SQL-BATCH-001",
                " BATCH-FT-001 ": "SQL-BATCH-ALIAS",
                "BATCH-FT-002": "SQL-BATCH-002",
            },
            "duplicate_graph_batch_mapping",
        ),
    ],
)
def test_configured_graph_to_sql_mapping_must_cover_requested_ids_once(
    mapping,
    error_code,
):
    class RecordingEngine:
        def __init__(self):
            self.connect_calls = 0

        def connect(self):
            self.connect_calls += 1
            raise AssertionError("invalid mapping must fail before database access")

    engine = RecordingEngine()
    result = SqlAlchemyExposureRepository(
        engine,
        graph_batch_to_sql_fk=mapping,
    ).lookup_exposures(["BATCH-FT-001", "BATCH-FT-002"])

    assert result.status == "invalid_configuration"
    assert result.error_code == error_code
    assert result.inventories == ()
    assert result.orders == ()
    assert engine.connect_calls == 0


def test_sql_adapter_rejects_rows_outside_mapped_request_scope():
    class FakeResult:
        def __init__(self, rows):
            self.rows = rows

        def mappings(self):
            return iter(self.rows)

    class FakeConnection:
        def __init__(self):
            self.execute_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, statement, parameters):
            self.execute_calls += 1
            if self.execute_calls == 1:
                return FakeResult(
                    [
                        {
                            "inventory_id": "INV-OUT",
                            "store_id": "STORE-OUT",
                            "production_batch_id": "SQL-BATCH-OUT",
                            "quantity": 5,
                        }
                    ]
                )
            return FakeResult(
                [
                    {
                        "order_id": "ORDER-OUT",
                        "store_id": "STORE-OUT",
                        "production_batch_id": "SQL-BATCH-OUT",
                        "quantity": 1,
                        "customer_ref": "CUSTOMER-HASH-OUT",
                        "ordered_at": "2026-07-25 12:00:00",
                    }
                ]
            )

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    result = SqlAlchemyExposureRepository(
        FakeEngine(),
        graph_batch_to_sql_fk={"BATCH-FT-001": "SQL-BATCH-001"},
    ).lookup_exposures(["BATCH-FT-001"])

    assert result.status == "invalid_dependency_response"
    assert result.error_code == "invalid_dependency_response"
    assert result.inventories == ()
    assert result.orders == ()


def test_dependency_failures_are_explicit_and_return_no_fake_rows():
    class FailingGraph:
        def query(self, statement, params=None):
            raise ConnectionError("neo4j unavailable")

    class FailingEngine:
        def connect(self):
            raise ConnectionError("mysql unavailable")

    graph_result = Neo4jGraphRepository(FailingGraph()).trace_batches("LOT-PEANUT-001")
    sql_result = SqlAlchemyExposureRepository(FailingEngine()).lookup_exposures(
        ["BATCH-FT-001"]
    )

    assert graph_result.status == "dependency_unavailable"
    assert graph_result.production_batch_ids == ()
    assert graph_result.error_code == "dependency_unavailable"
    assert sql_result.status == "dependency_unavailable"
    assert sql_result.inventories == ()
    assert sql_result.orders == ()
    assert sql_result.error_code == "dependency_unavailable"
