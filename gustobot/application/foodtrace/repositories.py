from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import field_validator

from .graph_queries import build_batch_trace_query
from .models import (
    FoodTraceFixture,
    FoodTraceModel,
    OrderExposure,
    StoreInventory,
)
from .sql_queries import build_exposure_queries


RepositoryStatus = Literal[
    "ok",
    "not_found",
    "dependency_unavailable",
    "invalid_dependency_response",
    "invalid_configuration",
]


class GraphTraceResult(FoodTraceModel):
    status: RepositoryStatus
    ingredient_lot_id: str
    recipe_ids: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()
    production_batch_ids: tuple[str, ...] = ()
    error_code: str | None = None

    @field_validator(
        "recipe_ids",
        "product_ids",
        "production_batch_ids",
        mode="before",
    )
    @classmethod
    def normalize_ids(cls, value: object) -> tuple[str, ...]:
        return tuple(sorted({str(item) for item in value or ()}))


class ExposureLookupResult(FoodTraceModel):
    status: Literal[
        "ok",
        "dependency_unavailable",
        "invalid_dependency_response",
        "invalid_configuration",
    ]
    production_batch_ids: tuple[str, ...]
    inventories: tuple[StoreInventory, ...] = ()
    orders: tuple[OrderExposure, ...] = ()
    error_code: str | None = None


class FixtureGraphRepository:
    def __init__(self, fixture: FoodTraceFixture) -> None:
        self.fixture = fixture

    def trace_batches(self, ingredient_lot_id: str) -> GraphTraceResult:
        query = build_batch_trace_query(ingredient_lot_id)
        normalized_lot_id = str(query.parameters["ingredient_lot_id"])
        known_lot_ids = {lot.ingredient_lot_id for lot in self.fixture.ingredient_lots}
        if normalized_lot_id not in known_lot_ids:
            return GraphTraceResult(
                status="not_found",
                ingredient_lot_id=normalized_lot_id,
            )

        recipe_ids = {
            recipe.recipe_id
            for recipe in self.fixture.recipes
            if normalized_lot_id in recipe.ingredient_lot_ids
        }
        product_ids = {
            product.product_id
            for product in self.fixture.products
            if product.recipe_id in recipe_ids
        }
        batch_ids = {
            batch.production_batch_id
            for batch in self.fixture.production_batches
            if batch.product_id in product_ids
        }
        if not batch_ids:
            return GraphTraceResult(
                status="not_found",
                ingredient_lot_id=normalized_lot_id,
            )
        return GraphTraceResult(
            status="ok",
            ingredient_lot_id=normalized_lot_id,
            recipe_ids=recipe_ids,
            product_ids=product_ids,
            production_batch_ids=batch_ids,
        )


class Neo4jGraphRepository:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def trace_batches(self, ingredient_lot_id: str) -> GraphTraceResult:
        query = build_batch_trace_query(ingredient_lot_id)
        normalized_lot_id = str(query.parameters["ingredient_lot_id"])
        try:
            raw_rows = self.graph.query(query.statement, params=query.parameters)
        except Exception:
            return GraphTraceResult(
                status="dependency_unavailable",
                ingredient_lot_id=normalized_lot_id,
                error_code="dependency_unavailable",
            )
        try:
            rows = list(raw_rows)
        except Exception:
            return _invalid_graph_response(normalized_lot_id)
        if not rows:
            return GraphTraceResult(
                status="not_found",
                ingredient_lot_id=normalized_lot_id,
            )
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                return _invalid_graph_response(normalized_lot_id)
            normalized_row: dict[str, str] = {}
            for field in (
                "recipe_id",
                "product_id",
                "production_batch_id",
            ):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    return _invalid_graph_response(normalized_lot_id)
                normalized_row[field] = value.strip()
            normalized_rows.append(normalized_row)
        return GraphTraceResult(
            status="ok",
            ingredient_lot_id=normalized_lot_id,
            recipe_ids={row["recipe_id"] for row in normalized_rows},
            product_ids={row["product_id"] for row in normalized_rows},
            production_batch_ids={
                row["production_batch_id"] for row in normalized_rows
            },
        )


class FixtureSqlRepository:
    def __init__(self, fixture: FoodTraceFixture) -> None:
        self.fixture = fixture

    def lookup_exposures(
        self,
        production_batch_ids: list[str] | tuple[str, ...],
    ) -> ExposureLookupResult:
        queries = build_exposure_queries(production_batch_ids)
        sql_batch_ids, mapping_error = _map_graph_batch_ids(
            queries.production_batch_ids,
            self.fixture.graph_batch_to_sql_fk,
        )
        if mapping_error:
            return ExposureLookupResult(
                status="invalid_configuration",
                production_batch_ids=queries.production_batch_ids,
                error_code=mapping_error,
            )
        sql_batch_id_set = set(sql_batch_ids)
        return ExposureLookupResult(
            status="ok",
            production_batch_ids=queries.production_batch_ids,
            inventories=tuple(
                row
                for row in self.fixture.store_inventories
                if row.production_batch_id in sql_batch_id_set
            ),
            orders=tuple(
                row
                for row in self.fixture.order_exposures
                if row.production_batch_id in sql_batch_id_set
            ),
        )


class SqlAlchemyExposureRepository:
    def __init__(
        self,
        engine: Any,
        *,
        graph_batch_to_sql_fk: Mapping[str, str] | None = None,
    ) -> None:
        self.engine = engine
        self.mapping_is_configured = graph_batch_to_sql_fk is not None
        self.graph_batch_to_sql_fk = dict(graph_batch_to_sql_fk or {})

    def lookup_exposures(
        self,
        production_batch_ids: list[str] | tuple[str, ...],
    ) -> ExposureLookupResult:
        graph_queries = build_exposure_queries(production_batch_ids)
        if self.mapping_is_configured:
            sql_batch_ids, mapping_error = _map_graph_batch_ids(
                graph_queries.production_batch_ids,
                self.graph_batch_to_sql_fk,
            )
            if mapping_error:
                return ExposureLookupResult(
                    status="invalid_configuration",
                    production_batch_ids=graph_queries.production_batch_ids,
                    error_code=mapping_error,
                )
        else:
            sql_batch_ids = graph_queries.production_batch_ids
        queries = build_exposure_queries(sql_batch_ids)
        try:
            with self.engine.connect() as connection:
                inventory_rows = tuple(
                    connection.execute(
                        queries.inventory.statement,
                        queries.inventory.parameters,
                    ).mappings()
                )
                order_rows = tuple(
                    connection.execute(
                        queries.orders.statement,
                        queries.orders.parameters,
                    ).mappings()
                )
        except Exception:
            return ExposureLookupResult(
                status="dependency_unavailable",
                production_batch_ids=graph_queries.production_batch_ids,
                error_code="dependency_unavailable",
            )
        try:
            inventories = tuple(
                StoreInventory.model_validate(dict(row)) for row in inventory_rows
            )
            orders = tuple(
                OrderExposure.model_validate(dict(row)) for row in order_rows
            )
        except Exception:
            return ExposureLookupResult(
                status="invalid_dependency_response",
                production_batch_ids=graph_queries.production_batch_ids,
                error_code="invalid_dependency_response",
            )
        allowed_sql_batch_ids = set(sql_batch_ids)
        if any(
            row.production_batch_id not in allowed_sql_batch_ids
            for row in (*inventories, *orders)
        ):
            return ExposureLookupResult(
                status="invalid_dependency_response",
                production_batch_ids=graph_queries.production_batch_ids,
                error_code="invalid_dependency_response",
            )
        return ExposureLookupResult(
            status="ok",
            production_batch_ids=graph_queries.production_batch_ids,
            inventories=inventories,
            orders=orders,
        )


def _invalid_graph_response(ingredient_lot_id: str) -> GraphTraceResult:
    return GraphTraceResult(
        status="invalid_dependency_response",
        ingredient_lot_id=ingredient_lot_id,
        error_code="invalid_dependency_response",
    )


def _map_graph_batch_ids(
    graph_batch_ids: tuple[str, ...],
    mapping: Mapping[str, str],
) -> tuple[tuple[str, ...], str | None]:
    normalized_mapping: dict[str, str] = {}
    for graph_batch_id, sql_batch_id in mapping.items():
        if (
            not isinstance(graph_batch_id, str)
            or not graph_batch_id.strip()
            or not isinstance(sql_batch_id, str)
            or not sql_batch_id.strip()
        ):
            return (), "invalid_batch_mapping"
        normalized_graph_batch_id = graph_batch_id.strip()
        if normalized_graph_batch_id in normalized_mapping:
            return (), "duplicate_graph_batch_mapping"
        normalized_mapping[normalized_graph_batch_id] = sql_batch_id.strip()
    all_sql_batch_ids = list(normalized_mapping.values())
    if len(all_sql_batch_ids) != len(set(all_sql_batch_ids)):
        return (), "non_unique_batch_mapping"

    missing_ids = sorted(set(graph_batch_ids) - set(normalized_mapping))
    if missing_ids:
        return (), "incomplete_batch_mapping"
    sql_batch_ids = tuple(normalized_mapping[batch_id] for batch_id in graph_batch_ids)
    return tuple(sorted(sql_batch_ids)), None
