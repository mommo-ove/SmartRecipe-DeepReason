import json
import re
from pathlib import Path

import pytest
from sqlglot import exp, parse

from gustobot.application.foodtrace.fixtures import (
    DEFAULT_FOODTRACE_SEED,
    build_foodtrace_fixture,
    derive_recall_scope,
)


def test_same_seed_produces_identical_fixture_and_gold_scope():
    first = build_foodtrace_fixture(seed=DEFAULT_FOODTRACE_SEED)
    second = build_foodtrace_fixture(seed=DEFAULT_FOODTRACE_SEED)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_fixture_contains_two_affected_paths_and_an_unaffected_negative():
    fixture = build_foodtrace_fixture()
    contaminated_lot_id = fixture.incident.ingredient_lot_id
    affected_recipe_ids = {
        recipe.recipe_id
        for recipe in fixture.recipes
        if contaminated_lot_id in recipe.ingredient_lot_ids
    }
    unaffected_recipe_ids = {
        recipe.recipe_id
        for recipe in fixture.recipes
        if contaminated_lot_id not in recipe.ingredient_lot_ids
    }

    assert len(affected_recipe_ids) >= 2
    assert unaffected_recipe_ids
    assert len(fixture.production_batches) >= 3
    assert len({row.store_id for row in fixture.store_inventories}) == 3


def test_gold_scope_is_derived_from_graph_to_sql_batch_mapping():
    fixture = build_foodtrace_fixture()
    contaminated_lot_id = fixture.incident.ingredient_lot_id
    affected_recipe_ids = {
        recipe.recipe_id
        for recipe in fixture.recipes
        if contaminated_lot_id in recipe.ingredient_lot_ids
    }
    affected_product_ids = {
        product.product_id
        for product in fixture.products
        if product.recipe_id in affected_recipe_ids
    }
    affected_batch_ids = {
        batch.production_batch_id
        for batch in fixture.production_batches
        if batch.product_id in affected_product_ids
    }
    affected_store_ids = {
        row.store_id
        for row in fixture.store_inventories
        if row.production_batch_id in affected_batch_ids
    }
    affected_order_ids = {
        row.order_id
        for row in fixture.order_exposures
        if row.production_batch_id in affected_batch_ids
    }
    gold_scope = fixture.gold_cases[0].gold_scope

    assert set(gold_scope.impacted_product_ids) == affected_product_ids
    assert set(gold_scope.impacted_batch_ids) == affected_batch_ids
    assert set(gold_scope.impacted_store_ids) == affected_store_ids
    assert set(gold_scope.impacted_order_ids) == affected_order_ids
    assert fixture.graph_batch_to_sql_fk == {
        batch_id: batch_id for batch_id in sorted(affected_batch_ids | {"BATCH-FT-004"})
    }


def test_non_identity_graph_to_sql_mapping_preserves_exposure_scope():
    fixture = build_foodtrace_fixture()
    mapping = {
        batch.production_batch_id: f"SQL-{batch.production_batch_id}"
        for batch in fixture.production_batches
    }
    inventories = tuple(
        row.model_copy(update={"production_batch_id": mapping[row.production_batch_id]})
        for row in fixture.store_inventories
    )
    orders = tuple(
        row.model_copy(update={"production_batch_id": mapping[row.production_batch_id]})
        for row in fixture.order_exposures
    )

    scope = derive_recall_scope(
        incident=fixture.incident,
        recipes=fixture.recipes,
        products=fixture.products,
        production_batches=fixture.production_batches,
        store_inventories=inventories,
        order_exposures=orders,
        graph_batch_to_sql_fk=mapping,
    )

    assert scope == fixture.gold_cases[0].gold_scope


def test_graph_to_sql_mapping_must_be_total_and_one_to_one():
    fixture = build_foodtrace_fixture()
    incomplete_mapping = dict(fixture.graph_batch_to_sql_fk)
    incomplete_mapping.pop("BATCH-FT-004")
    duplicate_mapping = dict(fixture.graph_batch_to_sql_fk)
    duplicate_mapping["BATCH-FT-004"] = duplicate_mapping["BATCH-FT-003"]

    with pytest.raises(ValueError, match="every graph production batch"):
        derive_recall_scope(
            incident=fixture.incident,
            recipes=fixture.recipes,
            products=fixture.products,
            production_batches=fixture.production_batches,
            store_inventories=fixture.store_inventories,
            order_exposures=fixture.order_exposures,
            graph_batch_to_sql_fk=incomplete_mapping,
        )
    with pytest.raises(ValueError, match="one-to-one"):
        derive_recall_scope(
            incident=fixture.incident,
            recipes=fixture.recipes,
            products=fixture.products,
            production_batches=fixture.production_batches,
            store_inventories=fixture.store_inventories,
            order_exposures=fixture.order_exposures,
            graph_batch_to_sql_fk=duplicate_mapping,
        )


def test_committed_seed_files_and_benchmark_case_match_python_fixture():
    repo_root = Path(__file__).resolve().parents[1]
    fixture = build_foodtrace_fixture()
    cases = json.loads(
        (repo_root / "benchmark/foodtrace/cases.json").read_text(encoding="utf-8")
    )
    graph_seed = (repo_root / "gustobot/data/foodtrace/seed_graph.cypher").read_text(
        encoding="utf-8"
    )
    order_seed = (repo_root / "gustobot/data/foodtrace/seed_orders.sql").read_text(
        encoding="utf-8"
    )

    assert cases == [
        case.model_dump(mode="json", exclude_none=True) for case in fixture.gold_cases
    ]
    graph_nodes, graph_edges = _parse_graph_seed(graph_seed)
    expected_graph_nodes = {
        *(lot.ingredient_lot_id for lot in fixture.ingredient_lots),
        *(recipe.recipe_id for recipe in fixture.recipes),
        *(product.product_id for product in fixture.products),
        *(batch.production_batch_id for batch in fixture.production_batches),
    }
    expected_graph_edges = {
        *(
            (lot_id, "USED_IN_RECIPE", recipe.recipe_id)
            for recipe in fixture.recipes
            for lot_id in recipe.ingredient_lot_ids
        ),
        *(
            (product.recipe_id, "PRODUCES", product.product_id)
            for product in fixture.products
        ),
        *(
            (batch.product_id, "HAS_BATCH", batch.production_batch_id)
            for batch in fixture.production_batches
        ),
    }

    assert graph_nodes == expected_graph_nodes
    assert graph_edges == expected_graph_edges

    sql_rows = _parse_insert_rows(order_seed)
    assert sql_rows["foodtrace_store_inventory"] == [
        {
            "inventory_id": row.inventory_id,
            "store_id": row.store_id,
            "production_batch_id": row.production_batch_id,
            "quantity": str(row.quantity),
        }
        for row in fixture.store_inventories
    ]
    assert sql_rows["foodtrace_order_exposures"] == [
        {
            "order_id": row.order_id,
            "store_id": row.store_id,
            "production_batch_id": row.production_batch_id,
            "quantity": str(row.quantity),
            "customer_ref": row.customer_ref,
            "ordered_at": row.ordered_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for row in fixture.order_exposures
    ]


def _parse_graph_seed(text: str) -> tuple[set[str], set[tuple[str, str, str]]]:
    alias_to_id = {
        match.group("alias"): match.group("node_id")
        for match in re.finditer(
            r"MERGE \((?P<alias>\w+):FoodTrace\w+ " r"\{\w+: '(?P<node_id>[^']+)'\}\)",
            text,
        )
    }
    edges = {
        (
            alias_to_id[match.group("source")],
            match.group("relationship"),
            alias_to_id[match.group("target")],
        )
        for match in re.finditer(
            r"MERGE \((?P<source>\w+)\)-"
            r"\[:(?P<relationship>\w+)\]->"
            r"\((?P<target>\w+)\);",
            text,
        )
    }
    return set(alias_to_id.values()), edges


def _parse_insert_rows(text: str) -> dict[str, list[dict[str, str]]]:
    rows_by_table: dict[str, list[dict[str, str]]] = {}
    for statement in parse(text, read="mysql"):
        if not isinstance(statement, exp.Insert):
            continue
        schema = statement.this
        assert isinstance(schema, exp.Schema)
        table_name = schema.this.name
        columns = [column.name for column in schema.expressions]
        values = statement.expression
        assert isinstance(values, exp.Values)
        rows_by_table[table_name] = [
            {
                column: str(value.this)
                for column, value in zip(columns, row.expressions, strict=True)
            }
            for row in values.expressions
        ]
    return rows_by_table
