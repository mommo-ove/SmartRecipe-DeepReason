from __future__ import annotations

from datetime import datetime, timezone

from .models import (
    FoodTraceFixture,
    GoldRecallCase,
    IncidentNotice,
    IngredientLot,
    OrderExposure,
    Product,
    ProductionBatch,
    RecallScope,
    Recipe,
    StoreInventory,
)


DEFAULT_FOODTRACE_SEED = 20260726
_HAND_AUTHORED_GOLD_SCOPE = RecallScope(
    impacted_product_ids=("PRODUCT-FT-001", "PRODUCT-FT-002"),
    impacted_batch_ids=("BATCH-FT-001", "BATCH-FT-002", "BATCH-FT-003"),
    impacted_store_ids=("STORE-FT-001", "STORE-FT-002", "STORE-FT-003"),
    impacted_order_ids=("ORDER-FT-001", "ORDER-FT-002", "ORDER-FT-003"),
)


def build_foodtrace_fixture(
    seed: int = DEFAULT_FOODTRACE_SEED,
) -> FoodTraceFixture:
    """Build a deterministic, enterprise-shaped allergen recall dataset."""
    incident = IncidentNotice(
        incident_id="INC-PEANUT-001",
        supplier_id="SUP-PEANUT-001",
        ingredient_lot_id="LOT-PEANUT-001",
        allergen="peanut",
        declared_on_label=False,
        received_at=_utc(2026, 7, 26, 9),
    )
    ingredient_lots = (
        IngredientLot(
            ingredient_lot_id="LOT-PEANUT-001",
            ingredient_name="peanut paste",
            supplier_id="SUP-PEANUT-001",
            suspected_allergens=("peanut",),
        ),
        IngredientLot(
            ingredient_lot_id="LOT-CHICKEN-001",
            ingredient_name="chicken",
            supplier_id="SUP-PROTEIN-001",
        ),
        IngredientLot(
            ingredient_lot_id="LOT-NOODLE-001",
            ingredient_name="rice noodle",
            supplier_id="SUP-GRAIN-001",
        ),
        IngredientLot(
            ingredient_lot_id="LOT-TOMATO-001",
            ingredient_name="tomato",
            supplier_id="SUP-PRODUCE-001",
        ),
    )
    recipes = (
        Recipe(
            recipe_id="RECIPE-FT-001",
            recipe_name="spicy chicken bowl",
            ingredient_lot_ids=("LOT-PEANUT-001", "LOT-CHICKEN-001"),
        ),
        Recipe(
            recipe_id="RECIPE-FT-002",
            recipe_name="satay rice noodles",
            ingredient_lot_ids=("LOT-PEANUT-001", "LOT-NOODLE-001"),
        ),
        Recipe(
            recipe_id="RECIPE-FT-003",
            recipe_name="tomato soup",
            ingredient_lot_ids=("LOT-TOMATO-001",),
        ),
    )
    products = (
        Product(
            product_id="PRODUCT-FT-001",
            product_name="chicken bowl retail pack",
            recipe_id="RECIPE-FT-001",
        ),
        Product(
            product_id="PRODUCT-FT-002",
            product_name="satay noodle retail pack",
            recipe_id="RECIPE-FT-002",
        ),
        Product(
            product_id="PRODUCT-FT-003",
            product_name="tomato soup retail pack",
            recipe_id="RECIPE-FT-003",
        ),
    )
    production_batches = (
        ProductionBatch(
            production_batch_id="BATCH-FT-001",
            product_id="PRODUCT-FT-001",
            produced_at=_utc(2026, 7, 20, 8),
        ),
        ProductionBatch(
            production_batch_id="BATCH-FT-002",
            product_id="PRODUCT-FT-001",
            produced_at=_utc(2026, 7, 21, 8),
        ),
        ProductionBatch(
            production_batch_id="BATCH-FT-003",
            product_id="PRODUCT-FT-002",
            produced_at=_utc(2026, 7, 22, 8),
        ),
        ProductionBatch(
            production_batch_id="BATCH-FT-004",
            product_id="PRODUCT-FT-003",
            produced_at=_utc(2026, 7, 23, 8),
        ),
    )
    store_inventories = (
        StoreInventory(
            inventory_id="INV-FT-001",
            store_id="STORE-FT-001",
            production_batch_id="BATCH-FT-001",
            quantity=_seeded_quantity(seed, 0),
        ),
        StoreInventory(
            inventory_id="INV-FT-002",
            store_id="STORE-FT-001",
            production_batch_id="BATCH-FT-004",
            quantity=_seeded_quantity(seed, 1),
        ),
        StoreInventory(
            inventory_id="INV-FT-003",
            store_id="STORE-FT-002",
            production_batch_id="BATCH-FT-002",
            quantity=_seeded_quantity(seed, 2),
        ),
        StoreInventory(
            inventory_id="INV-FT-004",
            store_id="STORE-FT-003",
            production_batch_id="BATCH-FT-003",
            quantity=_seeded_quantity(seed, 3),
        ),
        StoreInventory(
            inventory_id="INV-FT-005",
            store_id="STORE-FT-003",
            production_batch_id="BATCH-FT-004",
            quantity=_seeded_quantity(seed, 4),
        ),
    )
    order_exposures = (
        OrderExposure(
            order_id="ORDER-FT-001",
            store_id="STORE-FT-001",
            production_batch_id="BATCH-FT-001",
            quantity=1,
            customer_ref="CUSTOMER-HASH-001",
            ordered_at=_utc(2026, 7, 23, 12),
        ),
        OrderExposure(
            order_id="ORDER-FT-002",
            store_id="STORE-FT-002",
            production_batch_id="BATCH-FT-002",
            quantity=2,
            customer_ref="CUSTOMER-HASH-002",
            ordered_at=_utc(2026, 7, 24, 12),
        ),
        OrderExposure(
            order_id="ORDER-FT-003",
            store_id="STORE-FT-003",
            production_batch_id="BATCH-FT-003",
            quantity=1,
            customer_ref="CUSTOMER-HASH-003",
            ordered_at=_utc(2026, 7, 25, 12),
        ),
        OrderExposure(
            order_id="ORDER-FT-004",
            store_id="STORE-FT-002",
            production_batch_id="BATCH-FT-004",
            quantity=1,
            customer_ref="CUSTOMER-HASH-004",
            ordered_at=_utc(2026, 7, 25, 13),
        ),
    )
    graph_batch_to_sql_fk = {
        batch.production_batch_id: batch.production_batch_id
        for batch in production_batches
    }
    gold_cases = (
        GoldRecallCase(
            case_id="allergen_peanut_001",
            incident=incident,
            gold_scope=_HAND_AUTHORED_GOLD_SCOPE,
            expected_gate="ALLOW_REPORT",
            scenario_tags=("happy_path", "unlabeled_peanut"),
        ),
    )
    return FoodTraceFixture(
        seed=seed,
        incident=incident,
        ingredient_lots=ingredient_lots,
        recipes=recipes,
        products=products,
        production_batches=production_batches,
        store_inventories=store_inventories,
        order_exposures=order_exposures,
        graph_batch_to_sql_fk=graph_batch_to_sql_fk,
        gold_cases=gold_cases,
    )


def derive_recall_scope(
    *,
    incident: IncidentNotice,
    recipes: tuple[Recipe, ...],
    products: tuple[Product, ...],
    production_batches: tuple[ProductionBatch, ...],
    store_inventories: tuple[StoreInventory, ...],
    order_exposures: tuple[OrderExposure, ...],
    graph_batch_to_sql_fk: dict[str, str],
) -> RecallScope:
    graph_batch_ids = {batch.production_batch_id for batch in production_batches}
    if set(graph_batch_to_sql_fk) != graph_batch_ids:
        raise ValueError("mapping must cover every graph production batch exactly once")
    sql_batch_ids = list(graph_batch_to_sql_fk.values())
    if len(sql_batch_ids) != len(set(sql_batch_ids)):
        raise ValueError("graph-to-SQL batch mapping must be one-to-one")
    unknown_sql_batch_ids = {
        row.production_batch_id
        for row in (*store_inventories, *order_exposures)
        if row.production_batch_id not in set(sql_batch_ids)
    }
    if unknown_sql_batch_ids:
        raise ValueError(
            "SQL exposure rows reference unmapped batch IDs: "
            + ", ".join(sorted(unknown_sql_batch_ids))
        )

    affected_recipe_ids = {
        recipe.recipe_id
        for recipe in recipes
        if incident.ingredient_lot_id in recipe.ingredient_lot_ids
    }
    affected_product_ids = {
        product.product_id
        for product in products
        if product.recipe_id in affected_recipe_ids
    }
    affected_batch_ids = {
        batch.production_batch_id
        for batch in production_batches
        if batch.product_id in affected_product_ids
    }
    affected_sql_batch_ids = {
        graph_batch_to_sql_fk[batch_id] for batch_id in affected_batch_ids
    }
    return RecallScope(
        impacted_product_ids=affected_product_ids,
        impacted_batch_ids=affected_batch_ids,
        impacted_store_ids={
            row.store_id
            for row in store_inventories
            if row.production_batch_id in affected_sql_batch_ids
        },
        impacted_order_ids={
            row.order_id
            for row in order_exposures
            if row.production_batch_id in affected_sql_batch_ids
        },
    )


def _seeded_quantity(seed: int, index: int) -> int:
    return 10 + ((seed + index * 17) % 23)


def _utc(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)
