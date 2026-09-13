from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FOODTRACE_BATCH_TRACE_CYPHER = """
MATCH (lot:FoodTraceIngredientLot {ingredient_lot_id: $ingredient_lot_id})
MATCH (lot)-[:USED_IN_RECIPE]->(recipe:FoodTraceRecipe)
MATCH (recipe)-[:PRODUCES]->(product:FoodTraceProduct)
MATCH (product)-[:HAS_BATCH]->(batch:FoodTraceProductionBatch)
RETURN DISTINCT
    recipe.recipe_id AS recipe_id,
    product.product_id AS product_id,
    batch.production_batch_id AS production_batch_id
ORDER BY recipe_id, product_id, production_batch_id
""".strip()


@dataclass(frozen=True)
class CypherQuery:
    statement: str
    parameters: dict[str, Any]


def build_batch_trace_query(ingredient_lot_id: str) -> CypherQuery:
    normalized_lot_id = ingredient_lot_id.strip()
    if not normalized_lot_id:
        raise ValueError("ingredient_lot_id cannot be blank")
    return CypherQuery(
        statement=FOODTRACE_BATCH_TRACE_CYPHER,
        parameters={"ingredient_lot_id": normalized_lot_id},
    )
