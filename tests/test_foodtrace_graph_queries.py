import pytest

from gustobot.application.foodtrace.graph_queries import (
    FOODTRACE_BATCH_TRACE_CYPHER,
    build_batch_trace_query,
)


def test_fixed_graph_trace_uses_only_the_declared_read_path():
    normalized = " ".join(FOODTRACE_BATCH_TRACE_CYPHER.split())

    assert (
        "(lot:FoodTraceIngredientLot {ingredient_lot_id: $ingredient_lot_id})"
        in normalized
    )
    assert "(lot)-[:USED_IN_RECIPE]->(recipe:FoodTraceRecipe)" in normalized
    assert "(recipe)-[:PRODUCES]->(product:FoodTraceProduct)" in normalized
    assert "(product)-[:HAS_BATCH]->(batch:FoodTraceProductionBatch)" in normalized
    assert all(
        keyword not in normalized.upper()
        for keyword in (" CREATE ", " MERGE ", " DELETE ", " SET ", " REMOVE ")
    )


def test_incident_lot_is_a_parameter_never_interpolated_into_cypher():
    hostile_lot_id = "LOT-001'}) MATCH (everything) RETURN everything //"

    query = build_batch_trace_query(hostile_lot_id)

    assert query.statement == FOODTRACE_BATCH_TRACE_CYPHER
    assert hostile_lot_id not in query.statement
    assert query.parameters == {"ingredient_lot_id": hostile_lot_id}


@pytest.mark.parametrize("invalid_lot_id", ["", "   "])
def test_graph_trace_rejects_blank_lot_scope(invalid_lot_id):
    with pytest.raises(ValueError, match="ingredient_lot_id"):
        build_batch_trace_query(invalid_lot_id)
