from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from gustobot.application.meal_planning.text2cypher.models import (
    CypherCandidate,
    CypherRunStatus,
    CypherSource,
)
from gustobot.application.meal_planning.text2cypher.neo4j_adapter import (
    Neo4jCypherAdapter,
)
from gustobot.application.meal_planning.text2cypher.schema import Neo4jSchemaLoader


@pytest.mark.integration
def test_real_meal_graph_exposes_the_text2cypher_schema():
    url = os.getenv("MEAL_PLANNING_NEO4J_URL")
    if not url:
        pytest.skip("set MEAL_PLANNING_NEO4J_URL to run the real Neo4j integration")

    with GraphDatabase.driver(url, auth=None) as driver:
        driver.verify_connectivity()
        snapshot = Neo4jSchemaLoader(
            driver,
            allowed_labels={"Recipe", "Ingredient"},
        ).load(refresh=True)

    assert "Recipe" in snapshot.node_properties
    assert "recipe_id" in snapshot.node_properties["Recipe"]
    assert any(
        relationship.start_label == "Recipe"
        and relationship.relationship_type == "HAS_INGREDIENT"
        and relationship.end_label == "Ingredient"
        for relationship in snapshot.relationships
    )


@pytest.mark.integration
def test_real_neo4j_explain_execute_and_syntax_failure():
    url = os.getenv("MEAL_PLANNING_NEO4J_URL")
    if not url:
        pytest.skip("set MEAL_PLANNING_NEO4J_URL to run the real Neo4j integration")

    valid = CypherCandidate(
        statement=(
            "MATCH (recipe:Recipe) "
            "RETURN recipe.recipe_id AS recipe_id, recipe.name AS recipe_name "
            "LIMIT $top_k"
        ),
        parameters={"top_k": 1},
        source=CypherSource.TEMPLATE,
    )
    malformed = CypherCandidate(
        statement="MATCH (recipe:Recipe RETURN recipe.recipe_id AS recipe_id LIMIT 1",
        source=CypherSource.DYNAMIC,
    )

    with GraphDatabase.driver(url, auth=None) as driver:
        driver.verify_connectivity()
        adapter = Neo4jCypherAdapter(driver)
        valid_explain = adapter.explain(valid)
        execution = adapter.execute(valid) if valid_explain.valid else None
        invalid_explain = adapter.explain(malformed)

    assert valid_explain.valid
    assert execution is not None
    assert execution.status is CypherRunStatus.SUCCESS
    assert execution.selected_recipe_ids
    assert not invalid_explain.valid
    assert invalid_explain.issues[0].code == "CYPHER_SYNTAX_ERROR"
