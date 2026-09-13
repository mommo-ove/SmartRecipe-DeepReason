from __future__ import annotations

import importlib
import os

import pytest
from neo4j import GraphDatabase

from gustobot.application.meal_planning.external_corpus import (
    ExternalRecipeDocument,
    RecipeCapabilities,
)


def recipe(recipe_id: str, ingredients: list[str]) -> ExternalRecipeDocument:
    return ExternalRecipeDocument(
        recipe_id=recipe_id,
        name=f"Recipe {recipe_id}",
        meal_types={"lunch"},
        ingredients=ingredients,
        calories_kcal=500,
        protein_g=35,
        total_minutes=25,
        source_ref=f"integration:{recipe_id}",
        capabilities=RecipeCapabilities(
            retrieval_ready=True,
            nutrition_ready=True,
            time_ready=True,
        ),
    )


@pytest.mark.integration
def test_real_neo4j_import_and_ingredient_intersection_query():
    url = os.getenv("MEAL_PLANNING_NEO4J_URL")
    if not url:
        pytest.skip("set MEAL_PLANNING_NEO4J_URL to run the real Neo4j integration")

    module = importlib.import_module(
        "gustobot.application.meal_planning.graph_retrieval"
    )
    store_type = getattr(module, "Neo4jMealGraphStore", None)
    assert store_type is not None, "real Neo4j graph store is not implemented"

    driver = GraphDatabase.driver(url, auth=None)
    store = store_type(driver, database="neo4j")
    projection = module.build_meal_graph_projection(
        [
            recipe("integration-r1", ["egg", "tomato"]),
            recipe("integration-r2", ["egg", "spinach"]),
        ],
        dataset_version="integration-v1",
    )
    try:
        driver.verify_connectivity()
        store.replace_projection(projection)
        results = store.search_by_ingredients(
            ["egg", "tomato"],
            dataset_version="integration-v1",
            meal_type="lunch",
            max_minutes=30,
            min_protein_g=25,
            top_k=10,
        )
        assert [result.recipe_id for result in results] == ["integration-r1"]
        assert results[0].matched_ingredients == ["egg", "tomato"]
        assert set(results[0].evidence_node_ids) == {
            "ingredient:egg",
            "ingredient:tomato",
        }
    finally:
        store.delete_dataset("integration-v1")
        driver.close()
