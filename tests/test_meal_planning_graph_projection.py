from __future__ import annotations

import importlib

import pytest

from gustobot.application.meal_planning.external_corpus import (
    ExternalRecipeDocument,
    RecipeCapabilities,
)


def recipe(
    recipe_id: str,
    ingredients: list[str],
    *,
    meal_types: set[str] | None = None,
    total_minutes: int = 25,
    protein_g: float = 35,
) -> ExternalRecipeDocument:
    return ExternalRecipeDocument(
        recipe_id=recipe_id,
        name=f"Recipe {recipe_id}",
        description="A test recipe",
        meal_types=meal_types or {"lunch"},
        ingredients=ingredients,
        tags=["high protein"],
        instructions=["Cook it"],
        calories_kcal=500,
        protein_g=protein_g,
        total_minutes=total_minutes,
        source_ref=f"test:{recipe_id}",
        capabilities=RecipeCapabilities(
            retrieval_ready=True,
            nutrition_ready=True,
            time_ready=True,
        ),
    )


def test_projection_reuses_one_canonical_ingredient_across_recipes():
    try:
        module = importlib.import_module(
            "gustobot.application.meal_planning.graph_retrieval"
        )
    except ModuleNotFoundError:
        pytest.fail("meal-planning graph projection is not implemented")

    projection = module.build_meal_graph_projection(
        [
            recipe("r1", ["Egg", "tomato"]),
            recipe("r2", [" egg ", "spinach"]),
        ],
        dataset_version="test-v1",
    )

    ingredient_ids = {item["ingredient_id"] for item in projection.ingredients}
    assert ingredient_ids == {"egg", "tomato", "spinach"}
    egg_edges = [
        edge for edge in projection.has_ingredients if edge["ingredient_id"] == "egg"
    ]
    assert {edge["recipe_id"] for edge in egg_edges} == {"r1", "r2"}


def test_local_graph_query_requires_all_ingredients_and_uses_parameters():
    module = importlib.import_module(
        "gustobot.application.meal_planning.graph_retrieval"
    )
    builder = getattr(module, "build_local_graph_query", None)
    assert builder is not None, "local graph query builder is not implemented"

    query = builder(
        [" Egg ", "tomato"],
        dataset_version="test-v1",
        meal_type="lunch",
        max_minutes=30,
        min_protein_g=25,
        top_k=10,
    )

    assert "[:HAS_INGREDIENT]" in query.statement
    assert "size(matched_ingredients) = size($ingredient_ids)" in query.statement
    assert "Egg" not in query.statement
    assert query.parameters == {
        "ingredient_ids": ["egg", "tomato"],
        "excluded_ingredient_ids": [],
        "dataset_version": "test-v1",
        "meal_type": "lunch",
        "max_minutes": 30,
        "min_protein_g": 25.0,
        "top_k": 10,
    }


def test_relation_case_gold_is_derived_from_all_structured_constraints():
    module = importlib.import_module(
        "gustobot.application.meal_planning.graph_retrieval"
    )
    spec_type = getattr(module, "GraphRelationCaseSpec", None)
    builder = getattr(module, "build_graph_relation_case", None)
    assert spec_type is not None and builder is not None

    documents = [
        recipe("eligible", ["egg", "tomato"], total_minutes=20, protein_g=30),
        recipe("too-slow", ["egg", "tomato"], total_minutes=50, protein_g=30),
        recipe("missing-ingredient", ["egg", "spinach"], total_minutes=20),
        recipe(
            "wrong-slot",
            ["egg", "tomato"],
            meal_types={"dinner"},
            total_minutes=20,
        ),
    ]
    spec = spec_type(
        case_id="graph-001",
        query="egg and tomato lunch under 30 minutes",
        ingredient_ids=["egg", "tomato"],
        meal_type="lunch",
        max_minutes=30,
        min_protein_g=25,
        split="test",
    )

    case = builder(spec, documents)

    assert case.relevant_recipe_ids == {"eligible"}
    assert case.metadata_filters["graph_ingredient_ids"] == ["egg", "tomato"]
    assert case.metadata_filters["meal_type"] == "lunch"


def test_graph_augmented_retriever_reranks_only_graph_valid_recipe_ids():
    module = importlib.import_module(
        "gustobot.application.meal_planning.graph_retrieval"
    )
    retriever_type = getattr(module, "GraphAugmentedRetriever", None)
    result_type = getattr(module, "GraphSearchResult", None)
    assert retriever_type is not None, "graph augmented retriever is not implemented"

    class TextRetriever:
        def search(self, query, metadata_filters=None, *, top_k=20):
            return ["r1", "r2"]

    class GraphStore:
        def search_by_ingredients(self, ingredient_ids, **kwargs):
            return [
                result_type(
                    recipe_id="r2",
                    name="R2",
                    source_ref="test:r2",
                    total_minutes=20,
                    protein_g=30,
                    matched_ingredients=["egg"],
                    evidence_node_ids=["ingredient:egg"],
                ),
                result_type(
                    recipe_id="r3",
                    name="R3",
                    source_ref="test:r3",
                    total_minutes=20,
                    protein_g=30,
                    matched_ingredients=["egg"],
                    evidence_node_ids=["ingredient:egg"],
                ),
            ]

    retriever = retriever_type(TextRetriever(), GraphStore(), dataset_version="test-v1")
    ranking = retriever.search(
        "egg lunch",
        {"graph_ingredient_ids": ["egg"], "meal_type": "lunch"},
        top_k=3,
    )

    assert ranking == ["r2", "r3"]


def test_relation_gold_and_cypher_exclude_forbidden_ingredient_edges():
    module = importlib.import_module(
        "gustobot.application.meal_planning.graph_retrieval"
    )
    spec = module.GraphRelationCaseSpec(
        case_id="graph-negative-001",
        query="egg and tomato lunch without peanut",
        ingredient_ids=["egg", "tomato"],
        excluded_ingredient_ids=["peanut"],
        meal_type="lunch",
    )
    case = module.build_graph_relation_case(
        spec,
        [
            recipe("safe", ["egg", "tomato"]),
            recipe("unsafe", ["egg", "tomato", "peanut"]),
        ],
    )
    query = module.build_local_graph_query(
        ["egg", "tomato"],
        excluded_ingredient_ids=["peanut"],
        dataset_version="test-v1",
    )

    assert case.relevant_recipe_ids == {"safe"}
    assert case.metadata_filters["excluded_graph_ingredient_ids"] == ["peanut"]
    assert query.parameters["excluded_ingredient_ids"] == ["peanut"]
    assert "NOT EXISTS" in query.statement


def test_graph_constraints_are_a_hard_gate_not_an_rrf_soft_signal():
    module = importlib.import_module(
        "gustobot.application.meal_planning.graph_retrieval"
    )

    class TextRetriever:
        def search(self, query, metadata_filters=None, *, top_k=20):
            return ["unsafe", "safe"]

    class GraphStore:
        def search_by_ingredients(self, ingredient_ids, **kwargs):
            return [
                module.GraphSearchResult(
                    recipe_id="safe",
                    name="Safe",
                    source_ref="test:safe",
                    total_minutes=20,
                    protein_g=30,
                    matched_ingredients=["egg", "tomato"],
                    evidence_node_ids=["ingredient:egg", "ingredient:tomato"],
                )
            ]

    retriever = module.GraphAugmentedRetriever(
        TextRetriever(), GraphStore(), dataset_version="test-v1"
    )

    ranking = retriever.search(
        "egg tomato without peanut",
        {
            "graph_ingredient_ids": ["egg", "tomato"],
            "excluded_graph_ingredient_ids": ["peanut"],
        },
        top_k=10,
    )

    assert ranking == ["safe"]
