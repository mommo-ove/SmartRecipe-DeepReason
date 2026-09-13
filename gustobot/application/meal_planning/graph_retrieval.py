from __future__ import annotations

import re
from typing import Any, Sequence

from pydantic import BaseModel, Field

from .benchmark import RetrievalBenchmarkCase
from .external_corpus import ExternalRecipeDocument
from .hybrid_retrieval import rrf_fuse


_WHITESPACE = re.compile(r"\s+")


def canonicalize_ingredient(value: str) -> str:
    """Return a stable cross-recipe ingredient identifier."""

    return _WHITESPACE.sub(" ", value.strip().casefold())


def apply_recipe_id_gate(
    ranking: Sequence[str],
    allowed_recipe_ids: Sequence[str] | None,
) -> list[str]:
    """Deduplicate a ranking and enforce an optional hard recipe-ID boundary."""

    allowed = None if allowed_recipe_ids is None else set(allowed_recipe_ids)
    seen: set[str] = set()
    gated: list[str] = []
    for recipe_id in ranking:
        if recipe_id in seen or allowed is not None and recipe_id not in allowed:
            continue
        seen.add(recipe_id)
        gated.append(recipe_id)
    return gated


class MealGraphProjection(BaseModel):
    dataset_version: str = Field(min_length=1)
    recipes: list[dict[str, Any]]
    ingredients: list[dict[str, Any]]
    has_ingredients: list[dict[str, Any]]


class CypherQueryRequest(BaseModel):
    statement: str = Field(min_length=1)
    parameters: dict[str, Any]


class GraphRelationCaseSpec(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    ingredient_ids: list[str] = Field(min_length=1)
    excluded_ingredient_ids: list[str] = Field(default_factory=list)
    meal_type: str | None = None
    max_minutes: int | None = Field(default=None, gt=0)
    min_protein_g: float | None = Field(default=None, ge=0)
    split: str = "development"


class GraphSearchResult(BaseModel):
    recipe_id: str
    name: str
    source_ref: str
    total_minutes: int
    protein_g: float
    matched_ingredients: list[str]
    evidence_node_ids: list[str]


_LOCAL_GRAPH_QUERY = """
UNWIND $ingredient_ids AS requested_ingredient
MATCH (ingredient:Ingredient {
  ingredient_id: requested_ingredient,
  dataset_version: $dataset_version
})
MATCH (recipe:Recipe {dataset_version: $dataset_version})
      -[:HAS_INGREDIENT]->(ingredient)
WITH recipe, collect(DISTINCT ingredient.ingredient_id) AS matched_ingredients
WHERE size(matched_ingredients) = size($ingredient_ids)
  AND NOT EXISTS {
    MATCH (recipe)-[:HAS_INGREDIENT]->(excluded:Ingredient {
      dataset_version: $dataset_version
    })
    WHERE excluded.ingredient_id IN $excluded_ingredient_ids
  }
  AND ($meal_type IS NULL OR $meal_type IN recipe.meal_types)
  AND ($max_minutes IS NULL OR recipe.total_minutes <= $max_minutes)
  AND ($min_protein_g IS NULL OR recipe.protein_g >= $min_protein_g)
RETURN recipe.recipe_id AS recipe_id,
       recipe.name AS name,
       recipe.source_ref AS source_ref,
       recipe.total_minutes AS total_minutes,
       recipe.protein_g AS protein_g,
       matched_ingredients,
       [ingredient_id IN matched_ingredients |
          'ingredient:' + ingredient_id] AS evidence_node_ids
ORDER BY size(matched_ingredients) DESC, recipe.protein_g DESC, recipe.recipe_id
LIMIT $top_k
""".strip()


def build_local_graph_query(
    ingredient_ids: Sequence[str],
    *,
    excluded_ingredient_ids: Sequence[str] = (),
    dataset_version: str,
    meal_type: str | None = None,
    max_minutes: int | None = None,
    min_protein_g: float | None = None,
    top_k: int = 20,
) -> CypherQueryRequest:
    normalized = list(
        dict.fromkeys(
            canonicalize_ingredient(value)
            for value in ingredient_ids
            if canonicalize_ingredient(value)
        )
    )
    if not normalized:
        raise ValueError("local graph retrieval requires at least one ingredient")
    if top_k < 1 or top_k > 100:
        raise ValueError("top_k must be between 1 and 100")
    excluded = list(
        dict.fromkeys(
            canonicalize_ingredient(value)
            for value in excluded_ingredient_ids
            if canonicalize_ingredient(value)
        )
    )
    return CypherQueryRequest(
        statement=_LOCAL_GRAPH_QUERY,
        parameters={
            "ingredient_ids": normalized,
            "excluded_ingredient_ids": excluded,
            "dataset_version": dataset_version,
            "meal_type": meal_type,
            "max_minutes": max_minutes,
            "min_protein_g": (
                float(min_protein_g) if min_protein_g is not None else None
            ),
            "top_k": top_k,
        },
    )


def build_graph_relation_case(
    spec: GraphRelationCaseSpec,
    documents: Sequence[ExternalRecipeDocument],
) -> RetrievalBenchmarkCase:
    ingredient_ids = list(
        dict.fromkeys(canonicalize_ingredient(value) for value in spec.ingredient_ids)
    )
    excluded_ingredient_ids = list(
        dict.fromkeys(
            canonicalize_ingredient(value) for value in spec.excluded_ingredient_ids
        )
    )
    required = set(ingredient_ids)
    excluded = set(excluded_ingredient_ids)
    relevant_recipe_ids = {
        document.recipe_id
        for document in documents
        if required
        <= {canonicalize_ingredient(value) for value in document.ingredients}
        and not excluded
        & {canonicalize_ingredient(value) for value in document.ingredients}
        and (spec.meal_type is None or spec.meal_type in document.meal_types)
        and (spec.max_minutes is None or document.total_minutes <= spec.max_minutes)
        and (spec.min_protein_g is None or document.protein_g >= spec.min_protein_g)
    }
    if not relevant_recipe_ids:
        raise ValueError(f"relation case {spec.case_id} has no relevant recipes")
    return RetrievalBenchmarkCase(
        case_id=spec.case_id,
        query=spec.query,
        relevant_recipe_ids=relevant_recipe_ids,
        metadata_filters={
            "graph_ingredient_ids": ingredient_ids,
            "excluded_graph_ingredient_ids": excluded_ingredient_ids,
            "meal_type": spec.meal_type,
            "max_meal_minutes": spec.max_minutes,
            "min_protein_g": spec.min_protein_g,
        },
        split=spec.split,
        authoring="structured_ground_truth",
    )


class Neo4jMealGraphStore:
    """Dataset-scoped Neo4j importer and deterministic local graph retriever."""

    def __init__(self, driver: Any, *, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    def _execute(
        self,
        statement: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[Any]:
        records, _, _ = self._driver.execute_query(
            statement,
            parameters_=parameters or {},
            database_=self._database,
        )
        return list(records)

    def delete_dataset(self, dataset_version: str) -> None:
        self._execute(
            """
            MATCH (node)
            WHERE node.dataset_version = $dataset_version
              AND (node:Recipe OR node:Ingredient)
            DETACH DELETE node
            """,
            {"dataset_version": dataset_version},
        )

    def replace_projection(
        self,
        projection: MealGraphProjection,
        *,
        batch_size: int = 500,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._execute(
            """
            CREATE CONSTRAINT meal_recipe_identity IF NOT EXISTS
            FOR (recipe:Recipe)
            REQUIRE (recipe.dataset_version, recipe.recipe_id) IS UNIQUE
            """
        )
        self._execute(
            """
            CREATE CONSTRAINT meal_ingredient_identity IF NOT EXISTS
            FOR (ingredient:Ingredient)
            REQUIRE (ingredient.dataset_version, ingredient.ingredient_id) IS UNIQUE
            """
        )
        self.delete_dataset(projection.dataset_version)
        self._write_batches(
            """
            UNWIND $rows AS row
            MERGE (recipe:Recipe {
              dataset_version: row.dataset_version,
              recipe_id: row.recipe_id
            })
            SET recipe.name = row.name,
                recipe.description = row.description,
                recipe.meal_types = row.meal_types,
                recipe.tags = row.tags,
                recipe.calories_kcal = row.calories_kcal,
                recipe.protein_g = row.protein_g,
                recipe.total_minutes = row.total_minutes,
                recipe.source_ref = row.source_ref
            """,
            projection.recipes,
            batch_size,
        )
        self._write_batches(
            """
            UNWIND $rows AS row
            MERGE (ingredient:Ingredient {
              dataset_version: row.dataset_version,
              ingredient_id: row.ingredient_id
            })
            SET ingredient.canonical_name = row.canonical_name
            """,
            projection.ingredients,
            batch_size,
        )
        self._write_batches(
            """
            UNWIND $rows AS row
            MATCH (recipe:Recipe {
              dataset_version: row.dataset_version,
              recipe_id: row.recipe_id
            })
            MATCH (ingredient:Ingredient {
              dataset_version: row.dataset_version,
              ingredient_id: row.ingredient_id
            })
            MERGE (recipe)-[edge:HAS_INGREDIENT]->(ingredient)
            SET edge.position = row.position
            """,
            projection.has_ingredients,
            batch_size,
        )

    def _write_batches(
        self,
        statement: str,
        rows: Sequence[dict[str, Any]],
        batch_size: int,
    ) -> None:
        for start in range(0, len(rows), batch_size):
            self._execute(statement, {"rows": list(rows[start : start + batch_size])})

    def search_by_ingredients(
        self,
        ingredient_ids: Sequence[str],
        *,
        excluded_ingredient_ids: Sequence[str] = (),
        dataset_version: str,
        meal_type: str | None = None,
        max_minutes: int | None = None,
        min_protein_g: float | None = None,
        top_k: int = 20,
    ) -> list[GraphSearchResult]:
        query = build_local_graph_query(
            ingredient_ids,
            excluded_ingredient_ids=excluded_ingredient_ids,
            dataset_version=dataset_version,
            meal_type=meal_type,
            max_minutes=max_minutes,
            min_protein_g=min_protein_g,
            top_k=top_k,
        )
        return [
            GraphSearchResult.model_validate(dict(record))
            for record in self._execute(query.statement, query.parameters)
        ]


class GraphAugmentedRetriever:
    """Fuse an existing text retriever with deterministic local graph results."""

    def __init__(
        self,
        text_retriever: Any,
        graph_store: Any,
        *,
        dataset_version: str,
        rrf_k: int = 60,
    ) -> None:
        self._text_retriever = text_retriever
        self._graph_store = graph_store
        self._dataset_version = dataset_version
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        *,
        top_k: int = 20,
    ) -> list[str]:
        filters = metadata_filters or {}
        route_k = max(top_k, 20)
        text_ranking = self._text_retriever.search(
            query,
            filters,
            top_k=route_k,
        )
        ingredient_ids = [
            canonicalize_ingredient(str(value))
            for value in filters.get("graph_ingredient_ids", [])
        ]
        if not ingredient_ids:
            return text_ranking[:top_k]
        graph_results = self._graph_store.search_by_ingredients(
            ingredient_ids,
            excluded_ingredient_ids=filters.get("excluded_graph_ingredient_ids", []),
            dataset_version=self._dataset_version,
            meal_type=filters.get("meal_type"),
            max_minutes=filters.get("max_meal_minutes"),
            min_protein_g=filters.get("min_protein_g"),
            top_k=route_k,
        )
        graph_ranking = [result.recipe_id for result in graph_results]
        text_ranking = apply_recipe_id_gate(text_ranking, graph_ranking)
        fused = rrf_fuse(
            {"text": text_ranking, "local_graph": graph_ranking},
            k=self._rrf_k,
        )
        return apply_recipe_id_gate(fused, graph_ranking)[:top_k]


def build_meal_graph_projection(
    documents: Sequence[ExternalRecipeDocument],
    *,
    dataset_version: str,
) -> MealGraphProjection:
    """Project recipe documents into shared graph nodes and relation rows."""

    recipes: list[dict[str, Any]] = []
    ingredient_nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for document in documents:
        recipes.append(
            {
                "recipe_id": document.recipe_id,
                "name": document.name,
                "description": document.description,
                "meal_types": sorted(document.meal_types),
                "tags": list(document.tags),
                "calories_kcal": document.calories_kcal,
                "protein_g": document.protein_g,
                "total_minutes": document.total_minutes,
                "source_ref": document.source_ref,
                "dataset_version": dataset_version,
            }
        )
        seen_for_recipe: set[str] = set()
        for position, raw_name in enumerate(document.ingredients):
            ingredient_id = canonicalize_ingredient(raw_name)
            if not ingredient_id or ingredient_id in seen_for_recipe:
                continue
            seen_for_recipe.add(ingredient_id)
            ingredient_nodes.setdefault(
                ingredient_id,
                {
                    "ingredient_id": ingredient_id,
                    "canonical_name": ingredient_id,
                    "dataset_version": dataset_version,
                },
            )
            edges.append(
                {
                    "recipe_id": document.recipe_id,
                    "ingredient_id": ingredient_id,
                    "position": position,
                    "dataset_version": dataset_version,
                }
            )

    return MealGraphProjection(
        dataset_version=dataset_version,
        recipes=sorted(recipes, key=lambda item: item["recipe_id"]),
        ingredients=sorted(
            ingredient_nodes.values(), key=lambda item: item["ingredient_id"]
        ),
        has_ingredients=sorted(
            edges,
            key=lambda item: (item["recipe_id"], item["position"]),
        ),
    )
