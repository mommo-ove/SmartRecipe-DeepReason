from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from ..graph_retrieval import canonicalize_ingredient
from .models import CypherCandidate, CypherSource


class TemplateOperation(str, Enum):
    SEARCH_RECIPES = "search_recipes"
    RECIPE_PROPERTIES = "recipe_properties"
    RECIPE_INGREDIENTS = "recipe_ingredients"
    COUNT_RECIPES = "count_recipes"


class TemplateRequest(BaseModel):
    operation: TemplateOperation
    dataset_version: str = Field(min_length=1)
    recipe_id: str | None = None
    required_ingredient_ids: list[str] = Field(default_factory=list)
    excluded_ingredient_ids: list[str] = Field(default_factory=list)
    meal_type: str | None = None
    max_minutes: int | None = Field(default=None, gt=0)
    min_protein_g: float | None = Field(default=None, ge=0)
    top_k: int = Field(default=20, ge=1, le=100)

    @field_validator("required_ingredient_ids", "excluded_ingredient_ids")
    @classmethod
    def normalize_ingredient_ids(cls, values: list[str]) -> list[str]:
        normalized = [canonicalize_ingredient(value) for value in values]
        return list(dict.fromkeys(value for value in normalized if value))


_SEARCH_RECIPES = """
MATCH (recipe:Recipe {dataset_version: $dataset_version})
WHERE all(ingredient_id IN $ingredient_ids WHERE EXISTS {
  MATCH (recipe)-[:HAS_INGREDIENT]->(required:Ingredient {
    dataset_version: $dataset_version
  })
  WHERE required.ingredient_id = ingredient_id
})
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
       recipe.name AS recipe_name,
       recipe.total_minutes AS total_minutes,
       recipe.protein_g AS protein_g
ORDER BY recipe.protein_g DESC, recipe.recipe_id
LIMIT $top_k
""".strip()

_RECIPE_PROPERTIES = """
MATCH (recipe:Recipe {dataset_version: $dataset_version})
WHERE recipe.recipe_id = $recipe_id
RETURN recipe.recipe_id AS recipe_id,
       recipe.name AS recipe_name,
       recipe.total_minutes AS total_minutes,
       recipe.calories_kcal AS calories_kcal,
       recipe.protein_g AS protein_g,
       recipe.meal_types AS meal_types,
       recipe.tags AS tags
LIMIT 1
""".strip()

_RECIPE_INGREDIENTS = """
MATCH (recipe:Recipe {dataset_version: $dataset_version})
      -[:HAS_INGREDIENT]->(ingredient:Ingredient {
        dataset_version: $dataset_version
      })
WHERE recipe.recipe_id = $recipe_id
RETURN recipe.recipe_id AS recipe_id,
       recipe.name AS recipe_name,
       ingredient.ingredient_id AS ingredient_id,
       ingredient.canonical_name AS ingredient_name
ORDER BY ingredient.ingredient_id
LIMIT 100
""".strip()

_COUNT_RECIPES = """
MATCH (recipe:Recipe {dataset_version: $dataset_version})
RETURN count(recipe) AS recipe_count
""".strip()


def build_template_candidate(request: TemplateRequest) -> CypherCandidate | None:
    if request.operation is TemplateOperation.SEARCH_RECIPES:
        has_constraint = any(
            (
                request.required_ingredient_ids,
                request.excluded_ingredient_ids,
                request.meal_type,
                request.max_minutes is not None,
                request.min_protein_g is not None,
            )
        )
        if not has_constraint:
            return None
        return CypherCandidate(
            statement=_SEARCH_RECIPES,
            parameters={
                "dataset_version": request.dataset_version,
                "ingredient_ids": request.required_ingredient_ids,
                "excluded_ingredient_ids": request.excluded_ingredient_ids,
                "meal_type": request.meal_type,
                "max_minutes": request.max_minutes,
                "min_protein_g": request.min_protein_g,
                "top_k": request.top_k,
            },
            source=CypherSource.TEMPLATE,
            template_id="search_recipes_by_constraints",
        )

    if request.operation is TemplateOperation.COUNT_RECIPES:
        return CypherCandidate(
            statement=_COUNT_RECIPES,
            parameters={"dataset_version": request.dataset_version},
            source=CypherSource.TEMPLATE,
            template_id="count_recipes",
        )

    if not request.recipe_id:
        return None
    statement = (
        _RECIPE_PROPERTIES
        if request.operation is TemplateOperation.RECIPE_PROPERTIES
        else _RECIPE_INGREDIENTS
    )
    return CypherCandidate(
        statement=statement,
        parameters={
            "dataset_version": request.dataset_version,
            "recipe_id": request.recipe_id,
        },
        source=CypherSource.TEMPLATE,
        template_id=request.operation.value,
    )
