from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from gustobot.application.meal_planning.models import IngredientAmount, RecipeCandidate


_PLANNING_FACTS_SQL = text("""
SELECT
    facts.recipe_id,
    facts.name,
    facts.meal_types,
    facts.calories_kcal,
    facts.protein_g,
    facts.total_minutes,
    facts.estimated_cost_cents,
    facts.allergens,
    facts.allergen_status_verified,
    facts.preference_tags,
    facts.source_refs,
    ingredient.ingredient_id,
    ingredient.amount,
    ingredient.unit,
    ingredient.ingredient_order
FROM meal_planning_recipe_facts AS facts
JOIN meal_planning_ingredient_amounts AS ingredient
  ON ingredient.recipe_id = facts.recipe_id
WHERE facts.recipe_id IN :recipe_ids
  AND facts.planning_eligible = 1
ORDER BY facts.recipe_id, ingredient.ingredient_order
""").bindparams(bindparam("recipe_ids", expanding=True))


class SqlAlchemyMealPlanningRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_planning_facts(
        self,
        recipe_ids: Sequence[str],
    ) -> list[RecipeCandidate]:
        normalized_ids = _normalize_recipe_ids(recipe_ids)
        if not normalized_ids:
            return []
        with self._engine.connect() as connection:
            rows = connection.execute(
                _PLANNING_FACTS_SQL,
                {"recipe_ids": normalized_ids},
            ).mappings().all()

        rows_by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            mapped = dict(row)
            recipe_id = str(mapped["recipe_id"])
            if recipe_id not in normalized_ids:
                raise ValueError("database returned a recipe outside the requested ID scope")
            rows_by_recipe[recipe_id].append(mapped)
        return [
            _map_recipe_rows(rows_by_recipe[recipe_id])
            for recipe_id in sorted(rows_by_recipe)
        ]


def _normalize_recipe_ids(recipe_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(recipe_ids, (str, bytes)):
        raise ValueError("recipe_ids must be a structured collection")
    normalized = [str(recipe_id).strip() for recipe_id in recipe_ids]
    if any(not recipe_id for recipe_id in normalized):
        raise ValueError("recipe_ids cannot contain a blank value")
    if len(set(normalized)) > 200:
        raise ValueError("recipe_ids cannot contain more than 200 IDs")
    return tuple(sorted(set(normalized)))


def _map_recipe_rows(rows: list[dict[str, Any]]) -> RecipeCandidate:
    first = rows[0]
    return RecipeCandidate(
        recipe_id=str(first["recipe_id"]),
        name=str(first["name"]),
        meal_types=set(_json_list(first["meal_types"])),
        calories_kcal=Decimal(str(first["calories_kcal"])),
        protein_g=Decimal(str(first["protein_g"])),
        total_minutes=int(first["total_minutes"]),
        estimated_cost_cents=int(first["estimated_cost_cents"]),
        allergens=set(_json_list(first["allergens"])),
        allergen_status_verified=bool(first["allergen_status_verified"]),
        ingredients_normalized=[
            IngredientAmount(
                ingredient_id=str(row["ingredient_id"]),
                amount=Decimal(str(row["amount"])),
                unit=str(row["unit"]),
            )
            for row in rows
        ],
        preference_tags=set(_json_list(first["preference_tags"])),
        source_refs=_json_list(first["source_refs"]),
    )


def _json_list(value: Any) -> list[str]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, list):
        raise ValueError("expected a JSON array from planning facts")
    return [str(item) for item in decoded]
