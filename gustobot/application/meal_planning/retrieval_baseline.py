from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from .external_corpus import ExternalRecipeDocument
from .models import MealSlot, RecipeCandidate


_ASCII_WORD = re.compile(r"[a-z0-9_]+")
_CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    normalized = text.casefold()
    tokens = _ASCII_WORD.findall(normalized)
    for run in _CHINESE_RUN.findall(normalized):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _document(recipe: RecipeCandidate) -> str:
    fields = [
        recipe.name,
        *(slot.value for slot in recipe.meal_types),
        *(ingredient.ingredient_id for ingredient in recipe.ingredients_normalized),
        *recipe.preference_tags,
    ]
    return " ".join(fields)


class FieldedBM25Retriever:
    """Small, deterministic lexical baseline for offline experiments."""

    def __init__(self, recipes: list[RecipeCandidate]):
        self._recipes = list(recipes)

    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        *,
        top_k: int = 20,
    ) -> list[str]:
        filters = metadata_filters or {}
        candidates = [
            recipe for recipe in self._recipes if self._matches(recipe, filters)
        ]
        if not candidates:
            return []

        corpus = [tokenize(_document(recipe)) for recipe in candidates]
        index = BM25Okapi(corpus)
        scores = index.get_scores(tokenize(query))
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].recipe_id),
        )
        return [recipe.recipe_id for recipe, _ in ranked[:top_k]]

    @staticmethod
    def _matches(recipe: RecipeCandidate, filters: dict[str, Any]) -> bool:
        if filters.get("planning_eligible") is True and not recipe.planning_eligible:
            return False

        meal_type = filters.get("meal_type")
        if meal_type is not None and MealSlot(str(meal_type)) not in recipe.meal_types:
            return False

        excluded = {str(item).casefold() for item in filters.get("excluded_allergens", [])}
        if recipe.allergens & excluded:
            return False

        max_minutes = filters.get("max_meal_minutes")
        if (
            max_minutes is not None
            and recipe.total_minutes is not None
            and recipe.total_minutes > int(max_minutes)
        ):
            return False
        return True


class ExternalBM25Retriever:
    def __init__(self, documents: list[ExternalRecipeDocument]):
        self._documents = list(documents)

    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        *,
        top_k: int = 20,
    ) -> list[str]:
        filters = metadata_filters or {}
        candidates = [
            document
            for document in self._documents
            if self._matches(document, filters)
        ]
        if not candidates:
            return []
        index = BM25Okapi([tokenize(item.search_text()) for item in candidates])
        scores = index.get_scores(tokenize(query))
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].recipe_id),
        )
        return [document.recipe_id for document, _ in ranked[:top_k]]

    @staticmethod
    def _matches(
        document: ExternalRecipeDocument,
        filters: dict[str, Any],
    ) -> bool:
        meal_type = filters.get("meal_type")
        if meal_type is not None and str(meal_type) not in document.meal_types:
            return False
        max_minutes = filters.get("max_meal_minutes")
        if max_minutes is not None and document.total_minutes > int(max_minutes):
            return False
        min_protein = filters.get("min_protein_g")
        if min_protein is not None and document.protein_g < float(min_protein):
            return False
        return True
