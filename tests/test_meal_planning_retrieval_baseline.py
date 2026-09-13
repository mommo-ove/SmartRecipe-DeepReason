from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.external_corpus import ExternalRecipeNormalizer
from gustobot.application.meal_planning.models import MealSlot
from gustobot.application.meal_planning.retrieval_baseline import (
    ExternalBM25Retriever,
    FieldedBM25Retriever,
)


DATA_DIR = Path(__file__).parents[1] / "gustobot" / "data" / "meal_planning"


def recipes():
    return load_seed_corpus(
        DATA_DIR / "recipes.v1.json",
        DATA_DIR / "manifest.v1.json",
    ).recipes


def test_bm25_applies_structured_filters_before_ranking():
    retriever = FieldedBM25Retriever(recipes())

    results = retriever.search(
        "鸡胸肉 高蛋白",
        {
            "planning_eligible": True,
            "meal_type": "lunch",
            "excluded_allergens": ["peanut"],
            "max_meal_minutes": 30,
        },
        top_k=5,
    )

    assert results[0] == "r201"
    assert all(
        MealSlot.LUNCH in next(r for r in recipes() if r.recipe_id == recipe_id).meal_types
        for recipe_id in results
    )


def test_planning_eligibility_is_computed_from_facts_not_predicted_by_bm25():
    candidates = recipes()
    target = candidates[0]
    broken = target.model_copy(update={"protein_g": None})
    retriever = FieldedBM25Retriever(
        [broken if item.recipe_id == target.recipe_id else item for item in candidates]
    )

    results = retriever.search(
        target.name,
        {"planning_eligible": True},
        top_k=12,
    )

    assert target.recipe_id not in results


def test_numeric_filters_do_not_become_semantic_query_words():
    retriever = FieldedBM25Retriever(recipes())

    results = retriever.search(
        "quick breakfast",
        {"meal_type": "breakfast", "max_meal_minutes": 10},
        top_k=10,
    )

    assert results == ["r104"]
    selected = next(item for item in recipes() if item.recipe_id == results[0])
    assert selected.total_minutes == 8
    assert selected.calories_kcal == Decimal("420")


def test_external_bm25_uses_public_corpus_fields_and_meal_slot_filter():
    normalizer = ExternalRecipeNormalizer()
    chicken = normalizer.normalize(
        {
            "RecipeId": 1,
            "Name": "Chicken Broccoli Bowl",
            "TotalTime": "PT25M",
            "Description": "high protein rice bowl",
            "RecipeCategory": "Chicken Breast",
            "Keywords": 'c("Lunch", "High Protein")',
            "RecipeIngredientParts": 'c("chicken breast", "broccoli", "rice")',
            "Calories": 520,
            "ProteinContent": 42,
            "RecipeInstructions": 'c("cook")',
        }
    )
    beef = normalizer.normalize(
        {
            "RecipeId": 2,
            "Name": "Beef Pepper Dinner",
            "TotalTime": "PT30M",
            "Description": "beef dinner",
            "RecipeCategory": "Meat",
            "Keywords": 'c("Dinner")',
            "RecipeIngredientParts": 'c("beef", "pepper")',
            "Calories": 600,
            "ProteinContent": 45,
            "RecipeInstructions": 'c("cook")',
        }
    )

    results = ExternalBM25Retriever([chicken, beef]).search(
        "chicken broccoli high protein",
        {"meal_type": "lunch", "max_meal_minutes": 30},
        top_k=5,
    )

    assert results == ["foodcom-1"]
