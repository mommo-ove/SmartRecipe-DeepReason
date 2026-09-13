from __future__ import annotations

from pathlib import Path

from gustobot.application.meal_planning.external_corpus import (
    ExternalRecipeNormalizer,
    collect_normalized,
    load_external_corpus,
    parse_iso8601_minutes,
    parse_r_vector,
)


EXTERNAL_DIR = (
    Path(__file__).parents[1] / "gustobot" / "data" / "meal_planning" / "external"
)


def source_row(**updates):
    row = {
        "RecipeId": 123,
        "Name": "Chicken and Broccoli Bowl",
        "TotalTime": "PT35M",
        "Description": "A quick high protein dinner.",
        "RecipeCategory": "Chicken",
        "Keywords": 'c("Lunch", "Main Dish", "High Protein")',
        "RecipeIngredientParts": 'c("chicken breast", "broccoli", "rice")',
        "Calories": 520.0,
        "ProteinContent": 42.0,
        "RecipeInstructions": 'c("Cook chicken", "Steam broccoli")',
    }
    row.update(updates)
    return row


def test_iso8601_duration_is_converted_to_minutes():
    assert parse_iso8601_minutes("PT1H15M") == 75
    assert parse_iso8601_minutes("PT35M") == 35
    assert parse_iso8601_minutes(None) is None


def test_r_vector_fields_are_parsed_without_evaluating_source_text():
    assert parse_r_vector('c("chicken breast", "broccoli")') == [
        "chicken breast",
        "broccoli",
    ]
    assert parse_r_vector(None) == []


def test_normalizer_creates_traceable_retrieval_and_nutrition_capabilities():
    document = ExternalRecipeNormalizer().normalize(source_row())

    assert document is not None
    assert document.recipe_id == "foodcom-123"
    assert document.meal_types == {"lunch"}
    assert document.capabilities.retrieval_ready is True
    assert document.capabilities.nutrition_ready is True
    assert document.capabilities.time_ready is True
    assert document.capabilities.budget_ready is False
    assert document.capabilities.full_planning_ready is False
    assert document.source_ref == "hf:untitledwebsite123/food-recipes:123"


def test_nutritionally_substantial_non_breakfast_defaults_to_lunch_and_dinner():
    document = ExternalRecipeNormalizer().normalize(
        source_row(RecipeCategory="Chicken Breast", Keywords='c("Easy")')
    )

    assert document is not None
    assert document.meal_types == {"lunch", "dinner"}


def test_normalizer_rejects_rows_outside_meal_planning_scope():
    normalizer = ExternalRecipeNormalizer()

    assert normalizer.normalize(source_row(Calories=0)) is None
    assert normalizer.normalize(source_row(ProteinContent=2)) is None
    assert normalizer.normalize(source_row(TotalTime="PT4H")) is None
    assert normalizer.normalize(source_row(RecipeCategory="Beverages")) is None


def test_collection_skips_invalid_rows_and_deduplicates_source_ids():
    rows = [
        source_row(RecipeId=1),
        source_row(RecipeId=1, Name="duplicate"),
        source_row(RecipeId=2, ProteinContent=1),
        source_row(RecipeId=3, Name="Beef Rice Bowl"),
    ]

    documents = collect_normalized(rows, target_count=2)

    assert [item.recipe_id for item in documents] == ["foodcom-1", "foodcom-3"]


def test_generated_external_corpus_matches_manifest_and_capability_counts():
    corpus = load_external_corpus(
        EXTERNAL_DIR / "retrieval_corpus.v1.jsonl",
        EXTERNAL_DIR / "manifest.v1.json",
    )

    assert corpus.manifest.record_count == 300
    assert len(corpus.documents) == 300
    assert corpus.manifest.capabilities["full_planning_ready"] == 0
    assert all(item.capabilities.retrieval_ready for item in corpus.documents)
