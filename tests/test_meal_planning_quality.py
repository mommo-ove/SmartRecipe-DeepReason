from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.quality import (
    DuplicateRecipeIdError,
    QualityIssueCode,
    assess_recipe_quality,
    partition_recipes,
    validate_unique_recipe_ids,
)
from gustobot.application.meal_planning.models import (
    IngredientAmount,
    MealSlot,
    RecipeCandidate,
)


DATA_DIR = Path(__file__).parents[1] / "gustobot" / "data" / "meal_planning"


def complete_recipe(**overrides) -> RecipeCandidate:
    values = {
        "recipe_id": "r101",
        "name": "西兰花鸡胸肉",
        "meal_types": {MealSlot.LUNCH, MealSlot.DINNER},
        "calories_kcal": Decimal("430"),
        "protein_g": Decimal("42"),
        "total_minutes": 25,
        "estimated_cost_cents": 1600,
        "allergens": set(),
        "allergen_status_verified": True,
        "ingredients_normalized": [
            IngredientAmount(
                ingredient_id="chicken_breast",
                amount=Decimal("150"),
                unit="g",
            )
        ],
        "source_refs": ["recipe:r101", "nutrition:fixture:r101"],
    }
    values.update(overrides)
    return RecipeCandidate(**values)


@pytest.mark.parametrize(
    ("overrides", "expected_issue"),
    [
        ({"calories_kcal": None}, QualityIssueCode.MISSING_CALORIES),
        ({"protein_g": None}, QualityIssueCode.MISSING_PROTEIN),
        (
            {"ingredients_normalized": []},
            QualityIssueCode.MISSING_NORMALIZED_INGREDIENTS,
        ),
        ({"allergen_status_verified": False}, QualityIssueCode.UNKNOWN_ALLERGEN_STATUS),
        ({"source_refs": []}, QualityIssueCode.MISSING_SOURCE_LINEAGE),
    ],
)
def test_quality_gate_rejects_incomplete_planning_facts(overrides, expected_issue):
    assessment = assess_recipe_quality(complete_recipe(**overrides))

    assert assessment.planning_eligible is False
    assert expected_issue in assessment.issues


def test_non_normalized_ingredient_unit_is_rejected_at_model_boundary():
    with pytest.raises(ValidationError):
        IngredientAmount(ingredient_id="egg", amount=2, unit="piece")


def test_noneligible_recipe_remains_searchable_but_not_solver_eligible():
    complete = complete_recipe()
    incomplete = complete_recipe(recipe_id="r102", calories_kcal=None)

    partition = partition_recipes([complete, incomplete])

    assert [recipe.recipe_id for recipe in partition.searchable] == ["r101", "r102"]
    assert [recipe.recipe_id for recipe in partition.planning_eligible] == ["r101"]


def test_duplicate_recipe_ids_are_rejected():
    duplicate = complete_recipe()

    with pytest.raises(DuplicateRecipeIdError, match="r101"):
        validate_unique_recipe_ids([duplicate, duplicate])


def test_versioned_seed_corpus_matches_manifest_hash_and_count():
    corpus = load_seed_corpus(
        DATA_DIR / "recipes.v1.json",
        DATA_DIR / "manifest.v1.json",
    )

    assert corpus.manifest.version == "1.0.0-dev"
    assert corpus.manifest.status == "development_fixture"
    assert corpus.manifest.record_count == len(corpus.recipes)
    assert len(corpus.recipes) >= 12
    assert all(
        assess_recipe_quality(recipe).planning_eligible for recipe in corpus.recipes
    )
