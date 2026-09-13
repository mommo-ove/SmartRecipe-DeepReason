from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel

from .models import RecipeCandidate


class QualityIssueCode(str, Enum):
    MISSING_CALORIES = "missing_calories"
    MISSING_PROTEIN = "missing_protein"
    MISSING_TIME = "missing_time"
    MISSING_COST = "missing_cost"
    MISSING_NORMALIZED_INGREDIENTS = "missing_normalized_ingredients"
    UNKNOWN_ALLERGEN_STATUS = "unknown_allergen_status"
    MISSING_SOURCE_LINEAGE = "missing_source_lineage"


class RecipeQualityAssessment(BaseModel):
    recipe_id: str
    planning_eligible: bool
    issues: list[QualityIssueCode]


class RecipePartition(BaseModel):
    searchable: list[RecipeCandidate]
    planning_eligible: list[RecipeCandidate]
    rejected: dict[str, list[QualityIssueCode]]


class DuplicateRecipeIdError(ValueError):
    """Raised when stable recipe identifiers are not unique."""


def assess_recipe_quality(recipe: RecipeCandidate) -> RecipeQualityAssessment:
    checks = (
        (recipe.calories_kcal is None, QualityIssueCode.MISSING_CALORIES),
        (recipe.protein_g is None, QualityIssueCode.MISSING_PROTEIN),
        (recipe.total_minutes is None, QualityIssueCode.MISSING_TIME),
        (recipe.estimated_cost_cents is None, QualityIssueCode.MISSING_COST),
        (
            not recipe.ingredients_normalized,
            QualityIssueCode.MISSING_NORMALIZED_INGREDIENTS,
        ),
        (
            not recipe.allergen_status_verified,
            QualityIssueCode.UNKNOWN_ALLERGEN_STATUS,
        ),
        (not recipe.source_refs, QualityIssueCode.MISSING_SOURCE_LINEAGE),
    )
    issues = [issue for failed, issue in checks if failed]
    return RecipeQualityAssessment(
        recipe_id=recipe.recipe_id,
        planning_eligible=not issues,
        issues=issues,
    )


def validate_unique_recipe_ids(recipes: list[RecipeCandidate]) -> None:
    counts = Counter(recipe.recipe_id for recipe in recipes)
    duplicates = sorted(recipe_id for recipe_id, count in counts.items() if count > 1)
    if duplicates:
        raise DuplicateRecipeIdError(
            f"duplicate recipe_id values: {', '.join(duplicates)}"
        )


def partition_recipes(recipes: list[RecipeCandidate]) -> RecipePartition:
    validate_unique_recipe_ids(recipes)
    eligible: list[RecipeCandidate] = []
    rejected: dict[str, list[QualityIssueCode]] = {}
    for recipe in recipes:
        assessment = assess_recipe_quality(recipe)
        if assessment.planning_eligible:
            eligible.append(recipe)
        else:
            rejected[recipe.recipe_id] = assessment.issues
    return RecipePartition(
        searchable=recipes,
        planning_eligible=eligible,
        rejected=rejected,
    )
