from gustobot.application.meal_planning.text2cypher.models import CypherSource
from gustobot.application.meal_planning.text2cypher.templates import (
    TemplateOperation,
    TemplateRequest,
    build_template_candidate,
)


def test_required_and_excluded_ingredients_use_parameters_not_interpolation():
    candidate = build_template_candidate(
        TemplateRequest(
            operation=TemplateOperation.SEARCH_RECIPES,
            dataset_version="foodcom-v1",
            required_ingredient_ids=["celery", "onions"],
            excluded_ingredient_ids=["chicken breasts"],
            top_k=20,
        )
    )

    assert candidate is not None
    assert candidate.source is CypherSource.TEMPLATE
    assert candidate.template_id == "search_recipes_by_constraints"
    assert "celery" not in candidate.statement
    assert "chicken breasts" not in candidate.statement
    assert "NOT EXISTS" in candidate.statement
    assert candidate.parameters["ingredient_ids"] == ["celery", "onions"]
    assert candidate.parameters["excluded_ingredient_ids"] == ["chicken breasts"]


def test_search_template_supports_meal_time_and_protein_filters():
    candidate = build_template_candidate(
        TemplateRequest(
            operation=TemplateOperation.SEARCH_RECIPES,
            dataset_version="foodcom-v1",
            meal_type="dinner",
            max_minutes=30,
            min_protein_g=25,
        )
    )

    assert candidate is not None
    assert "$meal_type IN recipe.meal_types" in candidate.statement
    assert "recipe.total_minutes <= $max_minutes" in candidate.statement
    assert "recipe.protein_g >= $min_protein_g" in candidate.statement
    assert candidate.parameters["meal_type"] == "dinner"


def test_recipe_properties_template_uses_stable_recipe_id():
    candidate = build_template_candidate(
        TemplateRequest(
            operation=TemplateOperation.RECIPE_PROPERTIES,
            dataset_version="foodcom-v1",
            recipe_id="foodcom-481",
        )
    )

    assert candidate is not None
    assert "recipe.recipe_id = $recipe_id" in candidate.statement
    assert "recipe.name AS recipe_name" in candidate.statement
    assert "foodcom-481" not in candidate.statement


def test_recipe_ingredients_template_returns_canonical_ingredient_ids():
    candidate = build_template_candidate(
        TemplateRequest(
            operation=TemplateOperation.RECIPE_INGREDIENTS,
            dataset_version="foodcom-v1",
            recipe_id="foodcom-481",
        )
    )

    assert candidate is not None
    assert "ingredient.ingredient_id AS ingredient_id" in candidate.statement
    assert "HAS_INGREDIENT" in candidate.statement


def test_recipe_count_template_is_dataset_scoped():
    candidate = build_template_candidate(
        TemplateRequest(
            operation=TemplateOperation.COUNT_RECIPES,
            dataset_version="foodcom-v1",
        )
    )

    assert candidate is not None
    assert "count(recipe) AS recipe_count" in candidate.statement
    assert candidate.parameters == {"dataset_version": "foodcom-v1"}


def test_ambiguous_search_without_structured_constraints_routes_to_dynamic():
    candidate = build_template_candidate(
        TemplateRequest(
            operation=TemplateOperation.SEARCH_RECIPES,
            dataset_version="foodcom-v1",
        )
    )

    assert candidate is None
