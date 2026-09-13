from __future__ import annotations

import json

from sqlalchemy import create_engine, text

from gustobot.infrastructure.persistence.meal_planning_repository import (
    SqlAlchemyMealPlanningRepository,
)


def build_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE meal_planning_recipe_facts (
                recipe_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                meal_types TEXT NOT NULL, calories_kcal NUMERIC NOT NULL,
                protein_g NUMERIC NOT NULL, total_minutes INTEGER NOT NULL,
                estimated_cost_cents INTEGER NOT NULL, allergens TEXT NOT NULL,
                allergen_status_verified INTEGER NOT NULL,
                preference_tags TEXT NOT NULL, source_refs TEXT NOT NULL,
                planning_eligible INTEGER NOT NULL
            )
        """))
        connection.execute(text("""
            CREATE TABLE meal_planning_ingredient_amounts (
                recipe_id TEXT NOT NULL, ingredient_id TEXT NOT NULL,
                amount NUMERIC NOT NULL, unit TEXT NOT NULL,
                ingredient_order INTEGER NOT NULL
            )
        """))
        for recipe_id, name in (("r1", "Recipe 1"), ("r2", "Recipe 2")):
            connection.execute(
                text("""
                    INSERT INTO meal_planning_recipe_facts VALUES
                    (:recipe_id, :name, :meal_types, 500, 35, 20, 900,
                     :allergens, 1, :tags, :sources, 1)
                """),
                {
                    "recipe_id": recipe_id,
                    "name": name,
                    "meal_types": json.dumps(["lunch"]),
                    "allergens": json.dumps([]),
                    "tags": json.dumps(["high_protein"]),
                    "sources": json.dumps([f"recipe:{recipe_id}"]),
                },
            )
            connection.execute(
                text("""
                    INSERT INTO meal_planning_ingredient_amounts VALUES
                    (:recipe_id, :ingredient_id, 150, 'g', 0)
                """),
                {"recipe_id": recipe_id, "ingredient_id": f"main-{recipe_id}"},
            )
    return engine


def test_repository_hydrates_only_the_explicit_recipe_id_closed_set():
    repository = SqlAlchemyMealPlanningRepository(build_engine())

    recipes = repository.get_planning_facts(["r1"])

    assert [recipe.recipe_id for recipe in recipes] == ["r1"]
    assert recipes[0].planning_eligible is True
    assert recipes[0].ingredients_normalized[0].ingredient_id == "main-r1"


def test_repository_deduplicates_and_normalizes_requested_ids():
    repository = SqlAlchemyMealPlanningRepository(build_engine())

    recipes = repository.get_planning_facts([" r2 ", "r1", "r2"])

    assert [recipe.recipe_id for recipe in recipes] == ["r1", "r2"]


def test_empty_id_scope_returns_empty_without_querying_the_whole_table():
    class EngineThatMustNotExecute:
        def connect(self):
            raise AssertionError("repository attempted an unscoped database query")

    repository = SqlAlchemyMealPlanningRepository(EngineThatMustNotExecute())

    assert repository.get_planning_facts([]) == []


def test_blank_recipe_id_is_rejected_before_sql_execution():
    repository = SqlAlchemyMealPlanningRepository(build_engine())

    try:
        repository.get_planning_facts(["r1", " "])
    except ValueError as error:
        assert "blank" in str(error)
    else:
        raise AssertionError("blank recipe ID must not reach the database")
