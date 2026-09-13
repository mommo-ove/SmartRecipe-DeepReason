from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from gustobot.infrastructure.persistence.meal_planning_repository import (
    SqlAlchemyMealPlanningRepository,
)


@pytest.mark.integration
def test_mysql_repository_can_read_seeded_planning_facts():
    database_url = os.getenv("MEAL_PLANNING_MYSQL_URL")
    if not database_url:
        pytest.skip("set MEAL_PLANNING_MYSQL_URL to run the real MySQL integration")

    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM meal_planning_ingredient_amounts WHERE recipe_id = 'r101'"))
        connection.execute(text("DELETE FROM meal_planning_recipe_facts WHERE recipe_id = 'r101'"))
        connection.execute(text("""
            INSERT INTO meal_planning_recipe_facts
            (recipe_id, name, meal_types, calories_kcal, protein_g,
             total_minutes, estimated_cost_cents, allergens,
             allergen_status_verified, preference_tags, source_refs,
             planning_eligible)
            VALUES ('r101', 'integration recipe', JSON_ARRAY('lunch'), 500, 35,
                    20, 900, JSON_ARRAY(), 1, JSON_ARRAY('high_protein'),
                    JSON_ARRAY('integration:r101'), 1)
        """))
        connection.execute(text("""
            INSERT INTO meal_planning_ingredient_amounts
            (recipe_id, ingredient_id, amount, unit, ingredient_order)
            VALUES ('r101', 'chicken_breast', 150, 'g', 0)
        """))

    repository = SqlAlchemyMealPlanningRepository(engine)
    recipes = repository.get_planning_facts(["r101"])

    assert [recipe.recipe_id for recipe in recipes] == ["r101"]
    assert recipes[0].planning_eligible is True
