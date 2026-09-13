import pytest

from gustobot.application.meal_planning.text2cypher.safety import (
    validate_cypher_safety,
)


SAFE_RECIPE_QUERY = """
MATCH (recipe:Recipe)
RETURN recipe.recipe_id AS recipe_id, recipe.name AS recipe_name
LIMIT $top_k
"""


def issue_codes(statement: str, *, require_recipe_ids: bool = True) -> set[str]:
    return {
        issue.code
        for issue in validate_cypher_safety(
            statement,
            require_recipe_ids=require_recipe_ids,
        ).issues
    }


def test_accepts_a_bounded_read_only_recipe_query():
    assert validate_cypher_safety(SAFE_RECIPE_QUERY).valid


@pytest.mark.parametrize(
    "clause",
    [
        "CREATE (n:Recipe)",
        "MERGE (n:Recipe {recipe_id: 'x'})",
        "MATCH (n) DELETE n",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) SET n.name = 'x'",
        "MATCH (n) REMOVE n.name",
        "DROP INDEX recipe_index",
        "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
        "CALL db.labels()",
        "FOREACH (x IN [] | CREATE (:Recipe))",
        "BEGIN TRANSACTION",
        "COMMIT",
        "ROLLBACK",
    ],
)
def test_rejects_write_admin_and_transaction_clauses(clause: str):
    assert "UNSAFE_CLAUSE" in issue_codes(clause, require_recipe_ids=False)


def test_keyword_scanning_ignores_property_names_strings_and_comments():
    statement = """
// CREATE (:Recipe) is an example, not an instruction
MATCH (recipe:Recipe)
WHERE recipe.created_at IS NOT NULL
  AND recipe.description = 'MERGE and DELETE are words here'
RETURN recipe.recipe_id AS recipe_id
LIMIT 20
"""

    assert validate_cypher_safety(statement).valid


def test_rejects_multiple_statements_but_not_semicolons_inside_strings():
    assert "MULTIPLE_STATEMENTS" in issue_codes(
        "MATCH (recipe:Recipe) RETURN recipe; DELETE recipe",
        require_recipe_ids=False,
    )
    assert validate_cypher_safety(
        "MATCH (recipe:Recipe) WHERE recipe.note = 'a;b' "
        "RETURN recipe.recipe_id AS recipe_id LIMIT 1"
    ).valid


def test_recipe_list_requires_stable_recipe_id_and_bounded_limit():
    missing_id = "MATCH (recipe:Recipe) RETURN recipe.name AS recipe_name LIMIT 10"
    missing_limit = "MATCH (recipe:Recipe) RETURN recipe.recipe_id AS recipe_id"
    oversized_limit = (
        "MATCH (recipe:Recipe) RETURN recipe.recipe_id AS recipe_id LIMIT 1000"
    )

    assert "MISSING_RECIPE_ID" in issue_codes(missing_id)
    assert "MISSING_LIMIT" in issue_codes(missing_limit)
    assert "LIMIT_TOO_HIGH" in issue_codes(oversized_limit)


def test_aggregate_query_can_opt_out_of_recipe_id_contract():
    report = validate_cypher_safety(
        "MATCH (recipe:Recipe) RETURN count(recipe) AS recipe_count",
        require_recipe_ids=False,
    )

    assert report.valid
