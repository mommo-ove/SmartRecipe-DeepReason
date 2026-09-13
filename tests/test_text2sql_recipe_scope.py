from pathlib import Path

from gustobot.application.agents.text2sql_sub_graph.sql_validation.node import (
    validate_recipe_scope,
)


def test_recipe_scope_rejects_sql_that_drops_upstream_ids():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "canonical_recipe_id" in errors[0]


def test_recipe_scope_accepts_exact_canonical_id_filter():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "WHERE canonical_recipe_id IN ('201003834', '201004552')",
        ["201003834", "201004552"],
    )

    assert ok is True
    assert errors == []


def test_recipe_scope_rejects_extra_recipe_ids():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "WHERE canonical_recipe_id IN ('201003834', '201004552', '999999999')",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "范围" in errors[0]


def test_recipe_scope_rejects_scope_widening_or_predicate():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "WHERE canonical_recipe_id IN ('201003834', '201004552') OR 1 = 1",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "OR" in errors[0]


def test_recipe_scope_ignores_ids_and_column_names_in_comments():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "/* canonical_recipe_id IN ('201003834', '201004552') */",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "canonical_recipe_id" in errors[0]


def test_recipe_scope_rejects_parenthesized_negated_filter():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "WHERE NOT (canonical_recipe_id IN ('201003834', '201004552'))",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "正向" in errors[0]


def test_recipe_scope_rejects_deeply_parenthesized_negated_filter():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "WHERE NOT (((canonical_recipe_id IN ('201003834', '201004552'))))",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "正向" in errors[0]


def test_recipe_scope_rejects_is_false_inversion():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "WHERE (canonical_recipe_id IN ('201003834', '201004552')) IS FALSE",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "正向" in errors[0]


def test_recipe_scope_rejects_equals_false_inversion():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "WHERE (canonical_recipe_id IN ('201003834', '201004552')) = FALSE",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "正向" in errors[0]


def test_recipe_scope_rejects_xor_inversion():
    ok, errors = validate_recipe_scope(
        "SELECT SUM(total_calories) FROM recipes "
        "WHERE (canonical_recipe_id IN ('201003834', '201004552')) XOR TRUE",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "正向" in errors[0]


def test_recipe_scope_rejects_uncorrelated_exists_subquery_scope():
    ok, errors = validate_recipe_scope(
        "SELECT * FROM recipes "
        "WHERE EXISTS ("
        "SELECT 1 FROM (SELECT '201003834' AS canonical_recipe_id) x "
        "WHERE canonical_recipe_id IN ('201003834', '201004552')"
        ")",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "顶层" in errors[0]


def test_recipe_scope_rejects_scalar_subquery_scope():
    ok, errors = validate_recipe_scope(
        "SELECT * FROM recipes "
        "WHERE (SELECT canonical_recipe_id IN ('201003834', '201004552') "
        "FROM (SELECT '201003834' AS canonical_recipe_id) x) = TRUE",
        ["201003834", "201004552"],
    )

    assert ok is False
    assert "顶层" in errors[0]


def test_mysql_schema_and_seed_share_the_graph_canonical_ids():
    repo_root = Path(__file__).resolve().parents[1]
    schema = (repo_root / "gustobot/data/init_mysql.sql").read_text(encoding="utf-8")
    seed = (repo_root / "gustobot/data/insert_sample_data.sql").read_text(encoding="utf-8")

    assert "canonical_recipe_id VARCHAR(64)" in schema
    assert "UNIQUE INDEX idx_recipe_canonical_id" in schema
    assert "MODIFY canonical_recipe_id VARCHAR(64) NOT NULL" in seed
    assert "201003834" in seed
    assert "201004552" in seed
