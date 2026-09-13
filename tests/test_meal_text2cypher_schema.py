from gustobot.application.meal_planning.text2cypher.schema import (
    GraphRelationship,
    GraphSchemaSnapshot,
    Neo4jSchemaLoader,
    validate_cypher_schema,
)


class FakeDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, statement, **kwargs):
        self.calls.append((statement, kwargs))
        if "collect(DISTINCT property)" in statement:
            return (
                [
                    {
                        "label": "Recipe",
                        "properties": ["recipe_id", "name", "protein_g"],
                    },
                    {
                        "label": "Ingredient",
                        "properties": ["ingredient_id", "canonical_name"],
                    },
                    {"label": "Unrelated", "properties": ["secret"]},
                ],
                None,
                None,
            )
        return (
            [
                {
                    "start_label": "Recipe",
                    "relationship_type": "HAS_INGREDIENT",
                    "end_label": "Ingredient",
                },
                {
                    "start_label": "Unrelated",
                    "relationship_type": "OTHER",
                    "end_label": "Recipe",
                },
            ],
            None,
            None,
        )


def schema() -> GraphSchemaSnapshot:
    return GraphSchemaSnapshot(
        node_properties={
            "Recipe": {"recipe_id", "name", "protein_g"},
            "Ingredient": {"ingredient_id", "canonical_name"},
        },
        relationships={
            GraphRelationship(
                start_label="Recipe",
                relationship_type="HAS_INGREDIENT",
                end_label="Ingredient",
            )
        },
    )


def issue_codes(statement: str) -> set[str]:
    return {issue.code for issue in validate_cypher_schema(statement, schema()).issues}


def test_loader_builds_and_caches_an_allowlisted_schema_snapshot():
    driver = FakeDriver()
    loader = Neo4jSchemaLoader(
        driver,
        database="neo4j",
        allowed_labels={"Recipe", "Ingredient"},
    )

    first = loader.load()
    second = loader.load()

    assert first is second
    assert set(first.node_properties) == {"Recipe", "Ingredient"}
    assert first.relationships == {
        GraphRelationship(
            start_label="Recipe",
            relationship_type="HAS_INGREDIENT",
            end_label="Ingredient",
        )
    }
    assert len(driver.calls) == 2

    loader.load(refresh=True)
    assert len(driver.calls) == 4


def test_schema_validation_accepts_current_recipe_graph():
    statement = """
MATCH (recipe:Recipe)-[:HAS_INGREDIENT]->(ingredient:Ingredient)
WHERE recipe.protein_g >= $min_protein_g
RETURN recipe.recipe_id AS recipe_id
LIMIT 20
"""

    assert validate_cypher_schema(statement, schema()).valid


def test_schema_validation_rejects_unknown_labels_relationships_and_properties():
    assert "UNKNOWN_LABEL" in issue_codes("MATCH (dish:Dish) RETURN dish.name")
    assert "UNKNOWN_RELATIONSHIP" in issue_codes(
        "MATCH (recipe:Recipe)-[:CONTAINS]->(ingredient:Ingredient) RETURN recipe"
    )
    assert "UNKNOWN_PROPERTY" in issue_codes(
        "MATCH (recipe:Recipe) RETURN recipe.price"
    )


def test_schema_validation_checks_relationship_direction():
    wrong = """
MATCH (ingredient:Ingredient)-[:HAS_INGREDIENT]->(recipe:Recipe)
RETURN recipe.recipe_id AS recipe_id
"""
    reverse_syntax_but_correct_direction = """
MATCH (ingredient:Ingredient)<-[:HAS_INGREDIENT]-(recipe:Recipe)
RETURN recipe.recipe_id AS recipe_id
"""

    assert "WRONG_RELATIONSHIP_DIRECTION" in issue_codes(wrong)
    assert validate_cypher_schema(reverse_syntax_but_correct_direction, schema()).valid


def test_schema_snapshot_formats_a_small_prompt_contract():
    rendered = schema().to_prompt_text()

    assert "Recipe(recipe_id, name, protein_g)" in rendered
    assert "Recipe-[:HAS_INGREDIENT]->Ingredient" in rendered
