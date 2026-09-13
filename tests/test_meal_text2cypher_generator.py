import pytest

from gustobot.application.meal_planning.text2cypher.generator import (
    Text2CypherGenerator,
)
from gustobot.application.meal_planning.text2cypher.models import (
    CypherCandidate,
    CypherSource,
    ValidationIssue,
)
from gustobot.application.meal_planning.text2cypher.schema import (
    GraphRelationship,
    GraphSchemaSnapshot,
)


class DeterministicTextModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    async def ainvoke(self, prompt):
        self.inputs.append(prompt)
        return self.responses.pop(0)


class Message:
    def __init__(self, content):
        self.content = content


def schema() -> GraphSchemaSnapshot:
    return GraphSchemaSnapshot(
        node_properties={
            "Recipe": {"recipe_id", "name", "protein_g"},
            "Ingredient": {"ingredient_id"},
        },
        relationships={
            GraphRelationship(
                start_label="Recipe",
                relationship_type="HAS_INGREDIENT",
                end_label="Ingredient",
            )
        },
    )


@pytest.mark.asyncio
async def test_generate_supplies_question_schema_examples_and_parameter_contract():
    model = DeterministicTextModel(
        [
            Message(
                "```cypher\nMATCH (recipe:Recipe) "
                "RETURN recipe.recipe_id AS recipe_id LIMIT $top_k\n```"
            )
        ]
    )
    generator = Text2CypherGenerator(model)

    candidate = await generator.generate(
        question="Find high-protein recipes",
        schema=schema(),
        examples=[
            (
                "Find recipes with celery",
                "MATCH (recipe:Recipe) RETURN recipe.recipe_id AS recipe_id LIMIT $top_k",
            )
        ],
        parameters={"min_protein_g": 25, "top_k": 20},
    )

    prompt = model.inputs[0]
    assert "Find high-protein recipes" in prompt
    assert "Recipe(recipe_id, name, protein_g)" in prompt
    assert "Find recipes with celery" in prompt
    assert "$min_protein_g" in prompt and "$top_k" in prompt
    assert candidate.statement.startswith("MATCH")
    assert "```" not in candidate.statement
    assert candidate.parameters == {"min_protein_g": 25, "top_k": 20}
    assert candidate.source is CypherSource.DYNAMIC


@pytest.mark.asyncio
async def test_repair_receives_structured_errors_and_preserves_parameters():
    model = DeterministicTextModel(
        [
            "MATCH (recipe:Recipe) WHERE recipe.protein_g >= $min_protein_g "
            "RETURN recipe.recipe_id AS recipe_id LIMIT $top_k"
        ]
    )
    generator = Text2CypherGenerator(model)
    invalid = CypherCandidate(
        statement=(
            "MATCH (recipe:Recipe) WHERE recipe.protein >= $min_protein_g "
            "RETURN recipe.recipe_id AS recipe_id LIMIT $top_k"
        ),
        parameters={"min_protein_g": 25, "top_k": 20},
        source=CypherSource.DYNAMIC,
        attempt=0,
    )

    repaired = await generator.repair(
        question="Find high-protein recipes",
        candidate=invalid,
        issues=[
            ValidationIssue(
                code="UNKNOWN_PROPERTY",
                stage="schema",
                message="Unknown property Recipe.protein",
            )
        ],
        schema=schema(),
    )

    prompt = model.inputs[0]
    assert "UNKNOWN_PROPERTY" in prompt
    assert "Unknown property Recipe.protein" in prompt
    assert invalid.statement in prompt
    assert repaired.attempt == 1
    assert repaired.parameters == invalid.parameters
    assert "recipe.protein_g" in repaired.statement


@pytest.mark.asyncio
async def test_repair_refuses_to_exceed_two_attempts():
    generator = Text2CypherGenerator(DeterministicTextModel(["MATCH (n) RETURN n"]))
    exhausted = CypherCandidate(
        statement="MATCH (recipe:Recipe) RETURN recipe.recipe_id AS recipe_id LIMIT 20",
        source=CypherSource.DYNAMIC,
        attempt=2,
    )

    with pytest.raises(ValueError, match="repair attempts exhausted"):
        await generator.repair(
            question="question",
            candidate=exhausted,
            issues=[],
            schema=schema(),
        )
