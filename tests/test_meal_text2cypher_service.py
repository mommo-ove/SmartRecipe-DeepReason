import pytest

from gustobot.application.meal_planning.text2cypher.models import (
    CypherCandidate,
    CypherExecutionResult,
    CypherRunStatus,
    CypherSource,
    ValidationIssue,
    ValidationReport,
)
from gustobot.application.meal_planning.text2cypher.schema import (
    GraphRelationship,
    GraphSchemaSnapshot,
)
from gustobot.application.meal_planning.text2cypher.service import Text2CypherService
from gustobot.application.meal_planning.text2cypher.templates import (
    TemplateOperation,
    TemplateRequest,
)


VALID_STATEMENT = (
    "MATCH (recipe:Recipe) "
    "RETURN recipe.recipe_id AS recipe_id LIMIT $top_k"
)


def graph_schema() -> GraphSchemaSnapshot:
    return GraphSchemaSnapshot(
        node_properties={
            "Recipe": {
                "recipe_id",
                "name",
                "dataset_version",
                "total_minutes",
                "protein_g",
                "calories_kcal",
                "meal_types",
                "tags",
            },
            "Ingredient": {
                "ingredient_id",
                "canonical_name",
                "dataset_version",
            },
        },
        relationships={
            GraphRelationship(
                start_label="Recipe",
                relationship_type="HAS_INGREDIENT",
                end_label="Ingredient",
            )
        },
    )


class StaticSchemaLoader:
    def __init__(self):
        self.calls = 0

    def load(self):
        self.calls += 1
        return graph_schema()


class DeterministicGenerator:
    def __init__(self, generated=None, repaired=None):
        self.generated = list(generated or [])
        self.repaired = list(repaired or [])
        self.generate_calls = []
        self.repair_calls = []

    async def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.generated.pop(0)

    async def repair(self, **kwargs):
        self.repair_calls.append(kwargs)
        statement = self.repaired.pop(0)
        previous = kwargs["candidate"]
        return CypherCandidate(
            statement=statement,
            parameters=previous.parameters,
            source=CypherSource.DYNAMIC,
            attempt=previous.attempt + 1,
        )


class DeterministicAdapter:
    def __init__(self, *, explain_reports=None, execution=None):
        self.explain_reports = list(explain_reports or [ValidationReport()])
        self.execution = execution or CypherExecutionResult(
            status=CypherRunStatus.SUCCESS,
            records=[{"recipe_id": "r101"}],
            selected_recipe_ids=["r101"],
        )
        self.explain_calls = []
        self.execute_calls = []

    def explain(self, candidate):
        self.explain_calls.append(candidate)
        return self.explain_reports.pop(0)

    def execute(self, candidate):
        self.execute_calls.append(candidate)
        return self.execution


def dynamic_candidate(statement=VALID_STATEMENT, *, attempt=0):
    return CypherCandidate(
        statement=statement,
        parameters={"top_k": 20},
        source=CypherSource.DYNAMIC,
        attempt=attempt,
    )


def service(generator, adapter=None):
    return Text2CypherService(
        generator=generator,
        adapter=adapter or DeterministicAdapter(),
        schema_loader=StaticSchemaLoader(),
    )


@pytest.mark.asyncio
async def test_template_hit_bypasses_generation_and_returns_deduplicated_ids():
    generator = DeterministicGenerator()
    adapter = DeterministicAdapter(
        execution=CypherExecutionResult(
            status=CypherRunStatus.SUCCESS,
            records=[
                {"recipe_id": "r101"},
                {"recipe_id": "r101"},
                {"recipe_id": "r205"},
            ],
            selected_recipe_ids=["r101", "r101", "r205"],
        )
    )

    result = await service(generator, adapter).run(
        question="推荐含鸡蛋、不含花生的菜",
        template_request=TemplateRequest(
            operation=TemplateOperation.SEARCH_RECIPES,
            dataset_version="foodcom-v1",
            required_ingredient_ids=["egg"],
            excluded_ingredient_ids=["peanut"],
        ),
    )

    assert generator.generate_calls == []
    assert result.status is CypherRunStatus.SUCCESS
    assert result.selected_recipe_ids == ["r101", "r205"]
    assert any(step.startswith("template_route:hit") for step in result.trace)


@pytest.mark.asyncio
async def test_template_miss_invokes_dynamic_generation():
    generator = DeterministicGenerator(generated=[dynamic_candidate()])

    result = await service(generator).run(
        question="找一些合适的菜",
        template_request=TemplateRequest(
            operation=TemplateOperation.SEARCH_RECIPES,
            dataset_version="foodcom-v1",
        ),
        parameters={"top_k": 20},
    )

    assert len(generator.generate_calls) == 1
    assert result.candidate.source is CypherSource.DYNAMIC
    assert any(step.startswith("template_route:miss") for step in result.trace)
    assert any(step.startswith("generate:ok") for step in result.trace)


@pytest.mark.asyncio
async def test_schema_failure_repairs_then_reruns_validation_from_safety():
    invalid = dynamic_candidate(
        "MATCH (recipe:Recipe) "
        "RETURN recipe.protein AS recipe_id LIMIT $top_k"
    )
    generator = DeterministicGenerator(
        generated=[invalid],
        repaired=[VALID_STATEMENT],
    )

    result = await service(generator).run(
        question="找高蛋白菜谱",
        parameters={"top_k": 20},
    )

    assert result.status is CypherRunStatus.SUCCESS
    assert len(generator.repair_calls) == 1
    assert sum(step.startswith("validate:safety") for step in result.trace) == 2
    assert sum(step.startswith("validate:schema") for step in result.trace) == 2
    assert sum(step.startswith("explain:ok") for step in result.trace) == 1


@pytest.mark.asyncio
async def test_repairable_safety_contract_failure_is_repaired():
    missing_limit = dynamic_candidate(
        "MATCH (recipe:Recipe) RETURN recipe.recipe_id AS recipe_id"
    )
    generator = DeterministicGenerator(
        generated=[missing_limit],
        repaired=[VALID_STATEMENT],
    )

    result = await service(generator).run(
        question="找菜谱",
        parameters={"top_k": 20},
    )

    assert result.status is CypherRunStatus.SUCCESS
    assert len(generator.repair_calls) == 1
    assert generator.repair_calls[0]["issues"][0].code == "MISSING_LIMIT"
    assert sum(step.startswith("validate:safety") for step in result.trace) == 2


@pytest.mark.asyncio
async def test_explain_failure_is_repaired_before_execution():
    explain_error = ValidationReport(
        issues=[
            ValidationIssue(
                code="CYPHER_SYNTAX_ERROR",
                stage="explain",
                message="invalid input",
            )
        ]
    )
    generator = DeterministicGenerator(
        generated=[dynamic_candidate()],
        repaired=[VALID_STATEMENT],
    )
    adapter = DeterministicAdapter(
        explain_reports=[explain_error, ValidationReport()]
    )

    result = await service(generator, adapter).run(
        question="找高蛋白菜谱",
        parameters={"top_k": 20},
    )

    assert result.status is CypherRunStatus.SUCCESS
    assert len(generator.repair_calls) == 1
    assert len(adapter.explain_calls) == 2
    assert len(adapter.execute_calls) == 1


@pytest.mark.asyncio
async def test_repairs_stop_after_two_attempts():
    invalid = (
        "MATCH (recipe:Recipe) "
        "RETURN recipe.unknown AS recipe_id LIMIT $top_k"
    )
    generator = DeterministicGenerator(
        generated=[dynamic_candidate(invalid)],
        repaired=[invalid, invalid],
    )
    adapter = DeterministicAdapter()

    result = await service(generator, adapter).run(
        question="找菜谱",
        parameters={"top_k": 20},
    )

    assert result.status is CypherRunStatus.VALIDATION_FAILED
    assert len(generator.repair_calls) == 2
    assert adapter.execute_calls == []
    assert result.candidate.attempt == 2


@pytest.mark.asyncio
async def test_unsafe_statement_is_never_repaired_explained_or_executed():
    unsafe = dynamic_candidate(
        "MATCH (recipe:Recipe) DELETE recipe "
        "RETURN recipe.recipe_id AS recipe_id LIMIT $top_k"
    )
    generator = DeterministicGenerator(
        generated=[unsafe],
        repaired=[VALID_STATEMENT],
    )
    adapter = DeterministicAdapter()

    result = await service(generator, adapter).run(
        question="删除后返回菜谱",
        parameters={"top_k": 20},
    )

    assert result.status is CypherRunStatus.VALIDATION_FAILED
    assert generator.repair_calls == []
    assert adapter.explain_calls == []
    assert adapter.execute_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("execution", "expected_status"),
    [
        (
            CypherExecutionResult(status=CypherRunStatus.EMPTY),
            CypherRunStatus.EMPTY,
        ),
        (
            CypherExecutionResult(
                status=CypherRunStatus.EXECUTION_FAILED,
                error="database timeout",
            ),
            CypherRunStatus.EXECUTION_FAILED,
        ),
    ],
)
async def test_empty_result_is_distinct_from_execution_failure(
    execution, expected_status
):
    adapter = DeterministicAdapter(execution=execution)
    result = await service(
        DeterministicGenerator(generated=[dynamic_candidate()]),
        adapter,
    ).run(
        question="找菜谱",
        parameters={"top_k": 20},
    )

    assert result.status is expected_status
    assert result.execution.status is expected_status
