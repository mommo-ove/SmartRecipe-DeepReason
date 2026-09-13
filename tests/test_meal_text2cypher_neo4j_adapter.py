from gustobot.application.meal_planning.text2cypher.models import (
    CypherCandidate,
    CypherRunStatus,
    CypherSource,
)
from gustobot.application.meal_planning.text2cypher.neo4j_adapter import (
    Neo4jCypherAdapter,
)


class FakeNeo4jError(Exception):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


class FakeDriver:
    def __init__(self, *, records=None, error=None):
        self.records = records or []
        self.error = error
        self.calls = []

    def execute_query(self, statement, **kwargs):
        self.calls.append((statement, kwargs))
        if self.error:
            raise self.error
        return self.records, None, None


def candidate(
    statement="MATCH (recipe:Recipe) RETURN recipe.recipe_id AS recipe_id LIMIT $top_k",
):
    return CypherCandidate(
        statement=statement,
        parameters={"top_k": 20},
        source=CypherSource.DYNAMIC,
    )


def test_explain_uses_the_same_parameters_without_executing_the_query():
    driver = FakeDriver()
    adapter = Neo4jCypherAdapter(driver, database="neo4j")

    report = adapter.explain(candidate())

    assert report.valid
    assert len(driver.calls) == 1
    statement, kwargs = driver.calls[0]
    assert statement.startswith("EXPLAIN MATCH")
    assert kwargs["parameters_"] == {"top_k": 20}
    assert kwargs["database_"] == "neo4j"


def test_explain_classifies_syntax_and_missing_parameter_errors():
    syntax_driver = FakeDriver(
        error=FakeNeo4jError(
            "Invalid input",
            "Neo.ClientError.Statement.SyntaxError",
        )
    )
    parameter_driver = FakeDriver(
        error=FakeNeo4jError(
            "Expected parameter(s): ingredient_ids",
            "Neo.ClientError.Statement.ParameterMissing",
        )
    )

    assert Neo4jCypherAdapter(syntax_driver).explain(candidate()).issues[0].code == (
        "CYPHER_SYNTAX_ERROR"
    )
    assert Neo4jCypherAdapter(parameter_driver).explain(candidate()).issues[0].code == (
        "MISSING_PARAMETER"
    )


def test_execute_normalizes_records_and_stable_recipe_ids():
    driver = FakeDriver(
        records=[
            {"recipe_id": "r101", "recipe_name": "A"},
            {"recipe_id": "r101", "recipe_name": "A"},
            {"recipe_id": "r205", "recipe_name": "B"},
        ]
    )

    result = Neo4jCypherAdapter(driver).execute(candidate())

    assert result.status is CypherRunStatus.SUCCESS
    assert result.selected_recipe_ids == ["r101", "r205"]
    assert result.records[0] == {"recipe_id": "r101", "recipe_name": "A"}
    assert result.latency_ms >= 0


def test_execute_distinguishes_empty_results_from_failures():
    empty = Neo4jCypherAdapter(FakeDriver()).execute(candidate())
    failed = Neo4jCypherAdapter(
        FakeDriver(error=TimeoutError("database timeout"))
    ).execute(candidate())

    assert empty.status is CypherRunStatus.EMPTY
    assert failed.status is CypherRunStatus.EXECUTION_FAILED
    assert failed.error == "database timeout"
