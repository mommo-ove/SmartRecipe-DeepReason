import pytest

from gustobot.application.deepreason.domain_agents import (
    DomainAgentRegistry,
    FunctionDomainAgent,
    build_demo_registry,
    build_gustobot_registry,
)
from gustobot.application.deepreason.direct_capabilities import GustoBotDirectExecutor
from gustobot.application.deepreason.models import AgentTask, Domain, TaskStatus


class RecordingGraph:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def ainvoke(self, state, config=None):
        self.calls.append((state, config))
        return self.result


@pytest.mark.asyncio
async def test_function_agent_normalizes_successful_result():
    async def handler(task, context):
        return {"answer": "宫保鸡丁", "sources": ["neo4j"]}

    agent = FunctionDomainAgent(name="recipe_agent", domain=Domain.RECIPE, handler=handler)
    task = AgentTask(task_id="recipe-1", domain=Domain.RECIPE, instruction="推荐鸡肉菜")

    handoff = await agent.execute(task, {})

    assert handoff.status == TaskStatus.SUCCESS
    assert handoff.summary == "宫保鸡丁"
    assert handoff.output["sources"] == ["neo4j"]


@pytest.mark.asyncio
async def test_function_agent_isolates_handler_failure():
    async def handler(task, context):
        raise RuntimeError("neo4j unavailable")

    agent = FunctionDomainAgent(name="recipe_agent", domain=Domain.RECIPE, handler=handler)
    task = AgentTask(task_id="recipe-1", domain=Domain.RECIPE, instruction="推荐鸡肉菜")

    handoff = await agent.execute(task, {})

    assert handoff.status == TaskStatus.FAILED
    assert "neo4j unavailable" in handoff.error


def test_registry_rejects_duplicate_domain():
    async def handler(task, context):
        return {"answer": "ok"}

    registry = DomainAgentRegistry()
    registry.register(FunctionDomainAgent(name="one", domain=Domain.RECIPE, handler=handler))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(FunctionDomainAgent(name="two", domain=Domain.RECIPE, handler=handler))


@pytest.mark.asyncio
async def test_demo_registry_runs_without_external_databases():
    registry = build_demo_registry()
    recipe = await registry.get(Domain.RECIPE).execute(
        AgentTask(task_id="recipe-1", domain=Domain.RECIPE, instruction="推荐低辣鸡肉菜"),
        {},
    )
    analytics = await registry.get(Domain.ANALYTICS).execute(
        AgentTask(task_id="analytics-1", domain=Domain.ANALYTICS, instruction="统计平均时长"),
        {},
    )

    assert "演示" in recipe.summary
    assert analytics.output["sql_statement"].startswith("SELECT")


@pytest.mark.asyncio
async def test_real_registry_dispatches_recipe_and_analytics_directly_to_subgraphs():
    recipe_graph = RecordingGraph(
        {
            "answer": "recipe answer",
            "cyphers": [
                {
                    "statement": "MATCH (r:Recipe) RETURN r LIMIT 1",
                    "records": [{"name": "kung pao chicken"}],
                }
            ],
            "steps": ["final_answer"],
        }
    )
    analytics_graph = RecordingGraph(
        {
            "answer": "342 recipes",
            "sql_statement": "SELECT COUNT(*) AS total FROM recipes",
            "execution_results": [{"total": 342}],
            "steps": ["format_answer"],
        }
    )
    executor = GustoBotDirectExecutor(
        recipe_graph=recipe_graph,
        analytics_graph=analytics_graph,
        node_handlers={},
    )
    registry = build_gustobot_registry(executor)
    context = {"session_id": "session-7", "user_id": "user-2"}

    recipe = await registry.get(Domain.RECIPE).execute(
        AgentTask(task_id="recipe-1", domain=Domain.RECIPE, instruction="find chicken recipes"),
        context,
    )
    analytics = await registry.get(Domain.ANALYTICS).execute(
        AgentTask(task_id="analytics-1", domain=Domain.ANALYTICS, instruction="count recipes"),
        context,
    )

    assert recipe_graph.calls[0][0] == {
        "question": "find chicken recipes",
        "history": [],
    }
    assert analytics_graph.calls[0][0] == {
        "question": "count recipes",
        "db_type": "MySQL",
        "max_rows": 1000,
        "max_retries": 3,
    }
    assert recipe_graph.calls[0][1]["configurable"]["thread_id"] == "session-7:recipe-1"
    assert analytics.output["execution_results"] == [{"total": 342}]
    assert recipe.evidence[0].source_type == "cypher"
    assert analytics.evidence[0].source_type == "sql"


@pytest.mark.asyncio
async def test_cypher_records_emit_atomic_recipe_fact_evidence():
    async def handler(task, context):
        return {
            "answer": "recipe facts",
            "cyphers": [
                {
                    "statement": "MATCH (r:Recipe) RETURN r",
                    "records": [
                        {
                            "recipe_id": "r101",
                            "total_minutes": 15,
                            "calories_kcal": 280,
                            "ingredients": ["egg", "tomato"],
                        }
                    ],
                }
            ],
        }

    handoff = await FunctionDomainAgent(
        name="recipe_agent",
        domain=Domain.RECIPE,
        handler=handler,
    ).execute(
        AgentTask(task_id="recipe-1", domain=Domain.RECIPE, instruction="facts"),
        {},
    )

    atoms = [item for item in handoff.evidence if item.source_type == "fact_atom"]
    facts = {item.metadata["field"]: item.metadata["value"] for item in atoms}
    assert facts["exists"] is True
    assert facts["total_minutes"] == 15
    assert facts["calories_kcal"] == 280
    ingredients = next(item for item in atoms if item.metadata["field"] == "ingredients")
    assert ingredients.metadata["value"] == ["egg", "tomato"]


@pytest.mark.asyncio
async def test_direct_vision_agent_passes_image_path_to_business_node():
    calls = []

    async def vision_node(state, *, config):
        calls.append((state, config))
        message = type("Message", (), {"content": "vision answer"})()
        return {"messages": [message]}

    executor = GustoBotDirectExecutor(
        recipe_graph=RecordingGraph({}),
        analytics_graph=RecordingGraph({}),
        node_handlers={"vision": vision_node},
    )
    agent = build_gustobot_registry(executor).get(Domain.VISION)
    handoff = await agent.execute(
        AgentTask(task_id="vision-1", domain=Domain.VISION, instruction="identify this dish"),
        {"session_id": "session-8", "image_path": "C:/tmp/dish.jpg"},
    )

    assert handoff.status == TaskStatus.SUCCESS
    assert handoff.summary == "vision answer"
    assert calls[0][1]["configurable"]["image_path"] == "C:/tmp/dish.jpg"
    assert calls[0][0].router.type == "image-query"


@pytest.mark.asyncio
async def test_recipe_agent_returns_ranked_canonical_ids_for_handoff():
    recipe_graph = RecordingGraph(
        {
            "answer": "recommendations",
            "documents": [
                {
                    "content": "first evidence",
                    "metadata": {
                        "node_id": "201003834",
                        "recipe_name": "Kung Pao Chicken",
                        "score": 0.91,
                    },
                },
                {
                    "content": "duplicate evidence",
                    "metadata": {
                        "node_id": "201003834",
                        "recipe_name": "Kung Pao Chicken",
                        "score": 0.87,
                    },
                },
            ],
            "cyphers": [
                {
                    "records": [
                        {
                            "recipe_id": "201004552",
                            "recipe_name": "Tomato Egg Soup",
                        }
                    ]
                }
            ],
        }
    )
    executor = GustoBotDirectExecutor(recipe_graph=recipe_graph)

    result = await executor.recipe(
        AgentTask(
            task_id="recipe-1",
            domain=Domain.RECIPE,
            instruction="select two recipes",
            result_limit=2,
        ),
        {"session_id": "handoff-session"},
    )

    assert result["selected_recipe_ids"] == ["201003834", "201004552"]
    assert result["recipe_candidates"] == [
        {
            "recipe_id": "201003834",
            "recipe_name": "Kung Pao Chicken",
            "rank": 1,
            "source": "retrieval",
            "score": 0.91,
        },
        {
            "recipe_id": "201004552",
            "recipe_name": "Tomato Egg Soup",
            "rank": 2,
            "source": "cypher",
            "score": None,
        },
    ]


@pytest.mark.asyncio
async def test_analytics_agent_passes_only_upstream_selected_recipe_ids():
    analytics_graph = RecordingGraph(
        {
            "answer": "440 kcal",
            "sql_statement": (
                "SELECT SUM(total_calories) FROM recipes "
                "WHERE canonical_recipe_id IN ('201003834', '201004552')"
            ),
        }
    )
    executor = GustoBotDirectExecutor(analytics_graph=analytics_graph)
    task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="calculate total calories",
        depends_on=["recipe-1"],
    )

    await executor.analytics(
        task,
        {
            "session_id": "handoff-session",
            "dependency_outputs": {
                "recipe-1": {
                    "selected_recipe_ids": [
                        "201003834",
                        "201004552",
                        "201003834",
                    ]
                },
                "unrelated-task": {"selected_recipe_ids": ["999999999"]},
            },
        },
    )

    assert analytics_graph.calls[0][0]["recipe_ids"] == [
        "201003834",
        "201004552",
    ]


@pytest.mark.asyncio
async def test_analytics_agent_blocks_broad_query_when_recipe_selection_is_empty():
    analytics_graph = RecordingGraph({"answer": "should not execute"})
    executor = GustoBotDirectExecutor(analytics_graph=analytics_graph)
    task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="calculate total calories",
        depends_on=["recipe-1"],
    )

    with pytest.raises(ValueError, match="selected_recipe_ids"):
        await executor.analytics(
            task,
            {
                "dependency_outputs": {
                    "recipe-1": {
                        "selected_recipe_ids": [],
                        "recipe_candidates": [],
                    }
                }
            },
        )

    assert analytics_graph.calls == []
