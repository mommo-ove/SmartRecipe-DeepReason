import pytest

from gustobot.application.deepreason.direct_capabilities import GustoBotDirectExecutor
from gustobot.application.deepreason.domain_agents import build_gustobot_registry
from gustobot.application.deepreason.models import (
    AgentTask,
    Domain,
    ExecutionPlan,
    GateStatus,
)
from gustobot.application.deepreason.orchestrator import DeepReasonOrchestrator


class RecordingGraph:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def ainvoke(self, state, config=None):
        self.calls.append((state, config))
        return self.result


class StaticPlanner:
    def __init__(self, plan):
        self.plan = plan

    async def acoordinate(self, query, **kwargs):
        return self.plan

    async def areview(self, query, candidate_plan):
        return candidate_plan


@pytest.mark.asyncio
async def test_recipe_selection_drives_scoped_nutrition_query(tmp_path):
    recipe_graph = RecordingGraph(
        {
            "answer": "selected two recipes",
            "documents": [
                {
                    "content": "recipe one",
                    "metadata": {"node_id": "201003834", "recipe_name": "recipe one"},
                },
                {
                    "content": "recipe two",
                    "metadata": {"node_id": "201004552", "recipe_name": "recipe two"},
                },
            ],
        }
    )
    analytics_graph = RecordingGraph(
        {
            "answer": "total calories: 440",
            "sql_statement": (
                "SELECT SUM(total_calories) FROM recipes "
                "WHERE canonical_recipe_id IN ('201003834', '201004552')"
            ),
            "execution_results": [{"total_calories": 440}],
        }
    )
    recipe_task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="select two recipes",
        result_limit=2,
    )
    nutrition_task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="calculate their total calories",
        depends_on=["recipe-1"],
    )
    plan = ExecutionPlan(
        query="select two recipes and calculate their total calories",
        tasks=[recipe_task, nutrition_task],
    )
    executor = GustoBotDirectExecutor(
        recipe_graph=recipe_graph,
        analytics_graph=analytics_graph,
        node_handlers={},
    )
    orchestrator = DeepReasonOrchestrator(
        registry=build_gustobot_registry(executor),
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
    )

    result = await orchestrator.run(plan.query, session_id="handoff-integration")

    assert result.gate.status == GateStatus.ALLOW
    assert analytics_graph.calls[0][0]["recipe_ids"] == [
        "201003834",
        "201004552",
    ]
    assert [handoff.task_id for handoff in result.handoffs] == [
        "recipe-1",
        "nutrition-2",
    ]
