import pytest

from gustobot.application.deepreason.domain_agents import (
    DomainAgentRegistry,
    FunctionDomainAgent,
)
from gustobot.application.deepreason.models import (
    AgentTask,
    Domain,
    ExecutionPlan,
    GateStatus,
    TaskStatus,
)
from gustobot.application.deepreason.orchestrator import DeepReasonOrchestrator


class StaticPlanner:
    def __init__(self, plan):
        self.plan = plan

    async def acoordinate(self, query, **kwargs):
        return self.plan

    async def areview(self, query, candidate_plan):
        return candidate_plan


def test_dispatches_only_tasks_whose_dependencies_are_satisfied(tmp_path):
    orchestrator = DeepReasonOrchestrator(
        registry=DomainAgentRegistry(),
        ledger_path=tmp_path / "ledger.jsonl",
    )
    recipe_task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="选择两道符合约束的菜谱",
    )
    nutrition_task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="统计已选菜谱的总热量",
        depends_on=["recipe-1"],
    )

    routes = orchestrator._dispatch_pending_tasks(
        {
            "pending_tasks": [recipe_task, nutrition_task],
            "handoffs": [],
        }
    )

    dispatched_ids = [route.arg["active_task"].task_id for route in routes]
    assert dispatched_ids == ["recipe-1"]


@pytest.mark.asyncio
async def test_dependent_task_runs_after_upstream_and_receives_its_output(tmp_path):
    recipe_task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="选择两道符合约束的菜谱",
    )
    nutrition_task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="统计已选菜谱的总热量",
        depends_on=["recipe-1"],
    )
    plan = ExecutionPlan(query="推荐两道菜并计算热量", tasks=[recipe_task, nutrition_task])

    execution_order = []

    async def recipe_handler(task, context):
        execution_order.append(task.task_id)
        return {
            "answer": "已选择两道菜",
            "selected_recipe_ids": ["r101", "r205"],
            "documents": [{"content": "r101和r205符合约束"}],
        }

    async def nutrition_handler(task, context):
        execution_order.append(task.task_id)
        assert context["dependency_outputs"]["recipe-1"]["selected_recipe_ids"] == [
            "r101",
            "r205",
        ]
        return {
            "answer": "总热量440千卡",
            "sql_statement": "SELECT SUM(calories) FROM recipes",
        }

    registry = DomainAgentRegistry()
    registry.register(FunctionDomainAgent("recipe_agent", Domain.RECIPE, recipe_handler))
    registry.register(FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, nutrition_handler))
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
    )

    result = await orchestrator.run("推荐两道菜并计算热量", session_id="dependency-session")

    assert execution_order == ["recipe-1", "nutrition-2"]
    assert [handoff.task_id for handoff in result.handoffs] == ["recipe-1", "nutrition-2"]


@pytest.mark.asyncio
async def test_missing_dependency_is_denied_before_any_agent_runs(tmp_path):
    task = AgentTask(
        task_id="nutrition-1",
        domain=Domain.ANALYTICS,
        instruction="calculate nutrition",
        depends_on=["recipe-does-not-exist"],
    )
    plan = ExecutionPlan(query="calculate nutrition", tasks=[task])
    executions = []

    async def analytics_handler(agent_task, context):
        executions.append(agent_task.task_id)
        return {"answer": "should not run"}

    registry = DomainAgentRegistry()
    registry.register(FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, analytics_handler))
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
    )

    result = await orchestrator.run("calculate nutrition", session_id="missing-dependency")

    assert executions == []
    assert result.gate.status == GateStatus.DENY
    assert any(finding.code == "dependency_missing" for finding in result.findings)


@pytest.mark.asyncio
async def test_dependency_cycle_is_denied_instead_of_looping(tmp_path):
    recipe_task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="select recipes",
        depends_on=["nutrition-2"],
    )
    nutrition_task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="calculate nutrition",
        depends_on=["recipe-1"],
    )
    plan = ExecutionPlan(query="cyclic plan", tasks=[recipe_task, nutrition_task])
    orchestrator = DeepReasonOrchestrator(
        registry=DomainAgentRegistry(),
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
    )

    result = await orchestrator.run("cyclic plan", session_id="dependency-cycle")

    assert result.handoffs == []
    assert result.gate.status == GateStatus.DENY
    assert any(finding.code == "dependency_cycle" for finding in result.findings)


@pytest.mark.asyncio
async def test_duplicate_task_ids_are_denied_before_any_agent_runs(tmp_path):
    first = AgentTask(
        task_id="duplicate-1",
        domain=Domain.RECIPE,
        instruction="select recipes",
    )
    second = AgentTask(
        task_id="duplicate-1",
        domain=Domain.ANALYTICS,
        instruction="calculate nutrition",
    )
    plan = ExecutionPlan(query="invalid duplicate plan", tasks=[first, second])
    orchestrator = DeepReasonOrchestrator(
        registry=DomainAgentRegistry(),
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
    )

    result = await orchestrator.run(
        "invalid duplicate plan",
        session_id="duplicate-task-id",
    )

    assert result.handoffs == []
    assert result.gate.status == GateStatus.DENY
    assert any(finding.code == "duplicate_task_id" for finding in result.findings)


@pytest.mark.asyncio
async def test_exhausted_upstream_failure_skips_dependent_task(tmp_path):
    recipe_task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="select recipes",
    )
    nutrition_task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="calculate nutrition",
        depends_on=["recipe-1"],
    )
    plan = ExecutionPlan(query="recommend and calculate", tasks=[recipe_task, nutrition_task])
    nutrition_executed = False

    async def failing_recipe_handler(task, context):
        raise RuntimeError("recipe store unavailable")

    async def nutrition_handler(task, context):
        nonlocal nutrition_executed
        nutrition_executed = True
        return {"answer": "should not run"}

    registry = DomainAgentRegistry()
    registry.register(
        FunctionDomainAgent("recipe_agent", Domain.RECIPE, failing_recipe_handler)
    )
    registry.register(
        FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, nutrition_handler)
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
        max_retries=0,
    )

    result = await orchestrator.run(
        "recommend and calculate",
        session_id="dependency-failure",
    )

    handoff_by_task = {handoff.task_id: handoff for handoff in result.handoffs}
    assert nutrition_executed is False
    assert handoff_by_task["recipe-1"].status == TaskStatus.FAILED
    assert handoff_by_task["nutrition-2"].status == TaskStatus.SKIPPED
    assert "recipe-1" in handoff_by_task["nutrition-2"].error


@pytest.mark.asyncio
async def test_missing_evidence_task_retries_and_replaces_handoff(tmp_path):
    task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="select recipes with evidence",
    )
    plan = ExecutionPlan(query="select recipes", tasks=[task])
    attempts = 0

    async def evidence_on_retry_handler(agent_task, context):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {}
        return {"answer": "supported result"}

    registry = DomainAgentRegistry()
    registry.register(
        FunctionDomainAgent(
            "recipe_agent",
            Domain.RECIPE,
            evidence_on_retry_handler,
        )
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
        max_retries=1,
    )

    result = await orchestrator.run(
        "select recipes",
        session_id="missing-evidence-retry",
    )

    assert attempts == 2
    assert result.gate.status == GateStatus.ALLOW
    assert len(result.handoffs) == 1
    assert result.handoffs[0].evidence


@pytest.mark.asyncio
async def test_dependent_waits_for_evidence_backed_retry_output(tmp_path):
    recipe_task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="select recipes with evidence",
    )
    nutrition_task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="calculate nutrition",
        depends_on=["recipe-1"],
        evidence_required=False,
    )
    plan = ExecutionPlan(query="select and calculate", tasks=[recipe_task, nutrition_task])
    recipe_attempts = 0
    execution_order = []

    async def evidence_on_retry_handler(agent_task, context):
        nonlocal recipe_attempts
        recipe_attempts += 1
        execution_order.append(agent_task.task_id)
        if recipe_attempts == 1:
            return {"selected_recipe_ids": ["stale"]}
        return {
            "answer": "supported",
            "selected_recipe_ids": ["verified"],
        }

    async def nutrition_handler(agent_task, context):
        execution_order.append(agent_task.task_id)
        assert context["dependency_outputs"]["recipe-1"]["selected_recipe_ids"] == [
            "verified"
        ]
        return {"answer": "calculated"}

    registry = DomainAgentRegistry()
    registry.register(
        FunctionDomainAgent("recipe_agent", Domain.RECIPE, evidence_on_retry_handler)
    )
    registry.register(
        FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, nutrition_handler)
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
        max_retries=1,
    )

    result = await orchestrator.run(
        "select and calculate",
        session_id="evidence-backed-dependency",
    )

    assert execution_order == ["recipe-1", "recipe-1", "nutrition-2"]
    assert result.gate.status == GateStatus.ALLOW


@pytest.mark.asyncio
async def test_successful_retry_unblocks_and_runs_dependent_task(tmp_path):
    recipe_task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="select recipes",
        evidence_required=False,
    )
    nutrition_task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="calculate nutrition",
        depends_on=["recipe-1"],
        evidence_required=False,
    )
    plan = ExecutionPlan(query="recommend and calculate", tasks=[recipe_task, nutrition_task])
    recipe_attempts = 0
    execution_order = []

    async def flaky_recipe_handler(task, context):
        nonlocal recipe_attempts
        recipe_attempts += 1
        execution_order.append(task.task_id)
        if recipe_attempts == 1:
            raise RuntimeError("temporary recipe store outage")
        return {"answer": "selected", "selected_recipe_ids": ["r101"]}

    async def nutrition_handler(task, context):
        execution_order.append(task.task_id)
        assert context["dependency_outputs"]["recipe-1"]["selected_recipe_ids"] == ["r101"]
        return {"answer": "calculated"}

    registry = DomainAgentRegistry()
    registry.register(
        FunctionDomainAgent("recipe_agent", Domain.RECIPE, flaky_recipe_handler)
    )
    registry.register(
        FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, nutrition_handler)
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
        max_retries=1,
    )

    result = await orchestrator.run(
        "recommend and calculate",
        session_id="dependency-retry-success",
    )

    assert execution_order == ["recipe-1", "recipe-1", "nutrition-2"]
    assert result.gate.status == GateStatus.ALLOW
    assert {
        handoff.task_id: handoff.status for handoff in result.handoffs
    } == {
        "recipe-1": TaskStatus.SUCCESS,
        "nutrition-2": TaskStatus.SUCCESS,
    }


@pytest.mark.asyncio
async def test_failed_retry_stops_after_budget_and_skips_dependent_task(tmp_path):
    recipe_task = AgentTask(
        task_id="recipe-1",
        domain=Domain.RECIPE,
        instruction="select recipes",
        evidence_required=False,
    )
    nutrition_task = AgentTask(
        task_id="nutrition-2",
        domain=Domain.ANALYTICS,
        instruction="calculate nutrition",
        depends_on=["recipe-1"],
        evidence_required=False,
    )
    plan = ExecutionPlan(query="recommend and calculate", tasks=[recipe_task, nutrition_task])
    recipe_attempts = 0

    async def failing_recipe_handler(task, context):
        nonlocal recipe_attempts
        recipe_attempts += 1
        raise RuntimeError("recipe store unavailable")

    registry = DomainAgentRegistry()
    registry.register(
        FunctionDomainAgent("recipe_agent", Domain.RECIPE, failing_recipe_handler)
    )
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        planner=StaticPlanner(plan),
        ledger_path=tmp_path / "ledger.jsonl",
        max_retries=1,
    )

    result = await orchestrator.run(
        "recommend and calculate",
        session_id="dependency-retry-exhausted",
    )

    handoff_by_task = {handoff.task_id: handoff for handoff in result.handoffs}
    assert recipe_attempts == 2
    assert result.gate.status == GateStatus.DENY
    assert handoff_by_task["recipe-1"].status == TaskStatus.FAILED
    assert handoff_by_task["nutrition-2"].status == TaskStatus.SKIPPED
