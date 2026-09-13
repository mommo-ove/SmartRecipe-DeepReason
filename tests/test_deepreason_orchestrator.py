import asyncio

import pytest

from gustobot.application.deepreason.domain_agents import DomainAgentRegistry, FunctionDomainAgent
from gustobot.application.deepreason.models import Domain, GateStatus, TaskStatus
from gustobot.application.deepreason.orchestrator import DeepReasonOrchestrator


def test_top_level_orchestrator_is_an_explicit_langgraph(tmp_path):
    registry = DomainAgentRegistry()
    orchestrator = DeepReasonOrchestrator(
        registry=registry,
        ledger_path=tmp_path / "ledger.jsonl",
    )

    assert {
        "coordinator",
        "reviewer",
        "domain_agent",
        "critic_gate",
        "prepare_retry",
        "synthesizer",
    }.issubset(orchestrator.graph.nodes)


@pytest.mark.asyncio
async def test_independent_domain_agents_execute_concurrently(tmp_path):
    recipe_started = asyncio.Event()
    analytics_started = asyncio.Event()

    async def recipe_handler(task, context):
        recipe_started.set()
        await asyncio.wait_for(analytics_started.wait(), timeout=0.5)
        return {"answer": "推荐宫保鸡丁", "documents": [{"content": "低辣鸡肉菜"}]}

    async def analytics_handler(task, context):
        analytics_started.set()
        await asyncio.wait_for(recipe_started.wait(), timeout=0.5)
        return {"answer": "平均 25 分钟", "sql_statement": "SELECT AVG(cook_time) FROM recipes"}

    registry = DomainAgentRegistry()
    registry.register(FunctionDomainAgent("recipe_agent", Domain.RECIPE, recipe_handler))
    registry.register(FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, analytics_handler))
    orchestrator = DeepReasonOrchestrator(registry=registry, ledger_path=tmp_path / "ledger.jsonl")

    result = await orchestrator.run(
        "推荐一道低辣鸡肉菜，并统计这类菜的平均烹饪时长", session_id="session-1"
    )

    assert result.plan.is_multi_agent is True
    assert {handoff.status for handoff in result.handoffs} == {TaskStatus.SUCCESS}
    assert result.gate.status == GateStatus.ALLOW
    assert "宫保鸡丁" in result.answer
    assert "25 分钟" in result.answer


@pytest.mark.asyncio
async def test_failed_task_is_retried_once_and_then_allowed(tmp_path):
    attempts = 0

    async def flaky_handler(task, context):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary database error")
        return {"answer": "共有 342 道菜", "sql_statement": "SELECT COUNT(*) FROM recipes"}

    registry = DomainAgentRegistry()
    registry.register(FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, flaky_handler))
    orchestrator = DeepReasonOrchestrator(
        registry=registry, ledger_path=tmp_path / "ledger.jsonl", max_retries=1
    )

    result = await orchestrator.run("统计菜谱总数", session_id="session-2")

    assert attempts == 2
    assert result.handoffs[0].status == TaskStatus.SUCCESS
    assert result.gate.status == GateStatus.ALLOW
    assert any(event["kind"] == "gate_retry" for event in result.events)


@pytest.mark.asyncio
async def test_unsafe_sql_evidence_is_denied(tmp_path):
    async def unsafe_handler(task, context):
        return {"answer": "已删除", "sql_statement": "DELETE FROM recipes"}

    registry = DomainAgentRegistry()
    registry.register(FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, unsafe_handler))
    orchestrator = DeepReasonOrchestrator(registry=registry, ledger_path=tmp_path / "ledger.jsonl")

    result = await orchestrator.run("统计菜谱数量", session_id="session-3")

    assert result.gate.status == GateStatus.DENY
    assert any(finding.code == "unsafe_sql" for finding in result.findings)
