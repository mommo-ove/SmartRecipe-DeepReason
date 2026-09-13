from __future__ import annotations

import pytest

from gustobot.application.deepreason.models import Domain, GateStatus
from scripts.run_deepreason_demo import execute_demo, render_demo_trace


@pytest.mark.asyncio
async def test_demo_runs_existing_orchestrator_agents_ledger_and_benchmark(tmp_path):
    ledger_path = tmp_path / "demo-ledger.jsonl"

    demo = await execute_demo(ledger_path=ledger_path)

    assert [task.domain for task in demo.result.plan.tasks] == [
        Domain.RECIPE,
        Domain.ANALYTICS,
    ]
    assert demo.result.plan.review_status == "approved"
    assert {handoff.agent for handoff in demo.result.handoffs} == {
        "demo_recipe_agent",
        "demo_analytics_agent",
    }
    assert demo.agents_overlapped is True
    assert demo.result.gate.status == GateStatus.ALLOW
    assert {item.evidence_id for item in demo.ledger_items} == {
        item.evidence_id for item in demo.result.evidence
    }
    assert {item.source_type for item in demo.result.evidence} >= {
        "retrieval",
        "sql",
        "fact_atom",
    }
    assert demo.benchmark["exact_matches"] == demo.benchmark["total"] == 30


@pytest.mark.asyncio
async def test_demo_trace_explains_every_interview_stage(tmp_path):
    demo = await execute_demo(ledger_path=tmp_path / "demo-ledger.jsonl")

    trace = render_demo_trace(demo)

    for stage in (
        "[1/7] Coordinator",
        "[2/7] Planner",
        "[3/7] Reviewer",
        "[4/7] 并发领域 Agents",
        "[5/7] Evidence Ledger",
        "[6/7] Critic / 安全 Gate",
        "[7/7] Synthesizer",
    ):
        assert stage in trace
    assert "recipe-1" in trace
    assert "analytics-2" in trace
    assert "并发重叠: 是" in trace
    assert "demo_recipe_kg" in trace
    assert "mysql_readonly" in trace
    assert "Gate: ALLOW" in trace
    assert "30/30" in trace
