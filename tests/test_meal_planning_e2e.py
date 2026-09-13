from pathlib import Path

import pytest

from gustobot.application.deepreason.domain_agents import build_gustobot_registry
from gustobot.application.deepreason.models import Domain
from gustobot.application.deepreason.orchestrator import DeepReasonOrchestrator
from gustobot.application.meal_planning.gateway import MealRequestGateway
from gustobot.application.meal_planning.routing import MealIntentRouter


@pytest.mark.asyncio
async def test_explicit_chinese_request_runs_real_local_meal_planning_pipeline(tmp_path):
    orchestrator = DeepReasonOrchestrator(
        registry=build_gustobot_registry(),
        ledger_path=Path(tmp_path) / "ledger.jsonl",
        request_gateway=MealRequestGateway(MealIntentRouter(None), None),
    )

    result = await orchestrator.run(
        "规划3天餐单，每天1600到1800千卡、蛋白质100克以上、"
        "不含花生、每餐30分钟内",
        session_id="real-local-e2e",
    )

    assert result.plan.tasks[0].domain is Domain.MEAL_PLANNING
    assert result.handoffs[0].agent == "meal_planning_agent"
    assert result.handoffs[0].output["status"] == "verified"
    assert len(result.handoffs[0].output["meal_plan"]["days"]) == 3
    assert result.gate.status.value == "allow"
