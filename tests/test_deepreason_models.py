from gustobot.application.deepreason.models import (
    AgentTask,
    Domain,
    EvidenceItem,
    GateDecision,
    GateStatus,
    Handoff,
    TaskStatus,
)


def test_evidence_id_is_stable_for_same_source_and_content():
    first = EvidenceItem.create(
        task_id="recipe-1", source_type="neo4j", source="recipe_graph", content="宫保鸡丁包含鸡肉"
    )
    second = EvidenceItem.create(
        task_id="recipe-1", source_type="neo4j", source="recipe_graph", content="宫保鸡丁包含鸡肉"
    )

    assert first.evidence_id == second.evidence_id


def test_handoff_and_gate_are_serializable():
    handoff = Handoff(
        task_id="sql-1",
        agent="analytics_agent",
        domain=Domain.ANALYTICS,
        status=TaskStatus.SUCCESS,
        summary="平均烹饪时长为 25 分钟",
    )
    gate = GateDecision(status=GateStatus.ALLOW, reasons=["all required tasks completed"])

    assert handoff.model_dump(mode="json")["domain"] == "analytics"
    assert gate.model_dump(mode="json")["status"] == "allow"


def test_agent_task_rejects_self_dependency():
    try:
        AgentTask(
            task_id="recipe-1",
            domain=Domain.RECIPE,
            instruction="查询菜谱",
            depends_on=["recipe-1"],
        )
    except ValueError as exc:
        assert "depend on itself" in str(exc)
    else:
        raise AssertionError("self dependency must be rejected")

