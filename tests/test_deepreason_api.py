from fastapi import FastAPI
from fastapi.testclient import TestClient

from gustobot.application.deepreason.domain_agents import DomainAgentRegistry, FunctionDomainAgent
from gustobot.application.deepreason.models import Domain
from gustobot.application.deepreason.orchestrator import DeepReasonOrchestrator
from gustobot.interfaces.http import deepreason as deepreason_http


def test_deepreason_chat_exposes_plan_handoffs_and_gate(tmp_path, monkeypatch):
    async def analytics_handler(task, context):
        return {"answer": "共有 342 道菜", "sql_statement": "SELECT COUNT(*) FROM recipes"}

    registry = DomainAgentRegistry()
    registry.register(FunctionDomainAgent("analytics_agent", Domain.ANALYTICS, analytics_handler))
    orchestrator = DeepReasonOrchestrator(registry=registry, ledger_path=tmp_path / "ledger.jsonl")
    monkeypatch.setattr(deepreason_http, "get_orchestrator", lambda: orchestrator)
    app = FastAPI()
    app.include_router(deepreason_http.router)
    client = TestClient(app)

    response = client.post(
        "/deepreason/chat",
        json={"message": "统计菜谱总数", "session_id": "api-session"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["tasks"][0]["domain"] == "analytics"
    assert payload["handoffs"][0]["agent"] == "analytics_agent"
    assert payload["gate"]["status"] == "allow"


def test_deepreason_status_reports_registered_domains(tmp_path, monkeypatch):
    registry = DomainAgentRegistry()

    async def handler(task, context):
        return {"answer": "ok"}

    registry.register(FunctionDomainAgent("general_agent", Domain.GENERAL, handler))
    orchestrator = DeepReasonOrchestrator(registry=registry, ledger_path=tmp_path / "ledger.jsonl")
    monkeypatch.setattr(deepreason_http, "get_orchestrator", lambda: orchestrator)
    app = FastAPI()
    app.include_router(deepreason_http.router)

    payload = TestClient(app).get("/deepreason/status").json()

    assert payload == {"initialized": True, "domains": ["general"], "last_run": None}

