from gustobot.application.services import deepreason_service


class FakeModel:
    def __init__(self, name):
        self.name = name

    def with_structured_output(self, schema, *, include_raw=False):
        return self


def test_real_service_separates_structured_and_generation_clients(monkeypatch, tmp_path):
    calls = []

    def fake_get_llm(tags=None, *, structured_output=False):
        calls.append((tuple(tags or ()), structured_output))
        return FakeModel("structured" if structured_output else "generation")

    monkeypatch.setattr(deepreason_service, "get_llm", fake_get_llm)
    monkeypatch.setattr(deepreason_service, "build_gustobot_registry", lambda: object())
    monkeypatch.setattr(deepreason_service, "_orchestrator", None)

    orchestrator = deepreason_service.init_deepreason(demo_mode=False)

    assert calls == [
        (("deepreason", "structured"), True),
        (("deepreason", "generation"), False),
    ]
    assert orchestrator.planner.model.name == "structured"
    assert orchestrator.request_gateway._router._model.name == "structured"
