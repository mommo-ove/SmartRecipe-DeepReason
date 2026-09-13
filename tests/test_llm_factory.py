from gustobot.application.agents.utils.llm_factory import get_llm


def test_structured_llm_disables_deepseek_thinking_mode(monkeypatch):
    monkeypatch.setattr("gustobot.application.agents.utils.llm_factory.settings.LLM_API_KEY", "secret")
    monkeypatch.setattr("gustobot.application.agents.utils.llm_factory.settings.LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr("gustobot.application.agents.utils.llm_factory.settings.LLM_BASE_URL", "https://api.deepseek.com")

    model = get_llm(structured_output=True)

    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_normal_llm_does_not_disable_thinking_mode(monkeypatch):
    monkeypatch.setattr("gustobot.application.agents.utils.llm_factory.settings.LLM_API_KEY", "secret")
    monkeypatch.setattr("gustobot.application.agents.utils.llm_factory.settings.LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr("gustobot.application.agents.utils.llm_factory.settings.LLM_BASE_URL", "https://api.deepseek.com")

    model = get_llm()

    assert model.extra_body is None
