import numpy as np

from gustobot.application.meal_planning.routing_error_mining import (
    select_diverse_route_errors,
)


class FakeEncoder:
    def encode(self, texts):
        vectors = {
            "推荐两道菜并算热量": [1.0, 0.0],
            "推荐三道菜并统计卡路里": [0.99, 0.01],
            "把周二晚餐换掉": [0.0, 1.0],
            "修改明天的午餐": [0.01, 0.99],
        }
        return np.asarray([vectors[text] for text in texts], dtype=float)


def test_error_mining_keeps_confusion_coverage_and_drops_semantic_duplicates():
    errors = [
        _error("a", "推荐两道菜并算热量", "recipe_lookup", "recipe_lookup", "workflow", "direct"),
        _error("b", "推荐三道菜并统计卡路里", "recipe_lookup", "recipe_lookup", "workflow", "direct"),
        _error("c", "把周二晚餐换掉", "plan_edit", "unknown", "workflow", "clarify"),
        _error("d", "修改明天的午餐", "plan_edit", "unknown", "workflow", "clarify"),
    ]

    selected = select_diverse_route_errors(errors, encoder=FakeEncoder(), limit=2)

    assert len(selected) == 2
    assert {item["expected_intent"] for item in selected} == {
        "recipe_lookup",
        "plan_edit",
    }


def test_error_mining_returns_every_error_when_under_limit():
    errors = [
        _error("a", "推荐两道菜并算热量", "recipe_lookup", "recipe_lookup", "workflow", "direct"),
        _error("c", "把周二晚餐换掉", "plan_edit", "unknown", "workflow", "clarify"),
    ]

    selected = select_diverse_route_errors(errors, encoder=FakeEncoder(), limit=5)

    assert [item["case_id"] for item in selected] == ["a", "c"]


def _error(case_id, query, expected_intent, predicted_intent, expected_policy, predicted_policy):
    return {
        "case_id": case_id,
        "query": query,
        "expected_intent": expected_intent,
        "predicted_intent": predicted_intent,
        "expected_policy": expected_policy,
        "predicted_policy": predicted_policy,
    }
