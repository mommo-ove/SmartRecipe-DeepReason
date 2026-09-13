from __future__ import annotations

from decimal import Decimal

from gustobot.application.meal_planning.extraction import MealConstraintExtractor
from gustobot.application.meal_planning.models import MealPlanDraft
from gustobot.application.meal_planning.planning_entry import (
    MealPlanningEntry,
    MealPlanningEntryStatus,
)


class FakeStructuredChain:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeModel:
    def __init__(self, responses):
        self.chain = FakeStructuredChain(responses)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.chain


class UsageMessage:
    usage_metadata = {"input_tokens": 80, "output_tokens": 20}


class UsageChain:
    def invoke(self, messages):
        return {
            "parsed": complete_draft(),
            "raw": UsageMessage(),
            "parsing_error": None,
        }


class UsageModel:
    def with_structured_output(self, schema, *, include_raw=False):
        assert schema is MealPlanDraft
        assert include_raw is True
        return UsageChain()


def complete_draft(**updates):
    values = {
        "days": 7,
        "daily_calories_min": 1600,
        "daily_calories_max": 1800,
        "daily_protein_min_g": Decimal("100"),
        "excluded_allergens": {"peanut"},
        "max_meal_minutes": 30,
        "weekly_budget_cents": 30000,
        "preferred_tags": {"light", "high_protein"},
        "requested_ingredients": {"chicken_breast"},
    }
    values.update(updates)
    return MealPlanDraft(**values)


def build_entry(*responses):
    model = FakeModel(responses)
    return MealPlanningEntry(MealConstraintExtractor(model)), model


def test_complete_extraction_becomes_validated_constraints_and_retrieval_query():
    entry, model = build_entry(complete_draft())

    result = entry.handle("做七天清淡高蛋白鸡胸肉餐，不含花生")

    assert result.status is MealPlanningEntryStatus.READY
    assert result.constraints is not None
    assert result.constraints.days == 7
    assert result.constraints.excluded_allergens == {"peanut"}
    assert result.retrieval_query == "chicken_breast high_protein light"
    assert model.schema is MealPlanDraft


def test_constraint_extractor_records_token_usage_for_gateway_evaluation():
    extractor = MealConstraintExtractor(UsageModel())

    result = extractor.extract("complete plan")

    assert result.days == 7
    assert extractor.last_fallback_used is False
    assert extractor.last_input_tokens == 80
    assert extractor.last_output_tokens == 20


def test_missing_critical_fields_routes_to_deterministic_clarification():
    entry, _ = build_entry(
        MealPlanDraft(days=7, preferred_tags={"low_fat"})
    )

    result = entry.handle("帮我做七天减脂餐")

    assert result.status is MealPlanningEntryStatus.CLARIFY
    assert result.constraints is None
    assert result.missing_fields == [
        "daily_calories_range",
        "daily_protein_min_g",
        "allergen_status",
    ]
    assert result.questions == [
        "你的每日目标热量范围是多少？",
        "你希望每天至少摄入多少克蛋白质？",
        "有没有需要排除的过敏食材？如果没有，请明确回复没有。",
    ]


def test_explicit_no_allergens_is_different_from_unknown_allergen_status():
    entry, _ = build_entry(complete_draft(excluded_allergens=set()))

    result = entry.handle("我没有食物过敏")

    assert result.status is MealPlanningEntryStatus.READY
    assert result.constraints.excluded_allergens == set()


def test_second_turn_merges_new_fields_into_checkpointed_draft():
    first_draft = complete_draft(excluded_allergens=None)
    allergen_patch = MealPlanDraft(excluded_allergens=set())
    entry, _ = build_entry(first_draft, allergen_patch)

    first = entry.handle("做七天餐单")
    second = entry.handle("我没有食物过敏", previous_draft=first.draft)

    assert first.status is MealPlanningEntryStatus.CLARIFY
    assert second.status is MealPlanningEntryStatus.READY
    assert second.constraints.days == 7
    assert second.constraints.excluded_allergens == set()


def test_model_or_schema_failure_routes_to_retry_without_constraints():
    entry, _ = build_entry(RuntimeError("provider timeout: secret-token"))

    result = entry.handle("做七天餐单")

    assert result.status is MealPlanningEntryStatus.RETRY
    assert result.constraints is None
    assert result.error_code == "constraint_extraction_failed"
    assert "secret-token" not in result.model_dump_json()


def test_type_valid_but_implausible_calories_route_to_clarification():
    entry, _ = build_entry(
        complete_draft(
            daily_calories_min=6000,
            daily_calories_max=6000,
        )
    )

    result = entry.handle("每天一千六左右")

    assert result.status is MealPlanningEntryStatus.CLARIFY
    assert result.semantic_issue_codes == ["daily_calories_out_of_supported_range"]
    assert result.questions == [
        "当前识别出的每日热量目标为6000～6000千卡，请确认这个数值是否正确。"
    ]
