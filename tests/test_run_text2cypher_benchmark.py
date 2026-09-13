from __future__ import annotations

from scripts.run_text2cypher_benchmark import (
    CONFIGURATION_SPECS,
    PROMPT_VERSION,
    await_case_result,
    case_examples,
    case_template_request,
)
import asyncio

import pytest
from gustobot.application.meal_planning.text2cypher.benchmark import (
    Text2CypherBenchmarkCase,
)


def case(case_id: str, split: str, question: str) -> Text2CypherBenchmarkCase:
    return Text2CypherBenchmarkCase(
        case_id=case_id,
        split=split,
        question=question,
        gold_recipe_ids=["r1"],
        expected_route="template",
        dataset_sha256="a" * 64,
        parameters={"ingredient_ids": ["egg"]},
        template_request={
            "operation": "search_recipes",
            "dataset_version": "v1",
            "required_ingredient_ids": ["egg"],
            "top_k": 20,
        },
    )


def test_runner_declares_the_four_frozen_ablation_configurations():
    assert list(CONFIGURATION_SPECS) == [
        "deepseek_direct",
        "deepseek_schema",
        "deepseek_schema_fewshot",
        "template_first_validated",
    ]
    assert CONFIGURATION_SPECS["deepseek_direct"].include_schema is False
    assert CONFIGURATION_SPECS["deepseek_schema"].include_schema is True
    assert CONFIGURATION_SPECS["deepseek_schema_fewshot"].include_examples is True
    assert CONFIGURATION_SPECS["template_first_validated"].template_first is True
    assert PROMPT_VERSION


def test_runner_only_supplies_template_request_to_template_first_configuration():
    item = case("text2cypher-001", "test", "test question")

    assert case_template_request(item, CONFIGURATION_SPECS["deepseek_direct"]) is None
    assert case_template_request(
        item, CONFIGURATION_SPECS["template_first_validated"]
    ) is not None


def test_few_shot_configuration_uses_development_only_and_leaves_current_case_out():
    current = case("text2cypher-001", "development", "current dev question")
    other_dev = case("text2cypher-002", "development", "other dev question")
    frozen = case("text2cypher-003", "test", "frozen test question")

    examples = case_examples(
        current,
        [current, other_dev, frozen],
        CONFIGURATION_SPECS["deepseek_schema_fewshot"],
    )
    rendered = "\n".join(text for pair in examples for text in pair)

    assert "other dev question" in rendered
    assert "current dev question" not in rendered
    assert "frozen test question" not in rendered
    assert case_examples(
        current,
        [current, other_dev, frozen],
        CONFIGURATION_SPECS["deepseek_schema"],
    ) == []


def test_case_timeout_prevents_one_llm_request_from_blocking_the_benchmark():
    async def too_slow():
        await asyncio.sleep(0.05)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(await_case_result(too_slow(), timeout_seconds=0.001))
