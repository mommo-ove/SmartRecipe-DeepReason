from __future__ import annotations

from gustobot.application.meal_planning.graph_benchmark import (
    ingredient_exclusion_violation_at_k,
    mine_relation_benchmark_cases,
)
from tests.test_meal_planning_graph_projection import recipe


def test_mined_relation_cases_have_exhaustive_structured_gold_and_frozen_split():
    documents = [
        recipe("safe-1", ["egg", "tomato", "basil"]),
        recipe("safe-2", ["egg", "tomato", "spinach"]),
        recipe("unsafe", ["egg", "tomato", "peanut"]),
    ]

    cases = mine_relation_benchmark_cases(
        documents,
        case_count=1,
        development_count=0,
        min_pair_frequency=3,
        max_pair_frequency=3,
        stop_ingredients=set(),
    )

    assert len(cases) == 1
    assert cases[0].split == "test"
    assert cases[0].authoring == "structured_ground_truth"
    assert cases[0].relevant_recipe_ids
    excluded = cases[0].metadata_filters["excluded_graph_ingredient_ids"]
    assert len(excluded) == 1
    assert excluded[0] in cases[0].query


def test_exclusion_violation_metric_counts_forbidden_ranked_recipes():
    documents = [
        recipe("safe", ["egg", "tomato"]),
        recipe("unsafe", ["egg", "tomato", "peanut"]),
    ]
    cases = mine_relation_benchmark_cases(
        documents,
        case_count=1,
        development_count=0,
        min_pair_frequency=2,
        max_pair_frequency=2,
        stop_ingredients=set(),
    )

    score = ingredient_exclusion_violation_at_k(
        cases,
        {cases[0].case_id: ["unsafe", "safe"]},
        documents,
        k=2,
    )

    assert score == 0.5
