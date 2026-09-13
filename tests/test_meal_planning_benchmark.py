from __future__ import annotations

from pathlib import Path

from gustobot.application.meal_planning.benchmark import (
    RetrievalBenchmarkCase,
    evaluate_retrieval,
    load_retrieval_benchmark,
    reciprocal_rank,
    recall_at_k,
    run_retrieval_benchmark,
)
from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.retrieval_baseline import FieldedBM25Retriever


BENCHMARK_DIR = Path(__file__).parents[1] / "benchmark" / "meal_planning"
EXTERNAL_BENCHMARK_DIR = BENCHMARK_DIR / "external"
CURATED_BENCHMARK_DIR = BENCHMARK_DIR / "curated"


def test_recall_at_k_counts_unique_gold_recipes_found_in_top_k():
    assert recall_at_k(["r1", "r3", "r2"], {"r1", "r2"}, k=2) == 0.5
    assert recall_at_k(["r1", "r3", "r2"], {"r1", "r2"}, k=3) == 1.0


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(["r3", "r2", "r1"], {"r1", "r2"}) == 0.5
    assert reciprocal_rank(["r3"], {"r1", "r2"}) == 0.0


def test_evaluator_reports_same_metrics_for_each_named_configuration():
    cases = [
        RetrievalBenchmarkCase(
            case_id="q1",
            query="high protein lunch",
            relevant_recipe_ids={"r1", "r2"},
        ),
        RetrievalBenchmarkCase(
            case_id="q2",
            query="quick breakfast",
            relevant_recipe_ids={"r3"},
        ),
    ]
    rankings = {
        "q1": ["r1", "r4", "r2"],
        "q2": ["r4", "r3"],
    }

    result = evaluate_retrieval("bm25", cases, rankings, ks=(1, 2, 3))

    assert result.configuration == "bm25"
    assert result.case_count == 2
    assert result.metrics["recall@1"] == 0.25
    assert result.metrics["recall@2"] == 0.75
    assert result.metrics["recall@3"] == 1.0
    assert result.metrics["mrr"] == 0.75
    assert 0 < result.metrics["ndcg@3"] <= 1


def test_evaluator_rejects_missing_rankings_instead_of_silently_skipping_cases():
    cases = [
        RetrievalBenchmarkCase(
            case_id="q1",
            query="breakfast",
            relevant_recipe_ids={"r1"},
        )
    ]

    try:
        evaluate_retrieval("bm25", cases, {})
    except ValueError as error:
        assert "q1" in str(error)
    else:
        raise AssertionError("missing rankings must fail the benchmark")


def test_versioned_retrieval_smoke_set_matches_manifest():
    benchmark = load_retrieval_benchmark(
        BENCHMARK_DIR / "retrieval_cases.v1.jsonl",
        BENCHMARK_DIR / "manifest.v1.json",
    )

    assert benchmark.manifest.version == "1.0.0-dev"
    assert benchmark.manifest.status == "smoke_fixture"
    assert benchmark.manifest.resume_metric_eligible is False
    assert benchmark.manifest.case_count == len(benchmark.cases)
    assert len(benchmark.cases) >= 12


def test_runner_executes_the_same_retriever_for_every_versioned_case():
    benchmark = load_retrieval_benchmark(
        BENCHMARK_DIR / "retrieval_cases.v1.jsonl",
        BENCHMARK_DIR / "manifest.v1.json",
    )
    data_dir = Path(__file__).parents[1] / "gustobot" / "data" / "meal_planning"
    corpus = load_seed_corpus(
        data_dir / "recipes.v1.json",
        data_dir / "manifest.v1.json",
    )

    result = run_retrieval_benchmark(
        "bm25",
        benchmark.cases,
        FieldedBM25Retriever(corpus.recipes),
        top_k=20,
    )

    assert result.configuration == "bm25"
    assert result.case_count == benchmark.manifest.case_count
    assert set(result.metrics) == {"recall@5", "recall@10", "recall@20", "mrr", "ndcg@20"}


def test_external_known_item_set_is_larger_but_not_a_resume_metric():
    benchmark = load_retrieval_benchmark(
        EXTERNAL_BENCHMARK_DIR / "retrieval_cases.v1.jsonl",
        EXTERNAL_BENCHMARK_DIR / "manifest.v1.json",
    )

    assert benchmark.manifest.case_count == 30
    assert benchmark.manifest.status == "synthetic_known_item"
    assert benchmark.manifest.human_reviewed_count == 0
    assert benchmark.manifest.resume_metric_eligible is False


def test_curated_user_style_set_waits_for_explicit_human_review():
    benchmark = load_retrieval_benchmark(
        CURATED_BENCHMARK_DIR / "retrieval_cases.v1.jsonl",
        CURATED_BENCHMARK_DIR / "manifest.v1.json",
    )

    assert benchmark.manifest.case_count == 40
    assert benchmark.manifest.status == "awaiting_user_review"
    assert benchmark.manifest.human_reviewed_count == 0
    assert benchmark.manifest.resume_metric_eligible is False
    assert all(case.authoring == "curated_draft" for case in benchmark.cases)
