from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.benchmark import (
    load_retrieval_benchmark,
    run_retrieval_benchmark,
)
from gustobot.application.meal_planning.fixtures import load_seed_corpus
from gustobot.application.meal_planning.retrieval_baseline import FieldedBM25Retriever


def main() -> None:
    benchmark_dir = ROOT / "benchmark" / "meal_planning"
    data_dir = ROOT / "gustobot" / "data" / "meal_planning"
    benchmark = load_retrieval_benchmark(
        benchmark_dir / "retrieval_cases.v1.jsonl",
        benchmark_dir / "manifest.v1.json",
    )
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
    payload = {
        "benchmark_version": benchmark.manifest.version,
        "benchmark_status": benchmark.manifest.status,
        "resume_metric_eligible": benchmark.manifest.resume_metric_eligible,
        **result.model_dump(mode="json"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
