from __future__ import annotations

import argparse
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
from gustobot.application.meal_planning.external_corpus import load_external_corpus
from gustobot.application.meal_planning.hybrid_retrieval import (
    build_external_retriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration",
        choices=("bm25", "dense", "rrf", "rrf_rerank"),
        default="bm25",
    )
    parser.add_argument(
        "--benchmark",
        choices=("curated", "external"),
        default="curated",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = ROOT / "gustobot" / "data" / "meal_planning" / "external"
    benchmark_dir = ROOT / "benchmark" / "meal_planning" / args.benchmark
    corpus = load_external_corpus(
        corpus_dir / "retrieval_corpus.v1.jsonl",
        corpus_dir / "manifest.v1.json",
    )
    benchmark = load_retrieval_benchmark(
        benchmark_dir / "retrieval_cases.v1.jsonl",
        benchmark_dir / "manifest.v1.json",
    )
    retriever = build_external_retriever(
        args.configuration,
        corpus.documents,
        device=args.device,
    )
    result = run_retrieval_benchmark(
        args.configuration,
        benchmark.cases,
        retriever,
    )
    print(
        json.dumps(
            {
                "corpus_count": corpus.manifest.record_count,
                "benchmark_status": benchmark.manifest.status,
                "resume_metric_eligible": benchmark.manifest.resume_metric_eligible,
                **result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
