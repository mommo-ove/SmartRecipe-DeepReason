from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.benchmark import (
    evaluate_retrieval,
    load_retrieval_benchmark,
)
from gustobot.application.meal_planning.external_corpus import load_external_corpus
from gustobot.application.meal_planning.graph_benchmark import (
    ingredient_exclusion_violation_at_k,
)
from gustobot.application.meal_planning.graph_retrieval import (
    GraphAugmentedRetriever,
    Neo4jMealGraphStore,
)
from gustobot.application.meal_planning.hybrid_retrieval import (
    BgeM3Encoder,
    DenseExternalRetriever,
    RRFExternalRetriever,
)
from gustobot.application.meal_planning.retrieval_baseline import (
    ExternalBM25Retriever,
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--neo4j-url", default="bolt://localhost:17687")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=("bm25", "dense_bge_m3", "rrf_bm25_dense", "rrf_with_graph_gate"),
        default=("bm25", "dense_bge_m3", "rrf_bm25_dense", "rrf_with_graph_gate"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "benchmark"
        / "meal_planning"
        / "graph_relations"
        / "runs"
        / "test.v1.json",
    )
    args = parser.parse_args()

    corpus_dir = ROOT / "gustobot" / "data" / "meal_planning" / "external"
    benchmark_dir = ROOT / "benchmark" / "meal_planning" / "graph_relations"
    corpus = load_external_corpus(
        corpus_dir / "retrieval_corpus.v1.jsonl",
        corpus_dir / "manifest.v1.json",
    )
    benchmark = load_retrieval_benchmark(
        benchmark_dir / "retrieval_cases.v1.jsonl",
        benchmark_dir / "manifest.v1.json",
    )
    cases = [case for case in benchmark.cases if case.split == args.split]

    with ExitStack() as stack:
        bm25 = ExternalBM25Retriever(corpus.documents)
        retrievers: dict[str, object] = {}
        requested = set(args.configurations)
        if "bm25" in requested:
            retrievers["bm25"] = bm25
        dense = None
        rrf = None
        if requested - {"bm25"}:
            encoder = BgeM3Encoder(model_name=args.model, device=args.device)
            dense = DenseExternalRetriever(corpus.documents, encoder)
        if "dense_bge_m3" in requested:
            retrievers["dense_bge_m3"] = dense
        if requested & {"rrf_bm25_dense", "rrf_with_graph_gate"}:
            rrf = RRFExternalRetriever({"bm25": bm25, "dense": dense})
        if "rrf_bm25_dense" in requested:
            retrievers["rrf_bm25_dense"] = rrf
        if "rrf_with_graph_gate" in requested:
            driver = stack.enter_context(
                GraphDatabase.driver(args.neo4j_url, auth=None)
            )
            driver.verify_connectivity()
            graph = Neo4jMealGraphStore(driver, database=args.database)
            retrievers["rrf_with_graph_gate"] = GraphAugmentedRetriever(
                rrf,
                graph,
                dataset_version=corpus.manifest.version,
            )
        results: dict[str, object] = {}
        raw_rankings: dict[str, dict[str, list[str]]] = {}
        for name, retriever in retrievers.items():
            rankings: dict[str, list[str]] = {}
            latencies_ms: list[float] = []
            for case in cases:
                started = perf_counter()
                rankings[case.case_id] = retriever.search(
                    case.query,
                    case.metadata_filters,
                    top_k=20,
                )
                latencies_ms.append((perf_counter() - started) * 1000)
            result = evaluate_retrieval(name, cases, rankings)
            results[name] = {
                **result.model_dump(mode="json"),
                "ingredient_exclusion_violation@20": (
                    ingredient_exclusion_violation_at_k(
                        cases, rankings, corpus.documents, k=20
                    )
                ),
                "latency_ms": {
                    "mean": float(np.mean(latencies_ms)),
                    "p50": percentile(latencies_ms, 50),
                    "p95": percentile(latencies_ms, 95),
                },
            }
            raw_rankings[name] = rankings

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "structured include/exclude relation retrieval",
        "split": args.split,
        "corpus": corpus.manifest.model_dump(mode="json"),
        "benchmark": benchmark.manifest.model_dump(mode="json"),
        "model": args.model,
        "device": args.device,
        "results": results,
        "rankings": raw_rankings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
