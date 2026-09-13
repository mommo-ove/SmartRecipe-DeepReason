from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Protocol

from pydantic import BaseModel, Field


class RetrievalBenchmarkCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_recipe_ids: set[str] = Field(min_length=1)
    metadata_filters: dict[str, object] = Field(default_factory=dict)
    split: str = "development"
    authoring: str = "human_draft"


class RetrievalBenchmarkResult(BaseModel):
    configuration: str
    case_count: int = Field(ge=1)
    metrics: dict[str, float]


class RetrievalBenchmarkManifest(BaseModel):
    version: str
    status: str
    case_count: int = Field(ge=1)
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_reviewed_count: int = Field(ge=0)
    resume_metric_eligible: bool = False
    notes: str = ""


class RetrievalBenchmark(BaseModel):
    manifest: RetrievalBenchmarkManifest
    cases: list[RetrievalBenchmarkCase]


class SearchAdapter(Protocol):
    def search(
        self,
        query: str,
        metadata_filters: dict[str, object] | None = None,
        *,
        top_k: int = 20,
    ) -> list[str]: ...


def load_retrieval_benchmark(
    cases_path: Path,
    manifest_path: Path,
) -> RetrievalBenchmark:
    raw_cases = cases_path.read_bytes()
    manifest = RetrievalBenchmarkManifest.model_validate_json(
        manifest_path.read_text("utf-8")
    )
    actual_hash = hashlib.sha256(raw_cases).hexdigest()
    if actual_hash != manifest.cases_sha256:
        raise ValueError("retrieval benchmark hash does not match manifest")

    cases = [
        RetrievalBenchmarkCase.model_validate(json.loads(line))
        for line in raw_cases.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(cases) != manifest.case_count:
        raise ValueError("retrieval benchmark count does not match manifest")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("retrieval benchmark case ids must be unique")
    return RetrievalBenchmark(manifest=manifest, cases=cases)


def recall_at_k(ranking: list[str], gold: set[str], *, k: int) -> float:
    if not gold:
        raise ValueError("gold recipe ids cannot be empty")
    return len(set(ranking[:k]) & gold) / len(gold)


def reciprocal_rank(ranking: list[str], gold: set[str]) -> float:
    for rank, recipe_id in enumerate(ranking, start=1):
        if recipe_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: list[str], gold: set[str], *, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, recipe_id in enumerate(ranking[:k], start=1)
        if recipe_id in gold
    )
    ideal_hits = min(len(gold), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_retrieval(
    configuration: str,
    cases: list[RetrievalBenchmarkCase],
    rankings: dict[str, list[str]],
    *,
    ks: tuple[int, ...] = (5, 10, 20),
) -> RetrievalBenchmarkResult:
    if not cases:
        raise ValueError("benchmark requires at least one case")
    missing = [case.case_id for case in cases if case.case_id not in rankings]
    if missing:
        raise ValueError(f"missing rankings for cases: {', '.join(missing)}")

    metrics = {
        f"recall@{k}": fmean(
            recall_at_k(
                rankings[case.case_id], case.relevant_recipe_ids, k=k
            )
            for case in cases
        )
        for k in ks
    }
    metrics["mrr"] = fmean(
        reciprocal_rank(rankings[case.case_id], case.relevant_recipe_ids)
        for case in cases
    )
    largest_k = max(ks)
    metrics[f"ndcg@{largest_k}"] = fmean(
        ndcg_at_k(
            rankings[case.case_id], case.relevant_recipe_ids, k=largest_k
        )
        for case in cases
    )
    return RetrievalBenchmarkResult(
        configuration=configuration,
        case_count=len(cases),
        metrics=metrics,
    )


def run_retrieval_benchmark(
    configuration: str,
    cases: list[RetrievalBenchmarkCase],
    retriever: SearchAdapter,
    *,
    top_k: int = 20,
) -> RetrievalBenchmarkResult:
    rankings = {
        case.case_id: retriever.search(
            case.query,
            case.metadata_filters,
            top_k=top_k,
        )
        for case in cases
    }
    return evaluate_retrieval(configuration, cases, rankings)
