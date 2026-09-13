from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from statistics import fmean
from typing import Mapping, Sequence

from .benchmark import RetrievalBenchmarkCase
from .external_corpus import ExternalRecipeDocument
from .graph_retrieval import (
    GraphRelationCaseSpec,
    build_graph_relation_case,
    canonicalize_ingredient,
)


DEFAULT_STOP_INGREDIENTS = {
    "butter",
    "egg",
    "eggs",
    "flour",
    "garlic",
    "milk",
    "oil",
    "olive oil",
    "onion",
    "pepper",
    "salt",
    "sugar",
    "water",
}


def mine_relation_benchmark_cases(
    documents: Sequence[ExternalRecipeDocument],
    *,
    case_count: int = 40,
    development_count: int = 10,
    min_pair_frequency: int = 3,
    max_pair_frequency: int = 8,
    stop_ingredients: set[str] | None = None,
) -> list[RetrievalBenchmarkCase]:
    """Mine deterministic include-two/exclude-one cases with exhaustive gold IDs."""

    if case_count < 1 or not 0 <= development_count <= case_count:
        raise ValueError("invalid benchmark case or development count")
    stop = DEFAULT_STOP_INGREDIENTS if stop_ingredients is None else stop_ingredients
    pair_documents: dict[tuple[str, str], list[ExternalRecipeDocument]] = defaultdict(
        list
    )
    for document in documents:
        ingredients = sorted(
            {
                canonicalize_ingredient(value)
                for value in document.ingredients
                if canonicalize_ingredient(value)
                and canonicalize_ingredient(value) not in stop
            }
        )
        for pair in combinations(ingredients, 2):
            pair_documents[pair].append(document)

    candidates: list[tuple[int, int, tuple[str, str], str]] = []
    for pair, matches in pair_documents.items():
        if not min_pair_frequency <= len(matches) <= max_pair_frequency:
            continue
        extras = Counter(
            ingredient
            for document in matches
            for ingredient in {
                canonicalize_ingredient(value) for value in document.ingredients
            }
            if ingredient not in pair and ingredient not in stop
        )
        for excluded, excluded_count in extras.items():
            safe_count = len(matches) - excluded_count
            if excluded_count and safe_count:
                balance = min(excluded_count, safe_count)
                candidates.append((balance, len(matches), pair, excluded))

    candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    selected: list[tuple[tuple[str, str], str]] = []
    used_pairs: set[tuple[str, str]] = set()
    for _, _, pair, excluded in candidates:
        if pair in used_pairs:
            continue
        selected.append((pair, excluded))
        used_pairs.add(pair)
        if len(selected) == case_count:
            break
    if len(selected) < case_count:
        raise ValueError(
            f"only {len(selected)} relation cases could be mined; requested {case_count}"
        )

    cases: list[RetrievalBenchmarkCase] = []
    for index, (pair, excluded) in enumerate(selected, start=1):
        split = "development" if index <= development_count else "test"
        spec = GraphRelationCaseSpec(
            case_id=f"graph-rel-{index:03d}",
            query=(
                f"Find recipes containing {pair[0]} and {pair[1]}, "
                f"but without {excluded}."
            ),
            ingredient_ids=list(pair),
            excluded_ingredient_ids=[excluded],
            split=split,
        )
        cases.append(build_graph_relation_case(spec, documents))
    return cases


def ingredient_exclusion_violation_at_k(
    cases: Sequence[RetrievalBenchmarkCase],
    rankings: Mapping[str, Sequence[str]],
    documents: Sequence[ExternalRecipeDocument],
    *,
    k: int = 20,
) -> float:
    """Average fraction of returned recipes containing a forbidden ingredient."""

    ingredient_sets = {
        document.recipe_id: {
            canonicalize_ingredient(value) for value in document.ingredients
        }
        for document in documents
    }
    rates: list[float] = []
    for case in cases:
        returned = [
            recipe_id
            for recipe_id in rankings.get(case.case_id, [])[:k]
            if recipe_id in ingredient_sets
        ]
        excluded = set(case.metadata_filters.get("excluded_graph_ingredient_ids", []))
        violations = sum(
            bool(ingredient_sets[recipe_id] & excluded) for recipe_id in returned
        )
        rates.append(violations / len(returned) if returned else 0.0)
    if not rates:
        raise ValueError("benchmark requires at least one case")
    return fmean(rates)
