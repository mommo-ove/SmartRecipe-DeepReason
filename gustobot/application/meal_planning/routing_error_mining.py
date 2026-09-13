from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

import numpy as np


class QueryEncoder(Protocol):
    def encode(self, texts: list[str]) -> Any: ...


class BgeM3QueryEncoder:
    def __init__(self, model_path: str, *, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_path, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )


def select_diverse_route_errors(
    errors: list[dict],
    *,
    encoder: QueryEncoder,
    limit: int = 12,
    duplicate_similarity: float = 0.92,
) -> list[dict]:
    if len(errors) <= limit:
        return list(errors)

    vectors = _normalize(np.asarray(encoder.encode([item["query"] for item in errors])))
    groups: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, item in enumerate(errors):
        groups[_confusion_key(item)].append(index)

    selected: list[int] = []
    # First cover each distinct expected/predicted failure pattern.
    for indices in groups.values():
        selected.append(indices[0])
        if len(selected) == limit:
            break

    # Then maximize semantic distance while suppressing near-duplicates.
    remaining = [index for index in range(len(errors)) if index not in selected]
    while remaining and len(selected) < limit:
        candidates = []
        for index in remaining:
            max_similarity = max(float(vectors[index] @ vectors[chosen]) for chosen in selected)
            if max_similarity < duplicate_similarity:
                candidates.append((max_similarity, index))
        if not candidates:
            break
        _, chosen = min(candidates)
        selected.append(chosen)
        remaining.remove(chosen)

    return [errors[index] for index in selected]


def _confusion_key(item: dict) -> tuple[str, ...]:
    return (
        str(item["expected_intent"]),
        str(item["predicted_intent"]),
        str(item["expected_policy"]),
        str(item["predicted_policy"]),
    )


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
