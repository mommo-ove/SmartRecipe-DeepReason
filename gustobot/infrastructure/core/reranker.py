"""Unified text reranking through Alibaba Cloud Model Studio."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from gustobot.config.settings import settings
from gustobot.infrastructure.core.logger import get_logger

logger = get_logger(service="reranker")


def _build_qwen3_payload(
    query: str,
    documents: List[str],
    *,
    model: str,
    top_n: int,
) -> Dict[str, Any]:
    """Build the current qwen3-rerank compatible API payload."""
    return {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }


def _extract_ranked_results(data: Dict[str, Any]) -> List[Tuple[int, float]]:
    """Accept both the current top-level and the legacy nested response."""
    results_raw = data.get("results")
    if results_raw is None:
        results_raw = data.get("output", {}).get("results", [])

    ranked = [
        (int(item.get("index", 0)), float(item.get("relevance_score", 0.0)))
        for item in results_raw
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


async def rerank_documents(
    query: str,
    documents: List[str],
    *,
    top_n: Optional[int] = None,
    model: Optional[str] = None,
) -> List[Tuple[int, float]]:
    """Return ``(original_index, relevance_score)`` in descending order."""
    if not documents:
        return []

    if not settings.RERANK_ENABLED:
        logger.debug("Reranker disabled; preserving retrieval order")
        return [(index, 1.0) for index in range(len(documents))]

    api_key = settings.RERANK_API_KEY
    if not api_key:
        logger.warning("RERANK_API_KEY is missing; preserving retrieval order")
        return [(index, 1.0) for index in range(len(documents))]

    resolved_top_n = top_n or settings.RERANK_TOP_N
    resolved_model = model or settings.RERANK_MODEL
    url = f"{settings.RERANK_BASE_URL.rstrip('/')}/{settings.RERANK_ENDPOINT.lstrip('/')}"
    truncated = documents[: settings.RERANK_MAX_CANDIDATES]
    payload = _build_qwen3_payload(
        query,
        truncated,
        model=resolved_model,
        top_n=resolved_top_n,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info(
        "Rerank started: model=%s top_n=%d candidates=%d/%d",
        resolved_model,
        resolved_top_n,
        len(truncated),
        len(documents),
    )

    try:
        started_at = time.perf_counter()
        timeout = aiohttp.ClientTimeout(total=settings.RERANK_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("Rerank API returned %d: %s", response.status, error_text[:200])
                    return [(index, 1.0) for index in range(len(documents))]
                data = await response.json()

        ranked = _extract_ranked_results(data)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Rerank completed: %d candidates -> %d results (%.1fms)",
            len(truncated),
            len(ranked),
            elapsed_ms,
        )
        return ranked
    except Exception as exc:
        logger.error("Rerank API call failed: %s", exc)
        return [(index, 1.0) for index in range(len(documents))]
