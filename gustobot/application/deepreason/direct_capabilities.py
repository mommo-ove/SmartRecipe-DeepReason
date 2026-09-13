from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage

from .models import AgentTask


NodeHandler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class GustoBotDirectExecutor:
    """Invoke GustoBot business capabilities without entering its top-level router."""

    recipe_graph: Any | None = None
    analytics_graph: Any | None = None
    node_handlers: dict[str, NodeHandler] | None = None
    _recipe_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _analytics_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def recipe(self, task: AgentTask, context: dict[str, Any]) -> dict[str, Any]:
        graph = await self._get_recipe_graph()
        result = await graph.ainvoke(
            {"question": task.instruction, "history": []},
            config=self._config(task, context),
        )
        return normalize_recipe_result(result, result_limit=task.result_limit)

    async def analytics(self, task: AgentTask, context: dict[str, Any]) -> dict[str, Any]:
        recipe_ids = collect_dependency_recipe_ids(task, context)
        dependency_outputs = context.get("dependency_outputs", {})
        has_recipe_selection_handoff = any(
            dependency_id in dependency_outputs
            and "selected_recipe_ids" in dependency_outputs[dependency_id]
            for dependency_id in task.depends_on
        )
        if has_recipe_selection_handoff and not recipe_ids:
            raise ValueError(
                "recipe dependency did not produce selected_recipe_ids; "
                "refusing an unscoped analytics query"
            )
        graph = await self._get_analytics_graph()
        result = await graph.ainvoke(
            {
                "question": task.instruction,
                "db_type": "MySQL",
                "max_rows": 1000,
                "max_retries": 3,
                **({"recipe_ids": recipe_ids} if recipe_ids else {}),
            },
            config=self._config(task, context),
        )
        return normalize_capability_result(result)

    async def vision(self, task: AgentTask, context: dict[str, Any]) -> dict[str, Any]:
        return await self._invoke_business_node("vision", task, context)

    async def file(self, task: AgentTask, context: dict[str, Any]) -> dict[str, Any]:
        return await self._invoke_business_node("file", task, context)

    async def general(self, task: AgentTask, context: dict[str, Any]) -> dict[str, Any]:
        return await self._invoke_business_node("general", task, context)

    async def _get_recipe_graph(self) -> Any:
        if self.recipe_graph is not None:
            return self.recipe_graph
        async with self._recipe_lock:
            if self.recipe_graph is None:
                self.recipe_graph = await asyncio.to_thread(_build_recipe_graph)
        return self.recipe_graph

    async def _get_analytics_graph(self) -> Any:
        if self.analytics_graph is not None:
            return self.analytics_graph
        async with self._analytics_lock:
            if self.analytics_graph is None:
                self.analytics_graph = _build_analytics_graph()
        return self.analytics_graph

    async def _invoke_business_node(
        self,
        capability: str,
        task: AgentTask,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        from gustobot.application.agents.lg_states import AgentState, Router

        handler = self._get_node_handlers()[capability]
        router_types = {
            "vision": "image-query",
            "file": "file-query",
            "general": "general-query",
        }
        state = AgentState(
            messages=[HumanMessage(content=task.instruction)],
            router=Router(
                type=router_types[capability],
                logic="DeepReason direct domain dispatch",
                question=task.instruction,
            ),
        )
        result = await handler(state, config=self._config(task, context))
        return normalize_capability_result(result)

    def _get_node_handlers(self) -> dict[str, NodeHandler]:
        if self.node_handlers is None:
            from gustobot.application.agents.lg_builder import (
                process_file_query,
                process_general_query,
                process_image_query,
            )

            self.node_handlers = {
                "vision": process_image_query,
                "file": process_file_query,
                "general": process_general_query,
            }
        return self.node_handlers

    @staticmethod
    def _config(task: AgentTask, context: dict[str, Any]) -> dict[str, Any]:
        session_id = str(context.get("session_id", "deepreason"))
        configurable: dict[str, Any] = {
            "thread_id": f"{session_id}:{task.task_id}",
        }
        for key in ("user_id", "image_path", "file_path"):
            if context.get(key):
                configurable[key] = context[key]
        return {
            "configurable": configurable,
            "tags": ["deepreason", "direct-dispatch", task.domain.value],
        }


def _build_recipe_graph() -> Any:
    from gustobot.application.agents.rag_sub_graph.components.predefined_cypher.cypher_dict import (
        predefined_cypher_dict,
    )
    from gustobot.application.agents.rag_sub_graph.rag_builder import build_rag_subgraph
    from gustobot.application.agents.utils.neo4j_connect import get_neo4j_graph
    from gustobot.application.prompts.lg_prompts import SCOPE_DESCRIPTION

    return build_rag_subgraph(
        graph=get_neo4j_graph(),
        predefined_cypher_dict=predefined_cypher_dict,
        scope_description=SCOPE_DESCRIPTION,
    )


def _build_analytics_graph() -> Any:
    from gustobot.application.agents.text2sql_sub_graph.builder import build_text2sql_subgraph

    return build_text2sql_subgraph()


def normalize_capability_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in (
        "answer",
        "sources",
        "documents",
        "sql_statement",
        "execution_results",
        "execution_error",
        "cyphers",
        "steps",
    ):
        if result.get(key) is not None:
            normalized[key] = result[key]

    normalized.setdefault("answer", "")
    normalized.setdefault("sources", [])
    normalized.setdefault("documents", [])
    if not normalized["answer"]:
        for message in reversed(result.get("messages", []) or []):
            content = getattr(message, "content", "")
            if content:
                normalized["answer"] = str(content)
                break
    return normalized


def normalize_recipe_result(
    result: dict[str, Any],
    *,
    result_limit: int | None = None,
) -> dict[str, Any]:
    """Add a deterministic, cross-store recipe selection to the RAG result."""
    normalized = normalize_capability_result(result)
    candidates = extract_recipe_candidates(result)
    if result_limit is not None:
        candidates = candidates[:result_limit]
    normalized["recipe_candidates"] = candidates
    normalized["selected_recipe_ids"] = [
        candidate["recipe_id"] for candidate in candidates
    ]
    return normalized


def extract_recipe_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Read ranked recipe identities from retrieval metadata and Cypher records."""
    raw_candidates: list[tuple[str, str, str, float | None]] = []

    for document in result.get("documents", []) or []:
        if isinstance(document, dict):
            metadata = document.get("metadata", {})
        else:
            metadata = getattr(document, "metadata", {})
        if not isinstance(metadata, dict):
            continue
        recipe_id = metadata.get("recipe_id") or metadata.get("node_id")
        if recipe_id:
            raw_candidates.append(
                (
                    str(recipe_id),
                    str(metadata.get("recipe_name") or metadata.get("name") or ""),
                    "retrieval",
                    _optional_float(metadata.get("score")),
                )
            )

    for cypher in result.get("cyphers", []) or []:
        if not isinstance(cypher, dict):
            continue
        for record in cypher.get("records", []) or []:
            if not isinstance(record, dict):
                continue
            recipe_id = record.get("recipe_id") or record.get("node_id")
            if recipe_id:
                raw_candidates.append(
                    (
                        str(recipe_id),
                        str(record.get("recipe_name") or record.get("name") or ""),
                        "cypher",
                        _optional_float(record.get("score")),
                    )
                )

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for recipe_id, recipe_name, source, score in raw_candidates:
        if recipe_id in seen:
            continue
        seen.add(recipe_id)
        candidates.append(
            {
                "recipe_id": recipe_id,
                "recipe_name": recipe_name,
                "rank": len(candidates) + 1,
                "source": source,
                "score": score,
            }
        )
    return candidates


def collect_dependency_recipe_ids(
    task: AgentTask,
    context: dict[str, Any],
) -> list[str]:
    """Collect only IDs produced by dependencies declared on this task."""
    dependency_outputs = context.get("dependency_outputs", {})
    ids: list[str] = []
    seen: set[str] = set()
    for dependency_id in task.depends_on:
        output = dependency_outputs.get(dependency_id, {})
        for recipe_id in output.get("selected_recipe_ids", []) or []:
            normalized_id = str(recipe_id)
            if normalized_id.isdigit() and normalized_id not in seen:
                ids.append(normalized_id)
                seen.add(normalized_id)
    return ids


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
