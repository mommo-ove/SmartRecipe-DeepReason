from __future__ import annotations

import asyncio
import json
from pathlib import Path
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable, Protocol

from .direct_capabilities import GustoBotDirectExecutor
from .models import AgentTask, Domain, EvidenceItem, Handoff, TaskStatus


class DomainAgent(Protocol):
    name: str
    domain: Domain

    async def execute(self, task: AgentTask, context: dict[str, Any]) -> Handoff:
        ...


Handler = Callable[[AgentTask, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class FunctionDomainAgent:
    name: str
    domain: Domain
    handler: Handler
    timeout_seconds: float = 90.0

    async def execute(self, task: AgentTask, context: dict[str, Any]) -> Handoff:
        started = perf_counter()
        try:
            result = await asyncio.wait_for(self.handler(task, context), timeout=self.timeout_seconds)
            evidence = _evidence_from_result(task, self.name, result)
            return Handoff(
                task_id=task.task_id,
                agent=self.name,
                domain=self.domain,
                status=TaskStatus.SUCCESS,
                summary=_result_summary(result),
                output=result,
                evidence=evidence,
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
        except Exception as exc:
            return Handoff(
                task_id=task.task_id,
                agent=self.name,
                domain=self.domain,
                status=TaskStatus.FAILED,
                error=str(exc),
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )


class DomainAgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[Domain, DomainAgent] = {}

    def register(self, agent: DomainAgent) -> None:
        if agent.domain in self._agents:
            raise ValueError(f"domain {agent.domain.value} already registered")
        self._agents[agent.domain] = agent

    def get(self, domain: Domain) -> DomainAgent:
        try:
            return self._agents[domain]
        except KeyError as exc:
            raise KeyError(f"no agent registered for domain {domain.value}") from exc

    @property
    def domains(self) -> list[Domain]:
        return list(self._agents)


def build_gustobot_registry(
    executor: GustoBotDirectExecutor | None = None,
) -> DomainAgentRegistry:
    executor = executor or GustoBotDirectExecutor()
    registry = DomainAgentRegistry()
    agents = {
        Domain.RECIPE: ("recipe_retrieval_agent", executor.recipe),
        Domain.ANALYTICS: ("data_analysis_agent", executor.analytics),
        Domain.VISION: ("vision_agent", executor.vision),
        Domain.FILE: ("file_ingestion_agent", executor.file),
        Domain.GENERAL: ("general_agent", executor.general),
    }
    for domain, (name, handler) in agents.items():
        registry.register(FunctionDomainAgent(name=name, domain=domain, handler=handler))
    registry.register(_build_local_meal_planning_agent())
    return registry


def build_demo_registry() -> DomainAgentRegistry:
    """Offline adapters for learning the orchestration path without infrastructure."""

    async def demo_handler(task: AgentTask, context: dict[str, Any]) -> dict[str, Any]:
        if task.domain == Domain.RECIPE:
            return {
                "answer": "演示结果：推荐低辣版宫保鸡丁，主要食材为鸡肉、花生和青椒。",
                "documents": [
                    {
                        "content": "演示菜谱：低辣宫保鸡丁，难度简单，预计 25 分钟。",
                        "metadata": {"source": "demo_recipe_kg"},
                    }
                ],
            }
        if task.domain == Domain.ANALYTICS:
            return {
                "answer": "演示统计：同类菜谱平均烹饪时长为 25 分钟。",
                "sql_statement": "SELECT AVG(total_time_minutes) FROM recipes WHERE main_ingredient = '鸡肉'",
                "execution_results": [{"avg_minutes": 25}],
            }
        if task.domain == Domain.VISION:
            return {"answer": "演示图像分析：识别到鸡肉、花生和青椒。"}
        if task.domain == Domain.FILE:
            return {"answer": "演示文件处理：文件结构校验通过，已生成待导入记录。"}
        return {"answer": "演示模式已启用，可以测试 DeepReason 多智能体工作流。"}

    registry = DomainAgentRegistry()
    names = {
        Domain.RECIPE: "demo_recipe_agent",
        Domain.ANALYTICS: "demo_analytics_agent",
        Domain.VISION: "demo_vision_agent",
        Domain.FILE: "demo_file_agent",
        Domain.GENERAL: "demo_general_agent",
    }
    for domain, name in names.items():
        registry.register(FunctionDomainAgent(name=name, domain=domain, handler=demo_handler))
    registry.register(_build_local_meal_planning_agent())
    return registry


def _build_local_meal_planning_agent():
    from gustobot.application.meal_planning.agent import (
        InMemoryMealCandidateSource,
        MealPlanningAgent,
    )
    from gustobot.application.meal_planning.fixtures import load_seed_corpus

    data_dir = Path(__file__).resolve().parents[2] / "data" / "meal_planning"
    corpus = load_seed_corpus(
        data_dir / "recipes.v1.json",
        data_dir / "manifest.v1.json",
    )
    return MealPlanningAgent(InMemoryMealCandidateSource(corpus.recipes))


def _result_summary(result: dict[str, Any]) -> str:
    answer = result.get("answer") or result.get("summary")
    if answer:
        return str(answer)
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


def _evidence_from_result(task: AgentTask, agent_name: str, result: dict[str, Any]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    documents = result.get("documents", []) or []
    for index, document in enumerate(documents[:10]):
        if isinstance(document, dict):
            content = str(document.get("page_content") or document.get("content") or document)
            metadata = document.get("metadata", {}) if isinstance(document.get("metadata"), dict) else {}
        else:
            content = str(getattr(document, "page_content", document))
            metadata = getattr(document, "metadata", {}) or {}
        evidence.append(
            EvidenceItem.create(
                task_id=task.task_id,
                source_type="retrieval",
                source=str(metadata.get("source", f"{agent_name}:document:{index}")),
                content=content,
                metadata=metadata,
            )
        )
    if result.get("sql_statement"):
        sql_evidence = EvidenceItem.create(
            task_id=task.task_id,
            source_type="sql",
            source="mysql_readonly",
            content=str(result["sql_statement"]),
            metadata={"rows": result.get("execution_results", [])},
        )
        evidence.append(sql_evidence)
        for record in result.get("execution_results", []) or []:
            if isinstance(record, dict):
                evidence.extend(
                    _atomic_facts_from_record(
                        task,
                        record,
                        source="mysql_readonly",
                        provenance_id=sql_evidence.evidence_id,
                    )
                )
    for index, cypher in enumerate((result.get("cyphers", []) or [])[:10]):
        if not isinstance(cypher, dict):
            continue
        statement = str(cypher.get("statement") or cypher.get("cypher") or "")
        records = cypher.get("records", []) or []
        if statement or records:
            cypher_evidence = EvidenceItem.create(
                task_id=task.task_id,
                source_type="cypher",
                source="neo4j_readonly",
                content=statement or f"Cypher result set {index + 1}",
                metadata={"records": records},
            )
            evidence.append(cypher_evidence)
            for record in records:
                if isinstance(record, dict):
                    evidence.extend(
                        _atomic_facts_from_record(
                            task,
                            record,
                            source="neo4j_readonly",
                            provenance_id=cypher_evidence.evidence_id,
                        )
                    )
    if not evidence and result.get("answer"):
        evidence.append(
            EvidenceItem.create(
                task_id=task.task_id,
                source_type="agent_output",
                source=agent_name,
                content=str(result["answer"]),
            )
        )
    return evidence


def _atomic_facts_from_record(
    task: AgentTask,
    record: dict[str, Any],
    *,
    source: str,
    provenance_id: str,
) -> list[EvidenceItem]:
    recipe_id = record.get("recipe_id") or record.get("canonical_recipe_id")
    entity_id = str(recipe_id or task.task_id)
    entity_type = "recipe" if recipe_id else "query_result"
    provenance = {"query_evidence_id": provenance_id}
    facts = [
        EvidenceItem.create_fact(
            task_id=task.task_id,
            source_type="fact_atom",
            source=source,
            entity_type=entity_type,
            entity_id=entity_id,
            field="exists",
            value=True,
            provenance=provenance,
        )
    ]
    aliases = {"recipe_name": "name", "name": "name"}
    supported_fields = {
        "total_minutes",
        "calories_kcal",
        "protein_g",
        "ingredients",
        "ingredient_id",
        "meal_types",
        "tags",
        "total_calories",
    }
    for raw_field, value in record.items():
        field = aliases.get(raw_field, raw_field)
        if field not in supported_fields or value is None:
            continue
        field_provenance = dict(provenance)
        if field == "ingredients":
            field_provenance["complete"] = bool(
                record.get("ingredients_complete", False)
            )
        facts.append(
            EvidenceItem.create_fact(
                task_id=task.task_id,
                source_type="fact_atom",
                source=source,
                entity_type=entity_type,
                entity_id=entity_id,
                field=field,
                value=value,
                provenance=field_provenance,
            )
        )
    return facts
