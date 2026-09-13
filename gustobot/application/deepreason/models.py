from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Domain(str, Enum):
    GENERAL = "general"
    RECIPE = "recipe"
    ANALYTICS = "analytics"
    MEAL_PLANNING = "meal_planning"
    VISION = "vision"
    FILE = "file"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateStatus(str, Enum):
    ALLOW = "allow"
    RETRY = "retry"
    DENY = "deny"


class AgentTask(BaseModel):
    task_id: str
    domain: Domain
    instruction: str
    depends_on: list[str] = Field(default_factory=list)
    evidence_required: bool = True
    priority: int = 0
    result_limit: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def reject_self_dependency(self) -> "AgentTask":
        if self.task_id in self.depends_on:
            raise ValueError("task cannot depend on itself")
        return self


class ExecutionPlan(BaseModel):
    query: str
    tasks: list[AgentTask]
    rationale: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    review_status: str = "pending"
    reviewer_notes: list[str] = Field(default_factory=list)

    @property
    def is_multi_agent(self) -> bool:
        return len(self.tasks) > 1


class ReviewDecision(BaseModel):
    status: Literal["approve", "revise", "escalate"] = "approve"
    required_domains: list[Domain] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str
    task_id: str
    source_type: str
    source: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        source_type: str,
        source: str,
        content: str,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EvidenceItem":
        fingerprint = "\x1f".join((task_id, source_type, source, content))
        evidence_id = "ev_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        return cls(
            evidence_id=evidence_id,
            task_id=task_id,
            source_type=source_type,
            source=source,
            content=content,
            score=score,
            metadata=metadata or {},
        )

    @classmethod
    def create_fact(
        cls,
        *,
        task_id: str,
        source_type: str,
        source: str,
        entity_type: str,
        entity_id: str,
        field: str,
        value: Any,
        provenance: dict[str, Any] | None = None,
    ) -> "EvidenceItem":
        metadata = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "field": field,
            "value": value,
            **(provenance or {}),
        }
        return cls.create(
            task_id=task_id,
            source_type=source_type,
            source=source,
            content=json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str),
            metadata=metadata,
        )


class Handoff(BaseModel):
    task_id: str
    agent: str
    domain: Domain
    status: TaskStatus
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0


class CriticFinding(BaseModel):
    severity: str
    code: str
    message: str
    task_id: str | None = None


class GateDecision(BaseModel):
    status: GateStatus
    reasons: list[str] = Field(default_factory=list)
    retry_task_ids: list[str] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    run_id: str
    session_id: str
    answer: str
    plan: ExecutionPlan
    handoffs: list[Handoff]
    evidence: list[EvidenceItem]
    findings: list[CriticFinding]
    gate: GateDecision
    events: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = 0.0
