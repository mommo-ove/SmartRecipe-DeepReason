from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CypherSource(str, Enum):
    TEMPLATE = "template"
    DYNAMIC = "dynamic"


class CypherRunStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_FAILED = "execution_failed"


class CypherCandidate(BaseModel):
    statement: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: CypherSource
    template_id: str | None = None
    attempt: int = Field(default=0, ge=0, le=2)

    @field_validator("statement")
    @classmethod
    def statement_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("statement must not be blank")
        return value


class ValidationIssue(BaseModel):
    code: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ValidationReport(BaseModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues


def _ordered_unique_ids(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class CypherExecutionResult(BaseModel):
    status: CypherRunStatus
    records: list[dict[str, Any]] = Field(default_factory=list)
    selected_recipe_ids: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0)
    error: str | None = None

    @field_validator("selected_recipe_ids")
    @classmethod
    def normalize_selected_recipe_ids(cls, value: list[str]) -> list[str]:
        return _ordered_unique_ids(value)


class Text2CypherResult(BaseModel):
    question: str = Field(min_length=1)
    status: CypherRunStatus
    candidate: CypherCandidate | None = None
    validation_reports: list[ValidationReport] = Field(default_factory=list)
    execution: CypherExecutionResult | None = None
    selected_recipe_ids: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("selected_recipe_ids")
    @classmethod
    def normalize_selected_recipe_ids(cls, value: list[str]) -> list[str]:
        return _ordered_unique_ids(value)
