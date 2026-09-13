from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

from gustobot.application.agents.utils.llm_factory import get_llm

from .models import CypherCandidate, CypherSource, ValidationIssue
from .prompts import build_generation_prompt, build_repair_prompt
from .schema import GraphSchemaSnapshot


class AsyncTextModel(Protocol):
    async def ainvoke(self, prompt: str) -> Any: ...


_OPENING_FENCE = re.compile(r"^\s*```(?:cypher)?\s*", re.I)
_CLOSING_FENCE = re.compile(r"\s*```\s*$")


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _clean_cypher(response: Any) -> str:
    text = _response_text(response).strip()
    text = _OPENING_FENCE.sub("", text)
    return _CLOSING_FENCE.sub("", text).strip()


class Text2CypherGenerator:
    def __init__(self, model: AsyncTextModel) -> None:
        self._model = model

    async def generate(
        self,
        *,
        question: str,
        schema: GraphSchemaSnapshot,
        examples: Sequence[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> CypherCandidate:
        response = await self._model.ainvoke(
            build_generation_prompt(
                question=question,
                schema=schema,
                examples=examples,
                parameters=parameters,
            )
        )
        return CypherCandidate(
            statement=_clean_cypher(response),
            parameters=parameters,
            source=CypherSource.DYNAMIC,
        )

    async def repair(
        self,
        *,
        question: str,
        candidate: CypherCandidate,
        issues: Sequence[ValidationIssue],
        schema: GraphSchemaSnapshot,
    ) -> CypherCandidate:
        if candidate.attempt >= 2:
            raise ValueError("repair attempts exhausted")
        response = await self._model.ainvoke(
            build_repair_prompt(
                question=question,
                candidate=candidate,
                issues=issues,
                schema=schema,
            )
        )
        return CypherCandidate(
            statement=_clean_cypher(response),
            parameters=candidate.parameters,
            source=CypherSource.DYNAMIC,
            attempt=candidate.attempt + 1,
        )


def create_deepseek_text2cypher_generator() -> Text2CypherGenerator:
    return Text2CypherGenerator(get_llm(tags=["meal_text2cypher"]))
