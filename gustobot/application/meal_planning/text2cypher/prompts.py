from __future__ import annotations

import json
from typing import Any, Sequence

from .models import CypherCandidate, ValidationIssue
from .schema import GraphSchemaSnapshot


def build_generation_prompt(
    *,
    question: str,
    schema: GraphSchemaSnapshot,
    examples: Sequence[tuple[str, str]],
    parameters: dict[str, Any],
) -> str:
    rendered_examples = "\n\n".join(
        f"Question: {example_question}\nCypher: {cypher}"
        for example_question, cypher in examples
    )
    parameter_contract = "\n".join(
        f"- ${name}: {json.dumps(value, ensure_ascii=False)}"
        for name, value in sorted(parameters.items())
    )
    return f"""You generate one read-only Neo4j Cypher query for a recipe graph.
Return Cypher only, without Markdown fences or explanation.
Use only labels, properties and directed relationships in the schema.
Use the supplied $parameters instead of copying user values into string literals.
Recipe-list queries must return recipe.recipe_id AS recipe_id and use a bounded LIMIT.
Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, LOAD CSV, CALL or FOREACH.

SCHEMA
{schema.to_prompt_text()}

AVAILABLE PARAMETERS
{parameter_contract or '(none)'}

RELEVANT EXAMPLES
{rendered_examples or '(none)'}

USER QUESTION
{question}

CYPHER
"""


def build_repair_prompt(
    *,
    question: str,
    candidate: CypherCandidate,
    issues: Sequence[ValidationIssue],
    schema: GraphSchemaSnapshot,
) -> str:
    rendered_issues = "\n".join(
        f"- [{issue.stage}/{issue.code}] {issue.message}" for issue in issues
    )
    parameter_contract = "\n".join(
        f"- ${name}: {json.dumps(value, ensure_ascii=False)}"
        for name, value in sorted(candidate.parameters.items())
    )
    return f"""Repair one invalid read-only Neo4j Cypher query.
Return the corrected Cypher only, without Markdown fences or explanation.
Do not add or rename parameters. Never introduce a write/admin clause.
Recipe-list queries must return recipe.recipe_id AS recipe_id and use a bounded LIMIT.

FRESH SCHEMA
{schema.to_prompt_text()}

AVAILABLE PARAMETERS
{parameter_contract or '(none)'}

ORIGINAL QUESTION
{question}

INVALID CYPHER
{candidate.statement}

VALIDATION ISSUES
{rendered_issues or '(none provided)'}

CORRECTED CYPHER
"""
