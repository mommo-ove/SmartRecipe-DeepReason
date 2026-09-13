from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import ValidationIssue, ValidationReport


class GraphRelationship(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_label: str = Field(min_length=1)
    relationship_type: str = Field(min_length=1)
    end_label: str = Field(min_length=1)


class GraphSchemaSnapshot(BaseModel):
    node_properties: dict[str, set[str]]
    relationships: set[GraphRelationship]

    def to_prompt_text(self) -> str:
        node_lines = [
            f"{label}({', '.join(sorted(properties, key=lambda value: (not value.endswith('_id'), value)))})"
            for label, properties in sorted(self.node_properties.items())
        ]
        relationship_lines = [
            f"{item.start_label}-[:{item.relationship_type}]->{item.end_label}"
            for item in sorted(
                self.relationships,
                key=lambda value: (
                    value.start_label,
                    value.relationship_type,
                    value.end_label,
                ),
            )
        ]
        return (
            "Nodes:\n"
            + "\n".join(node_lines)
            + "\nRelationships:\n"
            + "\n".join(relationship_lines)
        )


_NODE_PROPERTIES_QUERY = """
MATCH (node)
UNWIND labels(node) AS label
UNWIND keys(node) AS property
RETURN label, collect(DISTINCT property) AS properties
ORDER BY label
""".strip()

_RELATIONSHIPS_QUERY = """
MATCH (start)-[relationship]->(end)
UNWIND labels(start) AS start_label
UNWIND labels(end) AS end_label
RETURN DISTINCT start_label,
       type(relationship) AS relationship_type,
       end_label
ORDER BY start_label, relationship_type, end_label
""".strip()


class Neo4jSchemaLoader:
    def __init__(
        self,
        driver: Any,
        *,
        database: str = "neo4j",
        allowed_labels: set[str] | None = None,
    ) -> None:
        self._driver = driver
        self._database = database
        self._allowed_labels = allowed_labels
        self._cached: GraphSchemaSnapshot | None = None

    def load(self, *, refresh: bool = False) -> GraphSchemaSnapshot:
        if self._cached is not None and not refresh:
            return self._cached
        node_records, _, _ = self._driver.execute_query(
            _NODE_PROPERTIES_QUERY,
            database_=self._database,
        )
        relationship_records, _, _ = self._driver.execute_query(
            _RELATIONSHIPS_QUERY,
            database_=self._database,
        )
        nodes = {
            str(record["label"]): {str(value) for value in record["properties"]}
            for record in node_records
            if self._is_allowed(str(record["label"]))
        }
        relationships = {
            GraphRelationship.model_validate(dict(record))
            for record in relationship_records
            if self._is_allowed(str(record["start_label"]))
            and self._is_allowed(str(record["end_label"]))
        }
        self._cached = GraphSchemaSnapshot(
            node_properties=nodes,
            relationships=relationships,
        )
        return self._cached

    def _is_allowed(self, label: str) -> bool:
        return self._allowed_labels is None or label in self._allowed_labels


_LABEL = re.compile(
    r"\(\s*(?P<variable>[A-Za-z_]\w*)?\s*:\s*(?P<label>[A-Za-z_]\w*)",
    re.I,
)
_RELATIONSHIP_TYPE = re.compile(r"\[[^\]]*:\s*(?P<type>[A-Za-z_]\w*)", re.I)
_PROPERTY = re.compile(r"\b(?P<variable>[A-Za-z_]\w*)\.(?P<property>[A-Za-z_]\w*)\b")
_RELATIONSHIP_PATTERN = re.compile(
    r"\((?P<left>[^()]*)\)\s*(?P<left_arrow><-|-)\s*"
    r"\[[^\]]*:\s*(?P<type>[A-Za-z_]\w*)[^\]]*\]\s*"
    r"(?P<right_arrow>->|-)\s*\((?P<right>[^()]*)\)",
    re.I,
)


def _schema_issue(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, stage="schema", message=message)


def _node_label(node_text: str, variable_labels: dict[str, str]) -> str | None:
    label_match = re.search(r":\s*([A-Za-z_]\w*)", node_text)
    if label_match:
        return label_match.group(1)
    variable_match = re.match(r"\s*([A-Za-z_]\w*)", node_text)
    return variable_labels.get(variable_match.group(1)) if variable_match else None


def validate_cypher_schema(
    statement: str,
    snapshot: GraphSchemaSnapshot,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    variable_labels = {
        match.group("variable"): match.group("label")
        for match in _LABEL.finditer(statement)
        if match.group("variable")
    }

    for label in sorted({match.group("label") for match in _LABEL.finditer(statement)}):
        if label not in snapshot.node_properties:
            issues.append(_schema_issue("UNKNOWN_LABEL", f"Unknown label {label}"))

    known_relationship_types = {
        relationship.relationship_type for relationship in snapshot.relationships
    }
    for relationship_type in sorted(
        {match.group("type") for match in _RELATIONSHIP_TYPE.finditer(statement)}
    ):
        if relationship_type not in known_relationship_types:
            issues.append(
                _schema_issue(
                    "UNKNOWN_RELATIONSHIP",
                    f"Unknown relationship {relationship_type}",
                )
            )

    for match in _PROPERTY.finditer(statement):
        variable = match.group("variable")
        property_name = match.group("property")
        label = variable_labels.get(variable)
        if (
            label in snapshot.node_properties
            and property_name not in snapshot.node_properties[label]
        ):
            issues.append(
                _schema_issue(
                    "UNKNOWN_PROPERTY",
                    f"Unknown property {label}.{property_name}",
                )
            )

    for match in _RELATIONSHIP_PATTERN.finditer(statement):
        relationship_type = match.group("type")
        if relationship_type not in known_relationship_types:
            continue
        left_label = _node_label(match.group("left"), variable_labels)
        right_label = _node_label(match.group("right"), variable_labels)
        if not left_label or not right_label:
            continue
        if match.group("left_arrow") == "<-":
            start_label, end_label = right_label, left_label
        else:
            start_label, end_label = left_label, right_label
        expected = GraphRelationship(
            start_label=start_label,
            relationship_type=relationship_type,
            end_label=end_label,
        )
        if expected not in snapshot.relationships:
            issues.append(
                _schema_issue(
                    "WRONG_RELATIONSHIP_DIRECTION",
                    f"Invalid direction {start_label}-[:{relationship_type}]->{end_label}",
                )
            )

    return ValidationReport(issues=issues)
