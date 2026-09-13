from __future__ import annotations

import re

from .models import ValidationIssue, ValidationReport


_UNSAFE_CLAUSE = re.compile(
    r"\b(?:DETACH\s+DELETE|LOAD\s+CSV|BEGIN\s+TRANSACTION|"
    r"CREATE|MERGE|DELETE|SET|REMOVE|DROP|CALL|FOREACH|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)
_ALLOWED_START = re.compile(r"^\s*(?:OPTIONAL\s+MATCH|MATCH|WITH|UNWIND)\b", re.I)
_RECIPE_ID_RETURN = re.compile(r"\bAS\s+recipe_id\b", re.I)
_LIMIT = re.compile(r"\bLIMIT\s+(\$[A-Za-z_]\w*|\d+)\b", re.I)


def _mask_literals_comments_and_identifiers(statement: str) -> str:
    """Mask content that must not be interpreted as Cypher clauses."""

    result = list(statement)
    index = 0
    state: str | None = None
    while index < len(statement):
        current = statement[index]
        following = statement[index + 1] if index + 1 < len(statement) else ""

        if state == "line_comment":
            if current in "\r\n":
                state = None
            else:
                result[index] = " "
            index += 1
            continue
        if state == "block_comment":
            result[index] = " "
            if current == "*" and following == "/":
                result[index + 1] = " "
                state = None
                index += 2
            else:
                index += 1
            continue
        if state in {"single_quote", "double_quote", "backtick"}:
            result[index] = " "
            delimiter = {
                "single_quote": "'",
                "double_quote": '"',
                "backtick": "`",
            }[state]
            if current == "\\" and following:
                result[index + 1] = " "
                index += 2
            elif current == delimiter:
                if following == delimiter:
                    result[index + 1] = " "
                    index += 2
                else:
                    state = None
                    index += 1
            else:
                index += 1
            continue

        if current == "/" and following == "/":
            result[index] = result[index + 1] = " "
            state = "line_comment"
            index += 2
        elif current == "/" and following == "*":
            result[index] = result[index + 1] = " "
            state = "block_comment"
            index += 2
        elif current in {"'", '"', "`"}:
            result[index] = " "
            state = {
                "'": "single_quote",
                '"': "double_quote",
                "`": "backtick",
            }[current]
            index += 1
        else:
            index += 1
    return "".join(result)


def _issue(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, stage="safety", message=message)


def validate_cypher_safety(
    statement: str,
    *,
    require_recipe_ids: bool = True,
    max_literal_limit: int = 100,
) -> ValidationReport:
    masked = _mask_literals_comments_and_identifiers(statement)
    issues: list[ValidationIssue] = []

    stripped_without_trailing_terminator = masked.strip().removesuffix(";")
    if ";" in stripped_without_trailing_terminator:
        issues.append(
            _issue("MULTIPLE_STATEMENTS", "Only one Cypher statement is allowed")
        )

    unsafe = sorted(
        {match.group(0).upper() for match in _UNSAFE_CLAUSE.finditer(masked)}
    )
    if unsafe:
        issues.append(
            _issue("UNSAFE_CLAUSE", f"Unsafe Cypher clause: {', '.join(unsafe)}")
        )

    if not _ALLOWED_START.search(masked):
        issues.append(
            _issue(
                "INVALID_START",
                "Read queries must start with MATCH, OPTIONAL MATCH, WITH or UNWIND",
            )
        )

    if require_recipe_ids:
        if not _RECIPE_ID_RETURN.search(masked):
            issues.append(
                _issue(
                    "MISSING_RECIPE_ID",
                    "Recipe-list queries must return a stable AS recipe_id field",
                )
            )
        limit_match = _LIMIT.search(masked)
        if limit_match is None:
            issues.append(
                _issue("MISSING_LIMIT", "Recipe-list queries require a bounded LIMIT")
            )
        elif (
            limit_match.group(1).isdigit()
            and int(limit_match.group(1)) > max_literal_limit
        ):
            issues.append(
                _issue(
                    "LIMIT_TOO_HIGH",
                    f"Literal LIMIT exceeds the maximum of {max_literal_limit}",
                )
            )

    return ValidationReport(issues=issues)
