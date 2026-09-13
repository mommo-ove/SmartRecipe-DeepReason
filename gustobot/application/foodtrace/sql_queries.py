from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import bindparam, text
from sqlalchemy.sql.elements import TextClause


_MAX_BATCH_SCOPE = 100

_INVENTORY_SQL = """
SELECT inventory_id, store_id, production_batch_id, quantity
FROM foodtrace_store_inventory
WHERE production_batch_id IN :production_batch_ids
ORDER BY inventory_id
""".strip()

_ORDERS_SQL = """
SELECT
    order_id,
    store_id,
    production_batch_id,
    quantity,
    customer_ref,
    ordered_at
FROM foodtrace_order_exposures
WHERE production_batch_id IN :production_batch_ids
ORDER BY order_id
""".strip()


@dataclass(frozen=True)
class ParameterizedSqlQuery:
    statement: TextClause
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ExposureQueries:
    production_batch_ids: tuple[str, ...]
    inventory: ParameterizedSqlQuery
    orders: ParameterizedSqlQuery


def build_exposure_queries(
    production_batch_ids: Iterable[str],
) -> ExposureQueries:
    normalized_ids = _normalize_batch_ids(production_batch_ids)
    parameters = {"production_batch_ids": normalized_ids}
    return ExposureQueries(
        production_batch_ids=normalized_ids,
        inventory=ParameterizedSqlQuery(
            statement=text(_INVENTORY_SQL).bindparams(
                bindparam("production_batch_ids", expanding=True)
            ),
            parameters=parameters,
        ),
        orders=ParameterizedSqlQuery(
            statement=text(_ORDERS_SQL).bindparams(
                bindparam("production_batch_ids", expanding=True)
            ),
            parameters=parameters,
        ),
    )


def _normalize_batch_ids(production_batch_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(production_batch_ids, (str, bytes)):
        raise ValueError("production_batch_ids must be a structured collection")
    normalized_ids = {str(batch_id).strip() for batch_id in production_batch_ids}
    if not normalized_ids or "" in normalized_ids:
        raise ValueError("production_batch_ids cannot be empty or blank")
    if len(normalized_ids) > _MAX_BATCH_SCOPE:
        raise ValueError(
            f"production_batch_ids cannot contain more than {_MAX_BATCH_SCOPE} IDs"
        )
    return tuple(sorted(normalized_ids))
