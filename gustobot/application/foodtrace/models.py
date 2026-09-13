from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FoodTraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IncidentNotice(FoodTraceModel):
    incident_id: str = Field(min_length=1)
    supplier_id: str = Field(min_length=1)
    ingredient_lot_id: str = Field(min_length=1)
    allergen: Literal["peanut"] = "peanut"
    declared_on_label: Literal[False] = False
    received_at: datetime


class IngredientLot(FoodTraceModel):
    ingredient_lot_id: str = Field(min_length=1)
    ingredient_name: str = Field(min_length=1)
    supplier_id: str = Field(min_length=1)
    suspected_allergens: tuple[str, ...] = ()


class Recipe(FoodTraceModel):
    recipe_id: str = Field(min_length=1)
    recipe_name: str = Field(min_length=1)
    ingredient_lot_ids: tuple[str, ...] = Field(min_length=1)


class Product(FoodTraceModel):
    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    recipe_id: str = Field(min_length=1)
    label_allergens: tuple[str, ...] = ()


class ProductionBatch(FoodTraceModel):
    production_batch_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    produced_at: datetime


class StoreInventory(FoodTraceModel):
    inventory_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    production_batch_id: str = Field(min_length=1)
    quantity: int = Field(ge=0)


class OrderExposure(FoodTraceModel):
    order_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    production_batch_id: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    customer_ref: str = Field(min_length=1)
    ordered_at: datetime


class RecallScope(FoodTraceModel):
    impacted_product_ids: tuple[str, ...] = ()
    impacted_batch_ids: tuple[str, ...] = ()
    impacted_store_ids: tuple[str, ...] = ()
    impacted_order_ids: tuple[str, ...] = ()

    @field_validator(
        "impacted_product_ids",
        "impacted_batch_ids",
        "impacted_store_ids",
        "impacted_order_ids",
        mode="before",
    )
    @classmethod
    def normalize_ids(cls, value: object) -> tuple[str, ...]:
        return _normalize_nonblank_collection(value, label="ID")


class GoldRecallCase(FoodTraceModel):
    case_id: str = Field(min_length=1)
    incident: IncidentNotice
    gold_scope: RecallScope
    expected_gate: Literal["ALLOW_REPORT", "HUMAN_REVIEW", "BLOCKED"]
    scenario_tags: tuple[str, ...] = ()

    @field_validator("scenario_tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> tuple[str, ...]:
        return _normalize_nonblank_collection(value, label="scenario tag")


class FoodTraceFixture(FoodTraceModel):
    seed: int
    incident: IncidentNotice
    ingredient_lots: tuple[IngredientLot, ...]
    recipes: tuple[Recipe, ...]
    products: tuple[Product, ...]
    production_batches: tuple[ProductionBatch, ...]
    store_inventories: tuple[StoreInventory, ...]
    order_exposures: tuple[OrderExposure, ...]
    graph_batch_to_sql_fk: dict[str, str]
    gold_cases: tuple[GoldRecallCase, ...]


def _normalize_nonblank_collection(
    value: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} collection cannot be a scalar string")
    try:
        normalized = {str(item).strip() for item in value}
    except TypeError as exc:
        raise ValueError(f"{label} collection must be iterable") from exc
    if "" in normalized:
        raise ValueError(f"{label} values cannot be blank")
    return tuple(sorted(normalized))
