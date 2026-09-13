from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    StrictInt,
    computed_field,
    field_validator,
    model_validator,
)


class MealSlot(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


class IngredientAmount(BaseModel):
    ingredient_id: str = Field(min_length=1)
    amount: Decimal = Field(gt=0)
    unit: Literal["g", "ml"]

    @field_validator("ingredient_id")
    @classmethod
    def normalize_ingredient_id(cls, value: str) -> str:
        return value.strip().casefold()


def _normalize_labels(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {str(value).strip().casefold() for value in values if str(value).strip()}


class MealPlanDraft(BaseModel):
    """Partially extracted constraints; None means the user has not confirmed it."""

    days: int | None = Field(default=None, ge=1, le=31)
    daily_calories_min: int | None = Field(default=None, gt=0)
    daily_calories_max: int | None = Field(default=None, gt=0)
    daily_protein_min_g: Decimal | None = Field(default=None, ge=0)
    excluded_allergens: set[str] | None = None
    max_meal_minutes: int | None = Field(default=None, gt=0)
    weekly_budget_cents: StrictInt | None = Field(default=None, gt=0)
    preferred_tags: set[str] = Field(default_factory=set)
    requested_ingredients: set[str] = Field(default_factory=set)

    @field_validator("excluded_allergens", mode="before")
    @classmethod
    def normalize_optional_allergens(cls, value: object) -> set[str] | None:
        if value is None:
            return None
        return _normalize_labels(value)

    @field_validator("preferred_tags", "requested_ingredients", mode="before")
    @classmethod
    def normalize_draft_labels(cls, value: object) -> set[str]:
        return _normalize_labels(value)

    @model_validator(mode="after")
    def validate_partial_calorie_range(self) -> "MealPlanDraft":
        if (
            self.daily_calories_min is not None
            and self.daily_calories_max is not None
            and self.daily_calories_max < self.daily_calories_min
        ):
            raise ValueError("daily calorie range is inverted")
        return self


class MealPlanConstraints(BaseModel):
    daily_calories_min: int = Field(gt=0)
    daily_calories_max: int = Field(gt=0)
    daily_protein_min_g: Decimal = Field(ge=0)
    excluded_allergens: set[str] = Field(default_factory=set)
    preferred_tags: set[str] = Field(default_factory=set)
    max_meal_minutes: int | None = Field(default=None, gt=0)
    weekly_budget_cents: StrictInt | None = Field(default=None, gt=0)
    max_recipe_repeats: int = Field(default=2, ge=1)
    max_main_ingredient_repeats: int = Field(default=4, ge=1)
    days: int = Field(default=7, ge=1, le=31)

    @field_validator("excluded_allergens", "preferred_tags", mode="before")
    @classmethod
    def normalize_constraint_labels(cls, value: object) -> set[str]:
        return _normalize_labels(value)

    @model_validator(mode="after")
    def validate_calorie_range(self) -> "MealPlanConstraints":
        if self.daily_calories_max < self.daily_calories_min:
            raise ValueError(
                "daily_calories_max must be greater than or equal to daily_calories_min"
            )
        return self


class RecipeCandidate(BaseModel):
    recipe_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    meal_types: set[MealSlot] = Field(min_length=1)
    calories_kcal: Decimal | None = Field(default=None, ge=0)
    protein_g: Decimal | None = Field(default=None, ge=0)
    total_minutes: int | None = Field(default=None, ge=0)
    estimated_cost_cents: StrictInt | None = Field(default=None, ge=0)
    allergens: set[str] = Field(default_factory=set)
    allergen_status_verified: bool = False
    ingredients_normalized: list[IngredientAmount] = Field(default_factory=list)
    preference_tags: set[str] = Field(default_factory=set)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("allergens", "preference_tags", mode="before")
    @classmethod
    def normalize_labels(cls, value: object) -> set[str]:
        return _normalize_labels(value)

    @field_validator("source_refs")
    @classmethod
    def reject_blank_source_refs(cls, value: list[str]) -> list[str]:
        cleaned = [source.strip() for source in value if source.strip()]
        return list(dict.fromkeys(cleaned))

    @computed_field
    @property
    def planning_eligible(self) -> bool:
        return (
            all(
                value is not None
                for value in (
                    self.calories_kcal,
                    self.protein_g,
                    self.total_minutes,
                    self.estimated_cost_cents,
                )
            )
            and self.allergen_status_verified
            and bool(self.ingredients_normalized)
            and bool(self.source_refs)
        )


class MealAssignment(BaseModel):
    day: int = Field(ge=1, le=31)
    slot: MealSlot
    recipe_id: str = Field(min_length=1)
    recipe_name: str = Field(min_length=1)
    calories_kcal: Decimal = Field(ge=0)
    protein_g: Decimal = Field(ge=0)
    total_minutes: int = Field(ge=0)
    estimated_cost_cents: StrictInt | None = Field(default=None, ge=0)
    evidence_ids: list[str] = Field(default_factory=list)


class DailyPlan(BaseModel):
    day: int = Field(ge=1, le=31)
    assignments: list[MealAssignment]

    @model_validator(mode="after")
    def validate_assignments(self) -> "DailyPlan":
        if any(assignment.day != self.day for assignment in self.assignments):
            raise ValueError("assignment day must match daily plan day")
        slots = [assignment.slot for assignment in self.assignments]
        if len(slots) != len(set(slots)):
            raise ValueError("meal slots must be unique within a day")
        return self

    @computed_field
    @property
    def calories_kcal(self) -> Decimal:
        return sum(
            (assignment.calories_kcal for assignment in self.assignments), Decimal("0")
        )

    @computed_field
    @property
    def protein_g(self) -> Decimal:
        return sum(
            (assignment.protein_g for assignment in self.assignments), Decimal("0")
        )

    @computed_field
    @property
    def total_cost_cents(self) -> int | None:
        costs = [assignment.estimated_cost_cents for assignment in self.assignments]
        if any(cost is None for cost in costs):
            return None
        return sum(cost for cost in costs if cost is not None)


class MealPlan(BaseModel):
    status: SolveStatus
    days: list[DailyPlan] = Field(default_factory=list)
    conflict_constraints: list[str] = Field(default_factory=list)
    solver_wall_time_ms: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_unique_days(self) -> "MealPlan":
        day_numbers = [day.day for day in self.days]
        if len(day_numbers) != len(set(day_numbers)):
            raise ValueError("day numbers must be unique")
        return self

    @computed_field
    @property
    def total_cost_cents(self) -> int | None:
        costs = [day.total_cost_cents for day in self.days]
        if any(cost is None for cost in costs):
            return None
        return sum(cost for cost in costs if cost is not None)
