"""FoodTrace allergen-recall domain contracts and deterministic fixtures."""

from .fixtures import DEFAULT_FOODTRACE_SEED, build_foodtrace_fixture
from .models import (
    FoodTraceFixture,
    GoldRecallCase,
    IncidentNotice,
    RecallScope,
)

__all__ = [
    "DEFAULT_FOODTRACE_SEED",
    "FoodTraceFixture",
    "GoldRecallCase",
    "IncidentNotice",
    "RecallScope",
    "build_foodtrace_fixture",
]
