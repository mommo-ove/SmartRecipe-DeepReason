from __future__ import annotations

import csv
import hashlib
import json
import re
from io import StringIO
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, computed_field


_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?$"
)
_EXCLUDED_CATEGORIES = {
    "beverages",
    "breads",
    "candy",
    "chutneys",
    "condiments, etc.",
    "cookies",
    "dessert",
    "frozen desserts",
    "salad dressings",
    "sauces",
    "spreads",
}


class RecipeCapabilities(BaseModel):
    retrieval_ready: bool = False
    nutrition_ready: bool = False
    time_ready: bool = False
    allergen_ready: bool = False
    budget_ready: bool = False

    @computed_field
    @property
    def full_planning_ready(self) -> bool:
        return all(
            (
                self.retrieval_ready,
                self.nutrition_ready,
                self.time_ready,
                self.allergen_ready,
                self.budget_ready,
            )
        )


class ExternalRecipeDocument(BaseModel):
    recipe_id: str
    name: str
    description: str = ""
    meal_types: set[str] = Field(min_length=1)
    ingredients: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    calories_kcal: float = Field(gt=0)
    protein_g: float = Field(gt=0)
    total_minutes: int = Field(gt=0)
    source_ref: str
    capabilities: RecipeCapabilities

    def search_text(self) -> str:
        return " ".join(
            [
                self.name,
                self.description,
                *sorted(self.meal_types),
                *self.ingredients,
                *self.tags,
            ]
        )


class ExternalCorpusManifest(BaseModel):
    version: str
    status: str
    source_dataset: str
    source_url: str
    record_count: int = Field(ge=1)
    source_rows_scanned: int = Field(ge=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: dict[str, int]
    resume_metric_eligible: bool = False
    notes: str = ""


class ExternalRecipeCorpus(BaseModel):
    manifest: ExternalCorpusManifest
    documents: list[ExternalRecipeDocument]


def load_external_corpus(
    corpus_path: Path,
    manifest_path: Path,
) -> ExternalRecipeCorpus:
    raw = corpus_path.read_bytes()
    manifest = ExternalCorpusManifest.model_validate_json(
        manifest_path.read_text("utf-8")
    )
    if hashlib.sha256(raw).hexdigest() != manifest.corpus_sha256:
        raise ValueError("external recipe corpus hash does not match manifest")
    documents = [
        ExternalRecipeDocument.model_validate(json.loads(line))
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(documents) != manifest.record_count:
        raise ValueError("external recipe corpus count does not match manifest")
    return ExternalRecipeCorpus(manifest=manifest, documents=documents)


def parse_iso8601_minutes(value: str | None) -> int | None:
    if not value:
        return None
    match = _DURATION.fullmatch(value.strip().upper())
    if match is None:
        return None
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    return days * 1440 + hours * 60 + minutes


def parse_r_vector(value: str | None) -> list[str]:
    if not value:
        return []
    text = value.strip()
    if not (text.startswith("c(") and text.endswith(")")):
        return [text.strip('"')]
    inner = text[2:-1]
    return [
        item.strip()
        for item in next(csv.reader(StringIO(inner), skipinitialspace=True), [])
        if item.strip()
    ]


class ExternalRecipeNormalizer:
    def __init__(
        self,
        *,
        min_calories: float = 250,
        max_calories: float = 800,
        min_protein_g: float = 15,
        max_minutes: int = 60,
    ) -> None:
        self._min_calories = min_calories
        self._max_calories = max_calories
        self._min_protein_g = min_protein_g
        self._max_minutes = max_minutes

    def normalize(self, row: dict[str, Any]) -> ExternalRecipeDocument | None:
        name = str(row.get("Name") or "").strip()
        category = str(row.get("RecipeCategory") or "").strip().casefold()
        calories = _number(row.get("Calories"))
        protein = _number(row.get("ProteinContent"))
        total_minutes = parse_iso8601_minutes(row.get("TotalTime"))
        ingredients = [item.casefold() for item in parse_r_vector(row.get("RecipeIngredientParts"))]
        tags = [item.casefold() for item in parse_r_vector(row.get("Keywords"))]
        instructions = parse_r_vector(row.get("RecipeInstructions"))

        if (
            not name
            or category in _EXCLUDED_CATEGORIES
            or calories is None
            or not self._min_calories <= calories <= self._max_calories
            or protein is None
            or protein < self._min_protein_g
            or total_minutes is None
            or not 0 < total_minutes <= self._max_minutes
            or not ingredients
        ):
            return None

        meal_types = _infer_meal_types(tags, category)
        if not meal_types:
            return None
        source_id = str(row.get("RecipeId"))
        return ExternalRecipeDocument(
            recipe_id=f"foodcom-{source_id}",
            name=name,
            description=str(row.get("Description") or "").strip(),
            meal_types=meal_types,
            ingredients=ingredients,
            tags=tags,
            instructions=instructions,
            calories_kcal=calories,
            protein_g=protein,
            total_minutes=total_minutes,
            source_ref=f"hf:untitledwebsite123/food-recipes:{source_id}",
            capabilities=RecipeCapabilities(
                retrieval_ready=True,
                nutrition_ready=True,
                time_ready=True,
                allergen_ready=False,
                budget_ready=False,
            ),
        )


def collect_normalized(
    rows: Iterable[dict[str, Any]],
    *,
    target_count: int,
    normalizer: ExternalRecipeNormalizer | None = None,
) -> list[ExternalRecipeDocument]:
    parser = normalizer or ExternalRecipeNormalizer()
    documents: list[ExternalRecipeDocument] = []
    seen: set[str] = set()
    for row in rows:
        document = parser.normalize(row)
        if document is None or document.recipe_id in seen:
            continue
        documents.append(document)
        seen.add(document.recipe_id)
        if len(documents) >= target_count:
            break
    return documents


def _infer_meal_types(tags: list[str], category: str) -> set[str]:
    values = set(tags)
    if category == "breakfast":
        return {"breakfast"}
    if category == "lunch/snacks":
        return {"lunch"}
    slots = {
        slot
        for slot in ("breakfast", "lunch", "dinner")
        if slot in values
    }
    if slots:
        return slots
    return {"lunch", "dinner"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None
