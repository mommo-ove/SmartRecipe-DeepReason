from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field


class CatalogEntry(BaseModel):
    recipe_id: str
    recipe_name: str
    commons_query: str
    commons_title: str | None = None
    split: Literal["catalog", "query"]
    ordinal: int = Field(ge=1)
    search_offset: int = Field(default=0, ge=0)
    license_allowlist: list[str] = Field(min_length=1)

    def source_fingerprint(self) -> str:
        source_config = {
            "recipe_id": self.recipe_id,
            "split": self.split,
            "ordinal": self.ordinal,
            "commons_query": self.commons_query,
            "commons_title": self.commons_title,
            "search_offset": self.search_offset,
            "license_allowlist": self.license_allowlist,
        }
        canonical = json.dumps(
            source_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CatalogValidationReport(BaseModel):
    recipe_count: int
    entry_count: int
    errors: list[str]


class ImageFileMetadata(BaseModel):
    width: int
    height: int
    format: str
    sha256: str


def load_catalog(path: str | Path) -> list[CatalogEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    return [CatalogEntry.model_validate(entry) for entry in entries]


def validate_catalog(
    entries: list[CatalogEntry],
    *,
    concepts_path: str | Path,
) -> CatalogValidationReport:
    known_recipes = _load_recipe_names(Path(concepts_path))
    errors: list[str] = []
    keys = [(entry.recipe_id, entry.split, entry.ordinal) for entry in entries]
    for key, count in Counter(keys).items():
        if count > 1:
            errors.append(f"duplicate catalog key: {key[0]}/{key[1]}/{key[2]}")
    for entry in entries:
        expected_name = known_recipes.get(entry.recipe_id)
        if expected_name is None:
            errors.append(f"unknown recipe_id: {entry.recipe_id}")
        elif expected_name != entry.recipe_name:
            errors.append(
                f"recipe name mismatch for {entry.recipe_id}: "
                f"expected {expected_name}, got {entry.recipe_name}"
            )
    return CatalogValidationReport(
        recipe_count=len({entry.recipe_id for entry in entries}),
        entry_count=len(entries),
        errors=errors,
    )


def validate_image_file(path: str | Path) -> ImageFileMetadata:
    image_path = Path(path)
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        image_format = str(image.format or "").upper()
        image.convert("RGB")
    return ImageFileMetadata(
        width=width,
        height=height,
        format=image_format,
        sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
    )


def _load_recipe_names(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["concept_id"]: row["name"]
            for row in csv.DictReader(handle)
            if row.get("concept_type") == "Recipe"
        }
