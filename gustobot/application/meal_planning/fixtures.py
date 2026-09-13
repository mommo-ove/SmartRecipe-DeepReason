from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from .models import RecipeCandidate
from .quality import validate_unique_recipe_ids


class SeedCorpusManifest(BaseModel):
    version: str
    status: str
    record_count: int = Field(ge=0)
    recipes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_reviewer: str | None = None
    notes: str = ""


class SeedCorpus(BaseModel):
    manifest: SeedCorpusManifest
    recipes: list[RecipeCandidate]


def load_seed_corpus(recipes_path: Path, manifest_path: Path) -> SeedCorpus:
    raw_recipes = recipes_path.read_bytes()
    manifest = SeedCorpusManifest.model_validate_json(manifest_path.read_text("utf-8"))
    actual_hash = hashlib.sha256(raw_recipes).hexdigest()
    if actual_hash != manifest.recipes_sha256:
        raise ValueError("recipe corpus hash does not match manifest")

    recipes = [RecipeCandidate.model_validate(item) for item in json.loads(raw_recipes)]
    validate_unique_recipe_ids(recipes)
    if len(recipes) != manifest.record_count:
        raise ValueError("recipe corpus count does not match manifest")
    return SeedCorpus(manifest=manifest, recipes=recipes)
