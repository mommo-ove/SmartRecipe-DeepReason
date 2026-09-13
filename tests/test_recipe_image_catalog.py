from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from gustobot.application.multimodal.catalog import (
    CatalogEntry,
    load_catalog,
    validate_catalog,
    validate_image_file,
)
from scripts.download_recipe_images import (
    build_commons_params,
    find_duplicate_split_checksums,
    get_with_retry,
    merge_attribution_records,
    portable_local_path,
    select_entries,
    select_pending_entries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_committed_manifest_maps_to_existing_recipe_ids() -> None:
    entries = load_catalog(PROJECT_ROOT / "configs" / "recipe_image_sources.json")

    report = validate_catalog(
        entries,
        concepts_path=PROJECT_ROOT / "gustobot" / "data" / "kg_output" / "concepts.csv",
    )

    assert report.recipe_count == 10
    assert report.entry_count == 20
    assert report.errors == []
    assert {entry.split for entry in entries} == {"catalog", "query"}


def test_validate_catalog_rejects_duplicate_recipe_split_ordinal(tmp_path: Path) -> None:
    concepts = tmp_path / "concepts.csv"
    concepts.write_text(
        "concept_id,concept_type,name\n201000001,Recipe,recipe-a\n",
        encoding="utf-8",
    )
    duplicate = CatalogEntry(
        recipe_id="201000001",
        recipe_name="recipe-a",
        commons_query="recipe a",
        split="catalog",
        ordinal=1,
        license_allowlist=["CC BY-SA 4.0"],
    )

    report = validate_catalog([duplicate, duplicate.model_copy()], concepts_path=concepts)

    assert any("duplicate" in error for error in report.errors)


def test_validate_catalog_rejects_unknown_recipe_id(tmp_path: Path) -> None:
    concepts = tmp_path / "concepts.csv"
    concepts.write_text(
        "concept_id,concept_type,name\n201000001,Recipe,recipe-a\n",
        encoding="utf-8",
    )
    entry = CatalogEntry(
        recipe_id="999999999",
        recipe_name="missing",
        commons_query="missing dish",
        split="query",
        ordinal=1,
        license_allowlist=["Public domain"],
    )

    report = validate_catalog([entry], concepts_path=concepts)

    assert report.errors == ["unknown recipe_id: 999999999"]


def test_validate_image_file_accepts_decodable_rgb_image(tmp_path: Path) -> None:
    image_path = tmp_path / "dish.png"
    Image.new("RGB", (32, 24), color=(220, 30, 20)).save(image_path)

    metadata = validate_image_file(image_path)

    assert metadata.width == 32
    assert metadata.height == 24
    assert metadata.format == "PNG"
    assert len(metadata.sha256) == 64


def test_load_catalog_rejects_unexpected_manifest_shape(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": "not-a-list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="entries must be a list"):
        load_catalog(manifest)


def test_exact_commons_title_bypasses_noisy_full_text_search() -> None:
    entry = CatalogEntry(
        recipe_id="201000001",
        recipe_name="recipe-a",
        commons_query="ambiguous words",
        commons_title="File:Curated dish.jpg",
        split="catalog",
        ordinal=1,
        license_allowlist=["CC BY-SA"],
    )

    params = build_commons_params(entry)

    assert params["titles"] == "File:Curated dish.jpg"
    assert params["iiurlwidth"] == 1280
    assert "generator" not in params


def test_duplicate_image_across_catalog_and_query_is_reported() -> None:
    records = [
        {"recipe_id": "201000001", "split": "catalog", "sha256": "same"},
        {"recipe_id": "201000001", "split": "query", "sha256": "same"},
    ]

    assert find_duplicate_split_checksums(records) == ["201000001: same"]


def test_download_retries_rate_limit_using_retry_after() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, request=request)
        return httpx.Response(200, content=b"ok", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = get_with_retry(client, "https://example.test/image.jpg", sleep=sleeps.append)

    assert response.content == b"ok"
    assert attempts == 2
    assert sleeps == [2.0]


def test_download_retries_transient_transport_error() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadError("server disconnected", request=request)
        return httpx.Response(200, content=b"ok", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = get_with_retry(client, "https://example.test/image.jpg", sleep=sleeps.append)

    assert response.content == b"ok"
    assert attempts == 2
    assert sleeps == [1.0]


def test_select_entries_can_retry_only_failed_recipe_ids() -> None:
    entries = [
        CatalogEntry(
            recipe_id=recipe_id,
            recipe_name=recipe_id,
            commons_query=recipe_id,
            split="catalog",
            ordinal=1,
            license_allowlist=["CC BY"],
        )
        for recipe_id in ("1", "2", "3")
    ]

    selected = select_entries(entries, recipe_ids={"2", "3"})

    assert [entry.recipe_id for entry in selected] == ["2", "3"]


def test_merge_attribution_preserves_unmodified_downloads() -> None:
    existing = [
        {"recipe_id": "1", "split": "catalog", "ordinal": 1, "sha256": "old-1"},
        {"recipe_id": "2", "split": "catalog", "ordinal": 1, "sha256": "old-2"},
    ]
    replacement = [
        {"recipe_id": "2", "split": "catalog", "ordinal": 1, "sha256": "new-2"}
    ]

    merged = merge_attribution_records(existing, replacement)

    assert [record["sha256"] for record in merged] == ["old-1", "new-2"]


def test_pending_selection_reuses_existing_checksum_verified_image(tmp_path: Path) -> None:
    image_path = tmp_path / "existing.png"
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(image_path)
    image_metadata = validate_image_file(image_path)
    entry = CatalogEntry(
        recipe_id="1",
        recipe_name="one",
        commons_query="one",
        split="catalog",
        ordinal=1,
        license_allowlist=["CC BY"],
    )
    existing = [{
        "recipe_id": "1",
        "split": "catalog",
        "ordinal": 1,
        "local_path": "existing.png",
        "sha256": image_metadata.sha256,
        "source_fingerprint": entry.source_fingerprint(),
    }]

    pending = select_pending_entries([entry], existing, project_root=tmp_path)

    assert pending == []


def test_pending_selection_invalidates_cache_when_source_configuration_changes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "existing.png"
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(image_path)
    image_metadata = validate_image_file(image_path)
    old_entry = CatalogEntry(
        recipe_id="1",
        recipe_name="one",
        commons_query="old search",
        commons_title="File:Old dish.jpg",
        split="catalog",
        ordinal=1,
        license_allowlist=["CC BY"],
    )
    new_entry = old_entry.model_copy(
        update={"commons_query": "new search", "commons_title": "File:New dish.jpg"}
    )
    existing = [{
        "recipe_id": "1",
        "split": "catalog",
        "ordinal": 1,
        "local_path": "existing.png",
        "sha256": image_metadata.sha256,
        "source_fingerprint": old_entry.source_fingerprint(),
    }]

    pending = select_pending_entries([new_entry], existing, project_root=tmp_path)

    assert pending == [new_entry]


def test_attribution_path_is_relative_to_project_root(tmp_path: Path) -> None:
    image_path = tmp_path / "gustobot" / "data" / "recipe_images" / "dish.jpg"

    assert portable_local_path(image_path, project_root=tmp_path) == (
        "gustobot/data/recipe_images/dish.jpg"
    )
