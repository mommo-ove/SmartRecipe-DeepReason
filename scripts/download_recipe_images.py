from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gustobot.application.multimodal.catalog import (  # noqa: E402
    CatalogEntry,
    load_catalog,
    validate_catalog,
    validate_image_file,
)


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SmartRecipe-Educational-Image-Corpus/1.0"


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, {})
    raw = value.get("value", "") if isinstance(value, dict) else ""
    return html.unescape(re.sub(r"<[^>]+>", "", str(raw))).strip()


def _license_allowed(license_name: str, allowlist: list[str]) -> bool:
    normalized = license_name.casefold()
    return any(item.casefold() in normalized for item in allowlist)


def build_commons_params(entry: CatalogEntry) -> dict[str, Any]:
    common = {
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 1280,
        "format": "json",
        "formatversion": 2,
        "origin": "*",
    }
    if entry.commons_title:
        return {**common, "titles": entry.commons_title}
    return {
        **common,
        "generator": "search",
        "gsrsearch": entry.commons_query,
        "gsrnamespace": 6,
        "gsrlimit": 20,
    }


def find_duplicate_split_checksums(records: list[dict[str, Any]]) -> list[str]:
    seen: dict[tuple[str, str], str] = {}
    duplicates: list[str] = []
    for record in records:
        recipe_id = str(record["recipe_id"])
        checksum = str(record["sha256"])
        key = (recipe_id, checksum)
        previous_split = seen.get(key)
        if previous_split is not None and previous_split != record["split"]:
            duplicates.append(f"{recipe_id}: {checksum}")
        seen[key] = str(record["split"])
    return list(dict.fromkeys(duplicates))


def select_entries(
    entries: list[CatalogEntry],
    *,
    recipe_ids: set[str] | None = None,
    limit_recipes: int | None = None,
    images_per_split: int = 1,
) -> list[CatalogEntry]:
    selected = entries
    if recipe_ids:
        selected = [entry for entry in selected if entry.recipe_id in recipe_ids]
    elif limit_recipes:
        ordered_ids = list(dict.fromkeys(entry.recipe_id for entry in selected))
        allowed = set(ordered_ids[:limit_recipes])
        selected = [entry for entry in selected if entry.recipe_id in allowed]
    return [entry for entry in selected if entry.ordinal <= images_per_split]


def merge_attribution_records(
    existing: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> tuple[str, str, int]:
        return str(record["recipe_id"]), str(record["split"]), int(record["ordinal"])

    merged = {key(record): record for record in existing}
    for record in replacements:
        merged[key(record)] = record
    return list(merged.values())


def select_pending_entries(
    entries: list[CatalogEntry],
    existing_records: list[dict[str, Any]],
    *,
    project_root: Path,
) -> list[CatalogEntry]:
    existing = {
        (str(record["recipe_id"]), str(record["split"]), int(record["ordinal"])): record
        for record in existing_records
    }
    pending: list[CatalogEntry] = []
    for entry in entries:
        record = existing.get((entry.recipe_id, entry.split, entry.ordinal))
        if record is None:
            pending.append(entry)
            continue
        try:
            local_path = Path(record["local_path"])
            if not local_path.is_absolute():
                local_path = project_root / local_path
            metadata = validate_image_file(local_path)
        except (OSError, KeyError, ValueError):
            pending.append(entry)
            continue
        if metadata.sha256 != record.get("sha256"):
            pending.append(entry)
            continue
        if record.get("source_fingerprint") != entry.source_fingerprint():
            pending.append(entry)
    return pending


def portable_local_path(path: Path, *, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    sleep=time.sleep,
    max_attempts: int = 3,
    **kwargs: Any,
) -> httpx.Response:
    response: httpx.Response | None = None
    last_transport_error: httpx.TransportError | None = None
    for attempt in range(max_attempts):
        try:
            response = client.get(url, **kwargs)
        except httpx.TransportError as exc:
            last_transport_error = exc
            if attempt + 1 < max_attempts:
                sleep(min(2.0**attempt, 10.0))
                continue
            raise
        if response.status_code not in {429, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt + 1 < max_attempts:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0**attempt
            sleep(min(delay, 10.0))
    if response is None and last_transport_error is not None:
        raise last_transport_error
    assert response is not None
    response.raise_for_status()
    return response


def search_commons(client: httpx.Client, entry: CatalogEntry) -> list[dict[str, Any]]:
    response = get_with_retry(
        client,
        COMMONS_API,
        params=build_commons_params(entry),
    )
    pages = response.json().get("query", {}).get("pages", [])
    candidates: list[dict[str, Any]] = []
    for page in sorted(pages, key=lambda value: value.get("index", 10_000)):
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        mime = str(info.get("mime", ""))
        license_name = _metadata_value(info.get("extmetadata", {}), "LicenseShortName")
        if not mime.startswith("image/") or not _license_allowed(license_name, entry.license_allowlist):
            continue
        if int(info.get("width", 0)) < 320 or int(info.get("height", 0)) < 240:
            continue
        candidates.append({"page": page, "info": info, "license": license_name})
    return candidates


def download_entry(
    client: httpx.Client,
    entry: CatalogEntry,
    *,
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    candidates = search_commons(client, entry)
    if len(candidates) <= entry.search_offset:
        raise RuntimeError(f"no licensed image found for {entry.recipe_name}/{entry.split}")
    candidate = candidates[entry.search_offset]
    page = candidate["page"]
    info = candidate["info"]
    download_url = info.get("thumburl") or info["url"]
    image_response = get_with_retry(client, download_url)
    suffix = mimetypes.guess_extension(info.get("mime", "")) or Path(info["url"]).suffix or ".jpg"
    target_dir = output_dir / entry.split / entry.recipe_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{entry.ordinal:02d}{suffix}"
    target.write_bytes(image_response.content)
    file_metadata = validate_image_file(target)
    metadata = info.get("extmetadata", {})
    return {
        "recipe_id": entry.recipe_id,
        "recipe_name": entry.recipe_name,
        "split": entry.split,
        "ordinal": entry.ordinal,
        "source_fingerprint": entry.source_fingerprint(),
        "local_path": portable_local_path(target, project_root=project_root),
        "source_page": info.get("descriptionurl") or info.get("descriptionshorturl"),
        "source_file": page.get("title"),
        "download_url": download_url,
        "author": _metadata_value(metadata, "Artist"),
        "credit": _metadata_value(metadata, "Credit"),
        "license": candidate["license"],
        "license_url": _metadata_value(metadata, "LicenseUrl"),
        **file_metadata.model_dump(),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=project_root / "configs/recipe_image_sources.json")
    parser.add_argument("--output-dir", type=Path, default=project_root / "gustobot/data/recipe_images")
    parser.add_argument("--attribution", type=Path, default=project_root / "configs/recipe_image_attribution.jsonl")
    parser.add_argument("--limit-recipes", type=int)
    parser.add_argument("--recipe-id", action="append", dest="recipe_ids")
    parser.add_argument("--images-per-split", type=int, default=1)
    args = parser.parse_args()

    entries = load_catalog(args.manifest)
    report = validate_catalog(entries, concepts_path=project_root / "gustobot/data/kg_output/concepts.csv")
    if report.errors:
        raise SystemExit("; ".join(report.errors))
    entries = select_entries(
        entries,
        recipe_ids=set(args.recipe_ids or []),
        limit_recipes=args.limit_recipes,
        images_per_split=args.images_per_split,
    )

    existing_records: list[dict[str, Any]] = []
    if args.attribution.exists():
        existing_records = [
            json.loads(line)
            for line in args.attribution.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    entries = select_pending_entries(entries, existing_records, project_root=project_root)

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        for entry in entries:
            try:
                records.append(
                    download_entry(
                        client,
                        entry,
                        output_dir=args.output_dir,
                        project_root=project_root,
                    )
                )
            except Exception as exc:
                failures.append(f"{entry.recipe_id}/{entry.split}: {exc}")

    all_records = merge_attribution_records(existing_records, records)
    for duplicate in find_duplicate_split_checksums(all_records):
        failures.append(f"duplicate catalog/query image: {duplicate}")

    args.attribution.parent.mkdir(parents=True, exist_ok=True)
    args.attribution.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in all_records),
        encoding="utf-8",
    )
    print(json.dumps({"downloaded": len(records), "failed": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
