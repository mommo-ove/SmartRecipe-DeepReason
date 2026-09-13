from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import sleep
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.external_corpus import (
    ExternalRecipeNormalizer,
)


DATASET = "untitledwebsite123/food-recipes"
VIEWER_URL = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a traceable Food.com sample corpus")
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--page-size", type=int, default=100)
    return parser.parse_args()


def fetch_page(offset: int, length: int, cache_dir: Path) -> list[dict]:
    cache_path = cache_dir / f"rows-{offset:07d}-{length}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text("utf-8"))

    query = urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = Request(
                f"{VIEWER_URL}?{query}",
                headers={"User-Agent": "SmartRecipe-evaluation-corpus/1.0"},
            )
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            rows = [item["row"] for item in payload.get("rows", [])]
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )
            return rows
        except HTTPError as error:
            last_error = error
            if error.code != 429:
                break
            retry_after = int(error.headers.get("Retry-After", "0") or 0)
            sleep(min(15, max(retry_after, 3 * (attempt + 1))))
        except Exception as error:
            last_error = error
            sleep(min(10, 2 * (attempt + 1)))
    raise RuntimeError(f"dataset viewer failed at offset {offset}") from last_error


def main() -> None:
    args = parse_args()
    normalizer = ExternalRecipeNormalizer()
    documents = []
    seen = set()
    scanned = 0
    cache_dir = ROOT / ".cache" / "meal_planning_hf"
    for offset in range(0, args.max_scan, args.page_size):
        rows = fetch_page(offset, args.page_size, cache_dir)
        if not rows:
            break
        scanned += len(rows)
        for row in rows:
            document = normalizer.normalize(row)
            if document is None or document.recipe_id in seen:
                continue
            documents.append(document)
            seen.add(document.recipe_id)
            if len(documents) >= args.target:
                break
        if len(documents) >= args.target:
            break

    if len(documents) < args.target:
        raise RuntimeError(
            f"only collected {len(documents)} records after scanning {scanned} rows"
        )

    output_dir = ROOT / "gustobot" / "data" / "meal_planning" / "external"
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "retrieval_corpus.v1.jsonl"
    raw = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for item in documents
    ).encode("utf-8")
    corpus_path.write_bytes(raw)
    manifest = {
        "version": "1.0.0-dev",
        "status": "external_retrieval_fixture",
        "source_dataset": DATASET,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "record_count": len(documents),
        "source_rows_scanned": scanned,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "capabilities": {
            "retrieval_ready": len(documents),
            "nutrition_ready": len(documents),
            "time_ready": len(documents),
            "allergen_ready": 0,
            "budget_ready": 0,
            "full_planning_ready": 0,
        },
        "resume_metric_eligible": False,
        "notes": (
            "Public-source development corpus. Allergen and cost facts require a "
            "separate verified enrichment step before full meal-plan solving."
        ),
    }
    (output_dir / "manifest.v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
