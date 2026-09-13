from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def main() -> None:
    corpus_path = (
        ROOT / "gustobot" / "data" / "meal_planning" / "external"
        / "retrieval_corpus.v1.jsonl"
    )
    benchmark_dir = ROOT / "benchmark" / "meal_planning" / "curated"
    cases_path = benchmark_dir / "retrieval_cases.v1.jsonl"
    output_path = benchmark_dir / "review_sheet.v1.csv"

    documents = {item["recipe_id"]: item for item in _read_jsonl(corpus_path)}
    rows = []
    for case in _read_jsonl(cases_path):
        gold_ids = case["relevant_recipe_ids"]
        gold_documents = [documents[recipe_id] for recipe_id in gold_ids]
        rows.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "relevant_recipe_ids": "|".join(gold_ids),
                "gold_recipe_name": "|".join(doc["name"] for doc in gold_documents),
                "gold_ingredients": "|".join(
                    "; ".join(doc["ingredients"][:8]) for doc in gold_documents
                ),
                "metadata_filters": json.dumps(
                    case.get("metadata_filters", {}), ensure_ascii=False
                ),
                "review_status": "pending",
                "review_notes": "",
            }
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported {len(rows)} cases to {output_path}")


if __name__ == "__main__":
    main()
