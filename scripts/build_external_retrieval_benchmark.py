from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.benchmark import RetrievalBenchmarkCase
from gustobot.application.meal_planning.external_corpus import load_external_corpus


def main() -> None:
    corpus_dir = ROOT / "gustobot" / "data" / "meal_planning" / "external"
    corpus = load_external_corpus(
        corpus_dir / "retrieval_corpus.v1.jsonl",
        corpus_dir / "manifest.v1.json",
    )
    cases = []
    for index, document in enumerate(corpus.documents[:30], start=1):
        slot = sorted(document.meal_types)[0]
        query_terms = [*document.ingredients[:2], slot]
        case = RetrievalBenchmarkCase(
            case_id=f"external-known-{index:03d}",
            query=" ".join(query_terms),
            relevant_recipe_ids={document.recipe_id},
            metadata_filters={
                "meal_type": slot,
                "max_meal_minutes": document.total_minutes,
            },
            split="development",
            authoring="synthetic_known_item",
        )
        cases.append(case)

    output_dir = ROOT / "benchmark" / "meal_planning" / "external"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "retrieval_cases.v1.jsonl"
    raw = "".join(
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for case in cases
    ).encode("utf-8")
    cases_path.write_bytes(raw)
    manifest = {
        "version": "1.0.0-dev",
        "status": "synthetic_known_item",
        "case_count": len(cases),
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "human_reviewed_count": 0,
        "resume_metric_eligible": False,
        "notes": (
            "Automatically generated known-item queries. Useful for regression "
            "testing, not for recommendation-quality claims."
        ),
    }
    (output_dir / "manifest.v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
