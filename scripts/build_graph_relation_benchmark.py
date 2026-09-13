from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.external_corpus import load_external_corpus
from gustobot.application.meal_planning.graph_benchmark import (
    mine_relation_benchmark_cases,
)


def main() -> None:
    corpus_dir = ROOT / "gustobot" / "data" / "meal_planning" / "external"
    output_dir = ROOT / "benchmark" / "meal_planning" / "graph_relations"
    corpus = load_external_corpus(
        corpus_dir / "retrieval_corpus.v1.jsonl",
        corpus_dir / "manifest.v1.json",
    )
    cases = mine_relation_benchmark_cases(corpus.documents)
    raw = "".join(
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        + "\n"
        for case in cases
    ).encode("utf-8")
    manifest = {
        "version": "1.0.0-frozen",
        "status": "structured_relation_test",
        "case_count": len(cases),
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "human_reviewed_count": 0,
        "resume_metric_eligible": True,
        "notes": (
            "Gold recipe IDs are exhaustively derived from frozen structured "
            "ingredient fields. This benchmark measures include/exclude relation "
            "retrieval, not end-to-end natural-language entity extraction."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_cases.v1.jsonl").write_bytes(raw)
    (output_dir / "manifest.v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
