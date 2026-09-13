from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(items: list[dict]) -> str:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    source_dir = ROOT / "benchmark" / "meal_planning" / "graph_relations"
    target_dir = ROOT / "benchmark" / "meal_planning" / "text2cypher"
    corpus_manifest = json.loads(
        (
            ROOT
            / "gustobot"
            / "data"
            / "meal_planning"
            / "external"
            / "manifest.v1.json"
        ).read_text("utf-8")
    )
    source_cases = [
        json.loads(line)
        for line in (source_dir / "retrieval_cases.v1.jsonl")
        .read_text("utf-8")
        .splitlines()
        if line.strip()
    ]
    cases: list[dict] = []
    for index, source in enumerate(source_cases, start=1):
        filters = source["metadata_filters"]
        parameters = {
            "dataset_version": corpus_manifest["version"],
            "ingredient_ids": filters["graph_ingredient_ids"],
            "excluded_ingredient_ids": filters["excluded_graph_ingredient_ids"],
            "meal_type": filters.get("meal_type"),
            "max_minutes": filters.get("max_meal_minutes"),
            "min_protein_g": filters.get("min_protein_g"),
            "top_k": 20,
        }
        cases.append(
            {
                "case_id": f"text2cypher-{index:03d}",
                "split": source["split"],
                "question": source["query"],
                "gold_recipe_ids": source["relevant_recipe_ids"],
                "expected_route": "template",
                "dataset_sha256": corpus_manifest["corpus_sha256"],
                "parameters": parameters,
                "template_request": {
                    "operation": "search_recipes",
                    "dataset_version": corpus_manifest["version"],
                    "required_ingredient_ids": filters["graph_ingredient_ids"],
                    "excluded_ingredient_ids": filters[
                        "excluded_graph_ingredient_ids"
                    ],
                    "meal_type": filters.get("meal_type"),
                    "max_minutes": filters.get("max_meal_minutes"),
                    "min_protein_g": filters.get("min_protein_g"),
                    "top_k": 20,
                },
                "source_case_id": source["case_id"],
            }
        )

    security_statements = [
        ("CREATE (n:Recipe) RETURN n.recipe_id AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("MERGE (n:Recipe {recipe_id:'x'}) RETURN n.recipe_id AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("MATCH (n:Recipe) DELETE n RETURN n.recipe_id AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("MATCH (n:Recipe) DETACH DELETE n RETURN n.recipe_id AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("MATCH (n:Recipe) SET n.name='x' RETURN n.recipe_id AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("MATCH (n:Recipe) REMOVE n.name RETURN n.recipe_id AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("DROP CONSTRAINT recipe_id RETURN 'x' AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("LOAD CSV FROM 'file:///x.csv' AS row RETURN row.id AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("CALL db.labels() YIELD label RETURN label AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("MATCH (n:Recipe) FOREACH (x IN [1] | SET n.x=x) RETURN n.recipe_id AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("COMMIT RETURN 'x' AS recipe_id LIMIT 1", "UNSAFE_CLAUSE"),
        ("MATCH (n:Recipe) RETURN n.recipe_id AS recipe_id; MATCH (m) RETURN m LIMIT 1", "MULTIPLE_STATEMENTS"),
        ("MATCH (n:Recipe) RETURN n.recipe_id AS recipe_id", "MISSING_LIMIT"),
        ("MATCH (n:Recipe) RETURN n.name LIMIT 10", "MISSING_RECIPE_ID"),
        ("MATCH (n:Recipe) RETURN n.recipe_id AS recipe_id LIMIT 1000", "LIMIT_TOO_HIGH"),
    ]
    security_cases = [
        {
            "case_id": f"text2cypher-security-{index:03d}",
            "split": "test",
            "statement": statement,
            "expected_issue_codes": [issue],
        }
        for index, (statement, issue) in enumerate(security_statements, start=1)
    ]

    target_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = _jsonl(cases).encode("utf-8")
    security_payload = _jsonl(security_cases).encode("utf-8")
    (target_dir / "cases.v1.jsonl").write_bytes(cases_payload)
    (target_dir / "security_cases.v1.jsonl").write_bytes(security_payload)
    manifest = {
        "version": "1.0.0-frozen",
        "scope": "Text2Cypher generation with pre-resolved structured entities",
        "case_count": len(cases),
        "development_count": sum(item["split"] == "development" for item in cases),
        "test_count": sum(item["split"] == "test" for item in cases),
        "security_case_count": len(security_cases),
        "cases_sha256": _sha256(cases_payload),
        "security_cases_sha256": _sha256(security_payload),
        "dataset_sha256": corpus_manifest["corpus_sha256"],
        "source_benchmark": "graph_relations/retrieval_cases.v1.jsonl",
        "entity_extraction_in_scope": False,
        "test_cases_allowed_in_few_shot": False,
    }
    (target_dir / "manifest.v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
