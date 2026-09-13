from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.external_corpus import load_external_corpus
from gustobot.application.meal_planning.graph_retrieval import (
    Neo4jMealGraphStore,
    build_meal_graph_projection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-url", default="bolt://localhost:17687")
    parser.add_argument("--database", default="neo4j")
    args = parser.parse_args()
    corpus_dir = ROOT / "gustobot" / "data" / "meal_planning" / "external"
    corpus = load_external_corpus(
        corpus_dir / "retrieval_corpus.v1.jsonl",
        corpus_dir / "manifest.v1.json",
    )
    projection = build_meal_graph_projection(
        corpus.documents,
        dataset_version=corpus.manifest.version,
    )
    with GraphDatabase.driver(args.neo4j_url, auth=None) as driver:
        driver.verify_connectivity()
        store = Neo4jMealGraphStore(driver, database=args.database)
        store.replace_projection(projection)
        counts, _, _ = driver.execute_query(
            """
            MATCH (recipe:Recipe {dataset_version: $version})
            OPTIONAL MATCH (recipe)-[edge:HAS_INGREDIENT]->(ingredient:Ingredient)
            RETURN count(DISTINCT recipe) AS recipes,
                   count(DISTINCT ingredient) AS ingredients,
                   count(edge) AS has_ingredient_edges
            """,
            version=corpus.manifest.version,
            database_=args.database,
        )
    print(
        json.dumps(
            {
                "dataset_version": corpus.manifest.version,
                **dict(counts[0]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
