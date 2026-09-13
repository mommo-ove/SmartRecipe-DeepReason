from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Awaitable, Sequence, TypeVar

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.text2cypher.benchmark import (
    Text2CypherBenchmarkCase,
    Text2CypherBenchmarkResult,
    Text2CypherSecurityCase,
    build_few_shot_examples,
    evaluate_security_cases,
    evaluate_text2cypher_results,
    load_jsonl_models,
)
from gustobot.application.meal_planning.text2cypher.generator import (
    Text2CypherGenerator,
    create_deepseek_text2cypher_generator,
)
from gustobot.application.meal_planning.text2cypher.neo4j_adapter import (
    Neo4jCypherAdapter,
)
from gustobot.application.meal_planning.text2cypher.schema import (
    GraphSchemaSnapshot,
    Neo4jSchemaLoader,
)
from gustobot.application.meal_planning.text2cypher.service import Text2CypherService
from gustobot.application.meal_planning.text2cypher.templates import TemplateRequest
from gustobot.config.settings import settings


PROMPT_VERSION = "meal-text2cypher-v1"
ResultT = TypeVar("ResultT")


async def await_case_result(
    operation: Awaitable[ResultT], *, timeout_seconds: float
) -> ResultT:
    """Bound one remote case so a long-tail API call cannot stall the run."""
    return await asyncio.wait_for(operation, timeout=timeout_seconds)


@dataclass(frozen=True)
class ConfigurationSpec:
    include_schema: bool
    include_examples: bool
    template_first: bool


CONFIGURATION_SPECS = {
    "deepseek_direct": ConfigurationSpec(False, False, False),
    "deepseek_schema": ConfigurationSpec(True, False, False),
    "deepseek_schema_fewshot": ConfigurationSpec(True, True, False),
    "template_first_validated": ConfigurationSpec(True, False, True),
}


class _UnavailableModel:
    async def ainvoke(self, prompt: str):
        raise RuntimeError("template-only evaluation unexpectedly invoked an LLM")


class _PromptAblationGenerator:
    def __init__(
        self,
        delegate: Text2CypherGenerator,
        spec: ConfigurationSpec,
    ) -> None:
        self._delegate = delegate
        self._spec = spec

    @staticmethod
    def _empty_schema() -> GraphSchemaSnapshot:
        return GraphSchemaSnapshot(node_properties={}, relationships=set())

    async def generate(
        self,
        *,
        question: str,
        schema: GraphSchemaSnapshot,
        examples: Sequence[tuple[str, str]],
        parameters: dict,
    ):
        return await self._delegate.generate(
            question=question,
            schema=schema if self._spec.include_schema else self._empty_schema(),
            examples=examples if self._spec.include_examples else (),
            parameters=parameters,
        )

    async def repair(self, *, question, candidate, issues, schema):
        return await self._delegate.repair(
            question=question,
            candidate=candidate,
            issues=issues,
            schema=schema if self._spec.include_schema else self._empty_schema(),
        )


def case_template_request(
    case: Text2CypherBenchmarkCase,
    spec: ConfigurationSpec,
) -> TemplateRequest | None:
    if not spec.template_first:
        return None
    return TemplateRequest.model_validate(case.template_request)


_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(text)}


def case_examples(
    current: Text2CypherBenchmarkCase,
    all_cases: list[Text2CypherBenchmarkCase],
    spec: ConfigurationSpec,
    *,
    limit: int = 3,
) -> list[tuple[str, str]]:
    if not spec.include_examples:
        return []
    current_tokens = _tokens(current.question)
    development = [
        case
        for case in all_cases
        if case.split == "development" and case.case_id != current.case_id
    ]
    ranked = sorted(
        development,
        key=lambda case: (
            -len(current_tokens & _tokens(case.question)),
            case.case_id,
        ),
    )[:limit]
    return build_few_shot_examples(ranked)


async def _run_configuration(
    *,
    name: str,
    spec: ConfigurationSpec,
    cases: list[Text2CypherBenchmarkCase],
    all_cases: list[Text2CypherBenchmarkCase],
    adapter: Neo4jCypherAdapter,
    schema_loader: Neo4jSchemaLoader,
    deepseek_generator: Text2CypherGenerator | None,
    case_timeout_seconds: float,
) -> tuple[dict, dict[str, dict]]:
    delegate = (
        Text2CypherGenerator(_UnavailableModel())
        if spec.template_first
        else deepseek_generator
    )
    if delegate is None:
        raise RuntimeError(f"{name} requires a configured DeepSeek model")
    service = Text2CypherService(
        generator=_PromptAblationGenerator(delegate, spec),
        adapter=adapter,
        schema_loader=schema_loader,
    )
    measured: dict[str, Text2CypherBenchmarkResult] = {}
    raw: dict[str, dict] = {}
    for case in cases:
        started = perf_counter()
        try:
            result = await await_case_result(
                service.run(
                    question=case.question,
                    template_request=case_template_request(case, spec),
                    parameters=case.parameters,
                    examples=case_examples(case, all_cases, spec),
                ),
                timeout_seconds=case_timeout_seconds,
            )
            run = Text2CypherBenchmarkResult(
                result=result,
                latency_ms=(perf_counter() - started) * 1000,
            )
        except Exception as error:
            run = Text2CypherBenchmarkResult(
                latency_ms=(perf_counter() - started) * 1000,
                error=str(error),
            )
        measured[case.case_id] = run
        raw[case.case_id] = {
            "question": case.question,
            "gold_recipe_ids": case.gold_recipe_ids,
            "expected_route": case.expected_route,
            **run.model_dump(mode="json"),
        }
    metrics = evaluate_text2cypher_results(
        configuration=name,
        cases=cases,
        results=measured,
    )
    return metrics.model_dump(mode="json"), raw


async def run(args: argparse.Namespace) -> dict:
    benchmark_dir = ROOT / "benchmark" / "meal_planning" / "text2cypher"
    all_cases = load_jsonl_models(
        benchmark_dir / "cases.v1.jsonl", Text2CypherBenchmarkCase
    )
    cases = [case for case in all_cases if case.split == args.split]
    security_cases = load_jsonl_models(
        benchmark_dir / "security_cases.v1.jsonl", Text2CypherSecurityCase
    )
    manifest = json.loads((benchmark_dir / "manifest.v1.json").read_text("utf-8"))
    requested = list(dict.fromkeys(args.configurations))
    needs_model = any(not CONFIGURATION_SPECS[name].template_first for name in requested)
    deepseek_generator = (
        create_deepseek_text2cypher_generator() if needs_model else None
    )

    driver = GraphDatabase.driver(args.neo4j_url, auth=None)
    try:
        driver.verify_connectivity()
        adapter = Neo4jCypherAdapter(driver, database=args.database)
        schema_loader = Neo4jSchemaLoader(driver, database=args.database)
        results: dict[str, dict] = {}
        raw_runs: dict[str, dict[str, dict]] = {}
        for name in requested:
            metrics, raw = await _run_configuration(
                name=name,
                spec=CONFIGURATION_SPECS[name],
                cases=cases,
                all_cases=all_cases,
                adapter=adapter,
                schema_loader=schema_loader,
                deepseek_generator=deepseek_generator,
                case_timeout_seconds=args.case_timeout_seconds,
            )
            results[name] = metrics
            raw_runs[name] = raw
    finally:
        driver.close()

    security = evaluate_security_cases(
        [case for case in security_cases if case.split == "test"]
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Text2Cypher generation with pre-resolved structured entities",
        "split": args.split,
        "model": settings.LLM_MODEL if needs_model else "not-invoked",
        "prompt_version": PROMPT_VERSION,
        "benchmark": manifest,
        "configurations": results,
        "security": security.model_dump(mode="json"),
        "raw_runs": raw_runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(CONFIGURATION_SPECS),
        default=tuple(CONFIGURATION_SPECS),
    )
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument("--neo4j-url", default="bolt://localhost:17687")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=60.0,
        help="Maximum wall time for one benchmark case (default: 60 seconds)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "benchmark"
        / "meal_planning"
        / "text2cypher"
        / "runs"
        / "test.v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
