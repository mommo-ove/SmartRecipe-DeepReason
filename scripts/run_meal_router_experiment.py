from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.agents.utils.llm_factory import get_llm
from gustobot.application.meal_planning.extraction import MealConstraintExtractor
from gustobot.application.meal_planning.gateway import MealRequestGateway
from gustobot.application.meal_planning.planning_entry import MealPlanningEntry
from gustobot.application.meal_planning.routing import (
    BusinessIntent,
    ExecutionPolicy,
    MealIntentRouter,
    RouterPromptVersion,
)
from gustobot.application.meal_planning.routing_benchmark import (
    RouteBenchmarkCase,
    load_frozen_route_benchmark,
    load_route_benchmark,
)
from gustobot.application.meal_planning.routing_experiment import (
    RouteObservation,
    summarize_route_experiment,
)
from gustobot.config.settings import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", choices=("fallback", "llm"), required=True)
    parser.add_argument(
        "--dataset",
        choices=("development", "frozen"),
        default="frozen",
        help="Use development for R1-R3 iteration; frozen only for baseline/final.",
    )
    parser.add_argument(
        "--prompt-version",
        choices=tuple(item.value for item in RouterPromptVersion),
        default=RouterPromptVersion.R0_ZERO_SHOT.value,
    )
    parser.add_argument("--mined-errors", type=Path, default=None)
    parser.add_argument("--input-cost-per-million", type=float, default=0)
    parser.add_argument("--output-cost-per-million", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_version = RouterPromptVersion(args.prompt_version)
    cases, dataset_metadata = _load_dataset(args, prompt_version)
    mined_examples = _load_mined_examples(args.mined_errors)

    if args.router == "llm":
        if not settings.LLM_API_KEY:
            raise SystemExit("LLM_API_KEY is required for a real LLM routing experiment")
        model = get_llm(
            tags=["meal-router-evaluation", prompt_version.value],
            structured_output=True,
        )
        extractor = MealConstraintExtractor(model)
        planning_entry = MealPlanningEntry(extractor)
        model_name = settings.LLM_MODEL
    else:
        model = None
        extractor = None
        planning_entry = None
        model_name = "deterministic_fallback"

    router = MealIntentRouter(
        model,
        prompt_version=prompt_version,
        mined_examples=mined_examples,
    )
    gateway = MealRequestGateway(router, planning_entry)
    observations: list[RouteObservation] = []
    for case in cases:
        if extractor is not None:
            extractor.last_input_tokens = 0
            extractor.last_output_tokens = 0
        started = perf_counter()
        decision = gateway.decide(case.query)
        latency_ms = (perf_counter() - started) * 1000
        observations.append(
            RouteObservation(
                case_id=case.case_id,
                predicted_intent=decision.route.business_intent,
                predicted_policy=decision.policy,
                confidence=decision.route.confidence,
                latency_ms=latency_ms,
                fallback_used=router.last_fallback_used,
                input_tokens=(
                    router.last_input_tokens
                    + (extractor.last_input_tokens if extractor is not None else 0)
                ),
                output_tokens=(
                    router.last_output_tokens
                    + (extractor.last_output_tokens if extractor is not None else 0)
                ),
            )
        )

    result = summarize_route_experiment(
        cases,
        observations,
        router_name=args.router,
        prompt_version=prompt_version.value,
        model_name=model_name,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    payload = result.model_dump(mode="json")
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **dataset_metadata,
            "pricing": {
                "input_cost_per_million": args.input_cost_per_million,
                "output_cost_per_million": args.output_cost_per_million,
                "source": "user_supplied_cli_values",
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_dataset(
    args: argparse.Namespace,
    prompt_version: RouterPromptVersion,
) -> tuple[list[RouteBenchmarkCase], dict[str, object]]:
    benchmark_root = ROOT / "benchmark" / "meal_planning" / "routing"
    if args.dataset == "frozen":
        if args.router == "llm" and prompt_version in {
            RouterPromptVersion.R1_FEW_SHOT,
            RouterPromptVersion.R2_HARD_NEGATIVE,
        }:
            raise SystemExit(
                "The frozen set cannot be used for intermediate prompt iteration; "
                "use --dataset development for R1/R2 and reserve frozen for R0/final R3."
            )
        cases_path = args.cases or benchmark_root / "frozen" / "cases.v1.jsonl"
        manifest_path = (
            args.manifest or benchmark_root / "frozen" / "manifest.v1.json"
        )
        benchmark = load_frozen_route_benchmark(cases_path, manifest_path)
        return benchmark.cases, {
            "dataset_version": benchmark.manifest.version,
            "dataset_status": benchmark.manifest.status,
            "dataset_sha256": benchmark.manifest.cases_sha256,
            "resume_metric_eligible": benchmark.manifest.resume_metric_eligible,
        }

    cases_path = args.cases or benchmark_root / "cases.v2.jsonl"
    raw = cases_path.read_bytes()
    # The former v2 development/test partitions predate the independent frozen
    # set. Both are now safe to use for prompt iteration; neither may be used as
    # a final score.
    return load_route_benchmark(cases_path), {
        "dataset_version": "development-v2",
        "dataset_status": "iteration_development_only",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "resume_metric_eligible": False,
    }


def _load_mined_examples(
    path: Path | None,
) -> list[tuple[str, BusinessIntent, set[str]]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for error in payload.get("errors", []):
        intent = BusinessIntent(error["expected_intent"])
        policy = ExecutionPolicy(error["expected_policy"])
        examples.append(
            (
                str(error["query"]),
                intent,
                _expected_capabilities(intent, policy),
            )
        )
    return examples


def _expected_capabilities(
    intent: BusinessIntent,
    policy: ExecutionPolicy,
) -> set[str]:
    if intent is BusinessIntent.RECIPE_LOOKUP:
        capabilities = {"recipe_retrieval"}
        if policy is ExecutionPolicy.WORKFLOW:
            capabilities.add("nutrition_query")
        return capabilities
    if intent is BusinessIntent.DATA_QUERY:
        return {"nutrition_query"}
    if intent in {BusinessIntent.MEAL_PLAN, BusinessIntent.PLAN_EDIT}:
        return {"constraint_solving"}
    return set()


if __name__ == "__main__":
    main()
