from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.routing import (
    MealIntentRouter,
    decide_execution_policy,
)
from gustobot.application.meal_planning.routing_benchmark import (
    evaluate_route_predictions,
    load_route_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "benchmark" / "meal_planning" / "routing" / "cases.v2.jsonl",
    )
    parser.add_argument(
        "--split",
        choices=("development", "test"),
        default="test",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=None,
        help="optional JSONL path for reviewable routing errors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_route_benchmark(args.cases, split=args.split)
    router = MealIntentRouter(model=None)
    predictions = {}
    for case in cases:
        route = router.route(case.query)
        predictions[case.case_id] = (
            route.business_intent,
            decide_execution_policy(route),
            route.confidence,
        )
    result = evaluate_route_predictions(cases, predictions)
    payload = asdict(result)
    payload["split"] = args.split
    payload["router"] = "deterministic_fallback_baseline"
    payload["human_review_complete"] = all(
        case.review_status == "reviewed" for case in cases
    )
    payload["metric_status"] = "engineering_smoke_only"
    payload["resume_metric_eligible"] = False
    if args.error_output is not None:
        args.error_output.parent.mkdir(parents=True, exist_ok=True)
        args.error_output.write_text(
            "\n".join(
                json.dumps(asdict(error), ensure_ascii=False, default=str)
                for error in result.errors
            )
            + ("\n" if result.errors else ""),
            encoding="utf-8",
        )
        payload["error_output"] = str(args.error_output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
