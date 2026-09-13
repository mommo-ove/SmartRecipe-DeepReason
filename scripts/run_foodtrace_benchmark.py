from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.foodtrace.benchmark import (
    load_foodtrace_benchmark_cases,
    run_foodtrace_benchmark,
    write_benchmark_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic FoodTrace safety benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark" / "foodtrace" / "results" / "latest.json",
        help="Raw JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_foodtrace_benchmark(load_foodtrace_benchmark_cases())
    write_benchmark_report(report, args.output)
    print(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
