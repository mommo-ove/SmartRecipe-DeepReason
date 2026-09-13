from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.foodtrace.cli import (
    render_demo_trace,
    run_foodtrace_case,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic FoodTrace allergen-recall demo."
    )
    parser.add_argument(
        "--case",
        default="allergen_peanut_001",
        help="Deterministic benchmark case ID.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured result as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        demo = run_foodtrace_case(args.case)
    except ValueError:
        print(
            "FoodTrace demo error: unknown case ID.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    if args.json:
        print(
            json.dumps(
                demo.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_demo_trace(demo))


if __name__ == "__main__":
    main()
