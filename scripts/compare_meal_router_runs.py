from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    dataset_hashes = {run.get("dataset_sha256") for run in runs}
    if len(dataset_hashes) != 1:
        raise SystemExit("Cannot compare runs from different dataset hashes")
    baseline = runs[0]
    headers = (
        "version",
        "macro_f1",
        "delta_f1",
        "route_error",
        "delta_error",
        "unsafe",
        "clarify",
        "fallback",
        "p95_ms",
        "tokens",
        "cost",
    )
    rows = []
    for run in runs:
        rows.append(
            (
                str(run["prompt_version"]),
                f'{run["intent_macro_f1"]:.4f}',
                f'{run["intent_macro_f1"] - baseline["intent_macro_f1"]:+.4f}',
                f'{run["policy_error_rate"]:.4f}',
                f'{run["policy_error_rate"] - baseline["policy_error_rate"]:+.4f}',
                f'{run["unsafe_route_rate"]:.4f}',
                f'{run["clarification_rate"]:.4f}',
                f'{run["fallback_rate"]:.4f}',
                f'{run["p95_latency_ms"]:.1f}',
                str(run["input_tokens"] + run["output_tokens"]),
                f'{run["estimated_cost"]:.6f}',
            )
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


if __name__ == "__main__":
    main()
