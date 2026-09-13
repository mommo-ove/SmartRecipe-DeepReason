from __future__ import annotations

from typing import Any

from .planning import HeuristicPlanner


def run_routing_benchmark(cases: list[dict[str, Any]]) -> dict[str, Any]:
    planner = HeuristicPlanner()
    failures: list[dict[str, Any]] = []
    for case in cases:
        plan = planner.plan(
            str(case["query"]),
            image_path=case.get("image_path"),
            file_path=case.get("file_path"),
        )
        actual = [task.domain.value for task in plan.tasks]
        expected = [str(domain) for domain in case["domains"]]
        if set(actual) != set(expected):
            failures.append({"query": case["query"], "expected": expected, "actual": actual})
    total = len(cases)
    exact_matches = total - len(failures)
    return {
        "metric": "exact domain-set match",
        "total": total,
        "exact_matches": exact_matches,
        "accuracy": round(exact_matches / total, 4) if total else 0.0,
        "failures": failures,
    }

