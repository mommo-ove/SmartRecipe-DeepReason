import json
from pathlib import Path

from gustobot.application.deepreason.benchmark import run_routing_benchmark


def test_offline_routing_benchmark_is_reproducible():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "configs" / "deepreason_benchmark_cases.json").read_text(encoding="utf-8"))

    report = run_routing_benchmark(cases)

    assert report["total"] == 30
    assert report["exact_matches"] >= 29
    assert report["accuracy"] >= 0.966
    assert report["failures"] == []

