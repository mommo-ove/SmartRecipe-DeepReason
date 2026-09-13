import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_meal_router_runs.py"


def test_compare_script_reports_delta_from_first_run(tmp_path):
    baseline = tmp_path / "r0.json"
    improved = tmp_path / "r1.json"
    baseline.write_text(json.dumps(_result("r0_zero_shot", 0.6, 0.3)), encoding="utf-8")
    improved.write_text(json.dumps(_result("r1_few_shot", 0.75, 0.2)), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(baseline), str(improved)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "r0_zero_shot" in completed.stdout
    assert "r1_few_shot" in completed.stdout
    assert "+0.1500" in completed.stdout
    assert "-0.1000" in completed.stdout


def _result(version: str, macro_f1: float, error_rate: float) -> dict:
    return {
        "prompt_version": version,
        "intent_macro_f1": macro_f1,
        "policy_error_rate": error_rate,
        "unsafe_route_rate": 0.01,
        "clarification_rate": 0.2,
        "fallback_rate": 0.0,
        "p95_latency_ms": 300,
        "input_tokens": 100,
        "output_tokens": 20,
        "estimated_cost": 0.001,
        "dataset_sha256": "abc",
    }
