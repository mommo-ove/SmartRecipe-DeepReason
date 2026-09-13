import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_meal_router_experiment.py"


def test_frozen_fallback_experiment_runs_and_persists_metrics(tmp_path):
    output = tmp_path / "fallback.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--router",
            "fallback",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["total"] == 60
    assert payload["router_name"] == "fallback"
    assert payload["dataset_status"] == "frozen_pending_human_review"
    assert payload["resume_metric_eligible"] is False
    assert payload["input_tokens"] == 0
    assert payload["estimated_cost"] == 0


def test_llm_experiment_refuses_to_fake_results_without_an_api_key(tmp_path):
    environment = os.environ.copy()
    environment["LLM_API_KEY"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--router",
            "llm",
            "--prompt-version",
            "r0_zero_shot",
            "--output",
            str(tmp_path / "llm.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode != 0
    assert "LLM_API_KEY" in completed.stderr


def test_development_experiment_uses_only_the_development_split(tmp_path):
    output = tmp_path / "development.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--router",
            "fallback",
            "--dataset",
            "development",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["total"] == 43
    assert payload["dataset_status"] == "iteration_development_only"
    assert payload["resume_metric_eligible"] is False


def test_frozen_set_rejects_intermediate_prompt_iterations(tmp_path):
    environment = os.environ.copy()
    environment["LLM_API_KEY"] = "test-key"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--router",
            "llm",
            "--dataset",
            "frozen",
            "--prompt-version",
            "r2_hard_negative",
            "--output",
            str(tmp_path / "invalid.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode != 0
    assert "frozen set" in completed.stderr
