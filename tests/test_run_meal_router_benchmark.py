from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def test_router_benchmark_can_export_a_reviewable_error_file(tmp_path):
    error_output = tmp_path / "errors.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_meal_router_benchmark.py"),
            "--split",
            "development",
            "--error-output",
            str(error_output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert error_output.exists()
    assert error_output.read_text(encoding="utf-8") == ""
    assert '"resume_metric_eligible": false' in completed.stdout
