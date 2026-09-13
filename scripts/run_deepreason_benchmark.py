from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.deepreason.benchmark import run_routing_benchmark


def main() -> None:
    cases_path = ROOT / "configs" / "deepreason_benchmark_cases.json"
    report_path = ROOT / "tests" / "deepreason_benchmark_report.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    report = run_routing_benchmark(cases)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
