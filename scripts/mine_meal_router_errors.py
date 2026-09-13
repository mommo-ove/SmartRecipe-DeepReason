from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.meal_planning.routing_error_mining import (
    BgeM3QueryEncoder,
    select_diverse_route_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--duplicate-similarity", type=float, default=0.92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = args.input.read_bytes()
    source = json.loads(raw)
    errors = list(source.get("errors", []))
    encoder = BgeM3QueryEncoder(args.model_path, device=args.device)
    selected = select_diverse_route_errors(
        errors,
        encoder=encoder,
        limit=args.limit,
        duplicate_similarity=args.duplicate_similarity,
    )
    payload = {
        "source_result": str(args.input),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "model_path": args.model_path,
        "selection_method": "confusion_coverage_then_bge_m3_diversity",
        "input_error_count": len(errors),
        "selected_error_count": len(selected),
        "errors": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
