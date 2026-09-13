from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from .models import EvidenceItem


class EvidenceLedger:
    """Append-only JSONL ledger with stable-id deduplication."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def append_many(self, items: list[EvidenceItem]) -> None:
        if not items:
            return
        with self._lock:
            existing = {item.evidence_id for item in self.list()}
            pending: list[EvidenceItem] = []
            for item in items:
                if item.evidence_id in existing:
                    continue
                pending.append(item)
                existing.add(item.evidence_id)
            if not pending:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                for item in pending:
                    handle.write(item.model_dump_json() + "\n")

    def list(self) -> list[EvidenceItem]:
        if not self.path.exists():
            return []
        items: list[EvidenceItem] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(EvidenceItem.model_validate(json.loads(line)))
        return items
