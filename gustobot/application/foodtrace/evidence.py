from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .models import FoodTraceModel


class EvidenceItem(FoodTraceModel):
    evidence_id: str
    kind: str
    source: str
    payload_json: str
    subject_ids: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)


def make_evidence(
    kind: str,
    source: str,
    payload: Mapping[str, Any],
    *,
    subject_ids: Sequence[str] = (),
) -> EvidenceItem:
    normalized_kind = _nonblank(kind, "kind")
    normalized_source = _nonblank(source, "source")
    if isinstance(subject_ids, (str, bytes)):
        raise ValueError("subject_ids must be a collection, not a scalar string")
    normalized_subject_ids = tuple(
        sorted({_nonblank(subject_id, "subject_id") for subject_id in subject_ids})
    )
    payload_json = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest_input = json.dumps(
        {
            "kind": normalized_kind,
            "payload": json.loads(payload_json),
            "source": normalized_source,
            "subject_ids": normalized_subject_ids,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    evidence_id = "EVD-" + hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    return EvidenceItem(
        evidence_id=evidence_id,
        kind=normalized_kind,
        source=normalized_source,
        payload_json=payload_json,
        subject_ids=normalized_subject_ids,
    )


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value.strip()
