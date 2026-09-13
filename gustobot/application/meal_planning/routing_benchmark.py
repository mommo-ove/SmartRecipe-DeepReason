from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, Field

from gustobot.application.meal_planning.routing import BusinessIntent, ExecutionPolicy


@dataclass(frozen=True)
class RouteBenchmarkCase:
    case_id: str
    query: str
    expected_intent: BusinessIntent
    expected_policy: ExecutionPolicy
    split: str = "development"
    review_status: str = "draft"


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class RouteError:
    case_id: str
    query: str
    expected_intent: BusinessIntent
    predicted_intent: BusinessIntent
    expected_policy: ExecutionPolicy
    predicted_policy: ExecutionPolicy
    confidence: float | None


@dataclass(frozen=True)
class RouteBenchmarkResult:
    total: int
    intent_accuracy: float
    policy_accuracy: float
    intent_macro_f1: float
    intent_per_class: dict[str, ClassificationMetrics]
    intent_confusion: dict[str, dict[str, int]]
    policy_confusion: dict[str, dict[str, int]]
    errors: tuple[RouteError, ...]
    high_confidence_errors: tuple[RouteError, ...]


class FrozenRouteManifest(BaseModel):
    version: str
    status: str
    record_count: int = Field(ge=1)
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resume_metric_eligible: bool = False
    human_reviewer: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class FrozenRouteBenchmark:
    manifest: FrozenRouteManifest
    cases: list[RouteBenchmarkCase]


def load_frozen_route_benchmark(
    cases_path: Path,
    manifest_path: Path,
) -> FrozenRouteBenchmark:
    raw = cases_path.read_bytes()
    manifest = FrozenRouteManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if hashlib.sha256(raw).hexdigest() != manifest.cases_sha256:
        raise ValueError("frozen route cases do not match manifest hash")
    cases = load_route_benchmark(cases_path, split="frozen")
    if len(cases) != manifest.record_count:
        raise ValueError("frozen route case count does not match manifest")
    return FrozenRouteBenchmark(manifest=manifest, cases=cases)


def load_route_benchmark(
    path: Path,
    *,
    split: str | None = None,
) -> list[RouteBenchmarkCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        case = RouteBenchmarkCase(
            case_id=str(payload["case_id"]),
            query=str(payload["query"]),
            expected_intent=BusinessIntent(payload["expected_intent"]),
            expected_policy=ExecutionPolicy(payload["expected_policy"]),
            split=str(payload.get("split", "development")),
            review_status=str(payload.get("review_status", "draft")),
        )
        if split is None or case.split == split:
            cases.append(case)
    if not cases:
        raise ValueError("route benchmark needs at least one case")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("route benchmark case ids must be unique")
    return cases


def evaluate_route_predictions(
    cases: Sequence[RouteBenchmarkCase],
    predictions: Mapping[
        str,
        tuple[BusinessIntent, ExecutionPolicy]
        | tuple[BusinessIntent, ExecutionPolicy, float],
    ],
    *,
    high_confidence_threshold: float = 0.9,
) -> RouteBenchmarkResult:
    if not cases:
        raise ValueError("route benchmark needs at least one case")

    intent_correct = 0
    policy_correct = 0
    intent_confusion: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    policy_confusion: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    errors: list[RouteError] = []

    for case in cases:
        if case.case_id not in predictions:
            raise ValueError(f"missing prediction for case_id={case.case_id}")
        prediction = predictions[case.case_id]
        predicted_intent, predicted_policy = prediction[:2]
        confidence = prediction[2] if len(prediction) == 3 else None
        intent_correct += predicted_intent == case.expected_intent
        policy_correct += predicted_policy == case.expected_policy
        intent_confusion[case.expected_intent.value][predicted_intent.value] += 1
        policy_confusion[case.expected_policy.value][predicted_policy.value] += 1
        if (
            predicted_intent != case.expected_intent
            or predicted_policy != case.expected_policy
        ):
            errors.append(
                RouteError(
                    case_id=case.case_id,
                    query=case.query,
                    expected_intent=case.expected_intent,
                    predicted_intent=predicted_intent,
                    expected_policy=case.expected_policy,
                    predicted_policy=predicted_policy,
                    confidence=confidence,
                )
            )

    labels = sorted(
        {case.expected_intent.value for case in cases}
        | {
            prediction[0].value
            for prediction in predictions.values()
        }
    )
    intent_per_class: dict[str, ClassificationMetrics] = {}
    for label in labels:
        true_positive = intent_confusion[label][label]
        support = sum(intent_confusion[label].values())
        predicted_count = sum(row[label] for row in intent_confusion.values())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        intent_per_class[label] = ClassificationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )

    high_confidence_errors = tuple(
        error
        for error in errors
        if error.confidence is not None
        and error.confidence >= high_confidence_threshold
    )

    return RouteBenchmarkResult(
        total=len(cases),
        intent_accuracy=intent_correct / len(cases),
        policy_accuracy=policy_correct / len(cases),
        intent_macro_f1=(
            sum(metric.f1 for metric in intent_per_class.values())
            / len(intent_per_class)
        ),
        intent_per_class=intent_per_class,
        intent_confusion={key: dict(value) for key, value in intent_confusion.items()},
        policy_confusion={key: dict(value) for key, value in policy_confusion.items()},
        errors=tuple(errors),
        high_confidence_errors=high_confidence_errors,
    )
