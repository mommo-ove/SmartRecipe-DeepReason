from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .clarification import build_clarification_questions, find_missing_fields
from .extraction import MealConstraintExtractor
from .models import MealPlanConstraints, MealPlanDraft
from .semantic_validation import validate_draft_semantics


class MealPlanningEntryStatus(str, Enum):
    READY = "ready"
    CLARIFY = "clarify"
    RETRY = "retry"


class MealPlanningEntryResult(BaseModel):
    status: MealPlanningEntryStatus
    draft: MealPlanDraft = Field(default_factory=MealPlanDraft)
    constraints: MealPlanConstraints | None = None
    retrieval_query: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    semantic_issue_codes: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    error_code: str | None = None


class MealPlanningEntry:
    def __init__(self, extractor: MealConstraintExtractor) -> None:
        self._extractor = extractor

    def handle(
        self,
        user_query: str,
        *,
        previous_draft: MealPlanDraft | None = None,
    ) -> MealPlanningEntryResult:
        try:
            patch = self._extractor.extract(user_query, previous_draft)
        except Exception:
            return MealPlanningEntryResult(
                status=MealPlanningEntryStatus.RETRY,
                draft=previous_draft or MealPlanDraft(),
                error_code="constraint_extraction_failed",
            )

        draft = _merge_drafts(previous_draft, patch)
        missing = find_missing_fields(draft)
        if missing:
            return MealPlanningEntryResult(
                status=MealPlanningEntryStatus.CLARIFY,
                draft=draft,
                missing_fields=missing,
                questions=build_clarification_questions(missing),
            )

        semantic_issues = validate_draft_semantics(draft)
        if semantic_issues:
            return MealPlanningEntryResult(
                status=MealPlanningEntryStatus.CLARIFY,
                draft=draft,
                semantic_issue_codes=[issue.code for issue in semantic_issues],
                questions=[issue.question for issue in semantic_issues],
            )

        constraints = MealPlanConstraints(
            days=draft.days,
            daily_calories_min=draft.daily_calories_min,
            daily_calories_max=draft.daily_calories_max,
            daily_protein_min_g=draft.daily_protein_min_g,
            excluded_allergens=draft.excluded_allergens,
            max_meal_minutes=draft.max_meal_minutes,
            weekly_budget_cents=draft.weekly_budget_cents,
            preferred_tags=draft.preferred_tags,
        )
        query_terms = sorted(draft.requested_ingredients | draft.preferred_tags)
        return MealPlanningEntryResult(
            status=MealPlanningEntryStatus.READY,
            draft=draft,
            constraints=constraints,
            retrieval_query=" ".join(query_terms),
        )


def _merge_drafts(
    previous: MealPlanDraft | None,
    patch: MealPlanDraft,
) -> MealPlanDraft:
    if previous is None:
        return patch
    return previous.model_copy(update=patch.model_dump(exclude_none=True))
