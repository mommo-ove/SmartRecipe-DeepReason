from __future__ import annotations

from pydantic import BaseModel

from .models import MealPlanDraft


class SemanticIssue(BaseModel):
    code: str
    question: str


def validate_draft_semantics(draft: MealPlanDraft) -> list[SemanticIssue]:
    issues: list[SemanticIssue] = []
    if (
        draft.daily_calories_min is not None
        and draft.daily_calories_max is not None
        and (
            draft.daily_calories_min < 800
            or draft.daily_calories_max > 5000
        )
    ):
        issues.append(
            SemanticIssue(
                code="daily_calories_out_of_supported_range",
                question=(
                    "当前识别出的每日热量目标为"
                    f"{draft.daily_calories_min}～{draft.daily_calories_max}千卡，"
                    "请确认这个数值是否正确。"
                ),
            )
        )
    return issues
