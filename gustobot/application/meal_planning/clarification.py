from __future__ import annotations

from .models import MealPlanDraft


_QUESTIONS = {
    "days": "你希望规划几天？",
    "daily_calories_range": "你的每日目标热量范围是多少？",
    "daily_protein_min_g": "你希望每天至少摄入多少克蛋白质？",
    "allergen_status": "有没有需要排除的过敏食材？如果没有，请明确回复没有。",
}


def find_missing_fields(draft: MealPlanDraft) -> list[str]:
    missing: list[str] = []
    if draft.days is None:
        missing.append("days")
    if draft.daily_calories_min is None or draft.daily_calories_max is None:
        missing.append("daily_calories_range")
    if draft.daily_protein_min_g is None:
        missing.append("daily_protein_min_g")
    if draft.excluded_allergens is None:
        missing.append("allergen_status")
    return missing


def build_clarification_questions(missing_fields: list[str]) -> list[str]:
    return [_QUESTIONS[field] for field in missing_fields]
