from __future__ import annotations

from typing import Any

from .models import MealPlanDraft


_SYSTEM_PROMPT = """You extract meal-planning constraints into the provided schema.
Use null for information the user did not explicitly provide. Do not invent medical,
allergen, calorie, protein, time, or budget values. An empty excluded_allergens list
is allowed only when the user explicitly confirms there are no food allergies.
Convert yuan to integer cents. Use normalized lowercase tags and ingredient IDs.
Return only structured data through the supplied schema."""


class MealConstraintExtractor:
    def __init__(self, model: Any) -> None:
        try:
            self._chain = model.with_structured_output(
                MealPlanDraft,
                include_raw=True,
            )
        except TypeError:
            self._chain = model.with_structured_output(MealPlanDraft)
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_fallback_used = False

    def extract(
        self,
        user_query: str,
        previous_draft: MealPlanDraft | None = None,
    ) -> MealPlanDraft:
        context = ""
        if previous_draft is not None:
            context = (
                "Previously confirmed draft: "
                + previous_draft.model_dump_json(exclude_none=True)
                + "\nExtract only information supplied or corrected in the new message."
            )
        result = self._chain.invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", f"{context}\nUser message: {user_query}".strip()),
            ]
        )
        if isinstance(result, dict) and "parsed" in result:
            if result.get("parsing_error") is not None or result.get("parsed") is None:
                raise ValueError("structured constraint parsing failed")
            raw = result.get("raw")
            usage = getattr(raw, "usage_metadata", None) or {}
            self.last_input_tokens = int(usage.get("input_tokens") or 0)
            self.last_output_tokens = int(usage.get("output_tokens") or 0)
            parsed = result["parsed"]
            if isinstance(parsed, MealPlanDraft):
                return parsed
            return MealPlanDraft.model_validate(parsed)
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        if isinstance(result, MealPlanDraft):
            return result
        return MealPlanDraft.model_validate(result)
