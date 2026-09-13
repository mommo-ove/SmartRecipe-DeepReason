from __future__ import annotations

from enum import Enum
import re
from typing import Any

from pydantic import BaseModel, Field

from .clarification import find_missing_fields
from .models import MealPlanDraft
from .semantic_validation import validate_draft_semantics


class BusinessIntent(str, Enum):
    RECIPE_LOOKUP = "recipe_lookup"
    DATA_QUERY = "data_query"
    MEAL_PLAN = "meal_plan"
    PLAN_EDIT = "plan_edit"
    GENERAL = "general"
    UNKNOWN = "unknown"


class ExecutionPolicy(str, Enum):
    DIRECT = "direct"
    WORKFLOW = "workflow"
    CLARIFY = "clarify"


class RouterPromptVersion(str, Enum):
    R0_ZERO_SHOT = "r0_zero_shot"
    R1_FEW_SHOT = "r1_few_shot"
    R2_HARD_NEGATIVE = "r2_hard_negative"
    R3_MINED_ERRORS = "r3_mined_errors"


class RecipeResponseMode(str, Enum):
    DETAIL = "detail"
    LIST = "list"


class RecipeResponseStrategy(BaseModel):
    mode: RecipeResponseMode
    result_limit: int = Field(ge=1, le=20)


class MealRouteDraft(BaseModel):
    business_intent: BusinessIntent
    confidence: float = Field(ge=0, le=1)
    normalized_query: str = ""
    constraints: MealPlanDraft = Field(default_factory=MealPlanDraft)
    requested_capabilities: set[str] = Field(default_factory=set)


class IntentRouteOutput(BaseModel):
    """LLM-owned semantic decision; execution fields are deliberately excluded."""

    business_intent: BusinessIntent
    confidence: float = Field(ge=0, le=1)
    normalized_query: str = ""
    requested_capabilities: set[str] = Field(default_factory=set)


class MealIntentRouter:
    def __init__(
        self,
        model: Any | None,
        *,
        prompt_version: RouterPromptVersion = RouterPromptVersion.R0_ZERO_SHOT,
        mined_examples: list[tuple[str, BusinessIntent, set[str]]] | None = None,
    ) -> None:
        self._model = model
        self._prompt_version = prompt_version
        self._mined_examples = mined_examples
        self.last_fallback_used = False
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def route(self, query: str) -> MealRouteDraft:
        if self._model is None:
            self.last_fallback_used = True
            self.last_input_tokens = 0
            self.last_output_tokens = 0
            return _heuristic_route(query)
        try:
            try:
                chain = self._model.with_structured_output(
                    IntentRouteOutput,
                    include_raw=True,
                )
            except TypeError:
                chain = self._model.with_structured_output(IntentRouteOutput)
            result = chain.invoke(
                [
                    (
                        "system",
                        build_router_system_prompt(
                            self._prompt_version,
                            mined_examples=self._mined_examples,
                        ),
                    ),
                    ("human", query),
                ]
            )
            if isinstance(result, dict) and "parsed" in result:
                if result.get("parsing_error") is not None or result.get("parsed") is None:
                    raise ValueError("structured router parsing failed")
                parsed = result["parsed"]
                raw = result.get("raw")
                usage = getattr(raw, "usage_metadata", None) or {}
                self.last_input_tokens = int(usage.get("input_tokens") or 0)
                self.last_output_tokens = int(usage.get("output_tokens") or 0)
                semantic = (
                    parsed
                    if isinstance(parsed, IntentRouteOutput)
                    else IntentRouteOutput.model_validate(parsed)
                )
            else:
                self.last_input_tokens = 0
                self.last_output_tokens = 0
                semantic = (
                    result
                    if isinstance(result, IntentRouteOutput)
                    else IntentRouteOutput.model_validate(result)
                )
            self.last_fallback_used = False
            return MealRouteDraft(**semantic.model_dump())
        except Exception:
            self.last_fallback_used = True
            self.last_input_tokens = 0
            self.last_output_tokens = 0
            return _heuristic_route(query)


_ROUTER_BASE_PROMPT = (
    "你只负责识别业务意图和所需能力，不负责决定执行策略，也不提取餐单约束。"
    "可选意图只有 recipe_lookup、data_query、meal_plan、plan_edit、general、unknown。"
    "查询做法、单菜搜索或推荐少量菜属于 recipe_lookup；独立的统计或营养数据查询属于 data_query；"
    "生成跨天、跨餐位组合属于 meal_plan；修改已有餐单属于 plan_edit。"
    "可选能力包括 recipe_retrieval、nutrition_query、constraint_solving。"
    "复合请求可以输出多个能力；缺少会话指代或无法可靠判断时输出 unknown。"
)

_R1_EXAMPLES = (
    ("番茄炒蛋怎么做？", "recipe_lookup", "recipe_retrieval"),
    ("帮我安排未来七天的早中晚餐", "meal_plan", "constraint_solving"),
    ("只替换周二晚餐", "plan_edit", "constraint_solving"),
    ("当前一共收录了多少道菜？", "data_query", "nutrition_query"),
    ("我应该吃什么？", "unknown", ""),
)

_R2_EXAMPLES = (
    ("推荐三道鸡胸肉菜", "recipe_lookup", "recipe_retrieval"),
    ("制定三天餐单", "meal_plan", "constraint_solving"),
    (
        "推荐两道菜并计算总热量",
        "recipe_lookup",
        "recipe_retrieval,nutrition_query",
    ),
    ("把这个换得健康一点", "unknown", ""),
)


def build_router_system_prompt(
    version: RouterPromptVersion,
    *,
    mined_examples: list[tuple[str, BusinessIntent, set[str]]] | None = None,
) -> str:
    if version is RouterPromptVersion.R0_ZERO_SHOT:
        return _ROUTER_BASE_PROMPT
    examples = list(_R1_EXAMPLES)
    if version in {
        RouterPromptVersion.R2_HARD_NEGATIVE,
        RouterPromptVersion.R3_MINED_ERRORS,
    }:
        examples.extend(_R2_EXAMPLES)
    if version is RouterPromptVersion.R3_MINED_ERRORS:
        if not mined_examples:
            raise ValueError("R3 requires mined hard-negative examples")
        examples.extend(
            (query, intent.value, ",".join(sorted(capabilities)))
            for query, intent, capabilities in mined_examples
        )
    rendered = "\n".join(
        f"- {query} => intent={intent}; capabilities={capabilities or 'none'}"
        for query, intent, capabilities in examples
    )
    return f"{_ROUTER_BASE_PROMPT}\nExamples:\n{rendered}"


def decide_execution_policy(
    route: MealRouteDraft,
    *,
    confidence_threshold: float = 0.7,
) -> ExecutionPolicy:
    if route.confidence < confidence_threshold:
        return ExecutionPolicy.CLARIFY
    if route.business_intent is BusinessIntent.UNKNOWN:
        return ExecutionPolicy.CLARIFY
    if route.business_intent is BusinessIntent.MEAL_PLAN:
        if find_missing_fields(route.constraints):
            return ExecutionPolicy.CLARIFY
        if validate_draft_semantics(route.constraints):
            return ExecutionPolicy.CLARIFY
        return ExecutionPolicy.WORKFLOW
    if route.business_intent is BusinessIntent.PLAN_EDIT:
        return ExecutionPolicy.WORKFLOW
    if len(route.requested_capabilities) > 1:
        return ExecutionPolicy.WORKFLOW
    return ExecutionPolicy.DIRECT


def decide_recipe_response_mode(query: str) -> RecipeResponseStrategy:
    detail_terms = ("怎么做", "做法", "步骤", "哪些食材", "需要准备")
    list_terms = ("推荐", "有哪些", "几道", "菜谱")
    if any(term in query for term in detail_terms):
        return RecipeResponseStrategy(mode=RecipeResponseMode.DETAIL, result_limit=1)
    count = _extract_chinese_count(query)
    if any(term in query for term in list_terms) or count is not None:
        return RecipeResponseStrategy(
            mode=RecipeResponseMode.LIST,
            result_limit=count or 3,
        )
    return RecipeResponseStrategy(mode=RecipeResponseMode.DETAIL, result_limit=1)


def _extract_chinese_count(query: str) -> int | None:
    numbers = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    match = re.search(r"(\d+|[一两二三四五六七八九十])\s*道", query)
    if not match:
        return None
    token = match.group(1)
    return min(int(token) if token.isdigit() else numbers[token], 20)


def _heuristic_route(query: str) -> MealRouteDraft:
    text = query.strip().casefold()
    edit_terms = (
        "换成",
        "替换",
        "修改",
        "换掉",
        "换一道",
        "其他不变",
        "保持不动",
        "原餐单",
        "昨天的计划",
        "重新排",
        "重新规划",
    )
    plan_terms = (
        "餐单",
        "菜单",
        "规划",
        "安排一下",
        "一周",
        "几天",
        "每日",
        "早午晚餐",
        "减脂餐计划",
        "餐计划",
    )
    lookup_terms = (
        "怎么做",
        "做法",
        "步骤",
        "菜谱",
        "食谱",
        "哪些食材",
        "多少材料",
        "需要准备",
        "能做什么菜",
        "有没有",
        "推荐",
        "给我找",
        "可以做哪些菜",
        "查一道",
    )
    general_terms = ("你好", "谢谢", "你是谁", "介绍一下", "什么功能", "心情")
    analytics_terms = (
        "统计",
        "总数",
        "平均",
        "排名",
        "总热量",
        "计算热量",
        "营养数据",
    )
    has_lookup = any(term in text for term in lookup_terms)
    has_analytics = any(term in text for term in analytics_terms)
    has_recipe_action = any(
        term in text
        for term in ("推荐", "给我找", "怎么做", "做法", "步骤", "哪些食材")
    )
    if any(term in text for term in edit_terms):
        intent = BusinessIntent.PLAN_EDIT
        confidence = 0.86
    elif any(term in text for term in plan_terms):
        intent = BusinessIntent.MEAL_PLAN
        confidence = 0.82
    elif has_lookup and (not has_analytics or has_recipe_action):
        intent = BusinessIntent.RECIPE_LOOKUP
        confidence = 0.8
    elif has_analytics:
        intent = BusinessIntent.DATA_QUERY
        confidence = 0.8
    elif any(term in text for term in general_terms):
        intent = BusinessIntent.GENERAL
        confidence = 0.8
    else:
        intent = BusinessIntent.UNKNOWN
        confidence = 0.55
    capabilities: set[str] = set()
    if intent is BusinessIntent.RECIPE_LOOKUP:
        capabilities.add("recipe_retrieval")
    if has_analytics:
        capabilities.add("nutrition_query")
    return MealRouteDraft(
        business_intent=intent,
        confidence=confidence,
        normalized_query=query.strip(),
        constraints=(
            _extract_explicit_constraints(text)
            if intent is BusinessIntent.MEAL_PLAN
            else MealPlanDraft()
        ),
        requested_capabilities=capabilities,
    )


def _extract_explicit_constraints(text: str) -> MealPlanDraft:
    values: dict[str, Any] = {}
    days = re.search(r"(\d+)\s*天", text)
    calories = re.search(
        r"(\d{3,4})\s*(?:到|至|[-~～—])\s*(\d{3,4})\s*千卡",
        text,
    )
    protein = re.search(r"蛋白质\s*(\d+(?:\.\d+)?)\s*克", text)
    minutes = re.search(r"(?:每餐)?\s*(\d+)\s*分钟(?:内|以内)", text)
    if days:
        values["days"] = int(days.group(1))
    if calories:
        values["daily_calories_min"] = int(calories.group(1))
        values["daily_calories_max"] = int(calories.group(2))
    if protein:
        values["daily_protein_min_g"] = protein.group(1)
    if minutes:
        values["max_meal_minutes"] = int(minutes.group(1))
    if "不含花生" in text or "不要花生" in text:
        values["excluded_allergens"] = {"peanut"}
    elif "没有过敏食材" in text or "无过敏" in text:
        values["excluded_allergens"] = set()
    return MealPlanDraft(**values)
