from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import AgentTask, Domain, ExecutionPlan, ReviewDecision
from .prompts import COORDINATOR_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT


ANALYTICS_TERMS = (
    "统计",
    "平均",
    "数量",
    "多少",
    "总数",
    "排名",
    "最多",
    "最少",
    "占比",
    "频率",
    "最常用",
    "top",
)
ANALYTICS_TERMS += ("热量", "卡路里", "营养", "蛋白质", "脂肪", "碳水", "计算")
VISION_TERMS = ("图片", "照片", "看图", "识别这道菜", "生成一张", "画一张")
FILE_TERMS = ("文件", "文档", "excel", "xlsx", "csv", "pdf", "上传", "导入")
RECIPE_TERMS = (
    "菜",
    "菜谱",
    "食谱",
    "食材",
    "配料",
    "做法",
    "怎么做",
    "如何做",
    "步骤",
    "烹饪",
    "推荐",
    "鸡肉",
    "猪肉",
    "牛肉",
    "口味",
)
RECIPE_STRONG_TERMS = tuple(term for term in RECIPE_TERMS if term not in {"菜", "菜谱", "食谱"})
RECIPE_COMPOUND_TERMS = ("推荐", "做法", "怎么做", "如何做", "步骤", "搭配", "替代")


@dataclass
class HeuristicPlanner:
    """Deterministic fallback used when structured LLM planning is unavailable."""

    def plan(
        self,
        query: str,
        *,
        image_path: str | None = None,
        file_path: str | None = None,
    ) -> ExecutionPlan:
        text = query.strip()
        lowered = text.lower()
        domains: list[Domain] = []

        if image_path or any(term in lowered for term in VISION_TERMS):
            domains.append(Domain.VISION)
        if file_path or any(term in lowered for term in FILE_TERMS):
            domains.append(Domain.FILE)
        has_analytics, has_recipe = _intent_flags(lowered)
        if has_recipe:
            domains.append(Domain.RECIPE)
        if has_analytics:
            domains.append(Domain.ANALYTICS)

        if not domains:
            domains.append(Domain.GENERAL)

        domains = list(dict.fromkeys(domains))
        tasks = [
            AgentTask(
                task_id=f"{domain.value}-{index + 1}",
                domain=domain,
                instruction=self._instruction(domain, text),
                evidence_required=domain not in {Domain.GENERAL},
                priority=index,
            )
            for index, domain in enumerate(domains)
        ]
        _apply_business_dependencies(text, tasks)
        return ExecutionPlan(
            query=text,
            tasks=tasks,
            rationale="heuristic domain decomposition",
            confidence=0.82 if len(tasks) > 1 else 0.78,
        )

    def review(self, query: str, plan: ExecutionPlan) -> ExecutionPlan:
        domains = {task.domain for task in plan.tasks}
        notes: list[str] = []
        lowered = query.lower()
        _, has_recipe = _intent_flags(lowered)
        if has_recipe and Domain.RECIPE not in domains:
            plan.tasks.insert(
                0,
                AgentTask(
                    task_id="recipe-review-1",
                    domain=Domain.RECIPE,
                    instruction=self._instruction(Domain.RECIPE, query),
                ),
            )
            notes.append("reviewer added missing recipe retrieval task")
            plan.review_status = "revised"
        else:
            plan.review_status = "approved"
        plan.reviewer_notes = notes
        return plan

    @staticmethod
    def _instruction(domain: Domain, query: str) -> str:
        prefixes = {
            Domain.RECIPE: "使用菜谱知识图谱和语义检索回答：",
            Domain.ANALYTICS: "使用只读 Text2SQL 完成统计分析：",
            Domain.VISION: "使用视觉能力分析或生成图片：",
            Domain.FILE: "解析并处理用户文件：",
            Domain.GENERAL: "直接回答用户：",
        }
        return prefixes[domain] + query


def _intent_flags(text: str) -> tuple[bool, bool]:
    has_analytics = any(term in text for term in ANALYTICS_TERMS)
    has_vision = any(term in text for term in VISION_TERMS)
    has_file = any(term in text for term in FILE_TERMS)
    if has_analytics:
        has_recipe = any(term in text for term in RECIPE_COMPOUND_TERMS)
    elif has_vision:
        has_recipe = any(term in text for term in RECIPE_STRONG_TERMS)
    else:
        has_recipe = any(term in text for term in RECIPE_STRONG_TERMS) or any(
            term in text for term in ("菜", "菜谱", "食谱")
        )
    if has_file and not has_analytics and any(term in text for term in ("菜谱", "食谱")):
        has_recipe = True
    return has_analytics, has_recipe


@dataclass
class StructuredPlanner:
    """LLM planner with a deterministic, interview-friendly fallback path."""

    model: Any | None = None
    fallback: HeuristicPlanner | None = None

    async def aplan(
        self,
        query: str,
        *,
        image_path: str | None = None,
        file_path: str | None = None,
    ) -> ExecutionPlan:
        plan = await self.acoordinate(query, image_path=image_path, file_path=file_path)
        return await self.areview(query, plan)

    async def acoordinate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        file_path: str | None = None,
    ) -> ExecutionPlan:
        """Create the initial plan without applying the reviewer step."""
        fallback = self.fallback or HeuristicPlanner()
        if self.model is None:
            plan = fallback.plan(query, image_path=image_path, file_path=file_path)
            plan.rationale += "; no model configured, used fallback"
            return plan
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            chain = self.model.with_structured_output(ExecutionPlan)
            plan = await chain.ainvoke(
                [
                    SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"用户问题：{query}\n"
                            f"image_path={image_path or ''}\nfile_path={file_path or ''}"
                        )
                    ),
                ]
            )
            if not isinstance(plan, ExecutionPlan):
                plan = ExecutionPlan.model_validate(plan)
            _apply_business_dependencies(query, plan.tasks)
            return plan
        except Exception:
            plan = fallback.plan(query, image_path=image_path, file_path=file_path)
            plan.rationale += "; structured LLM failed, used fallback"
            return plan

    async def areview(self, query: str, plan: ExecutionPlan) -> ExecutionPlan:
        """Review a coordinator plan as a separate orchestration step."""
        fallback = self.fallback or HeuristicPlanner()
        return await self._areview(query, plan, fallback)

    async def _areview(
        self,
        query: str,
        plan: ExecutionPlan,
        fallback: HeuristicPlanner,
    ) -> ExecutionPlan:
        if self.model is None:
            return fallback.review(query, plan)
        try:
            import json

            from langchain_core.messages import HumanMessage, SystemMessage

            chain = self.model.with_structured_output(ReviewDecision)
            decision = await chain.ainvoke(
                [
                    SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"用户问题：{query}\nCoordinator 计划："
                            f"{json.dumps(plan.model_dump(mode='json'), ensure_ascii=False)}"
                        )
                    ),
                ]
            )
            if not isinstance(decision, ReviewDecision):
                decision = ReviewDecision.model_validate(decision)
            existing = {task.domain for task in plan.tasks}
            for domain in decision.required_domains:
                if domain in existing:
                    continue
                plan.tasks.append(
                    AgentTask(
                        task_id=f"{domain.value}-review-{len(plan.tasks) + 1}",
                        domain=domain,
                        instruction=fallback._instruction(domain, query),
                        evidence_required=domain != Domain.GENERAL,
                        priority=len(plan.tasks),
                    )
                )
                existing.add(domain)
            plan.review_status = "revised" if decision.status != "approve" else "approved"
            plan.reviewer_notes = list(decision.findings)
            _apply_business_dependencies(query, plan.tasks)
            return plan
        except Exception:
            plan = fallback.review(query, plan)
            plan.reviewer_notes.append("structured reviewer failed, used fallback")
            return plan


_CHINESE_NUMBERS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _apply_business_dependencies(query: str, tasks: list[AgentTask]) -> None:
    """Normalize planner output with deterministic business data-flow rules."""
    recipe_tasks = [task for task in tasks if task.domain == Domain.RECIPE]
    analytics_tasks = [task for task in tasks if task.domain == Domain.ANALYTICS]
    if not recipe_tasks:
        return

    result_limit = _extract_recipe_limit(query)
    if result_limit is not None:
        for task in recipe_tasks:
            task.result_limit = result_limit

    if analytics_tasks and _requires_recipe_handoff(query):
        recipe_task_ids = [task.task_id for task in recipe_tasks]
        for task in analytics_tasks:
            task.depends_on = list(dict.fromkeys([*task.depends_on, *recipe_task_ids]))


def _extract_recipe_limit(query: str) -> int | None:
    match = re.search(r"(\d+|[一两二三四五六七八九十])\s*道", query)
    if not match:
        return None
    token = match.group(1)
    value = int(token) if token.isdigit() else _CHINESE_NUMBERS[token]
    return min(value, 100)


def _requires_recipe_handoff(query: str) -> bool:
    selected_set_terms = (
        "所选",
        "选中",
        "这些菜",
        "它们",
        "总热量",
        "合计热量",
    )
    return any(term in query for term in selected_set_terms) or bool(
        re.search(r"这\s*(?:\d+|[一两二三四五六七八九十])\s*道", query)
    )
