from __future__ import annotations

import asyncio
import operator
import re
import uuid
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from .domain_agents import DomainAgentRegistry
from .evidence import EvidenceLedger
from .models import (
    AgentTask,
    CriticFinding,
    Domain,
    ExecutionPlan,
    GateDecision,
    GateStatus,
    Handoff,
    TaskStatus,
    WorkflowResult,
)
from .planning import StructuredPlanner
from .scheduling import (
    build_skipped_handoffs,
    select_ready_tasks,
    validate_task_dependencies,
)


WRITE_SQL_PATTERN = re.compile(
    r"\b(delete|drop|update|insert|alter|truncate|create|replace|merge|grant|revoke)\b",
    re.IGNORECASE,
)

Synthesizer = Callable[[str, list[Handoff]], Awaitable[str]]


def _merge_handoffs(left: list[Handoff], right: list[Handoff]) -> list[Handoff]:
    """Merge parallel results and replace the previous attempt during retries."""
    merged = list(left)
    positions = {handoff.task_id: index for index, handoff in enumerate(merged)}
    for handoff in right:
        index = positions.get(handoff.task_id)
        if index is None:
            positions[handoff.task_id] = len(merged)
            merged.append(handoff)
        else:
            merged[index] = handoff
    return merged


def _merge_task_attempts(
    left: dict[str, int],
    right: dict[str, int],
) -> dict[str, int]:
    """Merge attempt counters emitted by parallel domain-agent branches."""
    merged = dict(left)
    for task_id, count in right.items():
        merged[task_id] = max(count, merged.get(task_id, 0))
    return merged


class DeepReasonGraphState(TypedDict, total=False):
    run_id: str
    query: str
    session_id: str
    user_id: str | None
    image_path: str | None
    file_path: str | None
    plan: ExecutionPlan
    gateway_plan: ExecutionPlan
    meal_planning_request: dict[str, Any]
    pending_tasks: list[AgentTask]
    active_task: AgentTask
    dependency_outputs: dict[str, dict[str, Any]]
    handoffs: Annotated[list[Handoff], _merge_handoffs]
    task_attempts: Annotated[dict[str, int], _merge_task_attempts]
    retry_task_ids: list[str]
    dependency_findings: list[CriticFinding]
    findings: list[CriticFinding]
    gate: GateDecision
    retry_count: int
    answer: str
    events: Annotated[list[dict[str, Any]], operator.add]


class DeepReasonOrchestrator:
    """Top-level Multi-Agent workflow implemented as a compiled LangGraph."""

    def __init__(
        self,
        *,
        registry: DomainAgentRegistry,
        ledger_path: str | Path = "evidence/deepreason-ledger.jsonl",
        planner: StructuredPlanner | None = None,
        synthesizer: Synthesizer | None = None,
        max_retries: int = 1,
        request_gateway: Any | None = None,
    ) -> None:
        self.registry = registry
        self.ledger = EvidenceLedger(ledger_path)
        self.planner = planner or StructuredPlanner()
        self.synthesizer = synthesizer
        self.max_retries = max(0, max_retries)
        self.request_gateway = request_gateway
        self.last_result: WorkflowResult | None = None
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(DeepReasonGraphState)
        builder.add_node("coordinator", self._coordinator_node)
        builder.add_node("reviewer", self._reviewer_node)
        builder.add_node("scheduler", self._scheduler_node)
        builder.add_node("domain_agent", self._domain_agent_node)
        builder.add_node("critic_gate", self._critic_gate_node)
        builder.add_node("prepare_retry", self._prepare_retry_node)
        builder.add_node("synthesizer", self._synthesizer_node)

        builder.add_edge(START, "coordinator")
        builder.add_edge("coordinator", "reviewer")
        builder.add_edge("reviewer", "scheduler")
        builder.add_conditional_edges(
            "scheduler",
            self._dispatch_pending_tasks,
            ["domain_agent", "critic_gate"],
        )
        builder.add_edge("domain_agent", "scheduler")
        builder.add_conditional_edges(
            "critic_gate",
            self._route_after_gate,
            {"retry": "prepare_retry", "finish": "synthesizer"},
        )
        builder.add_edge("prepare_retry", "scheduler")
        builder.add_edge("synthesizer", END)
        return builder.compile()

    async def run(
        self,
        query: str,
        *,
        session_id: str,
        user_id: str | None = None,
        image_path: str | None = None,
        file_path: str | None = None,
    ) -> WorkflowResult:
        started = perf_counter()
        run_id = "run_" + uuid.uuid4().hex[:12]
        route_events: list[dict[str, Any]] = []
        gateway_plan: ExecutionPlan | None = None
        meal_planning_request: dict[str, Any] | None = None
        if self.request_gateway is not None and image_path is None and file_path is None:
            decision = await asyncio.to_thread(self.request_gateway.decide, query)
            route_events.append(
                {
                    "kind": "route_decided",
                    "business_intent": decision.route.business_intent.value,
                    "policy": decision.policy.value,
                    "confidence": decision.route.confidence,
                    "reason_code": decision.reason_code,
                }
            )
            if decision.policy.value == "clarify":
                return self._clarification_result(
                    run_id=run_id,
                    session_id=session_id,
                    query=query,
                    questions=decision.questions,
                    events=route_events,
                    started=started,
                )
            if decision.policy.value == "direct" and decision.target_domain:
                return await self._run_direct(
                    run_id=run_id,
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    target_domain=decision.target_domain,
                    events=route_events,
                    started=started,
                )
            if (
                decision.policy.value == "workflow"
                and decision.route.business_intent.value == "meal_plan"
                and decision.planning_result is not None
                and decision.planning_result.constraints is not None
            ):
                gateway_plan = ExecutionPlan(
                    query=query,
                    tasks=[
                        AgentTask(
                            task_id="meal-planning-1",
                            domain=Domain.MEAL_PLANNING,
                            instruction=query,
                            evidence_required=True,
                        )
                    ],
                    rationale="validated meal-plan request mapped to specialized agent",
                    confidence=decision.route.confidence,
                    review_status="validated_by_request_gateway",
                )
                meal_planning_request = {
                    "constraints": decision.planning_result.constraints.model_dump(
                        mode="json"
                    ),
                    "retrieval_query": decision.planning_result.retrieval_query,
                }
        initial_state: DeepReasonGraphState = {
            "run_id": run_id,
            "query": query,
            "session_id": session_id,
            "user_id": user_id,
            "image_path": image_path,
            "file_path": file_path,
            "handoffs": [],
            "task_attempts": {},
            "retry_task_ids": [],
            "retry_count": 0,
            "events": [
                {"kind": "run_started", "run_id": run_id},
                *route_events,
            ],
            **({"gateway_plan": gateway_plan} if gateway_plan is not None else {}),
            **(
                {"meal_planning_request": meal_planning_request}
                if meal_planning_request is not None
                else {}
            ),
        }
        state = await self.graph.ainvoke(
            initial_state,
            config={
                "configurable": {"thread_id": session_id},
                "recursion_limit": 12 + self.max_retries * 4,
                "tags": ["deepreason", "top-level-langgraph"],
            },
        )
        handoffs = state.get("handoffs", [])
        result = WorkflowResult(
            run_id=run_id,
            session_id=session_id,
            answer=state.get("answer", ""),
            plan=state["plan"],
            handoffs=handoffs,
            evidence=[item for handoff in handoffs for item in handoff.evidence],
            findings=state.get("findings", []),
            gate=state["gate"],
            events=state.get("events", []),
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        self.last_result = result
        return result

    def _clarification_result(
        self,
        *,
        run_id: str,
        session_id: str,
        query: str,
        questions: list[str],
        events: list[dict[str, Any]],
        started: float,
    ) -> WorkflowResult:
        answer = "\n".join(questions) or "请补充你的具体需求。"
        result = WorkflowResult(
            run_id=run_id,
            session_id=session_id,
            answer=answer,
            plan=ExecutionPlan(
                query=query,
                tasks=[],
                rationale="request gateway requested clarification",
                review_status="bypassed_for_clarification",
            ),
            handoffs=[],
            evidence=[],
            findings=[],
            gate=GateDecision(
                status=GateStatus.ALLOW,
                reasons=["clarification response contains no business claim"],
            ),
            events=[
                {"kind": "run_started", "run_id": run_id},
                *events,
                {"kind": "clarification_requested"},
                {"kind": "run_completed"},
            ],
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        self.last_result = result
        return result

    async def _run_direct(
        self,
        *,
        run_id: str,
        session_id: str,
        user_id: str | None,
        query: str,
        target_domain: str,
        events: list[dict[str, Any]],
        started: float,
    ) -> WorkflowResult:
        domain = Domain(target_domain)
        task = AgentTask(
            task_id=f"direct-{domain.value}-1",
            domain=domain,
            instruction=query,
            evidence_required=domain is not Domain.GENERAL,
        )
        plan = ExecutionPlan(
            query=query,
            tasks=[task],
            rationale="single-capability request bypassed multi-agent planning",
            review_status="bypassed_for_direct_route",
        )
        handoff = await self._execute_task(
            task,
            {
                "run_id": run_id,
                "session_id": session_id,
                "user_id": user_id,
                "query": query,
                "dependency_outputs": {},
            },
        )
        handoffs = [handoff]
        self._persist_evidence(handoffs)
        findings = audit_handoffs(plan.tasks, handoffs)
        gate = decide_gate(
            handoffs,
            findings,
            retry_count=0,
            max_retries=0,
        )
        answer = await self._synthesize(query, handoffs, gate)
        result = WorkflowResult(
            run_id=run_id,
            session_id=session_id,
            answer=answer,
            plan=plan,
            handoffs=handoffs,
            evidence=list(handoff.evidence),
            findings=findings,
            gate=gate,
            events=[
                {"kind": "run_started", "run_id": run_id},
                *events,
                {"kind": "task_started", "task_id": task.task_id, "domain": domain.value},
                {
                    "kind": "task_completed",
                    "task_id": task.task_id,
                    "status": handoff.status.value,
                    "duration_ms": handoff.duration_ms,
                },
                {"kind": "critic_completed", "finding_count": len(findings), "gate_status": gate.status.value},
                {"kind": "run_completed"},
            ],
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        self.last_result = result
        return result

    async def _coordinator_node(self, state: DeepReasonGraphState) -> dict[str, Any]:
        if state.get("gateway_plan") is not None:
            plan = state["gateway_plan"]
            return {
                "plan": plan,
                "events": [
                    {
                        "kind": "coordinator_bypassed",
                        "reason": "gateway supplied validated specialized plan",
                        "task_ids": [task.task_id for task in plan.tasks],
                    }
                ],
            }
        plan = await self.planner.acoordinate(
            state["query"],
            image_path=state.get("image_path"),
            file_path=state.get("file_path"),
        )
        return {
            "plan": plan,
            "events": [
                {
                    "kind": "coordinator_completed",
                    "task_ids": [task.task_id for task in plan.tasks],
                }
            ],
        }

    async def _reviewer_node(self, state: DeepReasonGraphState) -> dict[str, Any]:
        if state.get("gateway_plan") is not None:
            plan = state["plan"]
        else:
            plan = await self.planner.areview(state["query"], state["plan"])
        return {
            "plan": plan,
            "pending_tasks": list(plan.tasks),
            "events": [
                {
                    "kind": "plan_created",
                    "task_ids": [task.task_id for task in plan.tasks],
                    "review_status": plan.review_status,
                }
            ],
        }

    def _scheduler_node(self, state: DeepReasonGraphState) -> dict[str, Any]:
        """Create a superstep boundary so completed Handoffs are visible before routing."""
        dependency_findings = validate_task_dependencies(state["plan"].tasks)
        if dependency_findings:
            return {"dependency_findings": dependency_findings}

        skipped_handoffs: list[Handoff] = []
        retry_count = state.get("retry_count", 0)
        retry_attempt_limit = 1 + retry_count
        attempts = state.get("task_attempts", {})
        retry_pending = any(
            attempts.get(task_id, 0) < retry_attempt_limit
            for task_id in state.get("retry_task_ids", [])
        )
        if retry_count >= self.max_retries and not retry_pending:
            skipped_handoffs = build_skipped_handoffs(
                state["plan"].tasks,
                state.get("handoffs", []),
            )
        return {
            "dependency_findings": [],
            "handoffs": skipped_handoffs,
            "events": [
                {
                    "kind": "task_skipped",
                    "task_id": handoff.task_id,
                    "reason": handoff.error,
                }
                for handoff in skipped_handoffs
            ],
        }

    def _dispatch_pending_tasks(self, state: DeepReasonGraphState) -> list[Send] | str:
        if state.get("dependency_findings"):
            return "critic_gate"
        tasks = state.get("pending_tasks", [])
        if not tasks:
            return "critic_gate"
        retry_count = state.get("retry_count", 0)
        ready_tasks = select_ready_tasks(
            tasks,
            state.get("handoffs", []),
            retry_task_ids=state.get("retry_task_ids", []),
            task_attempts=state.get("task_attempts", {}),
            retry_attempt_limit=1 + retry_count,
        )
        if not ready_tasks:
            return "critic_gate"
        return [
            Send(
                "domain_agent",
                {
                    **self._execution_context(state),
                    "active_task": task,
                    "dependency_outputs": self._dependency_outputs(task, state.get("handoffs", [])),
                    "task_attempts": state.get("task_attempts", {}),
                },
            )
            for task in ready_tasks
        ]

    async def _domain_agent_node(self, state: DeepReasonGraphState) -> dict[str, Any]:
        task = state["active_task"]
        handoff = await self._execute_task(task, self._execution_context(state))
        attempt = state.get("task_attempts", {}).get(task.task_id, 0) + 1
        return {
            "handoffs": [handoff],
            "task_attempts": {task.task_id: attempt},
            "events": [
                {"kind": "task_started", "task_id": task.task_id, "domain": task.domain.value},
                {
                    "kind": "task_completed",
                    "task_id": task.task_id,
                    "status": handoff.status.value,
                    "duration_ms": handoff.duration_ms,
                },
            ],
        }

    def _critic_gate_node(self, state: DeepReasonGraphState) -> dict[str, Any]:
        handoffs = state.get("handoffs", [])
        self._persist_evidence(handoffs)
        handoff_findings = audit_handoffs(
            state["plan"].tasks,
            handoffs,
            require_terminal_handoffs=False,
        )
        has_retriable_handoff = (
            any(handoff.status == TaskStatus.FAILED for handoff in handoffs)
            or any(finding.code == "missing_evidence" for finding in handoff_findings)
        )
        require_terminal_handoffs = (
            not has_retriable_handoff
            or state.get("retry_count", 0) >= self.max_retries
        )
        findings = [
            *state.get("dependency_findings", []),
            *(
                audit_handoffs(
                    state["plan"].tasks,
                    handoffs,
                    require_terminal_handoffs=True,
                )
                if require_terminal_handoffs
                else handoff_findings
            ),
        ]
        gate = decide_gate(
            handoffs,
            findings,
            retry_count=state.get("retry_count", 0),
            max_retries=self.max_retries,
        )
        return {
            "findings": findings,
            "gate": gate,
            "events": [
                {
                    "kind": "critic_completed",
                    "finding_count": len(findings),
                    "gate_status": gate.status.value,
                }
            ],
        }

    @staticmethod
    def _route_after_gate(state: DeepReasonGraphState) -> str:
        return "retry" if state["gate"].status == GateStatus.RETRY else "finish"

    def _prepare_retry_node(self, state: DeepReasonGraphState) -> dict[str, Any]:
        retry_ids = set(state["gate"].retry_task_ids)
        return {
            "pending_tasks": list(state["plan"].tasks),
            "retry_task_ids": sorted(retry_ids),
            "retry_count": state.get("retry_count", 0) + 1,
            "events": [{"kind": "gate_retry", "task_ids": sorted(retry_ids)}],
        }

    async def _synthesizer_node(self, state: DeepReasonGraphState) -> dict[str, Any]:
        answer = await self._synthesize(state["query"], state.get("handoffs", []), state["gate"])
        return {
            "answer": answer,
            "events": [
                {"kind": "gate_decided", "status": state["gate"].status.value},
                {"kind": "run_completed"},
            ],
        }

    @staticmethod
    def _execution_context(state: DeepReasonGraphState) -> dict[str, Any]:
        return {
            "run_id": state.get("run_id"),
            "session_id": state.get("session_id"),
            "user_id": state.get("user_id"),
            "image_path": state.get("image_path"),
            "file_path": state.get("file_path"),
            "query": state.get("query"),
            "dependency_outputs": state.get("dependency_outputs", {}),
            "meal_planning_request": state.get("meal_planning_request"),
        }

    @staticmethod
    def _dependency_outputs(
        task: AgentTask,
        handoffs: list[Handoff],
    ) -> dict[str, dict[str, Any]]:
        successful_outputs = {
            handoff.task_id: handoff.output
            for handoff in handoffs
            if handoff.status == TaskStatus.SUCCESS
        }
        return {
            dependency_id: successful_outputs[dependency_id]
            for dependency_id in task.depends_on
            if dependency_id in successful_outputs
        }

    async def _execute_task(self, task: AgentTask, context: dict[str, Any]) -> Handoff:
        try:
            agent = self.registry.get(task.domain)
        except KeyError as exc:
            return Handoff(
                task_id=task.task_id,
                agent="unassigned",
                domain=task.domain,
                status=TaskStatus.FAILED,
                error=str(exc),
            )
        return await agent.execute(task, context)

    def _persist_evidence(self, handoffs: list[Handoff]) -> None:
        self.ledger.append_many([item for handoff in handoffs for item in handoff.evidence])

    async def _synthesize(self, query: str, handoffs: list[Handoff], gate: GateDecision) -> str:
        if gate.status == GateStatus.DENY:
            return "该请求未通过安全或证据门禁，系统已停止生成业务结论。"
        successful = [handoff for handoff in handoffs if handoff.status == TaskStatus.SUCCESS]
        if not successful:
            return "本次任务没有获得可用结果，请稍后重试。"
        if self.synthesizer is not None:
            return await self.synthesizer(query, successful)
        if len(successful) == 1:
            return successful[0].summary
        labels = {
            "recipe": "菜谱检索",
            "analytics": "数据分析",
            "vision": "图像分析",
            "file": "文件处理",
            "general": "综合回答",
        }
        return "\n\n".join(
            f"{labels[handoff.domain.value]}：{handoff.summary}" for handoff in successful
        )


def audit_handoffs(
    tasks: list[AgentTask],
    handoffs: list[Handoff],
    *,
    require_terminal_handoffs: bool = True,
) -> list[CriticFinding]:
    task_map = {task.task_id: task for task in tasks}
    findings: list[CriticFinding] = []
    if require_terminal_handoffs:
        completed_ids = {handoff.task_id for handoff in handoffs}
        for task in tasks:
            if task.task_id not in completed_ids:
                findings.append(
                    CriticFinding(
                        severity="critical",
                        code="task_not_executed",
                        message="planned task reached the gate without a terminal handoff",
                        task_id=task.task_id,
                    )
                )
    for handoff in handoffs:
        task = task_map.get(handoff.task_id)
        if handoff.status == TaskStatus.FAILED:
            findings.append(
                CriticFinding(
                    severity="error",
                    code="task_failed",
                    message=handoff.error or "domain task failed",
                    task_id=handoff.task_id,
                )
            )
            continue
        statement = str(handoff.output.get("sql_statement", ""))
        if statement and WRITE_SQL_PATTERN.search(statement):
            findings.append(
                CriticFinding(
                    severity="critical",
                    code="unsafe_sql",
                    message="write operation detected in analytics handoff",
                    task_id=handoff.task_id,
                )
            )
        if task and task.evidence_required and not handoff.evidence:
            findings.append(
                CriticFinding(
                    severity="warning",
                    code="missing_evidence",
                    message="required task returned no evidence",
                    task_id=handoff.task_id,
                )
            )
    return findings


def decide_gate(
    handoffs: list[Handoff],
    findings: list[CriticFinding],
    *,
    retry_count: int,
    max_retries: int,
) -> GateDecision:
    if any(finding.severity == "critical" for finding in findings):
        return GateDecision(status=GateStatus.DENY, reasons=["critical safety finding"])
    failed_ids = [handoff.task_id for handoff in handoffs if handoff.status == TaskStatus.FAILED]
    missing_ids = [
        finding.task_id for finding in findings if finding.code == "missing_evidence" and finding.task_id
    ]
    retry_ids = list(dict.fromkeys([*failed_ids, *missing_ids]))
    if retry_ids and retry_count < max_retries:
        return GateDecision(
            status=GateStatus.RETRY,
            reasons=["failed or unsupported tasks can be retried"],
            retry_task_ids=retry_ids,
        )
    if retry_ids:
        return GateDecision(status=GateStatus.DENY, reasons=["retry budget exhausted"])
    return GateDecision(status=GateStatus.ALLOW, reasons=["all required tasks passed critic audit"])
