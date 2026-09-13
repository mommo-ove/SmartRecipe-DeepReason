from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gustobot.application.deepreason.benchmark import run_routing_benchmark
from gustobot.application.deepreason.domain_agents import (
    DomainAgent,
    DomainAgentRegistry,
    build_demo_registry,
)
from gustobot.application.deepreason.models import Domain, EvidenceItem, Handoff, WorkflowResult
from gustobot.application.deepreason.orchestrator import DeepReasonOrchestrator


DEMO_QUERY = "推荐一道低辣鸡肉菜，并统计这类菜的平均烹饪时长"
DEMO_DOMAINS = (Domain.RECIPE, Domain.ANALYTICS)


@dataclass(frozen=True)
class AgentSpan:
    agent: str
    domain: Domain
    started_ms: float
    ended_ms: float


@dataclass
class DemoExecution:
    result: WorkflowResult
    ledger_path: Path
    ledger_items: list[EvidenceItem]
    agent_spans: list[AgentSpan]
    benchmark: dict[str, Any]

    @property
    def agents_overlapped(self) -> bool:
        """True when at least two measured domain-agent intervals overlap."""
        for index, left in enumerate(self.agent_spans):
            for right in self.agent_spans[index + 1 :]:
                if max(left.started_ms, right.started_ms) < min(left.ended_ms, right.ended_ms):
                    return True
        return False


class _MeasuredAgent:
    """Demo-only observer that delegates execution to an existing domain agent."""

    def __init__(
        self,
        delegate: DomainAgent,
        spans: list[AgentSpan],
        demo_started: float,
    ) -> None:
        self.name = delegate.name
        self.domain = delegate.domain
        self._delegate = delegate
        self._spans = spans
        self._demo_started = demo_started

    async def execute(self, task: Any, context: dict[str, Any]) -> Handoff:
        started = perf_counter()
        # A tiny async boundary makes the LangGraph Send fan-out visible and measurable.
        await asyncio.sleep(0.03)
        handoff = await self._delegate.execute(task, context)
        ended = perf_counter()
        self._spans.append(
            AgentSpan(
                agent=self.name,
                domain=self.domain,
                started_ms=round((started - self._demo_started) * 1000, 2),
                ended_ms=round((ended - self._demo_started) * 1000, 2),
            )
        )
        return handoff


async def execute_demo(*, ledger_path: str | Path | None = None) -> DemoExecution:
    """Run the real top-level graph with the repository's offline demo adapters."""
    target_ledger = Path(ledger_path) if ledger_path else ROOT / "evidence" / "deepreason-demo.jsonl"
    base_registry = build_demo_registry()
    measured_registry = DomainAgentRegistry()
    spans: list[AgentSpan] = []
    demo_started = perf_counter()
    for domain in DEMO_DOMAINS:
        measured_registry.register(
            _MeasuredAgent(base_registry.get(domain), spans, demo_started)
        )

    orchestrator = DeepReasonOrchestrator(
        registry=measured_registry,
        ledger_path=target_ledger,
        max_retries=1,
    )
    result = await orchestrator.run(DEMO_QUERY, session_id="interview-demo")

    cases = json.loads(
        (ROOT / "configs" / "deepreason_benchmark_cases.json").read_text(encoding="utf-8")
    )
    benchmark = run_routing_benchmark(cases)
    current_evidence_ids = {item.evidence_id for item in result.evidence}
    ledger_items = [
        item for item in orchestrator.ledger.list() if item.evidence_id in current_evidence_ids
    ]
    return DemoExecution(
        result=result,
        ledger_path=target_ledger.resolve(),
        ledger_items=ledger_items,
        agent_spans=sorted(spans, key=lambda span: span.started_ms),
        benchmark=benchmark,
    )


def render_demo_trace(demo: DemoExecution) -> str:
    result = demo.result
    lines = [
        "=" * 72,
        "DeepReason 本地端到端面试演示（离线、确定性、无需模型和数据库）",
        "=" * 72,
        "",
        "[1/7] Coordinator",
        f"  接收问题: {DEMO_QUERY}",
        f"  创建运行: {result.run_id} / session={result.session_id}",
        "  职责: 接收复杂目标并调用 Planner 生成结构化执行计划，不直接回答。",
        "",
        "[2/7] Planner",
        f"  路径: {result.plan.rationale}",
        f"  置信度: {result.plan.confidence:.2f}",
    ]
    for task in result.plan.tasks:
        lines.append(
            f"  - {task.task_id}: domain={task.domain.value}, "
            f"evidence_required={str(task.evidence_required).lower()}"
        )
        lines.append(f"    指令: {task.instruction}")

    notes = "；".join(result.plan.reviewer_notes) or "无修改，计划覆盖所需领域"
    lines.extend(
        [
            "",
            "[3/7] Reviewer",
            f"  复核结果: {result.plan.review_status.upper()}",
            f"  复核说明: {notes}",
            "  权限边界: 只复核/补充任务，不能绕过确定性安全规则。",
            "",
            "[4/7] 并发领域 Agents",
        ]
    )
    for span in demo.agent_spans:
        matching = next(item for item in result.handoffs if item.agent == span.agent)
        lines.append(
            f"  - {span.agent} ({span.domain.value}): "
            f"{span.started_ms:.2f}ms -> {span.ended_ms:.2f}ms, "
            f"status={matching.status.value}"
        )
        lines.append(f"    Handoff: {matching.summary}")
    lines.append(f"  并发重叠: {'是' if demo.agents_overlapped else '否'}（LangGraph Send 同一轮扇出）")

    lines.extend(
        [
            "",
            "[5/7] Evidence Ledger",
            f"  落盘位置: {demo.ledger_path}",
            f"  本次证据: {len(demo.ledger_items)} 条（稳定哈希 ID，可去重、可审计）",
        ]
    )
    for item in demo.ledger_items:
        content = " ".join(item.content.split())
        if len(content) > 100:
            content = content[:97] + "..."
        lines.append(
            f"  - {item.evidence_id} | task={item.task_id} | "
            f"{item.source_type}:{item.source}"
        )
        lines.append(f"    {content}")

    lines.extend(["", "[6/7] Critic / 安全 Gate"])
    if result.findings:
        for finding in result.findings:
            lines.append(
                f"  - [{finding.severity}] {finding.code}: {finding.message}"
            )
    else:
        lines.append("  Critic: 未发现任务失败、缺失证据或危险写 SQL")
    lines.append(f"  Gate: {result.gate.status.value.upper()}")
    lines.append(f"  原因: {'；'.join(result.gate.reasons)}")

    lines.extend(
        [
            "",
            "[7/7] Synthesizer",
            "  仅汇总通过 Gate 的成功 Handoff：",
            _indent(result.answer, "  "),
            "",
            "[演示校验] 复用现有离线路由 benchmark",
            f"  exact domain-set match: {demo.benchmark['exact_matches']}/{demo.benchmark['total']} "
            f"({demo.benchmark['accuracy']:.1%})",
            f"  端到端耗时: {result.duration_ms:.2f}ms",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline DeepReason interview walkthrough."
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        help="Evidence JSONL path (default: evidence/deepreason-demo.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    demo = asyncio.run(execute_demo(ledger_path=args.ledger_path))
    print(render_demo_trace(demo))


if __name__ == "__main__":
    main()
