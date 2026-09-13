from __future__ import annotations

from collections.abc import Iterable
from collections import Counter, deque

from .models import AgentTask, CriticFinding, Handoff, TaskStatus


def select_ready_tasks(
    tasks: Iterable[AgentTask],
    handoffs: Iterable[Handoff],
    *,
    retry_task_ids: Iterable[str] = (),
    task_attempts: dict[str, int] | None = None,
    retry_attempt_limit: int = 1,
) -> list[AgentTask]:
    """Return tasks that have not run and whose dependencies all succeeded."""
    task_list = list(tasks)
    task_by_id = {task.task_id: task for task in task_list}
    handoff_by_task = {handoff.task_id: handoff for handoff in handoffs}
    retry_ids = set(retry_task_ids)
    attempts = task_attempts or {}
    ready: list[AgentTask] = []

    for task in task_list:
        current = handoff_by_task.get(task.task_id)
        if current is not None:
            retry_is_authorized = (
                task.task_id in retry_ids
                and attempts.get(task.task_id, 0) < retry_attempt_limit
            )
            if not retry_is_authorized:
                continue

        dependencies_succeeded = True
        for dependency_id in task.depends_on:
            dependency_handoff = handoff_by_task.get(dependency_id)
            dependency_task = task_by_id.get(dependency_id)
            if dependency_handoff is None or dependency_handoff.status != TaskStatus.SUCCESS:
                dependencies_succeeded = False
                break
            if (
                dependency_task is not None
                and dependency_task.evidence_required
                and not dependency_handoff.evidence
            ):
                dependencies_succeeded = False
                break
        if dependencies_succeeded:
            ready.append(task)

    return ready


def validate_task_dependencies(tasks: Iterable[AgentTask]) -> list[CriticFinding]:
    """Validate dependency references and detect cycles before execution."""
    task_list = list(tasks)
    duplicate_ids = sorted(
        task_id
        for task_id, count in Counter(task.task_id for task in task_list).items()
        if count > 1
    )
    if duplicate_ids:
        return [
            CriticFinding(
                severity="critical",
                code="duplicate_task_id",
                message=f"duplicate task IDs: {', '.join(duplicate_ids)}",
            )
        ]

    task_by_id = {task.task_id: task for task in task_list}
    findings: list[CriticFinding] = []

    for task in task_list:
        missing_ids = [
            dependency_id
            for dependency_id in task.depends_on
            if dependency_id not in task_by_id
        ]
        if missing_ids:
            findings.append(
                CriticFinding(
                    severity="critical",
                    code="dependency_missing",
                    message=f"unknown dependencies: {', '.join(missing_ids)}",
                    task_id=task.task_id,
                )
            )

    incoming_count = {
        task.task_id: sum(
            dependency_id in task_by_id for dependency_id in task.depends_on
        )
        for task in task_list
    }
    dependents: dict[str, list[str]] = {task_id: [] for task_id in task_by_id}
    for task in task_list:
        for dependency_id in task.depends_on:
            if dependency_id in dependents:
                dependents[dependency_id].append(task.task_id)

    queue = deque(
        task_id for task_id, count in incoming_count.items() if count == 0
    )
    visited: set[str] = set()
    while queue:
        task_id = queue.popleft()
        visited.add(task_id)
        for dependent_id in dependents[task_id]:
            incoming_count[dependent_id] -= 1
            if incoming_count[dependent_id] == 0:
                queue.append(dependent_id)

    cycle_ids = sorted(set(task_by_id) - visited)
    if cycle_ids:
        findings.append(
            CriticFinding(
                severity="critical",
                code="dependency_cycle",
                message=f"dependency cycle detected among: {', '.join(cycle_ids)}",
            )
        )

    return findings


def build_skipped_handoffs(
    tasks: Iterable[AgentTask],
    handoffs: Iterable[Handoff],
) -> list[Handoff]:
    """Propagate terminal upstream failures to all blocked descendants."""
    task_list = list(tasks)
    handoff_by_task = {handoff.task_id: handoff for handoff in handoffs}
    terminal_failures = {
        task_id
        for task_id, handoff in handoff_by_task.items()
        if handoff.status in {TaskStatus.FAILED, TaskStatus.SKIPPED}
    }
    skipped: list[Handoff] = []

    changed = True
    while changed:
        changed = False
        for task in task_list:
            if task.task_id in handoff_by_task:
                continue
            failed_dependencies = sorted(set(task.depends_on) & terminal_failures)
            if not failed_dependencies:
                continue
            handoff = Handoff(
                task_id=task.task_id,
                agent="dependency_scheduler",
                domain=task.domain,
                status=TaskStatus.SKIPPED,
                error=f"dependency_failed: {', '.join(failed_dependencies)}",
            )
            handoff_by_task[task.task_id] = handoff
            terminal_failures.add(task.task_id)
            skipped.append(handoff)
            changed = True

    return skipped
