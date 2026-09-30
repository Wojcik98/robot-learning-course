"""Decide when a paid job must stop from saved progress and time bounds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteProgress:
    """A remote job's saved work count and heartbeat."""

    phase: str
    completed: int
    total: int
    updated_at: float
    last_work_at: float


@dataclass(frozen=True)
class MonitorPlan:
    """Fixed cost and duration bounds saved before a VM starts."""

    started_at: float
    hard_deadline_at: float
    estimated_work_s: float
    hourly_usd: float
    approved_cost_usd: float
    startup_allowance_s: float = 900
    p95_work_unit_s: float = 60
    heartbeat_limit_s: float = 120


def stop_reason(plan: MonitorPlan, progress: RemoteProgress | None, *, now: float) -> str | None:
    """Return a stop reason; do not extend a saved deadline."""
    elapsed = now - plan.started_at
    if now >= plan.hard_deadline_at:
        return "hard_deadline"
    if elapsed * plan.hourly_usd / 3600 >= plan.approved_cost_usd:
        return "cost_limit"
    if progress is None:
        return "lost_heartbeat" if elapsed > plan.startup_allowance_s else None
    if progress.phase in ("succeeded", "failed", "cancelled"):
        return progress.phase
    if elapsed <= plan.startup_allowance_s:
        return None
    if now - progress.updated_at > plan.heartbeat_limit_s:
        return "lost_heartbeat"
    if now - progress.last_work_at > max(300, 5 * plan.p95_work_unit_s):
        return "stalled"
    if progress.completed > 0 and progress.total > progress.completed:
        projected_s = elapsed * progress.total / progress.completed
        if projected_s > 2 * plan.estimated_work_s and projected_s - plan.estimated_work_s >= 900:
            return "projected_overrun"
    return None
