"""Reject cloud jobs that exceed the funded course allowance."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CloudQuote:
    """One current, stoppable GPU type quote."""

    instance_type: str
    hourly_usd: float
    disk_hourly_usd: float
    vram_gb: float
    stoppable: bool
    checked_at: float


@dataclass(frozen=True)
class BillingState:
    """Billing facts checked in the user's active Brev organization."""

    credit_usd: float
    project_spent_usd: float
    currency: str
    auto_recharge: bool
    postpaid_overage: bool
    other_running_instances: int
    checked_at: float


@dataclass(frozen=True)
class CloudLimits:
    """Cost and time limits for funded cloud jobs."""

    project_cap_usd: float = 30.0
    reserve_usd: float = 5.0
    hourly_cap_usd: float = 3.0
    job_cap_usd: float = 5.0
    max_job_s: int = 7200
    startup_allowance_s: int = 900
    export_allowance_s: int = 120
    stop_allowance_s: int = 60
    max_check_age_s: int = 300


def plan_cloud_job(
    quote: CloudQuote,
    billing: BillingState,
    *,
    estimated_work_s: float,
    now: float | None = None,
    limits: CloudLimits | None = None,
) -> dict[str, Any]:
    """Calculate a deadline after the quote and billing checks pass."""
    if limits is None:
        limits = CloudLimits()
    current = time.time() if now is None else now
    if current - quote.checked_at > limits.max_check_age_s or current < quote.checked_at:
        raise ValueError("GPU quote is stale")
    if current - billing.checked_at > limits.max_check_age_s or current < billing.checked_at:
        raise ValueError("Brev billing check is stale")
    if billing.currency != "USD" or billing.auto_recharge or billing.postpaid_overage:
        raise ValueError("Brev billing is outside the funded US dollar credit boundary")
    if billing.other_running_instances != 0:
        raise ValueError("Another Brev instance is already running")
    if not quote.stoppable or quote.vram_gb < 24:
        raise ValueError("GPU type is not a stoppable 24 GB instance")
    hourly = quote.hourly_usd + quote.disk_hourly_usd
    if not math.isfinite(hourly) or hourly <= 0 or hourly > limits.hourly_cap_usd:
        raise ValueError("GPU quote exceeds the hourly price limit")
    if not math.isfinite(estimated_work_s) or estimated_work_s <= 0:
        raise ValueError("Estimated work time must be positive")
    available = min(
        limits.job_cap_usd,
        billing.credit_usd - limits.reserve_usd,
        limits.project_cap_usd - billing.project_spent_usd - limits.reserve_usd,
    )
    if available <= 0:
        raise ValueError("Funded credit is below the project reserve")
    cost_bound_s = (
        math.floor(available / hourly * 3600) - limits.export_allowance_s - limits.stop_allowance_s
    )
    deadline_s = min(limits.max_job_s, cost_bound_s, math.floor(2 * estimated_work_s + 600))
    minimum_s = (
        estimated_work_s
        + limits.startup_allowance_s
        + limits.export_allowance_s
        + limits.stop_allowance_s
    )
    if minimum_s > deadline_s:
        raise ValueError("Estimated work and stop reserve do not fit the job deadline")
    approved_cost = (
        hourly * (deadline_s + limits.export_allowance_s + limits.stop_allowance_s) / 3600
    )
    return {
        "instance_type": quote.instance_type,
        "hourly_usd": round(hourly, 6),
        "available_usd": round(available, 4),
        "hard_deadline_s": deadline_s,
        "approved_cost_usd": round(approved_cost, 4),
        "quote_checked_at": quote.checked_at,
        "billing_checked_at": billing.checked_at,
        "limits": asdict(limits),
    }
