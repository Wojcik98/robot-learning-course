"""Check the funded cloud boundary before a VM can start."""

from dataclasses import replace

import pytest

from vla_lab.cloud.budget import BillingState, CloudLimits, CloudQuote, plan_cloud_job

NOW = 1000.0
QUOTE = CloudQuote("g2-standard-4:nvidia-l4:1", 0.845421, 0.002, 24, True, NOW)
BILLING = BillingState(29.94, 0, "USD", False, False, 0, NOW)
LIMITS = CloudLimits(project_cap_usd=29.94)


def test_bounded_quote_sets_a_cost_and_duration_limit() -> None:
    plan = plan_cloud_job(QUOTE, BILLING, estimated_work_s=600, now=NOW, limits=LIMITS)
    assert plan["hard_deadline_s"] == 1800
    assert plan["approved_cost_usd"] < 5
    assert plan["hourly_usd"] < 3


@pytest.mark.parametrize(
    ("quote", "billing", "message"),
    [
        (replace(QUOTE, checked_at=600), BILLING, "stale"),
        (QUOTE, replace(BILLING, checked_at=600), "stale"),
        (replace(QUOTE, hourly_usd=3.1), BILLING, "hourly"),
        (replace(QUOTE, stoppable=False), BILLING, "stoppable"),
        (QUOTE, replace(BILLING, auto_recharge=True), "credit boundary"),
        (QUOTE, replace(BILLING, postpaid_overage=True), "credit boundary"),
        (QUOTE, replace(BILLING, other_running_instances=1), "already running"),
        (QUOTE, replace(BILLING, credit_usd=4), "reserve"),
    ],
)
def test_budget_rejects_unfunded_or_unstoppable_job(
    quote: CloudQuote, billing: BillingState, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_cloud_job(quote, billing, estimated_work_s=600, now=NOW, limits=LIMITS)


def test_duration_rejects_work_that_cannot_fit_export_and_stop() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        plan_cloud_job(QUOTE, BILLING, estimated_work_s=7000, now=NOW, limits=LIMITS)
