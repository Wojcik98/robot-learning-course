"""Check saved cloud progress against the hard stop rules."""

from vla_lab.cloud.monitor import MonitorPlan, RemoteProgress, stop_reason

PLAN = MonitorPlan(1000, 4000, 1000, 1, 2, startup_allowance_s=100, p95_work_unit_s=10)


def _progress(**changes: object) -> RemoteProgress:
    values = {
        "phase": "running",
        "completed": 1,
        "total": 10,
        "updated_at": 1200,
        "last_work_at": 1200,
    }
    values.update(changes)
    return RemoteProgress(**values)


def test_monitor_stops_at_saved_deadline_after_restart() -> None:
    assert stop_reason(PLAN, _progress(updated_at=3999), now=4000) == "hard_deadline"


def test_monitor_detects_stall_and_lost_heartbeat() -> None:
    assert stop_reason(PLAN, _progress(updated_at=1500, last_work_at=1100), now=1501) == "stalled"
    assert stop_reason(PLAN, _progress(updated_at=1100), now=1300) == "lost_heartbeat"


def test_monitor_checks_work_units_not_only_heartbeat() -> None:
    assert (
        stop_reason(PLAN, _progress(completed=1, total=20, updated_at=1200), now=1200)
        == "projected_overrun"
    )


def test_monitor_allows_initial_startup_and_reports_completion() -> None:
    assert stop_reason(PLAN, None, now=1050) is None
    assert stop_reason(PLAN, _progress(phase="succeeded"), now=1200) == "succeeded"
