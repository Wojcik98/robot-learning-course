"""Check wall-clock call timing and deadline accounting."""

import pytest

from vla_lab.runtime.timing import measure_calls


def test_measured_calls_keep_cold_start_and_deadline_misses_separate() -> None:
    ticks = iter((0.0, 0.08, 1.0, 1.02, 2.0, 2.06, 3.0, 3.03))
    calls = []

    report = measure_calls(
        lambda index: calls.append(index),
        count=4,
        deadline_ms=50,
        clock=lambda: next(ticks),
    )

    assert calls == [0, 1, 2, 3]
    assert report.cold_ms == pytest.approx(80)
    assert report.warm_ms == pytest.approx((20, 60, 30))
    assert report.warm_deadline_misses == 1
    assert report.warm_p50_ms == pytest.approx(30)
    assert report.warm_p95_ms == pytest.approx(57)


def test_timing_rejects_invalid_deadline_and_count() -> None:
    with pytest.raises(ValueError):
        measure_calls(lambda _: None, count=1, deadline_ms=0)
    with pytest.raises(ValueError):
        measure_calls(lambda _: None, count=0, deadline_ms=50)
