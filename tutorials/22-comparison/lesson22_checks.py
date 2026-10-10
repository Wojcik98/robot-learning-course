"""Behaviour checks for the Lesson 22 exercises.

These compare your statistics with values computed independently here, on inputs the
notebook checks do not use: unsorted samples, whole-number percentile positions,
latencies that are exact multiples of the period, mirrored success counts, and a brute-force
binomial sum. Read a failing assertion message first.
"""

import itertools
import math

import numpy as np


def _done(message: str) -> None:
    print(f"✓ {message}")


def _expect_value_error(call, message: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError(message)


def check_latency_summary(latency_summary) -> None:
    """Exercise 22.1: interpolated p95 for any order and size; Python types; inputs unchanged."""
    samples = np.array([0.30, 0.01, 0.20, 0.05, 0.02])
    original = samples.copy()
    report = latency_summary(samples)
    assert math.isclose(report["p95_s"], 0.20 + 0.8 * (0.30 - 0.20)), (
        f"sort first: sorted = [0.01, 0.02, 0.05, 0.20, 0.30], h = 3.8, p95 = 0.28; got {report['p95_s']}"
    )
    assert np.array_equal(samples, original), "do not sort the caller's array in place"
    assert type(report["count"]) is int and all(
        type(report[k]) is float for k in ("mean_s", "p95_s", "max_s")
    ), "count must be a Python int and the statistics Python floats"
    _done("unsorted input is sorted (without changing the caller's array)")

    whole = latency_summary(np.arange(21) / 100)  # M = 21 -> h = 19 exactly
    assert math.isclose(whole["p95_s"], 0.19), (
        f"h = 19 is a whole number: p95 = x_19 = 0.19, got {whole['p95_s']}"
    )
    pair = latency_summary([0.1, 0.3])  # h = 0.95
    assert math.isclose(pair["p95_s"], 0.1 + 0.95 * 0.2), "two samples: p95 = x0 + 0.95 (x1 - x0)"
    single = latency_summary([0.07])  # M = 1 -> h = 0
    assert single["p95_s"] == 0.07 and single["mean_s"] == 0.07, (
        "one sample: every statistic is that sample"
    )
    zeros = latency_summary((0.0, 0.0, 0.0))
    assert zeros["count"] == 3 and zeros["max_s"] == 0.0, "zero latencies are valid (not negative)"
    for size in (3, 10, 100):
        values = np.random.default_rng(size).exponential(0.05, size=size)
        assert math.isclose(latency_summary(values)["p95_s"], float(np.quantile(values, 0.95))), (
            f"{size} samples: p95 differs from linear interpolation"
        )
    _done("whole-number positions, two samples, zeros and several sizes match linear interpolation")

    _expect_value_error(
        lambda: latency_summary([0.1, float("inf")]), "an infinite sample must raise ValueError"
    )


def check_realtime_report(realtime_report) -> None:
    """Exercise 22.2: exact multiples of the period, zero latency, and unsorted calls."""
    # 7 * 0.05 is 0.35000000000000003 in floating point, and times 20 it is 7.000000000000001.
    report = realtime_report([7 * 0.05, 0.0, 0.1], 5)
    # 0.35 s spans 7 periods (6 missed), 0.0 s misses none, 0.1 s spans 2 (1 missed): 7 of 15.
    assert math.isclose(report["late_tick_fraction"], 7 / 15), (
        f"exact multiples of 0.05 s must not round up (subtract 1e-9 before ceil: 7 * 0.05 s "
        f"spans 7 periods); expected 7/15, got {report['late_tick_fraction']}"
    )
    assert math.isclose(report["async_fit_fraction"], 2 / 3), (
        "0.35 s > 5 * 0.05 s does not fit; 0 and 0.1 do"
    )
    assert math.isclose(report["mean_call_s"], 0.15) and math.isclose(
        report["amortized_s_per_tick"], 0.03
    )
    assert all(type(value) is float for value in report.values()), "return Python floats"
    _done("exact period multiples, zero latency, and the per-tick cost")

    one_step = realtime_report([0.04, 0.06], 1)
    assert math.isclose(one_step["late_tick_fraction"], 1 / 2), (
        "n = 1: 0.06 s misses one of two ticks"
    )
    assert math.isclose(one_step["async_fit_fraction"], 1 / 2), "n = 1: the budget is one period"
    _expect_value_error(
        lambda: realtime_report([0.1, float("nan")], 5), "a NaN latency must raise ValueError"
    )
    _expect_value_error(lambda: realtime_report([[0.1]], 5), "a 2-D input must raise ValueError")
    _done("n = 1 uses a one-period budget; invalid latencies raise ValueError")


def check_training_exposure(training_exposure) -> None:
    """Exercise 22.3: the three formulas, their types, and invalid inputs."""
    exposure = training_exposure(300, 4, 2)
    assert exposure["samples_seen"] == 1200 and type(exposure["samples_seen"]) is int
    assert type(exposure["samples_per_second"]) is float and math.isclose(
        exposure["samples_per_second"], 2.0
    ), "samples_per_second = batch_size / seconds_per_step, as a float"
    assert type(exposure["projected_hours"]) is float and math.isclose(
        exposure["projected_hours"], 600 / 3600
    ), "projected_hours = steps * seconds_per_step / 3600, as a float"
    for args in ((-5, 4, 0.1), (5, -1, 0.1), (5, 4, -0.1), (5, 4, float("inf"))):
        _expect_value_error(
            lambda args=args: training_exposure(*args),
            f"training_exposure{args} must raise ValueError",
        )
    _done("integer inputs still give float rates; negative or infinite values raise ValueError")


def check_wilson_interval(wilson_interval) -> None:
    """Exercise 22.4: the formula for several (s, N, z), mirror symmetry, and coverage of p-hat."""

    def formula(s: int, n: int, z: float) -> tuple[float, float]:
        p = s / n
        d = 1 + z * z / n
        center = (p + z * z / (2 * n)) / d
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return max(0.0, center - half), min(1.0, center + half)

    for s, n, z in ((0, 1, 1.96), (1, 1, 1.96), (3, 7, 1.0), (13, 50, 2.58), (49, 50, 1.96)):
        interval = wilson_interval(s, n, z)
        low, high = formula(s, n, z)
        assert math.isclose(interval["low"], low, abs_tol=1e-12) and math.isclose(
            interval["high"], high, abs_tol=1e-12
        ), (
            f"s={s}, N={n}, z={z}: expected [{low:.6f}, {high:.6f}], got [{interval['low']:.6f}, {interval['high']:.6f}]"
        )
        assert interval["low"] <= s / n <= interval["high"], (
            "the interval contains the observed rate"
        )
        mirrored = wilson_interval(n - s, n, z)
        assert math.isclose(mirrored["low"], 1 - interval["high"], abs_tol=1e-12), (
            "the interval for N - s failures mirrors the one for s successes"
        )
        assert all(type(v) is float for v in interval.values()), "return Python floats"
    _expect_value_error(lambda: wilson_interval(-1, 5), "negative successes must raise ValueError")
    _expect_value_error(lambda: wilson_interval(2, 5, z=-1.0), "negative z must raise ValueError")
    _done("matches the formula for five cases (z = 1, 1.96, 2.58), is symmetric, and covers p-hat")


def check_exact_sign_test(exact_sign_test) -> None:
    """Exercise 22.5: equals a brute-force count over all 2^n outcomes."""
    for left, right in itertools.product(range(7), repeat=2):
        n = left + right
        if n == 0:
            expected = 1.0
        else:
            k = min(left, right)
            extreme = sum(1 for flips in itertools.product((0, 1), repeat=n) if sum(flips) <= k)
            expected = min(1.0, 2 * extreme / 2**n)
        got = exact_sign_test(left, right)
        assert math.isclose(got, expected, abs_tol=1e-12), (
            f"left_only={left}, right_only={right}: expected p = {expected:.6f}, got {got:.6f}"
        )
        assert type(got) is float, "return a Python float"
    big = exact_sign_test(60, 40)
    assert 0 < big < 0.06, f"large counts must not overflow: p(60, 40) is about 0.057, got {big}"
    _done("matches brute-force enumeration for every pair up to 6 x 6; large counts work")
