"""Measure wall-clock calls apart from virtual controller availability."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CallTiming:
    """Cold and warm durations against one declared deadline."""

    deadline_ms: float
    cold_ms: float
    cold_deadline_miss: bool
    warm_ms: tuple[float, ...]
    warm_deadline_misses: int
    warm_p50_ms: float | None
    warm_p95_ms: float | None


def measure_calls(
    call: Callable[[int], object],
    *,
    count: int,
    deadline_ms: float,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] | None = None,
) -> CallTiming:
    """Time calls after optional device synchronization and count misses."""
    if count < 1 or not math.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError("Call count and deadline must be positive and finite")
    durations: list[float] = []
    for index in range(count):
        if synchronize is not None:
            synchronize()
        start = clock()
        call(index)
        if synchronize is not None:
            synchronize()
        elapsed_ms = (clock() - start) * 1000
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("Measured duration must be finite and nonnegative")
        durations.append(elapsed_ms)
    warm = tuple(durations[1:])
    return CallTiming(
        deadline_ms=deadline_ms,
        cold_ms=durations[0],
        cold_deadline_miss=durations[0] > deadline_ms,
        warm_ms=warm,
        warm_deadline_misses=sum(value > deadline_ms for value in warm),
        warm_p50_ms=float(np.percentile(warm, 50)) if warm else None,
        warm_p95_ms=float(np.percentile(warm, 95)) if warm else None,
    )
