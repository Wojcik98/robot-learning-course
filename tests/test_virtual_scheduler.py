"""Check exact virtual-tick response and controller behavior."""

import numpy as np
import pytest

from vla_lab.contracts import ActionChunk, ActionSchema
from vla_lab.runtime.virtual_scheduler import VirtualScheduler, latency_to_ticks


def chunk(origin: int, values: list[float], arrival: int, request: str) -> ActionChunk:
    actions = np.zeros((len(values), 7), dtype=np.float32)
    actions[:, 0] = values
    actions[:, 6] = -1
    return ActionChunk(
        episode_id="episode",
        reset_counter=0,
        request_id=request,
        observation_tick=origin,
        availability_tick=arrival,
        actions=actions,
        schema=ActionSchema.libero_relative(),
        controller_period_ms=50,
    )


def test_delay_skips_old_chunk_indices_and_underflow_holds_gripper() -> None:
    scheduler = VirtualScheduler("episode", 0, "bowl_plate")
    scheduler.submit(chunk(0, [0.1, 0.2, 0.3], 2, "first"))

    rows = [scheduler.step(tick) for tick in range(4)]

    assert [row.action[0] for row in rows] == pytest.approx([0, 0, 0.3, 0])
    assert [row.action[6] for row in rows] == pytest.approx([0, 0, -1, -1])
    assert [row.underflow for row in rows] == [True, True, False, True]
    assert rows[2].observation_age_ticks == 2


def test_reordered_and_duplicate_responses_cannot_replace_newer_chunk() -> None:
    scheduler = VirtualScheduler("episode", 0, "bowl_plate")
    scheduler.submit(chunk(0, [0.1, 0.2, 0.3], 2, "old"))
    scheduler.submit(chunk(1, [0.4, 0.5, 0.6], 1, "new"))
    scheduler.submit(chunk(1, [0.4, 0.5, 0.6], 1, "new"))

    rows = [scheduler.step(tick) for tick in range(3)]

    assert rows[1].action[0] == pytest.approx(0.4)
    assert rows[2].action[0] == pytest.approx(0.5)
    assert rows[1].rejected == ("duplicate",)
    assert rows[2].rejected == ("reordered",)


def test_policy_outage_defers_response_and_reset_discards_old_work() -> None:
    scheduler = VirtualScheduler("episode", 0, "bowl_plate")
    scheduler.submit(chunk(0, [0.1, 0.2, 0.3], 1, "pending"))
    scheduler.step(0)
    outage = scheduler.step(1, policy_available=False)
    assert outage.underflow
    assert outage.pending_responses == 1
    scheduler.reset("other-episode", 1, "bowl_plate")
    reset_row = scheduler.step(0)
    assert reset_row.underflow and reset_row.pending_responses == 0
    with pytest.raises(ValueError, match="episode or reset"):
        scheduler.submit(chunk(0, [0.1], 0, "stale"))


def test_fixed_jitter_rejects_late_old_reply_and_keeps_new_chunk() -> None:
    scheduler = VirtualScheduler("episode", 0, "bowl_plate")
    scheduler.submit(chunk(0, [0.1, 0.2, 0.3, 0.4], 3, "late-old"))
    scheduler.submit(chunk(2, [0.5, 0.6, 0.7], 2, "early-new"))

    rows = [scheduler.step(tick) for tick in range(4)]

    assert [row.action[0] for row in rows] == pytest.approx([0, 0, 0.5, 0.6])
    assert rows[2].accepted == ("early-new",)
    assert rows[3].rejected == ("reordered",)


def test_outage_can_exhaust_reply_and_camera_fault_is_reported() -> None:
    scheduler = VirtualScheduler("episode", 0, "bowl_plate")
    scheduler.submit(chunk(0, [0.1, 0.2, 0.3], 2, "exhausted"))
    rows = [
        scheduler.step(tick, policy_available=tick not in (2, 3), camera_available=tick != 1)
        for tick in range(5)
    ]

    assert rows[1].camera_available is False
    assert rows[2].pending_responses == 1
    assert rows[4].rejected == ("exhausted",)
    assert rows[4].underflow


def test_latency_quantizes_up_and_ticks_are_sequential() -> None:
    assert [latency_to_ticks(ms) for ms in (0, 1, 50, 51, 200)] == [0, 1, 1, 2, 4]
    scheduler = VirtualScheduler("episode", 0, "bowl_plate")
    with pytest.raises(ValueError, match="next controller tick"):
        scheduler.step(1)
