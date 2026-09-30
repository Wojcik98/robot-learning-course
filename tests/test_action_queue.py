"""Check chunk indexing, delayed actions, ensembling, and reset."""

import numpy as np
import pytest

from vla_lab.contracts import ActionChunk, ActionSchema
from vla_lab.runtime.action_queue import ActionQueue


def _chunk(origin: int, values: list[float], *, available: int | None = None) -> ActionChunk:
    return ActionChunk(
        episode_id="episode-1",
        reset_counter=0,
        request_id=f"request-{origin}",
        observation_tick=origin,
        availability_tick=origin if available is None else available,
        actions=np.repeat(np.asarray(values, dtype=np.float32)[:, None], 7, axis=1),
        schema=ActionSchema.libero_relative(),
        controller_period_ms=50,
    )


def test_chunk_index_follows_observation_tick() -> None:
    queue = ActionQueue("episode-1", 0, "bowl_plate")
    queue.push(_chunk(10, [0.1, 0.2, 0.3]))
    assert queue.action_at(11)[0] == pytest.approx(0.2)
    with pytest.raises(LookupError):
        queue.action_at(13)


def test_delayed_chunk_skips_past_actions() -> None:
    queue = ActionQueue("episode-1", 0, "bowl_plate")
    queue.push(_chunk(10, [0.1, 0.2, 0.3], available=12))
    with pytest.raises(LookupError):
        queue.action_at(11)
    assert queue.action_at(12)[0] == pytest.approx(0.3)


def test_temporal_ensemble_uses_aligned_predictions() -> None:
    queue = ActionQueue("episode-1", 0, "bowl_plate", ensemble_decay=0.0)
    queue.push(_chunk(10, [0.1, 0.2, 0.3]))
    queue.push(_chunk(11, [0.6, 0.7, 0.8]))
    assert queue.action_at(11, ensemble=True)[0] == pytest.approx(0.4)
    assert queue.action_at(11)[0] == pytest.approx(0.6)


def test_reset_clears_old_chunk_and_rejects_old_reset() -> None:
    queue = ActionQueue("episode-1", 0, "bowl_plate")
    old = _chunk(0, [0.1])
    queue.push(old)
    queue.reset("episode-2", 1, "bowl_stove")
    with pytest.raises(LookupError):
        queue.action_at(0)
    with pytest.raises(ValueError, match="episode or reset"):
        queue.push(old)


def test_queue_rejects_a_duplicate_request() -> None:
    queue = ActionQueue("episode-1", 0, "bowl_plate")
    chunk = _chunk(0, [0.1])
    queue.push(chunk)
    with pytest.raises(ValueError, match="repeats"):
        queue.push(chunk)
