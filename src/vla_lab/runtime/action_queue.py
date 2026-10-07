"""Align policy action chunks to the simulator's controller ticks."""

import math

import numpy as np

from vla_lab.contracts import ActionChunk


class ActionQueue:
    """Keep valid chunks for one task and one simulator reset."""

    def __init__(
        self, episode_id: str, reset_counter: int, task_id: str, *, ensemble_decay: float = 0.25
    ) -> None:
        if not math.isfinite(ensemble_decay) or ensemble_decay < 0:
            raise ValueError("Ensemble decay must be finite and nonnegative")
        self.ensemble_decay = ensemble_decay
        self.reset(episode_id, reset_counter, task_id)

    def reset(self, episode_id: str, reset_counter: int, task_id: str) -> None:
        """Drop all pending chunks after a simulator reset or task change."""
        if not episode_id or not task_id or reset_counter < 0:
            raise ValueError("Queue needs a valid episode, reset, and task")
        self.episode_id = episode_id
        self.reset_counter = reset_counter
        self.task_id = task_id
        self._chunks: list[ActionChunk] = []

    def push(self, chunk: ActionChunk) -> None:
        """Add one chunk only for the current episode and reset."""
        if (chunk.episode_id, chunk.reset_counter) != (self.episode_id, self.reset_counter):
            raise ValueError("Action chunk belongs to another episode or reset")
        if any(
            item.request_id == chunk.request_id or item.observation_tick == chunk.observation_tick
            for item in self._chunks
        ):
            raise ValueError("Action chunk repeats a request or observation tick")
        self._chunks.append(chunk)
        self._chunks.sort(key=lambda item: item.observation_tick)

    def action_at(self, tick: int, *, ensemble: bool = False) -> np.ndarray:
        """Select or average actions whose target and availability ticks match."""
        if tick < 0:
            raise ValueError("Controller tick must not be negative")
        candidates = [
            chunk
            for chunk in self._chunks
            if chunk.availability_tick <= tick < chunk.observation_tick + len(chunk.actions)
            and chunk.observation_tick <= tick
        ]
        if not candidates:
            raise LookupError(f"No valid action is available at controller tick {tick}")
        if not ensemble:
            chunk = candidates[-1]
            return chunk.actions[tick - chunk.observation_tick].copy()
        actions = np.stack([chunk.actions[tick - chunk.observation_tick] for chunk in candidates])
        weights = np.asarray(
            [
                math.exp(-self.ensemble_decay * (tick - chunk.observation_tick))
                for chunk in candidates
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        return np.average(actions, axis=0, weights=weights).astype(np.float32)
