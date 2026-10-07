"""Apply delayed action chunks on deterministic 50 ms simulator ticks."""

import math
from dataclasses import dataclass

import numpy as np

from vla_lab.contracts import ActionChunk
from vla_lab.runtime.action_queue import ActionQueue

CONTROL_PERIOD_MS = 50


def latency_to_ticks(latency_ms: float) -> int:
    """Round measured latency up to the next controller tick."""
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("Latency must be finite and nonnegative")
    return math.ceil(latency_ms / CONTROL_PERIOD_MS)


@dataclass(frozen=True)
class TickRecord:
    """One controller action with its virtual availability evidence."""

    tick: int
    action: np.ndarray
    underflow: bool
    observation_age_ticks: int | None
    queue_depth: int
    pending_responses: int
    accepted: tuple[str, ...]
    rejected: tuple[str, ...]
    policy_available: bool
    camera_available: bool


class VirtualScheduler:
    """Deliver policy responses and select actions in controller-tick order."""

    def __init__(
        self, episode_id: str, reset_counter: int, task_id: str, *, ensemble_decay: float = 0.25
    ) -> None:
        self.ensemble_decay = ensemble_decay
        self.reset(episode_id, reset_counter, task_id)

    def reset(self, episode_id: str, reset_counter: int, task_id: str) -> None:
        """Clear old requests, actions, and gripper state after a reset."""
        self.queue = ActionQueue(
            episode_id, reset_counter, task_id, ensemble_decay=self.ensemble_decay
        )
        self.episode_id = episode_id
        self.reset_counter = reset_counter
        self.task_id = task_id
        self._pending: list[tuple[int, ActionChunk]] = []
        self._accepted: list[ActionChunk] = []
        self._seen_requests: set[str] = set()
        self._latest_observation_tick = -1
        self._next_tick = 0
        self._last_gripper = 0.0
        self._submission_count = 0

    def submit(self, chunk: ActionChunk) -> None:
        """Schedule one response for its declared virtual availability tick."""
        if (chunk.episode_id, chunk.reset_counter) != (self.episode_id, self.reset_counter):
            raise ValueError("Action chunk belongs to another episode or reset")
        self._pending.append((self._submission_count, chunk))
        self._submission_count += 1

    def step(
        self,
        tick: int,
        *,
        ensemble: bool = False,
        policy_available: bool = True,
        camera_available: bool = True,
    ) -> TickRecord:
        """Deliver due replies, then apply the action aligned to this tick."""
        if tick != self._next_tick:
            raise ValueError(f"Expected next controller tick {self._next_tick}, got {tick}")
        self._next_tick += 1
        accepted: list[str] = []
        rejected: list[str] = []
        if policy_available:
            due = sorted(
                (item for item in self._pending if item[1].availability_tick <= tick),
                key=lambda item: (item[1].availability_tick, item[0]),
            )
            self._pending = [item for item in self._pending if item[1].availability_tick > tick]
            for _, chunk in due:
                if chunk.request_id in self._seen_requests:
                    rejected.append("duplicate")
                    continue
                self._seen_requests.add(chunk.request_id)
                if chunk.observation_tick <= self._latest_observation_tick:
                    rejected.append("reordered")
                    continue
                if tick >= chunk.observation_tick + len(chunk.actions):
                    rejected.append("exhausted")
                    continue
                self.queue.push(chunk)
                self._accepted.append(chunk)
                self._latest_observation_tick = chunk.observation_tick
                accepted.append(chunk.request_id)

        active = [
            chunk
            for chunk in self._accepted
            if chunk.availability_tick <= tick < chunk.observation_tick + len(chunk.actions)
        ]
        if active:
            action = self.queue.action_at(tick, ensemble=ensemble)
            age = tick - max(chunk.observation_tick for chunk in active)
            underflow = False
        else:
            action = np.zeros(7, dtype=np.float32)
            action[6] = self._last_gripper
            age = None
            underflow = True
        self._last_gripper = float(action[6])
        return TickRecord(
            tick=tick,
            action=action.copy(),
            underflow=underflow,
            observation_age_ticks=age,
            queue_depth=len(active),
            pending_responses=len(self._pending),
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            policy_available=policy_available,
            camera_available=camera_available,
        )
