"""Run one deterministic LIBERO episode on virtual controller ticks."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vla_lab.contracts import ActionChunk, ActionSchema
from vla_lab.runtime.virtual_scheduler import CONTROL_PERIOD_MS, TickRecord, VirtualScheduler
from vla_lab.teaching.interfaces import EpisodeEnvironment, LiberoObservation

Mode = Literal["synchronous", "naive_async", "temporal_ensemble", "rtc_hard", "rtc_soft"]
MODES = ("synchronous", "naive_async", "temporal_ensemble", "rtc_hard", "rtc_soft")
RTC_MODES = ("rtc_hard", "rtc_soft")
BOUNDARY_ACTIONS = 6

Predictor = Callable[
    [LiberoObservation, int, np.ndarray | None, int | None], tuple[np.ndarray, np.ndarray]
]


Env = EpisodeEnvironment[LiberoObservation]


@dataclass(frozen=True)
class EpisodeConfig:
    """Timing and mode choices for one episode."""

    mode: Mode
    horizon: int
    stride: int
    delay: int
    max_steps: int
    seed: int
    episode_id: str = "episode"
    task_id: str = "task"
    ensemble_decay: float = 0.25

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.horizon < 1 or self.stride < 1 or self.delay < 0 or self.max_steps < 1:
            raise ValueError(
                "Horizon, stride, and max steps must be positive; delay must not be negative"
            )
        if self.delay >= self.horizon:
            raise ValueError("Delay must be shorter than the horizon")
        if self.mode in RTC_MODES and not self.delay <= self.stride <= self.horizon - self.delay:
            raise ValueError("RTC needs delay <= stride <= horizon - delay")

    @property
    def effective_delay(self) -> int:
        """Synchronous mode applies each reply on the tick of its request."""
        return 0 if self.mode == "synchronous" else self.delay


@dataclass(frozen=True)
class TickLog:
    """Evidence for one controller tick."""

    tick: int
    action: np.ndarray
    underflow: bool
    observation_age_ticks: int | None
    queue_depth: int
    accepted: tuple[str, ...]
    rejected: tuple[str, ...]
    step_success: bool
    discontinuity: float | None


@dataclass(frozen=True)
class BoundaryRecord:
    """Action jumps over the first controller ticks after one accepted reply."""

    request_id: str
    start_tick: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class EpisodeResult:
    """All records from one episode."""

    episode_id: str
    reset_counter: int
    success: bool
    ticks: tuple[TickLog, ...]
    boundaries: tuple[BoundaryRecord, ...]


class EpisodeRunner:
    """Run episodes one after another and give each one a new reset counter."""

    def __init__(self, config: EpisodeConfig) -> None:
        self.config = config
        self._reset_counter = 0

    def run(self, env: Env, predictor: Predictor) -> EpisodeResult:
        """Reset the environment and scheduler, then step until success or max steps."""
        config = self.config
        reset_counter = self._reset_counter
        self._reset_counter += 1
        scheduler = VirtualScheduler(
            config.episode_id, reset_counter, config.task_id, ensemble_decay=config.ensemble_decay
        )
        observation, _ = env.reset(seed=config.seed)
        schema = ActionSchema.libero_relative()
        requested: list[tuple[int, np.ndarray]] = []
        logs: list[TickLog] = []
        tracked: dict[str, list[float]] = {}
        starts: dict[str, int] = {}
        previous_action: np.ndarray | None = None
        success = False

        for tick in range(config.max_steps):
            if tick % config.stride == 0:
                previous, origin = None, None
                if config.mode in RTC_MODES and requested:
                    origin, old = requested[-1]
                    previous = self._align(old, tick - origin)
                    if previous is None:
                        origin = None
                normalized, physical = predictor(observation, tick, previous, origin)
                self._check_chunk(normalized)
                self._check_chunk(physical)
                request_id = f"request-{tick}"
                scheduler.submit(
                    ActionChunk(
                        episode_id=config.episode_id,
                        reset_counter=reset_counter,
                        request_id=request_id,
                        observation_tick=tick,
                        availability_tick=tick + config.effective_delay,
                        actions=np.asarray(physical, dtype=np.float32).copy(),
                        schema=schema,
                        controller_period_ms=CONTROL_PERIOD_MS,
                    )
                )
                requested.append((tick, np.asarray(normalized, dtype=np.float32).copy()))

            record = scheduler.step(tick, ensemble=config.mode == "temporal_ensemble")
            jump = (
                None
                if previous_action is None
                else float(np.linalg.norm(record.action - previous_action))
            )
            previous_action = record.action
            for request_id in record.accepted:
                tracked[request_id] = []
                starts[request_id] = tick
            for values in tracked.values():
                if len(values) < BOUNDARY_ACTIONS and jump is not None:
                    values.append(jump)

            observation, _, terminated, truncated, info = env.step(record.action)
            success = bool(info.get("is_success", False))
            logs.append(self._log(record, success, jump))
            if success or terminated or truncated:
                break

        boundaries = tuple(
            BoundaryRecord(request_id, starts[request_id], tuple(values))
            for request_id, values in tracked.items()
        )
        return EpisodeResult(config.episode_id, reset_counter, success, tuple(logs), boundaries)

    def _check_chunk(self, chunk: np.ndarray) -> None:
        if chunk.shape != (self.config.horizon, 7):
            raise ValueError(f"Predictor chunk must have shape ({self.config.horizon}, 7)")

    def _align(self, chunk: np.ndarray, offset: int) -> np.ndarray | None:
        """Return only actions that remain at the new observation tick."""
        if offset >= len(chunk):
            return None
        return np.asarray(chunk[offset:], dtype=np.float32).copy()

    @staticmethod
    def _log(record: TickRecord, success: bool, jump: float | None) -> TickLog:
        return TickLog(
            tick=record.tick,
            action=record.action,
            underflow=record.underflow,
            observation_age_ticks=record.observation_age_ticks,
            queue_depth=record.queue_depth,
            accepted=record.accepted,
            rejected=record.rejected,
            step_success=success,
            discontinuity=jump,
        )


def run_episode(env: Env, predictor: Predictor, config: EpisodeConfig) -> EpisodeResult:
    """Run one episode with a new runner."""
    return EpisodeRunner(config).run(env, predictor)
