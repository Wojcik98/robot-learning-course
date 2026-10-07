"""Record keyboard actions with observations from the same simulator tick."""

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import BinaryIO, ClassVar, Protocol, cast, overload

import numpy as np

from vla_lab.contracts import ActionSchema
from vla_lab.teaching.interfaces import (
    EpisodeEnvironment,
    RawObservation,
    RawObservationValue,
    RecordedObservation,
)
from vla_lab.teaching.records import RecordingDefect
from vla_lab.timing import LIBERO_CONTROL_HZ, require_libero_rate

CONTROL_PERIOD_MS = 1000 // LIBERO_CONTROL_HZ
_TRANSLATION_STEP = 0.25
_ROTATION_STEP = 0.25


class KeyboardActionMapper:
    """Map key names to bounded seven-value LIBERO relative actions."""

    _AXES: ClassVar[dict[str, tuple[int, float, float]]] = {
        "w": (0, 1.0, _TRANSLATION_STEP),
        "s": (0, -1.0, _TRANSLATION_STEP),
        "d": (1, 1.0, _TRANSLATION_STEP),
        "a": (1, -1.0, _TRANSLATION_STEP),
        "r": (2, 1.0, _TRANSLATION_STEP),
        "f": (2, -1.0, _TRANSLATION_STEP),
        "u": (3, 1.0, _ROTATION_STEP),
        "o": (3, -1.0, _ROTATION_STEP),
        "j": (4, 1.0, _ROTATION_STEP),
        "l": (4, -1.0, _ROTATION_STEP),
        "i": (5, 1.0, _ROTATION_STEP),
        "k": (5, -1.0, _ROTATION_STEP),
    }

    def action_for(self, keys: Iterable[str]) -> np.ndarray:
        """Return one action for keys that are down during this tick."""
        pressed = {key.strip().lower() for key in keys}
        action = np.zeros(7, dtype=np.float32)
        for key, (axis, direction, scale) in self._AXES.items():
            if key in pressed:
                action[axis] += direction * scale
        if "open" in pressed and "close" not in pressed:
            action[6] = -1.0
        elif "close" in pressed and "open" not in pressed:
            action[6] = 1.0
        schema = ActionSchema.libero_relative()
        return np.clip(action, schema.lower, schema.upper).astype(np.float32, copy=False)


class ArrayArchiveWriter(Protocol):
    def __call__(self, file: BinaryIO, **arrays: np.ndarray) -> None: ...


@overload
def _copy_observation(value: Mapping[str, RawObservationValue]) -> RecordedObservation: ...


@overload
def _copy_observation(value: RawObservationValue) -> RecordedObservation | np.ndarray: ...


def _copy_observation(value: RawObservationValue) -> RecordedObservation | np.ndarray:
    """Copy numeric leaves from a simulator observation without Python objects."""
    if isinstance(value, Mapping):
        if not value or any(not isinstance(key, str) for key in value):
            raise ValueError("Observation mappings need string keys and at least one item")
        return {key: _copy_observation(item) for key, item in value.items()}
    array = np.asarray(value)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise ValueError("Observation leaves must be numeric arrays or values")
    if not np.isfinite(array).all():
        raise ValueError("Observation has a nonfinite value")
    return np.array(array, copy=True)


def _flatten_observation(
    observation: Mapping[str, RawObservationValue], prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], np.ndarray]:
    leaves: dict[tuple[str, ...], np.ndarray] = {}
    for key, value in observation.items():
        path = prefix + (key,)
        if isinstance(value, Mapping):
            leaves.update(_flatten_observation(value, path))
        else:
            leaves[path] = np.asarray(value)
    return leaves


def _nest_observation(leaves: Mapping[tuple[str, ...], np.ndarray]) -> RecordedObservation:
    root: RecordedObservation = {}
    for path, value in leaves.items():
        node = root
        for key in path[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError("Saved observation paths overlap")
            node = child
        if path[-1] in node:
            raise ValueError("Saved observation path is duplicated")
        node[path[-1]] = np.array(value, copy=True)
    return root


@dataclass(frozen=True)
class RecordedEpisode:
    """One episode with N applied actions and N+1 synchronized observations."""

    episode_id: str
    task_id: str
    actions: np.ndarray
    action_ticks: np.ndarray
    observation_ticks: np.ndarray
    observations: tuple[RecordedObservation, ...]
    defect: RecordingDefect | None = None
    reset_seed: int | None = None
    controller_period_ms: int = CONTROL_PERIOD_MS
    schema: ActionSchema = field(default_factory=ActionSchema.libero_relative)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.task_id:
            raise ValueError("Recording needs an episode ID and task ID")
        if self.reset_seed is not None and self.reset_seed < 0:
            raise ValueError("Reset seed must not be negative")
        if self.controller_period_ms != CONTROL_PERIOD_MS:
            raise ValueError("Recording must use the 50 ms simulator period")
        actions = np.asarray(self.actions, dtype=np.float32)
        action_ticks = np.asarray(self.action_ticks, dtype=np.int64)
        observation_ticks = np.asarray(self.observation_ticks, dtype=np.int64)
        if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 1:
            raise ValueError("Recording actions must have shape (steps, 7)")
        if not np.isfinite(actions).all():
            raise ValueError("Recording actions must be finite")
        if np.any(actions < self.schema.lower) or np.any(actions > self.schema.upper):
            raise ValueError("Recording actions exceed controller bounds")
        if action_ticks.shape != (len(actions),) or observation_ticks.shape != (len(actions) + 1,):
            raise ValueError("Recording tick counts do not match its actions")
        if observation_ticks[0] != 0 or np.any(np.diff(observation_ticks) != 1):
            raise ValueError("Observation ticks must start at zero and increase by one")
        if not self.observations or len(self.observations) != len(observation_ticks):
            raise ValueError("Recording needs one observation at each tick")
        copied = tuple(_copy_observation(item) for item in self.observations)
        paths = _flatten_observation(copied[0])
        if not paths:
            raise ValueError("Recording observations need numeric values")
        for item in copied[1:]:
            leaves = _flatten_observation(item)
            if leaves.keys() != paths.keys() or any(
                leaves[path].shape != paths[path].shape for path in paths
            ):
                raise ValueError("Recording observation structure changed during the episode")
        expected_ticks = observation_ticks[:-1]
        if self.defect is None:
            if not np.array_equal(action_ticks, expected_ticks):
                raise ValueError("Each action must use the tick of its observation")
        elif (
            self.defect.get("label") != "synthetic timing defect"
            or self.defect.get("kind") != "action_tick_offset"
            or not np.array_equal(
                action_ticks,
                expected_ticks + int(self.defect.get("offset_ticks", 0)),
            )
        ):
            raise ValueError("Recording defect metadata does not match its tick labels")
        object.__setattr__(self, "actions", np.array(actions, copy=True))
        object.__setattr__(self, "action_ticks", np.array(action_ticks, copy=True))
        object.__setattr__(self, "observation_ticks", np.array(observation_ticks, copy=True))
        object.__setattr__(self, "observations", copied)
        object.__setattr__(self, "defect", None if self.defect is None else dict(self.defect))

    @property
    def simulated_duration_s(self) -> float:
        """Return the duration of the saved control transitions in seconds."""
        return len(self.actions) * self.controller_period_ms / 1000

    def save(self, path: str | Path) -> Path:
        """Save arrays and metadata in a pickle-free compressed NPZ file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        first_leaves = _flatten_observation(self.observations[0])
        paths = sorted(first_leaves)
        metadata = {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "reset_seed": self.reset_seed,
            "controller_period_ms": self.controller_period_ms,
            "simulated_duration_s": self.simulated_duration_s,
            "saved_transition_hz": 1000 / self.controller_period_ms,
            "action_schema": {
                **vars(self.schema),
                "lower": list(self.schema.lower),
                "upper": list(self.schema.upper),
                "padding_mask": list(self.schema.padding_mask),
            },
            "observation_paths": [list(path) for path in paths],
            "defect": self.defect,
        }
        values: dict[str, np.ndarray] = {
            "actions": self.actions,
            "action_ticks": self.action_ticks,
            "observation_ticks": self.observation_ticks,
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        for index, path_tuple in enumerate(paths):
            values[f"observation_{index}"] = np.stack(
                [_flatten_observation(item)[path_tuple] for item in self.observations]
            )
        with target.open("wb") as stream:
            # These keys are array names, not NumPy control arguments.
            cast(ArrayArchiveWriter, np.savez_compressed)(stream, **values)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RecordedEpisode":
        """Load and validate an NPZ episode without enabling pickle."""
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if metadata.get("schema_version") != 1:
                raise ValueError("Unsupported recording schema version")
            paths = [tuple(item) for item in metadata["observation_paths"]]
            arrays = [
                np.array(saved[f"observation_{index}"], copy=True) for index in range(len(paths))
            ]
            observations = tuple(
                _nest_observation(
                    {path: values[row] for path, values in zip(paths, arrays, strict=True)}
                )
                for row in range(len(saved["observation_ticks"]))
            )
            return cls(
                episode_id=metadata["episode_id"],
                task_id=metadata["task_id"],
                actions=np.array(saved["actions"], copy=True),
                action_ticks=np.array(saved["action_ticks"], copy=True),
                observation_ticks=np.array(saved["observation_ticks"], copy=True),
                observations=observations,
                defect=metadata.get("defect"),
                reset_seed=metadata.get("reset_seed"),
                controller_period_ms=metadata["controller_period_ms"],
                schema=ActionSchema(**metadata["action_schema"]),
            )


def record_episode(
    environment: EpisodeEnvironment[RawObservation],
    keys_at_tick: Callable[[int, Mapping[str, RawObservationValue]], Iterable[str]],
    *,
    task_id: str,
    max_steps: int = 300,
    episode_id: str,
    reset_seed: int | None = None,
    control_hz: int = LIBERO_CONTROL_HZ,
) -> RecordedEpisode:
    """Record bounded keyboard actions with pre-action simulator observations."""
    mapper = KeyboardActionMapper()
    return record_action_episode(
        environment,
        lambda tick, observation: mapper.action_for(keys_at_tick(tick, observation)),
        task_id=task_id,
        max_steps=max_steps,
        episode_id=episode_id,
        reset_seed=reset_seed,
        control_hz=control_hz,
    )


def record_action_episode(
    environment: EpisodeEnvironment[RawObservation],
    action_at_tick: Callable[[int, Mapping[str, RawObservationValue]], np.ndarray],
    *,
    task_id: str,
    max_steps: int = 300,
    episode_id: str,
    reset_seed: int | None = None,
    control_hz: int = LIBERO_CONTROL_HZ,
) -> RecordedEpisode:
    """Record the bounded action applied after each matching observation."""
    require_libero_rate(control_hz)
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    reset = environment.reset(seed=reset_seed) if reset_seed is not None else environment.reset()
    observation = _copy_observation(reset[0] if isinstance(reset, tuple) else reset)
    schema = ActionSchema.libero_relative()
    observations = [observation]
    actions: list[np.ndarray] = []
    action_ticks: list[int] = []
    observation_ticks = [0]
    for tick in range(max_steps):
        proposed = np.asarray(
            action_at_tick(tick, _copy_observation(observation)), dtype=np.float32
        )
        if proposed.shape != (7,) or not np.isfinite(proposed).all():
            raise ValueError("Recorder action must have seven finite values")
        action = np.clip(proposed, schema.lower, schema.upper).astype(np.float32, copy=False)
        next_observation, _reward, terminated, truncated, info = environment.step(action.copy())
        actions.append(action.copy())
        action_ticks.append(tick)
        observation = _copy_observation(next_observation)
        observations.append(observation)
        observation_ticks.append(tick + 1)
        if bool(info.get("is_success", False)) or terminated or truncated:
            break
    return RecordedEpisode(
        episode_id=episode_id,
        task_id=task_id,
        actions=np.stack(actions),
        action_ticks=np.asarray(action_ticks),
        observation_ticks=np.asarray(observation_ticks),
        observations=tuple(observations),
        reset_seed=reset_seed,
        controller_period_ms=1000 // control_hz,
    )


def replay_episode(
    environment: EpisodeEnvironment[RawObservation], episode: RecordedEpisode
) -> tuple[RecordedObservation, ...]:
    """Replay saved applied actions and return observations at every control tick."""
    reset = (
        environment.reset(seed=episode.reset_seed)
        if episode.reset_seed is not None
        else environment.reset()
    )
    observations = [_copy_observation(reset[0] if isinstance(reset, tuple) else reset)]
    for action in episode.actions:
        observation, _reward, _terminated, _truncated, _info = environment.step(action.copy())
        observations.append(_copy_observation(observation))
    return tuple(observations)


def apply_synthetic_timing_defect(
    episode: RecordedEpisode, *, offset_ticks: int = 1
) -> RecordedEpisode:
    """Apply a labelled tick-label fault for the timing lesson."""
    if (
        episode.defect is not None
        or not isinstance(offset_ticks, Integral)
        or isinstance(offset_ticks, bool)
        or offset_ticks == 0
    ):
        raise ValueError("Synthetic timing defect needs a clean episode and nonzero offset")
    defect = {
        "label": "synthetic timing defect",
        "kind": "action_tick_offset",
        "offset_ticks": int(offset_ticks),
        "source_history": "unknown; this defect was created for the lesson",
    }
    return RecordedEpisode(
        episode_id=episode.episode_id,
        task_id=episode.task_id,
        actions=episode.actions,
        action_ticks=episode.action_ticks + int(offset_ticks),
        observation_ticks=episode.observation_ticks,
        observations=episode.observations,
        defect=defect,
        reset_seed=episode.reset_seed,
    )


def repair_synthetic_timing_defect(episode: RecordedEpisode) -> RecordedEpisode:
    """Repair the labelled synthetic tick offset and clear its fault label."""
    if episode.defect is None or episode.defect.get("label") != "synthetic timing defect":
        raise ValueError("Episode does not contain a labelled synthetic timing defect")
    offset = int(episode.defect["offset_ticks"])
    return RecordedEpisode(
        episode_id=episode.episode_id,
        task_id=episode.task_id,
        actions=episode.actions,
        action_ticks=episode.action_ticks - offset,
        observation_ticks=episode.observation_ticks,
        observations=episode.observations,
        reset_seed=episode.reset_seed,
    )
