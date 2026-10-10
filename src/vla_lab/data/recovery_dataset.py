"""Make ACT training samples from checked recovery recordings."""

from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.processor import LiberoProcessorStep
from torch.utils.data import Dataset

from vla_lab.contracts import ActionSchema
from vla_lab.data.recording import CONTROL_PERIOD_MS, RecordedEpisode
from vla_lab.teaching.interfaces import (
    ProcessorObservation,
    RawObservation,
    RawObservationValue,
    RecordedObservation,
    TransitionRowsFunction,
    WindowFunction,
)
from vla_lab.teaching.records import AlignedTransition

ACT_CHUNK_SIZE = 20
_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
_SAMPLE_KEYS = (*_IMAGE_KEYS, "observation.state")


def _single_env_batch(observation: Mapping[str, RawObservationValue]) -> RawObservation:
    """Add a batch axis to raw NumPy leaves for the pinned LIBERO path."""
    result: RawObservation = {}
    for key, value in observation.items():
        if isinstance(value, Mapping):
            result[key] = _single_env_batch(value)
        elif isinstance(value, np.ndarray):
            result[key] = np.expand_dims(value, axis=0)
        else:
            result[key] = value
    return result


def _values_equal(left: object, right: object) -> bool:
    """Compare copied nested observations without relying on object identity."""
    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, np.ndarray):
        return bool(result.all())
    return bool(result)


def _libero_observation_transform(
    observation: Mapping[str, RawObservationValue],
) -> dict[str, torch.Tensor]:
    """Use the same observation conversion as the ACT simulator path."""

    processed = LiberoProcessorStep().observation(
        cast(Callable[[RawObservation], ProcessorObservation], preprocess_observation)(
            _single_env_batch(observation)
        )
    )
    features: dict[str, torch.Tensor] = {}
    for key in _SAMPLE_KEYS:
        if key not in processed:
            raise ValueError(f"LIBERO observation conversion is missing {key}")
        value = torch.as_tensor(processed[key]).detach().cpu()
        if value.shape[:1] == (1,):
            value = value[0]
        features[key] = value.to(dtype=torch.float32).contiguous()
    return features


class RecoveryDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose each recorded 20 Hz controller tick as a 20-action ACT sample.

    The task ID and allowed reset seeds are explicit provenance checks. Pass a
    pure observation transform in tests or the default pinned LIBERO transform.
    """

    def __init__(
        self,
        episode_paths: Sequence[str | Path],
        *,
        task_id: str,
        expected_reset_seeds: Collection[int],
        observation_transform: Callable[
            [Mapping[str, RawObservationValue]], Mapping[str, torch.Tensor]
        ] = _libero_observation_transform,
        transition_rows_fn: TransitionRowsFunction | None = None,
        window_fn: WindowFunction | None = None,
    ) -> None:
        if not episode_paths:
            raise ValueError("Recovery dataset needs at least one recording")
        if not task_id:
            raise ValueError("Expected task ID must be nonempty")
        seeds = set(expected_reset_seeds)
        if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("Expected reset seeds must be nonnegative integers")

        episodes: list[RecordedEpisode] = []
        episode_ids: set[str] = set()
        for path in episode_paths:
            episode = RecordedEpisode.load(path)
            if episode.task_id != task_id:
                raise ValueError(f"Recording task {episode.task_id!r} does not match {task_id!r}")
            if episode.reset_seed is None or episode.reset_seed not in seeds:
                raise ValueError(f"Recording reset seed is missing or not allowed: {path}")
            if episode.episode_id in episode_ids:
                raise ValueError(f"Duplicate recording episode ID: {episode.episode_id}")
            episode_ids.add(episode.episode_id)
            if episode.defect is not None:
                raise ValueError(f"Recording has a declared timing defect: {episode.episode_id}")
            if episode.controller_period_ms != CONTROL_PERIOD_MS:
                raise ValueError("Recording does not use the pinned 50 ms controller period")
            schema = ActionSchema(
                **{
                    **vars(episode.schema),
                    "lower": tuple(episode.schema.lower),
                    "upper": tuple(episode.schema.upper),
                    "padding_mask": tuple(episode.schema.padding_mask),
                }
            )
            if schema != ActionSchema.libero_relative():
                raise ValueError("Recording action schema does not match the LIBERO controller")
            episodes.append(episode)

        self._episodes = episodes
        self._observation_transform = observation_transform
        self._window_fn = window_fn
        self._starts: list[tuple[int, int]] = []
        self._observations: list[Mapping[str, RawObservationValue]] = []
        self._actions_by_episode: list[np.ndarray] = []
        for episode_index, episode in enumerate(episodes):
            transitions: list[AlignedTransition[RecordedObservation]]
            if transition_rows_fn is None:
                transitions = [
                    AlignedTransition(
                        tick=tick,
                        observation=episode.observations[tick],
                        action=episode.actions[tick],
                    )
                    for tick in range(len(episode.actions))
                ]
            else:
                transitions = transition_rows_fn(
                    episode.observations,
                    episode.actions,
                    episode.observation_ticks,
                    episode.action_ticks,
                )
            if len(transitions) != len(episode.actions):
                raise ValueError(
                    "transition_rows_fn must return one aligned row per applied action"
                )
            if not all(isinstance(row, AlignedTransition) for row in transitions):
                raise TypeError("transition_rows_fn must return AlignedTransition rows")
            for tick, row in enumerate(transitions):
                action = np.asarray(row.action, dtype=np.float32)
                if action.shape != (7,) or not np.array_equal(action, episode.actions[tick]):
                    raise ValueError("transition_rows_fn changed an applied action or its order")
            selected = list(transitions)
            source_rows = {int(row.tick): row for row in transitions}
            selected_actions = []
            for selected_tick, selected_row in enumerate(selected):
                tick = int(selected_row.tick)
                if (
                    tick != selected_tick
                    or tick not in source_rows
                    or not _values_equal(selected_row.observation, episode.observations[tick])
                ):
                    raise ValueError("transition_rows_fn must select original aligned rows")
                if selected_tick and tick <= int(selected[selected_tick - 1].tick):
                    raise ValueError("transition_rows_fn must preserve transition order")
                action = np.asarray(selected_row.action, dtype=np.float32)
                if action.shape != (7,) or not np.array_equal(action, episode.actions[tick]):
                    raise ValueError("transition_rows_fn changed an applied action")
                selected_actions.append(action.copy())
                self._starts.append((episode_index, selected_tick))
                self._observations.append(selected_row.observation)
            self._actions_by_episode.append(np.stack(selected_actions))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, source_tick = self._starts[index]
        transformed = self._observation_transform(self._observations[index])
        sample: dict[str, torch.Tensor] = {}
        expected_shapes = {
            _IMAGE_KEYS[0]: (3, 256, 256),
            _IMAGE_KEYS[1]: (3, 256, 256),
            "observation.state": (8,),
        }
        for key, shape in expected_shapes.items():
            if key not in transformed:
                raise ValueError(f"Observation transform is missing {key}")
            value = torch.as_tensor(transformed[key], dtype=torch.float32).detach().cpu()
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Transformed {key} must be finite with shape {shape}")
            if key in _IMAGE_KEYS and (value.min() < 0 or value.max() > 1):
                raise ValueError(f"Transformed {key} values must be in [0, 1]")
            sample[key] = value.contiguous()

        if self._window_fn is not None:
            actions, padding = self._window_fn(
                self._actions_by_episode[episode_index],
                start=source_tick,
                horizon=ACT_CHUNK_SIZE,
            )
            actions = np.asarray(actions, dtype=np.float32)
            padding = np.asarray(padding, dtype=np.bool_)
            if actions.shape != (ACT_CHUNK_SIZE, 7) or padding.shape != (ACT_CHUNK_SIZE,):
                raise ValueError("window_fn must return a 20x7 action chunk and a 20-value mask")
        else:
            actions = np.zeros((ACT_CHUNK_SIZE, 7), dtype=np.float32)
            padding = np.ones(ACT_CHUNK_SIZE, dtype=np.bool_)
            source_indices = source_tick + np.arange(ACT_CHUNK_SIZE)
            valid = source_indices < len(self._actions_by_episode[episode_index])
            action_indices = source_indices[valid]
            actions[: len(action_indices)] = self._actions_by_episode[episode_index][action_indices]
            padding[: len(action_indices)] = False
            if len(action_indices):
                actions[len(action_indices) :] = actions[len(action_indices) - 1]
        sample["action"] = torch.from_numpy(actions)
        sample["action_is_pad"] = torch.from_numpy(padding)
        return sample
