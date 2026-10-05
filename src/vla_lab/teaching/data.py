"""Episode-safe data helpers for learner exercises."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vla_lab.data.stats import TrainingStats


def order_episode_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return rows ordered by episode and frame, rejecting duplicate keys."""
    keyed: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = (int(row["episode_index"]), int(row["frame_index"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("each row needs integer episode_index and frame_index") from error
        if min(key) < 0 or key in keyed:
            raise ValueError("episode/frame keys must be nonnegative and unique")
        keyed[key] = row
    return [keyed[key] for key in sorted(keyed)]


def extract_episode_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Read ordered model inputs from raw LeRobot rows for one episode."""
    ordered = order_episode_rows(rows)
    if not ordered:
        raise ValueError("at least one raw episode row is required")
    episode_ids = {int(row["episode_index"]) for row in ordered}
    if len(episode_ids) != 1:
        raise ValueError("raw rows must belong to one complete episode")
    try:
        actions = np.stack([np.asarray(row["action"], dtype=np.float32) for row in ordered])
        states = np.stack(
            [np.asarray(row["observation.state"], dtype=np.float32) for row in ordered]
        )
    except (KeyError, ValueError) as error:
        raise ValueError("raw rows need consistent action and observation.state vectors") from error
    if (
        actions.ndim != 2
        or states.ndim != 2
        or not np.isfinite(actions).all()
        or not np.isfinite(states).all()
    ):
        raise ValueError("raw action and state rows must be finite feature matrices")
    return {
        "episode_index": episode_ids.pop(),
        "frame_indices": np.asarray([int(row["frame_index"]) for row in ordered], dtype=np.int64),
        "actions": actions,
        "states": states,
    }


def action_window(
    episode_actions: object, *, start: int, horizon: int, action_dim: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Make a zero-padded action window and True-for-padding mask.

    No action can cross the supplied episode boundary. Rows beyond its end are
    zero and have ``mask=True``.
    """
    actions = np.asarray(episode_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("episode_actions must have shape (episode_steps, action_dim)")
    if action_dim is not None and actions.shape[1] != action_dim:
        raise ValueError("action_dim does not match the episode")
    if not np.isfinite(actions).all():
        raise ValueError("actions must be finite")
    if start < 0 or start >= len(actions) or horizon < 1:
        raise ValueError("start must select an episode row and horizon must be positive")
    window = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    mask = np.ones(horizon, dtype=bool)
    stop = min(start + horizon, len(actions))
    count = stop - start
    window[:count] = actions[start:stop]
    mask[:count] = False
    return window, mask


def split_episode_ids(
    episode_ids: Sequence[int], *, validation_fraction: float = 0.2
) -> tuple[list[int], list[int]]:
    """Split complete episode IDs in input order; keep at least one per side."""
    ids = list(map(int, episode_ids))
    if len(ids) < 2 or len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError("need at least two unique nonnegative episode IDs")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    validation_count = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
    boundary = len(ids) - validation_count
    return ids[:boundary], ids[boundary:]


def fit_training_statistics(
    training_episodes: Mapping[object, object],
    *,
    validation_episodes: Mapping[object, object] | None = None,
) -> TrainingStats:
    """Fit population mean/std on training samples and hash the result.

    The validation mapping is accepted by the teaching contract but is never
    read. Constant features use a standard deviation of one.
    """
    del validation_episodes
    if not training_episodes:
        raise ValueError("at least one training episode is required")
    arrays = [np.asarray(value, dtype=np.float64) for value in training_episodes.values()]
    if any(array.ndim < 1 for array in arrays):
        raise ValueError("each episode must have a feature dimension")
    feature_sizes = {array.shape[-1] for array in arrays}
    if len(feature_sizes) != 1:
        raise ValueError("all episodes must have the same feature dimension")
    samples = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays])
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("training samples must be non-empty and finite")
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    payload = json.dumps(
        {"mean": mean.tolist(), "std": std.tolist()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return TrainingStats(mean=mean, std=std, sha256=digest)


def read_local_episode_rows(
    task_alias: str,
    episode_ids: Sequence[int],
    *,
    workspace: str | Path | None = None,
    image_samples_per_episode: int = 4,
    row_fn: Any = extract_episode_arrays,
) -> dict[int, dict[str, Any]]:
    """Read full actions/states and a few images from selected local episodes.

    This helper never downloads files. IDs must belong to the frozen train or
    validation split. It keeps every action and state row so windows stop at
    true episode ends, and decodes images only at a few evenly-spaced starts.
    """
    from vla_lab.teaching.live import find_workspace

    if image_samples_per_episode < 1:
        raise ValueError("image_samples_per_episode must be positive")
    ids = list(map(int, episode_ids))
    if not ids or len(ids) > 8 or len(ids) != len(set(ids)):
        raise ValueError("episode_ids must contain one to eight unique IDs")
    root = find_workspace(workspace) if workspace is None else Path(workspace).resolve()
    configs = root / "configs"
    import yaml

    manifest = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    task = manifest["tasks"].get(task_alias)
    if task is None:
        raise ValueError(f"Unknown task alias: {task_alias}")
    allowed = set(task["train_episode_ids"]) | set(task["validation_episode_ids"])
    if not set(ids) <= allowed:
        raise ValueError("episode IDs must come from the frozen training or validation split")
    data_root = root / ".local-data"
    cache = data_root / "hub" / "lerobot-libero"
    if not (cache / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Local dataset is missing at {cache}; this helper never downloads")
    hf_root = data_root / "hf"
    (hf_root / "hub").mkdir(parents=True, exist_ok=True)
    (hf_root / "datasets").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(hf_root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_root / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "lerobot/libero",
        root=cache,
        episodes=ids,
        revision=manifest["dataset_revision"],
        video_backend="torchcodec",
    )
    raw_dataset = dataset.hf_dataset
    episode_positions: dict[int, list[int]] = {episode_id: [] for episode_id in ids}
    for index in range(len(raw_dataset)):
        episode_id = int(raw_dataset[index]["episode_index"])
        if episode_id in episode_positions:
            episode_positions[episode_id].append(index)
    if any(not positions for positions in episode_positions.values()):
        raise ValueError("a selected local episode contains no rows")

    def array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    episodes: dict[int, dict[str, Any]] = {}
    for episode_id, positions in episode_positions.items():
        raw_rows = [raw_dataset[index] for index in positions]
        extracted = row_fn(raw_rows)
        if int(extracted["episode_index"]) != episode_id:
            raise ValueError("row_fn returned a different episode ID")
        actions = np.asarray(extracted["actions"], dtype=np.float32)
        states = np.asarray(extracted["states"], dtype=np.float32)
        frame_indices = list(map(int, extracted["frame_indices"]))
        if frame_indices != list(range(len(frame_indices))):
            raise ValueError("selected episode frame rows must start at zero and be contiguous")
        sample_starts = np.linspace(
            0, len(positions) - 1, min(image_samples_per_episode, len(positions)), dtype=int
        )
        samples = []
        for local_index in sorted(set(map(int, sample_starts))):
            decoded = dataset[positions[local_index]]
            samples.append(
                {
                    "start": local_index,
                    "observation.images.image": array(decoded["observation.images.image"]),
                    "observation.images.image2": array(decoded["observation.images.image2"]),
                }
            )
        episodes[episode_id] = {
            "actions": actions,
            "states": states,
            "samples": samples,
            "frame_count": len(positions),
        }
    return episodes


def build_act_batches(
    training_episodes: Mapping[int, Mapping[str, Any]],
    validation_episodes: Mapping[int, Mapping[str, Any]],
    *,
    horizon: int,
    batch_size: int,
    window_fn: Any,
    stats_fn: Any,
) -> dict[str, Any]:
    """Build real ACT batches with learner-provided windows and statistics.

    Each episode mapping comes from ``read_local_episode_rows``. Validation
    episodes use statistics fitted only on training state/action rows.
    """
    import torch

    if not training_episodes or not validation_episodes:
        raise ValueError("training and validation episodes are both required")
    if set(training_episodes) & set(validation_episodes):
        raise ValueError("training and validation episode IDs overlap")
    if horizon < 1 or batch_size < 1:
        raise ValueError("horizon and batch_size must be positive")
    state_stats = stats_fn({key: value["states"] for key, value in training_episodes.items()})
    action_stats = stats_fn({key: value["actions"] for key, value in training_episodes.items()})

    def convert(episodes: Mapping[int, Mapping[str, Any]]) -> list[dict[str, torch.Tensor]]:
        samples: list[dict[str, np.ndarray]] = []
        for episode in episodes.values():
            actions = np.asarray(episode["actions"], dtype=np.float32)
            states = np.asarray(episode["states"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[1] != 7
                or states.ndim != 2
                or states.shape[1] != 8
            ):
                raise ValueError("ACT actions and states must have widths seven and eight")
            by_start = {int(item["start"]): item for item in episode["samples"]}
            for start, image_sample in by_start.items():
                window, mask = window_fn(actions, start=start, horizon=horizon)
                if window.shape != (horizon, 7) or mask.shape != (horizon,):
                    raise ValueError("window_fn returned an invalid ACT window or mask")
                item = {
                    key: np.asarray(image_sample[key], dtype=np.float32)
                    for key in ("observation.images.image", "observation.images.image2")
                }
                if any(image.ndim != 3 or image.shape[0] != 3 for image in item.values()):
                    raise ValueError("sample images must have shape (3, height, width)")
                if any(
                    not np.isfinite(image).all() or image.min() < 0 or image.max() > 1
                    for image in item.values()
                ):
                    raise ValueError("sample images must be finite float values in [0, 1]")
                item.update(
                    {
                        "observation.state": state_stats.normalize(states[start]).astype(
                            np.float32
                        ),
                        "action": action_stats.normalize(window).astype(np.float32),
                        "action_is_pad": np.asarray(mask, dtype=np.bool_),
                    }
                )
                samples.append(item)
        if not samples:
            raise ValueError("selected episodes contain no image/action training starts")
        batches = []
        for offset in range(0, len(samples), batch_size):
            group = samples[offset : offset + batch_size]
            batches.append(
                {key: torch.as_tensor(np.stack([row[key] for row in group])) for key in group[0]}
            )
        return batches

    return {
        "train": convert(training_episodes),
        "validation": convert(validation_episodes),
        "state_stats": state_stats,
        "action_stats": action_stats,
        "training_episode_ids": sorted(training_episodes),
        "validation_episode_ids": sorted(validation_episodes),
    }
