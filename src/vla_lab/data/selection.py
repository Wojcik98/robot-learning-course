"""Prepare a nested ACT demonstration set with its own statistics."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vla_lab.data.splits import partition_sha256, split_episodes
from vla_lab.data.stats import TrainingStats


def _selected_rows(
    configs_root: Path, data_root: Path, task_alias: str, selected_ids: list[int]
) -> dict[int, dict[str, np.ndarray]]:
    from vla_lab.training.act import _frozen_training_dataset

    dataset = _frozen_training_dataset(
        task_alias, configs_root, data_root, episode_ids=selected_ids
    )
    values: dict[int, dict[str, list[Any]]] = defaultdict(
        lambda: {"observation.state": [], "action": []}
    )
    for row in dataset.hf_dataset:
        episode_id = int(row["episode_index"])
        if episode_id not in selected_ids:
            raise ValueError("dataset returned an episode outside the selected training subset")
        for key in ("observation.state", "action"):
            values[episode_id][key].append(row[key])
    if set(values) != set(selected_ids):
        raise ValueError("selected training episodes are missing from the local dataset")
    return {
        episode_id: {key: np.asarray(rows) for key, rows in modalities.items()}
        for episode_id, modalities in values.items()
    }


def _modality_stats(
    episode_rows: Mapping[int, Mapping[str, np.ndarray]], selected_ids: list[int], key: str
) -> dict[str, Any]:
    arrays = {episode_id: np.asarray(episode_rows[episode_id][key]) for episode_id in selected_ids}
    fitted = TrainingStats.fit(arrays)
    joined = np.concatenate(list(arrays.values()), axis=0)
    return {
        **fitted.to_dict(),
        "count": int(joined.shape[0]),
        "min": joined.min(axis=0).tolist(),
        "max": joined.max(axis=0).tolist(),
    }


def prepare_act_selection(
    *,
    configs_root: str | Path,
    task_alias: str,
    episode_count: int,
    output_dir: str | Path,
    data_root: str | Path | None = None,
    episode_rows: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> Path:
    """Save a run-specific split and statistics under a new config root."""
    source = Path(configs_root).expanduser().resolve()
    manifest = yaml.safe_load((source / "data" / "episodes.yaml").read_text())
    task = manifest["tasks"][task_alias]
    train_ids = task["train_episode_ids"]
    validation_ids = task["validation_episode_ids"]
    frozen = split_episodes(train_ids + validation_ids)
    if frozen.sha256 != task["split_sha256"]:
        raise ValueError("source split hash differs from the frozen episode IDs")
    if not 1 <= episode_count <= len(train_ids):
        raise ValueError("episode_count must fit the frozen training episode count")
    target = Path(output_dir).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Selection output directory is not empty: {target}")
    selected = train_ids[:episode_count]
    selected_task = deepcopy(task)
    selected_task["train_episode_ids"] = selected
    selected_task["episode_count"] = episode_count + len(validation_ids)
    selected_task["split_sha256"] = partition_sha256(selected, validation_ids)
    if episode_count < len(train_ids):
        selected_task.update(
            {
                "selection_kind": "nested_train_subset",
                "source_train_episode_ids": train_ids,
                "source_validation_episode_ids": validation_ids,
                "source_split_sha256": task["split_sha256"],
            }
        )
        if episode_rows is None:
            if data_root is None:
                raise ValueError("data_root is required to calculate subset statistics")
            episode_rows = _selected_rows(source, Path(data_root), task_alias, selected)
        stats = {
            "schema_version": 1,
            "dataset_revision": manifest["dataset_revision"],
            "split_sha256": selected_task["split_sha256"],
            "modalities": {
                key: _modality_stats(episode_rows, selected, key)
                for key in ("observation.state", "action")
            },
        }
    else:
        stats = json.loads((source / "data" / "stats" / f"{task_alias}.json").read_text())
    selected_manifest = {
        "schema_version": manifest["schema_version"],
        "dataset_revision": manifest["dataset_revision"],
        "tasks": {task_alias: selected_task},
    }
    stats_path = target / "data" / "stats" / f"{task_alias}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    (target / "data" / "episodes.yaml").write_text(
        yaml.safe_dump(selected_manifest, sort_keys=False)
    )
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return target
