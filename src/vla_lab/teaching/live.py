"""Opt-in helpers for small, local examples with the frozen ACT data split."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data._utils.collate import default_collate


@dataclass(frozen=True)
class ActBatchSet:
    """Prepared CPU batches and the frozen split identity used to read them."""

    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_workspace(start: str | Path | None = None) -> Path:
    """Find the repository root from any path inside the working tree."""
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "manifests" / "act_references.yaml").is_file():
            return candidate
    raise FileNotFoundError("Could not find the workspace manifest above the current path")


def load_prepared_policy(
    method: str, workspace: str | Path | None = None, *, task_alias: str = "bowl_plate"
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load one hash-checked local ACT, SmolVLA, or Diffusion checkpoint.

    This helper reads the saved comparison recipe and checks the model, data
    split, and processor manifests. All weights must already exist on disk.
    The returned tuple is ``(policy, preprocessor, postprocessor, recipe)``.
    """
    root = find_workspace(workspace) if workspace is None else Path(workspace).resolve()
    cache = root / ".local-data" / "hf"
    (cache / "hub").mkdir(parents=True, exist_ok=True)
    (cache / "datasets").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if method == "smolvla_published":
        manifest = yaml.safe_load((root / "manifests" / "smolvla_reference.yaml").read_text())
        reference = manifest["published_reference"]
        learner_start = manifest["learner_start"]
        model_dir = (root / reference["local_dir"]).resolve()
        model_hash = _sha256_file(model_dir / "model.safetensors")
        if model_hash != reference["model_sha256"]:
            raise ValueError("Published SmolVLA model hash differs from the local manifest")
        vlm_dir = (root / learner_start["local_dir"]).resolve()
        vlm_hash = _sha256_file(vlm_dir / "model.safetensors")
        if vlm_hash != learner_start["model_sha256"]:
            raise ValueError("SmolVLM weights differ from the local manifest")
        from vla_lab.policy.smolvla import load_published_smolvla

        policy, preprocessor, postprocessor = load_published_smolvla(
            model_dir,
            vlm_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_action_steps=reference["selected_n_action_steps"],
        )
        recipe = {
            "method": method,
            "model_revision": reference["revision"],
            "model_sha256": model_hash,
            "training_data": reference["training_data"],
            "n_action_steps": reference["selected_n_action_steps"],
            "checkpoint": str(model_dir),
        }
        return policy, preprocessor, postprocessor, recipe
    from vla_lab.evaluation.comparison import _load_method

    configs = root / "configs"
    comparison = yaml.safe_load((configs / "experiments" / "comparison.yaml").read_text())
    if comparison["task_alias"] != task_alias:
        raise ValueError("Prepared comparison recipe uses another task")
    recipes = comparison["policies"]
    if method not in recipes:
        raise ValueError(f"Unknown prepared policy: {method}")
    recipe = dict(recipes[method])
    for key in ("checkpoint", "vlm_dir"):
        if key in recipe:
            recipe[key] = str((root / recipe[key]).resolve())
    episodes = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    task = episodes["tasks"][task_alias]
    policy, preprocessor, postprocessor = _load_method(
        method,
        recipe,
        task_alias,
        task["split_sha256"],
        episodes["dataset_revision"],
    )
    policy.eval()
    return policy, preprocessor, postprocessor, recipe


class _ACTRawModel(torch.nn.Module):
    """Expose ACT's predictions without running the policy's built-in loss."""

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict[str, Any]) -> Any:
        from lerobot.utils.constants import OBS_IMAGES

        model_input = dict(batch)
        model_input[OBS_IMAGES] = [batch[key] for key in self.policy.config.image_features]
        return self.policy.model(model_input)


def make_act_raw_model(policy: Any) -> torch.nn.Module:
    """Return ACT raw predictions so learner loss code can drive its update.

    The wrapper shares the policy parameters. Supply ``policy.get_optim_params()``
    to the optimizer. This function does not move the policy or batches between
    devices and it does not calculate a loss.
    """
    if not hasattr(policy, "model") or not hasattr(policy, "config"):
        raise TypeError("policy must be an ACT policy with model and config attributes")
    return _ACTRawModel(policy)


# Keep the noun-first spelling used by the course notebooks easy to discover.
make_raw_act_model = make_act_raw_model


def _cpu_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def load_act_batches(
    task_alias: str,
    workspace: str | Path | None = None,
    *,
    batch_size: int = 2,
    chunk_size: int = 20,
    train_batches: int = 2,
    validation_batches: int = 1,
    seed: int = 17,
) -> ActBatchSet:
    """Load fixed ACT train and validation batches without downloading data.

    The batch keys and shapes match ``ACTPolicy.model``: ``action`` is
    ``[B, T, 7]``, ``action_is_pad`` is boolean ``[B, T]``, the two RGB images
    are ``[B, 3, 256, 256]``, and state is ``[B, 8]``. All tensors stay on CPU.
    The function uses episode IDs and normalization statistics from the frozen
    manifests. It sets Hugging Face offline mode before importing LeRobot.
    """
    if batch_size < 1 or chunk_size < 1 or train_batches < 1 or validation_batches < 1:
        raise ValueError("batch size, chunk size, and batch counts must be positive")
    root = find_workspace(workspace) if workspace is None else Path(workspace).resolve()
    configs = root / "configs"
    data_root = root / ".local-data"
    split_manifest = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    try:
        task = split_manifest["tasks"][task_alias]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    cache = data_root / "hub" / "lerobot-libero"
    if not (cache / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Local LIBERO data is missing at {cache}; this helper never downloads it"
        )

    hf_cache = data_root / "hf"
    (hf_cache / "hub").mkdir(parents=True, exist_ok=True)
    (hf_cache / "datasets").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from vla_lab.policy.act import make_act_config, make_act_processors

    train_ids = list(task["train_episode_ids"])
    validation_ids = list(task["validation_episode_ids"])
    if not train_ids or not validation_ids or set(train_ids) & set(validation_ids):
        raise ValueError("Frozen train and validation episode lists must be non-empty and disjoint")
    required_train_rows = train_batches * batch_size
    required_validation_rows = validation_batches * batch_size

    def open_dataset(episodes: list[int]) -> Any:
        return LeRobotDataset(
            "lerobot/libero",
            root=cache,
            episodes=episodes,
            revision=split_manifest["dataset_revision"],
            delta_timestamps={"action": [index / 10 for index in range(chunk_size)]},
            video_backend="torchcodec",
        )

    training_data = open_dataset(train_ids)
    validation_data = open_dataset(validation_ids)
    if len(training_data) < required_train_rows or len(validation_data) < required_validation_rows:
        raise ValueError("The selected episode splits do not contain enough rows for these batches")

    config = make_act_config(device="cpu", chunk_size=chunk_size)
    preprocessor, _postprocessor = make_act_processors(config, task_alias, configs)
    generator = torch.Generator().manual_seed(seed)
    train_order = torch.randperm(len(training_data), generator=generator)[:required_train_rows]
    train_rows = [training_data[int(index)] for index in train_order]
    val_rows = [validation_data[index] for index in range(required_validation_rows)]
    train = tuple(
        _cpu_batch(preprocessor(default_collate(train_rows[start : start + batch_size])))
        for start in range(0, required_train_rows, batch_size)
    )
    validation = tuple(
        _cpu_batch(preprocessor(default_collate(val_rows[start : start + batch_size])))
        for start in range(0, required_validation_rows, batch_size)
    )

    # Keep only fields that the raw ACT model needs. The processors return
    # extra bookkeeping keys for the policy wrapper.
    def model_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in batch.items()
            if key in {"action", "action_is_pad", "observation.state", *config.image_features}
        }

    train = tuple(model_batch(batch) for batch in train)
    validation = tuple(model_batch(batch) for batch in validation)
    metadata = {
        "task_alias": task_alias,
        "dataset_revision": split_manifest["dataset_revision"],
        "split_sha256": task["split_sha256"],
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
        "seed": seed,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "train_rows": required_train_rows,
        "validation_rows": required_validation_rows,
        "normalization_source": str(configs / "data" / "stats" / f"{task_alias}.json"),
        "device": "cpu",
    }
    return ActBatchSet(train=train, validation=validation, metadata=metadata)
