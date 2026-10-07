"""Train a Diffusion Policy reference on the frozen demonstration split.

This module sets up the data windows, the training loop, and the checkpoint bundle.
It does not run a simulator.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.processor import load_pretrained_policy_processors
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from vla_lab.evaluation.states import selected_state_ids
from vla_lab.policy.diffusion import make_diffusion_config, make_diffusion_processors
from vla_lab.timing import frame_offset_seconds
from vla_lab.training.act import SAMPLER_VERSION, _batch_indices
from vla_lab.types import JSONValue

if TYPE_CHECKING:
    from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
    from lerobot.processor import PolicyProcessorPipeline

FPS = 10
# horizon - n_action_steps - n_obs_steps + 1 for the fixed 16/8/2 recipe.
DROP_N_LAST_FRAMES = 7
RESNET18_IMAGENET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESNET18_FILE = "resnet18-f37072fd.pth"
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
ACTION_KEY = "action"


@dataclass(frozen=True)
class DiffusionTrainConfig:
    """The fixed recipe, the training settings, and the development states."""

    task_alias: str
    seed: int
    device: str
    steps: int
    batch_size: int
    checkpoint_every: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    n_obs_steps: int
    horizon: int
    n_action_steps: int
    pretrained_backbone_weights: str
    train_episode_ids: tuple[int, ...]
    state_ids: tuple[int, ...]
    config_sha256: str


@dataclass(frozen=True)
class DiffusionTrainResult:
    """The completed step count, the last loss, and the output directory."""

    steps: int
    last_loss: float
    output_dir: Path


def _split_for(
    configs_root: Path, task_alias: str
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    manifest = yaml.safe_load((configs_root / "data" / "episodes.yaml").read_text())
    try:
        return manifest, manifest["tasks"][task_alias]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error


def load_diffusion_train_config(path: str | Path, configs_root: str | Path) -> DiffusionTrainConfig:
    """Read the settings. Reject a recipe that differs from the policy settings."""

    content = Path(path).read_bytes()
    payload = yaml.safe_load(content)
    if (
        payload.get("schema_version") != 1
        or payload.get("policy") != "diffusion"
        or payload.get("profile") != "local"
    ):
        raise ValueError(
            "Diffusion training needs schema version 1, policy diffusion, and the local profile"
        )
    configs = Path(configs_root)
    task_alias = str(payload["task_alias"])
    _, split = _split_for(configs, task_alias)

    reference = make_diffusion_config(device="cpu")
    for key in (
        "n_obs_steps",
        "horizon",
        "n_action_steps",
        "do_mask_loss_for_padding",
        "pretrained_backbone_weights",
    ):
        if payload.get(key) != getattr(reference, key):
            raise ValueError(f"{key} differs from the fixed Diffusion Policy recipe")
    if payload.get("fps") != FPS:
        raise ValueError(f"fps must be {FPS}")
    train_ids = tuple(int(value) for value in split["train_episode_ids"])
    if payload.get("train_episode_count") != len(train_ids):
        raise ValueError("train_episode_count differs from the frozen split")

    steps = int(payload["steps"])
    batch_size = int(payload["batch_size"])
    checkpoint_every = int(payload["checkpoint_every"])
    learning_rate = float(payload["learning_rate"])
    weight_decay = float(payload["weight_decay"])
    warmup_steps = int(payload["warmup_steps"])
    if min(steps, batch_size, checkpoint_every) < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("steps, batch_size, checkpoint_every, and learning_rate must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must not be negative")
    state_ids = tuple(
        selected_state_ids(configs, task_alias, "development", payload["development_state_ids"])
    )
    return DiffusionTrainConfig(
        task_alias=task_alias,
        seed=int(payload["seed"]),
        device=str(payload["device"]),
        steps=steps,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        n_obs_steps=reference.n_obs_steps,
        horizon=reference.horizon,
        n_action_steps=reference.n_action_steps,
        pretrained_backbone_weights=str(payload["pretrained_backbone_weights"]),
        train_episode_ids=train_ids,
        state_ids=state_ids,
        config_sha256=hashlib.sha256(content).hexdigest(),
    )


# --- Data windows -----------------------------------------------------------------------------


def validate_horizons(config: "DiffusionConfig") -> None:
    """Reject prediction and execution lengths that the policy cannot use."""
    if config.n_obs_steps < 1:
        raise ValueError("n_obs_steps must be positive")
    if not 1 <= config.n_action_steps <= config.horizon - config.n_obs_steps + 1:
        raise ValueError("n_action_steps must fit inside horizon - n_obs_steps + 1")
    if config.horizon % 2 ** len(config.down_dims) != 0:
        raise ValueError("horizon must be a multiple of the U-Net downsampling factor")
    expected_drop = config.horizon - config.n_action_steps - config.n_obs_steps + 1
    if config.drop_n_last_frames != expected_drop:
        raise ValueError(
            f"drop_n_last_frames must equal horizon - n_action_steps - n_obs_steps + 1 ({expected_drop})"
        )


def diffusion_delta_timestamps(config: "DiffusionConfig", fps: int) -> dict[str, list[float]]:
    """Map frame offsets to storage timestamp labels for LeRobot lookup.

    The observations end at the current frame. The action window starts one step before it,
    because it covers the same span as the observations and then the future.
    The storage FPS does not set control timing or action repetition.
    """
    validate_horizons(config)
    observation = frame_offset_seconds(config.observation_delta_indices, fps)
    deltas = {key: list(observation) for key in (*IMAGE_KEYS, STATE_KEY)}
    deltas[ACTION_KEY] = frame_offset_seconds(config.action_delta_indices, fps)
    return deltas


def executed_action_slice(config: "DiffusionConfig") -> tuple[int, int]:
    """Give the part of the action window that the policy executes.

    Index `n_obs_steps - 1` of the window is the action for the current frame.
    """
    start = config.n_obs_steps - 1
    return start, start + config.n_action_steps


def eligible_frame_indices(
    episode_index: Sequence[int], frame_index: Sequence[int], drop_n_last_frames: int
) -> list[int]:
    """Choose the dataset rows whose action window has little end padding.

    Each row must belong to its episode in order. The last `drop_n_last_frames` frames
    of every episode are not used as the current frame.
    """
    if len(episode_index) != len(frame_index):
        raise ValueError("Episode and frame columns must have the same length")
    lengths: dict[int, int] = {}
    for episode in episode_index:
        lengths[int(episode)] = lengths.get(int(episode), 0) + 1
    kept = [
        row
        for row, (episode, frame) in enumerate(zip(episode_index, frame_index, strict=True))
        if int(frame) < lengths[int(episode)] - drop_n_last_frames
    ]
    if not kept:
        raise ValueError("No frame is left after the last frames of each episode are dropped")
    return kept


def validate_window_batch(batch: Mapping[str, torch.Tensor], config: "DiffusionConfig") -> None:
    """Check the shape and values of one batch before training uses it."""
    state_shape = config.robot_state_feature.shape
    action_shape = config.action_feature.shape
    expected = {key: (config.n_obs_steps, *config.image_features[key].shape) for key in IMAGE_KEYS}
    expected[STATE_KEY] = (config.n_obs_steps, *state_shape)
    expected[ACTION_KEY] = (config.horizon, *action_shape)
    expected["action_is_pad"] = (config.horizon,)
    batch_size = None
    for key, shape in expected.items():
        if key not in batch:
            raise ValueError(f"Batch has no {key}")
        value = batch[key]
        if tuple(value.shape[1:]) != shape:
            raise ValueError(
                f"{key} must have shape (batch, {', '.join(map(str, shape))}); got {tuple(value.shape)}"
            )
        batch_size = value.shape[0] if batch_size is None else batch_size
        if value.shape[0] != batch_size:
            raise ValueError(f"{key} has another batch size")
        if key == "action_is_pad":
            if value.dtype != torch.bool:
                raise ValueError("action_is_pad must be a boolean tensor")
        elif not torch.isfinite(value).all():
            raise ValueError(f"{key} has nonfinite values")


def _frozen_diffusion_dataset(
    config: "DiffusionConfig", split: Mapping[str, JSONValue], revision: str, data_root: Path
) -> Dataset:
    """Open the frozen training episodes with the policy windows. Never download."""
    hub_cache = data_root / "hf" / "hub"
    datasets_cache = data_root / "hf" / "datasets"
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    dataset = LeRobotDataset(
        "lerobot/libero",
        root=data_root / "hub" / "lerobot-libero",
        episodes=list(split["train_episode_ids"]),
        revision=revision,
        delta_timestamps=diffusion_delta_timestamps(config, FPS),
        video_backend="torchcodec",
    )
    if dataset.fps != FPS:
        raise ValueError(f"Dataset frame rate must be {FPS}")
    return dataset


# --- Weights and bundle -----------------------------------------------------------------------


def verify_backbone_weights(torch_home: str | Path) -> str:
    """Check the cached ImageNet ResNet-18 file. Return its SHA-256."""
    path = Path(torch_home) / "hub" / "checkpoints" / RESNET18_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Pinned ResNet-18 weights are missing from the local cache: {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != RESNET18_IMAGENET_SHA256:
        raise ValueError(f"ResNet-18 weights hash differs from the pinned value: {path}")
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalizer_extrema(
    preprocessor: "PolicyProcessorPipeline",
) -> dict[str, dict[str, list[float]]]:
    for step in preprocessor.steps:
        if type(step).__name__ == "NormalizerProcessorStep":
            return {
                key: {name: step.stats[key][name].tolist() for name in ("min", "max")}
                for key in (STATE_KEY, ACTION_KEY)
            }
    raise ValueError("Preprocessor has no normalizer")


def save_diffusion_bundle(
    policy: "DiffusionPolicy",
    preprocessor: "PolicyProcessorPipeline",
    postprocessor: "PolicyProcessorPipeline",
    destination: str | Path,
    task_alias: str,
    configs_root: str | Path,
) -> None:
    """Save one complete local checkpoint. Write its manifest last."""
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Diffusion checkpoint directory is not empty: {target}")
    configs = Path(configs_root)
    split_manifest, split = _split_for(configs, task_alias)
    stats = json.loads((configs / "data" / "stats" / f"{task_alias}.json").read_text())
    if stats["split_sha256"] != split["split_sha256"]:
        raise ValueError(
            "Training statistics split hash differs from the frozen episode split hash"
        )
    if stats["dataset_revision"] != split_manifest["dataset_revision"]:
        raise ValueError("Training statistics dataset revision differs from the split revision")

    extrema = _normalizer_extrema(preprocessor)
    for key in (STATE_KEY, ACTION_KEY):
        for bound in ("min", "max"):
            expected = torch.tensor(stats["modalities"][key][bound])
            actual = torch.tensor(extrema[key][bound])
            if not torch.allclose(actual, expected, atol=1e-6):
                raise ValueError(f"Saved processor {key} {bound} differs from training statistics")

    config = policy.config
    weights = config.pretrained_backbone_weights
    target.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(target)
    preprocessor.save_pretrained(target)
    postprocessor.save_pretrained(target)
    (target / "training_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    processor_files = sorted(
        path
        for path in target.iterdir()
        if path.is_file() and path.name.startswith(("policy_preprocessor", "policy_postprocessor"))
    )
    manifest = {
        "schema_version": 1,
        "policy": "diffusion",
        "task_alias": task_alias,
        "dataset_revision": split_manifest["dataset_revision"],
        "split_sha256": split["split_sha256"],
        "training_stats_sha256": {
            key: stats["modalities"][key]["sha256"] for key in (STATE_KEY, ACTION_KEY)
        },
        "extrema": extrema,
        "n_obs_steps": config.n_obs_steps,
        "horizon": config.horizon,
        "n_action_steps": config.n_action_steps,
        "do_mask_loss_for_padding": config.do_mask_loss_for_padding,
        "pretrained_backbone_weights": weights,
        "vision_weights_sha256": RESNET18_IMAGENET_SHA256 if weights else None,
        "model_sha256": _sha256(target / "model.safetensors"),
        "config_sha256": _sha256(target / "config.json"),
        "training_stats_file_sha256": _sha256(target / "training_stats.json"),
        "processor_files_sha256": {path.name: _sha256(path) for path in processor_files},
    }
    temporary = target / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target / "manifest.json")


def load_diffusion_bundle(
    source: str | Path,
) -> (
    "tuple[DiffusionPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline, dict[str, JSONValue]]"
):
    """Load saved weights, processors, and the manifest from one directory.

    The saved weights replace the backbone, so loading needs no cached ImageNet file.
    """

    root = Path(source)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Diffusion checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("policy") != "diffusion":
        raise ValueError("Unsupported Diffusion checkpoint manifest")
    required_processors = {
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    listed_processors = set(manifest["processor_files_sha256"])
    present_processors = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name.startswith(("policy_preprocessor", "policy_postprocessor"))
    }
    if not required_processors <= listed_processors or listed_processors != present_processors:
        raise ValueError("Saved Diffusion processor file list is incomplete or changed")
    for name, expected in (
        ("model.safetensors", manifest["model_sha256"]),
        ("config.json", manifest["config_sha256"]),
        ("training_stats.json", manifest["training_stats_file_sha256"]),
        *manifest["processor_files_sha256"].items(),
    ):
        if _sha256(root / name) != expected:
            raise ValueError(f"Saved Diffusion file hash differs from the manifest: {name}")
    stats = json.loads((root / "training_stats.json").read_text())
    if any(
        stats["modalities"][key]["sha256"] != manifest["training_stats_sha256"][key]
        for key in (STATE_KEY, ACTION_KEY)
    ):
        raise ValueError("Training statistics differ from the saved manifest")

    config = DiffusionConfig.from_pretrained(root, local_files_only=True)
    weights = config.pretrained_backbone_weights
    config.pretrained_backbone_weights = None
    policy = DiffusionPolicy.from_pretrained(
        root, config=config, strict=True, local_files_only=True
    )
    policy.config.pretrained_backbone_weights = weights
    for key in (
        "n_obs_steps",
        "horizon",
        "n_action_steps",
        "do_mask_loss_for_padding",
        "pretrained_backbone_weights",
    ):
        if manifest[key] != getattr(policy.config, key):
            raise ValueError(f"Manifest {key} differs from the saved policy")

    preprocessor, postprocessor = load_pretrained_policy_processors(root)
    saved = _normalizer_extrema(preprocessor)
    for key, bounds in manifest["extrema"].items():
        for name, values in bounds.items():
            if not torch.allclose(torch.tensor(values), torch.tensor(saved[key][name]), atol=1e-6):
                raise ValueError(f"Manifest statistics differ from the saved processor for {key}")
    return policy, preprocessor, postprocessor, manifest


# --- Training ---------------------------------------------------------------------------------


def train_diffusion_reference(
    *,
    task_alias: str,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    steps: int,
    batch_size: int = 8,
    device: str = "cuda",
    seed: int = 20260930,
    checkpoint_every: int = 500,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    warmup_steps: int = 100,
    config_sha256: str | None = None,
    dataset: Dataset | None = None,
    sample_indices: Sequence[int] | None = None,
    policy: "DiffusionPolicy | None" = None,
) -> DiffusionTrainResult:
    """Train Diffusion Policy and save local checkpoints with the frozen processors."""
    if min(steps, batch_size, checkpoint_every) < 1:
        raise ValueError("steps, batch size, and checkpoint period must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must not be negative")
    configs = Path(configs_root)
    split_manifest, split = _split_for(configs, task_alias)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {output}")

    data = Path(data_root).expanduser().resolve()
    os.environ.setdefault("TORCH_HOME", str(data / "torch"))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    if policy is None:
        config = make_diffusion_config(device=device)
        if config.pretrained_backbone_weights:
            verify_backbone_weights(os.environ["TORCH_HOME"])
        policy = DiffusionPolicy(config)
    elif policy.config.device != device:
        raise ValueError("The supplied Diffusion policy uses another device")
    config = policy.config
    validate_horizons(config)
    config.optimizer_lr = learning_rate
    config.optimizer_weight_decay = weight_decay
    config.scheduler_warmup_steps = warmup_steps
    preprocessor, postprocessor = make_diffusion_processors(config, task_alias, configs)

    training_data = (
        dataset
        if dataset is not None
        else _frozen_diffusion_dataset(config, split, split_manifest["dataset_revision"], data)
    )
    if sample_indices is None:
        if dataset is None:
            table = training_data.hf_dataset
            sample_indices = eligible_frame_indices(
                [int(value) for value in table["episode_index"]],
                [int(value) for value in table["frame_index"]],
                config.drop_n_last_frames,
            )
        else:
            sample_indices = list(range(len(training_data)))
    samples = list(sample_indices)
    if not samples or not all(0 <= index < len(training_data) for index in samples):
        raise ValueError("Every sample index must address the training data")
    if len(samples) < batch_size:
        raise ValueError("Training data has fewer samples than the batch size")

    policy = policy.to(device)
    policy.train()
    optimizer = torch.optim.Adam(
        policy.get_optim_params(),
        lr=learning_rate,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=weight_decay,
    )
    scheduler = DiffuserSchedulerConfig(
        name=config.scheduler_name, num_warmup_steps=warmup_steps
    ).build(optimizer, steps)

    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_alias": task_alias,
                "dataset_revision": split_manifest["dataset_revision"],
                "train_episode_ids": split["train_episode_ids"],
                "split_sha256": split["split_sha256"],
                "seed": seed,
                "steps": steps,
                "batch_size": batch_size,
                "device": device,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "checkpoint_every": checkpoint_every,
                "n_obs_steps": config.n_obs_steps,
                "horizon": config.horizon,
                "n_action_steps": config.n_action_steps,
                "drop_n_last_frames": config.drop_n_last_frames,
                "do_mask_loss_for_padding": config.do_mask_loss_for_padding,
                "pretrained_backbone_weights": config.pretrained_backbone_weights,
                "n_parameters": sum(parameter.numel() for parameter in policy.parameters()),
                "sample_count": len(samples),
                "sampler_version": SAMPLER_VERSION,
                "video_backend": "torchcodec",
                "config_sha256": config_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    started = time.monotonic()
    last_loss = math.nan
    with (output / "metrics.jsonl").open("w") as metrics:
        for step in range(1, steps + 1):
            positions = _batch_indices(len(samples), batch_size, seed, step)
            raw_batch = default_collate(
                [training_data[samples[position]] for position in positions]
            )
            validate_window_batch(raw_batch, config)
            batch = preprocessor(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Diffusion loss is not finite at step {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Diffusion gradient is not finite at step {step}")
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.detach().cpu())
            metrics.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": last_loss,
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
                + "\n"
            )
            metrics.flush()
            if step % checkpoint_every == 0 or step == steps:
                save_diffusion_bundle(
                    policy,
                    preprocessor,
                    postprocessor,
                    output / f"checkpoint-{step:06d}",
                    task_alias,
                    configs,
                )
    return DiffusionTrainResult(steps, last_loss, output)


def main(argv: list[str] | None = None) -> None:
    """Run a bounded local training job from the terminal."""
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on the frozen episodes")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train/diffusion_bowl_plate.yaml")
    )
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="Replace the step count for a short probe")
    parser.add_argument("--device", help="Replace the device from the file")
    arguments = parser.parse_args(argv)
    config = load_diffusion_train_config(arguments.config, arguments.configs_root)
    result = train_diffusion_reference(
        task_alias=config.task_alias,
        configs_root=arguments.configs_root,
        data_root=arguments.data_root,
        output_dir=arguments.output,
        steps=arguments.steps or config.steps,
        batch_size=config.batch_size,
        device=arguments.device or config.device,
        seed=config.seed,
        checkpoint_every=config.checkpoint_every,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        config_sha256=config.config_sha256,
    )
    print(json.dumps({"steps": result.steps, "last_loss": result.last_loss}))


if __name__ == "__main__":
    main()
