"""Train a task-specific ACT reference on the frozen demonstration split."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

if TYPE_CHECKING:
    from lerobot.policies.act import ACTPolicy


@dataclass(frozen=True)
class TrainResult:
    """The completed step count and last training loss."""

    steps: int
    last_loss: float
    output_dir: Path


SAMPLER_VERSION = "epoch_randperm_v1"


def _batch_indices(length: int, batch_size: int, seed: int, step: int) -> list[int]:
    """Choose one batch from a fixed permutation for each training epoch."""
    batches_per_epoch = length // batch_size
    if batches_per_epoch < 1 or step < 1:
        raise ValueError("Training data must hold one batch, and step must be positive")
    epoch, batch = divmod(step - 1, batches_per_epoch)
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(length, generator=generator)
    return order[batch * batch_size : (batch + 1) * batch_size].tolist()


def _frozen_training_dataset(
    task_alias: str,
    configs_root: Path,
    data_root: Path,
    *,
    chunk_size: int = 20,
    episode_ids: list[int] | None = None,
) -> Dataset:
    split = yaml.safe_load((configs_root / "data" / "episodes.yaml").read_text())
    try:
        train_ids = split["tasks"][task_alias]["train_episode_ids"]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    if not train_ids:
        raise ValueError(f"No training episodes for {task_alias}")
    if episode_ids is not None:
        if not episode_ids or any(episode_id not in train_ids for episode_id in episode_ids):
            raise ValueError("Selected episodes must come from the frozen training split")
        train_ids = episode_ids
    hub_cache = data_root / "hf" / "hub"
    datasets_cache = data_root / "hf" / "datasets"
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        "lerobot/libero",
        root=data_root / "hub" / "lerobot-libero",
        episodes=train_ids,
        revision=split["dataset_revision"],
        delta_timestamps={"action": [index / 10 for index in range(chunk_size)]},
        video_backend="torchcodec",
    )


def train_act_reference(
    *,
    task_alias: str,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    steps: int,
    batch_size: int = 8,
    device: str = "cuda",
    seed: int = 20260930,
    checkpoint_every: int = 1000,
    chunk_size: int = 20,
    dataset: Dataset | None = None,
    policy: ACTPolicy | None = None,
    resume_from: str | Path | None = None,
) -> TrainResult:
    """Train ACT and save local checkpoints with the frozen processors."""
    if steps < 1 or batch_size < 1 or checkpoint_every < 1 or chunk_size < 1:
        raise ValueError("steps, batch size, checkpoint period, and chunk size must be positive")
    if resume_from is not None and policy is not None:
        raise ValueError("Supply either a policy or a resume checkpoint")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    configs = Path(configs_root)
    data = Path(data_root).expanduser().resolve()
    hub_cache = data / "hf" / "hub"
    datasets_cache = data / "hf" / "datasets"
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))

    from lerobot.policies.act import ACTPolicy

    from vla_lab.policy.act import make_act_config, make_act_processors
    from vla_lab.policy.bundle import load_act_bundle, save_act_bundle

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    split = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    source_run = None
    checkpoint_state = None
    start_step = 0
    if resume_from is not None:
        source = Path(resume_from).expanduser().resolve()
        source_run = json.loads((source.parent / "run.json").read_text())
        expected = {
            "task_alias": task_alias,
            "dataset_revision": split["dataset_revision"],
            "train_episode_ids": split["tasks"][task_alias]["train_episode_ids"],
            "split_sha256": split["tasks"][task_alias]["split_sha256"],
            "seed": seed,
            "batch_size": batch_size,
            "device": device,
            "chunk_size": chunk_size,
            "sampler_version": SAMPLER_VERSION,
        }
        for key, value in expected.items():
            if source_run.get(key) != value:
                raise ValueError(f"Resume {key} differs from the saved run")
        policy, preprocessor, postprocessor, bundle = load_act_bundle(source)
        if bundle["task_alias"] != task_alias or bundle["split_sha256"] != expected["split_sha256"]:
            raise ValueError("Resume checkpoint does not match the task and split")
        if bundle["chunk_size"] != chunk_size:
            raise ValueError("Resume chunk_size differs from the saved checkpoint")
        checkpoint_state = torch.load(
            source / "optimizer.pt", map_location="cpu", weights_only=True
        )
        start_step = int(checkpoint_state["step"])
        if steps <= start_step:
            raise ValueError("Resume target steps must exceed the saved step")
        config = policy.config
    else:
        config = make_act_config(device=device, chunk_size=chunk_size)
        if policy is None:
            policy = ACTPolicy(config)
        elif policy.config.device != device:
            raise ValueError("The supplied ACT policy uses another device")
        preprocessor, postprocessor = make_act_processors(config, task_alias, configs)
    policy = policy.to(device)
    policy.train()
    training_data = (
        dataset
        if dataset is not None
        else _frozen_training_dataset(task_alias, configs, data, chunk_size=chunk_size)
    )
    if len(training_data) < batch_size:
        raise ValueError("Training data has fewer frames than the batch size")
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-4)
    if checkpoint_state is not None:
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        torch.set_rng_state(checkpoint_state["torch_rng_state"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(checkpoint_state["cuda_rng_states"])
    (output / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_alias": task_alias,
                "dataset_revision": split["dataset_revision"],
                "train_episode_ids": split["tasks"][task_alias]["train_episode_ids"],
                "split_sha256": split["tasks"][task_alias]["split_sha256"],
                "seed": seed,
                "steps": steps,
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "device": device,
                "learning_rate": 1e-4,
                "backbone_learning_rate": config.optimizer_lr_backbone,
                "checkpoint_every": checkpoint_every,
                "video_backend": "torchcodec",
                "sampler_version": SAMPLER_VERSION,
                "resumed_from_step": start_step,
                "resumed_from": str(Path(resume_from).expanduser().resolve())
                if resume_from is not None
                else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    started = time.monotonic()
    last_loss = math.nan
    with (output / "metrics.jsonl").open("w") as metrics:
        for step in range(start_step + 1, steps + 1):
            indices = _batch_indices(len(training_data), batch_size, seed, step)
            raw_batch = default_collate([training_data[index] for index in indices])
            batch = preprocessor(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"ACT loss is not finite at step {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"ACT gradient is not finite at step {step}")
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            metrics.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": last_loss,
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
                + "\n"
            )
            metrics.flush()
            if step % checkpoint_every == 0 or step == steps:
                checkpoint = output / f"checkpoint-{step:06d}"
                save_act_bundle(
                    policy, preprocessor, postprocessor, checkpoint, task_alias, configs
                )
                torch.save(
                    {
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all()
                        if device.startswith("cuda")
                        else [],
                    },
                    checkpoint / "optimizer.pt",
                )
    return TrainResult(steps, last_loss, output)


def main(argv: list[str] | None = None) -> None:
    """Run a bounded local reference job from the terminal."""
    parser = argparse.ArgumentParser(description="Train a local ACT reference on frozen episodes")
    parser.add_argument("--task", choices=("bowl_plate", "bowl_stove"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    arguments = parser.parse_args(argv)
    result = train_act_reference(
        task_alias=arguments.task,
        configs_root=arguments.configs_root,
        data_root=arguments.data_root,
        output_dir=arguments.output,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        checkpoint_every=arguments.checkpoint_every,
        chunk_size=arguments.chunk_size,
        device=arguments.device,
        resume_from=arguments.resume_from,
    )
    print(json.dumps({"steps": result.steps, "last_loss": result.last_loss}))


if __name__ == "__main__":
    main()
