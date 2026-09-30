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

import numpy as np
import torch
import yaml
from lerobot.policies.act import ACTPolicy
from torch.utils.data import DataLoader, Dataset

from vla_lab.policy.act import make_act_config, make_act_processors
from vla_lab.policy.bundle import save_act_bundle


@dataclass(frozen=True)
class TrainResult:
    """The completed step count and last training loss."""

    steps: int
    last_loss: float
    output_dir: Path


def _frozen_training_dataset(task_alias: str, configs_root: Path, data_root: Path) -> Dataset:
    split = yaml.safe_load((configs_root / "data" / "episodes.yaml").read_text())
    try:
        train_ids = split["tasks"][task_alias]["train_episode_ids"]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    if not train_ids:
        raise ValueError(f"No training episodes for {task_alias}")
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
        delta_timestamps={"action": [index / 10 for index in range(20)]},
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
    dataset: Dataset | None = None,
    policy: ACTPolicy | None = None,
) -> TrainResult:
    """Train ACT and save local checkpoints with the frozen processors."""
    if steps < 1 or batch_size < 1 or checkpoint_every < 1:
        raise ValueError("steps, batch_size, and checkpoint_every must be positive")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    configs = Path(configs_root)
    data = Path(data_root).expanduser().resolve()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    config = make_act_config(device=device)
    if policy is None:
        policy = ACTPolicy(config)
    elif policy.config.device != device:
        raise ValueError("The supplied ACT policy uses another device")
    policy = policy.to(device)
    policy.train()
    preprocessor, postprocessor = make_act_processors(config, task_alias, configs)
    training_data = (
        dataset if dataset is not None else _frozen_training_dataset(task_alias, configs, data)
    )
    if len(training_data) < batch_size:
        raise ValueError("Training data has fewer frames than the batch size")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-4)
    split = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
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
                "device": device,
                "learning_rate": 1e-4,
                "backbone_learning_rate": config.optimizer_lr_backbone,
                "checkpoint_every": checkpoint_every,
                "video_backend": "torchcodec",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    iterator = iter(loader)
    started = time.monotonic()
    last_loss = math.nan
    with (output / "metrics.jsonl").open("w") as metrics:
        for step in range(1, steps + 1):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
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
                    {"step": step, "optimizer": optimizer.state_dict()}, checkpoint / "optimizer.pt"
                )
    return TrainResult(steps, last_loss, output)


def main(argv: list[str] | None = None) -> None:
    """Run a bounded local reference job from the terminal."""
    parser = argparse.ArgumentParser(description="Train a local ACT reference on frozen episodes")
    parser.add_argument("--task", choices=("bowl_plate", "bowl_stove"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = train_act_reference(
        task_alias=arguments.task,
        configs_root=Path("configs"),
        data_root=arguments.data_root,
        output_dir=arguments.output,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        checkpoint_every=arguments.checkpoint_every,
        device=arguments.device,
    )
    print(json.dumps({"steps": result.steps, "last_loss": result.last_loss}))


if __name__ == "__main__":
    main()
