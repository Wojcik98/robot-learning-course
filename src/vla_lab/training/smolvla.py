"""Train the pinned SmolVLA learner in two recorded stages."""

import argparse
import hashlib
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
from torch.utils.data._utils.collate import default_collate

from vla_lab.experiments.smolvla import load_smolvla_train_config
from vla_lab.policy.smolvla_learner import (
    _sha256,
    audit_learner_parameter_groups,
    build_learner_policy,
    load_learner_bundle,
    make_learner_processors,
    save_learner_bundle,
)
from vla_lab.training.act import SAMPLER_VERSION, _batch_indices, _frozen_training_dataset


@dataclass(frozen=True)
class SmolVLATrainResult:
    """The saved learner stage and its last loss."""

    stage: str
    steps: int
    last_loss: float
    checkpoint: Path


def _seed_all(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def _source_identity(configs_root: Path) -> tuple[str, str]:
    payload = yaml.safe_load(
        (configs_root.parent / "manifests" / "smolvla_reference.yaml").read_text()
    )
    source = payload["learner_start"]
    return source["revision"], source["model_sha256"]


def _verify_optimizer_checkpoint(checkpoint: Path, manifest: dict[str, object]) -> None:
    """Reject a missing or changed optimizer before policy loading."""
    optimizer = checkpoint / "optimizer.pt"
    if not optimizer.is_file() or not isinstance(manifest.get("optimizer_sha256"), str):
        raise ValueError("Saved learner optimizer and its hash are required for continuation")
    digest = hashlib.sha256()
    with optimizer.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["optimizer_sha256"]:
        raise ValueError("Optimizer hash differs from the saved learner model")


def _validate_resume_metadata(
    source_run: dict[str, object],
    source_manifest: dict[str, object],
    expected: dict[str, object],
    *,
    c1_steps: int,
    target_steps: int,
    split_sha256: str,
    vlm_revision: str,
    vlm_sha256: str,
) -> int:
    """Check model lineage and fixed training settings before C2 resumes."""
    for key, value in expected.items():
        if key != "config_sha256" and source_run.get(key) != value:
            raise ValueError(f"C2 {key} differs from the saved run")
    source_stage = source_manifest.get("stage")
    source_step = source_manifest.get("step")
    if (
        source_stage not in ("c1", "c2")
        or source_run.get("stage") != source_stage
        or source_run.get("steps") != source_step
        or not isinstance(source_step, int)
        or (source_stage == "c1" and source_step != c1_steps)
        or (source_stage == "c2" and source_step < c1_steps)
        or source_manifest.get("task_alias") != expected["task_alias"]
        or source_manifest.get("split_sha256") != split_sha256
        or source_manifest.get("vlm_revision") != vlm_revision
        or source_manifest.get("vlm_sha256") != vlm_sha256
        or source_manifest.get("chunk_size") != 50
        or source_manifest.get("n_action_steps") != 10
    ):
        raise ValueError("C2 source is not a compatible learner checkpoint")
    if (
        source_stage == "c2"
        and source_run.get("root_c1_steps", source_run.get("resumed_from_step")) != c1_steps
    ):
        raise ValueError("C2 source has another root C1 step")
    if target_steps <= source_step:
        raise ValueError("C2 target steps must exceed the saved step")
    return source_step


def train_smolvla_stage(
    *,
    config_path: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    vlm_dir: str | Path,
    output_dir: str | Path,
    stage: str,
    resume_from: str | Path | None = None,
) -> SmolVLATrainResult:
    """Train C1 from A or continue C2 from a checked learner state."""
    if stage not in ("c1", "c2"):
        raise ValueError("Learner stage must be c1 or c2")
    if stage == "c2" and resume_from is None:
        raise ValueError("C2 needs a C1 checkpoint")
    if stage == "c1" and resume_from is not None:
        raise ValueError("C1 starts from the pinned VLM source")

    configs = Path(configs_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    vlm = Path(vlm_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    config = load_smolvla_train_config(config_path, configs)
    target_steps = config.c1_steps if stage == "c1" else config.c2_steps
    split = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    task_split = split["tasks"][config.task_alias]
    revision, vlm_sha256 = _source_identity(configs)
    expected = {
        "task_alias": config.task_alias,
        "dataset_revision": split["dataset_revision"],
        "train_episode_ids": task_split["train_episode_ids"],
        "split_sha256": task_split["split_sha256"],
        "seed": config.seed,
        "batch_size": config.batch_size,
        "device": config.device,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "sampler_version": SAMPLER_VERSION,
        "config_sha256": config.config_sha256,
        "vlm_revision": revision,
        "vlm_sha256": vlm_sha256,
    }
    source = Path(resume_from).expanduser().resolve() if resume_from else None
    source_run = None
    source_manifest = None
    if source is not None:
        source_run = json.loads((source.parent / "run.json").read_text())
        source_manifest = json.loads((source / "manifest.json").read_text())
        _validate_resume_metadata(
            source_run,
            source_manifest,
            expected,
            c1_steps=config.c1_steps,
            target_steps=target_steps,
            split_sha256=task_split["split_sha256"],
            vlm_revision=revision,
            vlm_sha256=vlm_sha256,
        )
        _verify_optimizer_checkpoint(source, source_manifest)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Learner output directory is not empty: {output}")

    for name in ("hub", "datasets"):
        (data / "hf" / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(data / "hf" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(data / "hf" / "datasets"))

    if _sha256(vlm / "model.safetensors") != vlm_sha256:
        raise ValueError("Pinned VLM weights differ from the declared source hash")
    _seed_all(config.seed, config.device)
    if source is None:
        policy, matched = build_learner_policy(vlm, device=config.device)
        preprocessor, postprocessor = make_learner_processors(
            policy.config, config.task_alias, configs
        )
        start_step = 0
        state = None
    else:
        policy, preprocessor, postprocessor, loaded_manifest = load_learner_bundle(
            source, vlm, device=config.device
        )
        if loaded_manifest != source_manifest:
            raise ValueError("Learner checkpoint changed while it was loaded")
        matched = None
        state = torch.load(source / "optimizer.pt", map_location="cpu", weights_only=True)
        start_step = int(state["step"])
        if start_step != source_manifest["step"]:
            raise ValueError("Optimizer step differs from the saved learner model")
    parameter_groups = audit_learner_parameter_groups(policy)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
    policy.train()
    policy.model.vlm_with_expert.vlm.eval()
    training_data = _frozen_training_dataset(
        config.task_alias, configs, data, chunk_size=policy.config.chunk_size
    )
    if len(training_data) < config.batch_size:
        raise ValueError("Training data has fewer frames than the batch size")
    if state is not None:
        torch.set_rng_state(state["torch_rng_state"])
        if config.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    output.mkdir(parents=True, exist_ok=True)
    run = {
        "schema_version": 1,
        "policy": "smolvla_learner",
        "stage": stage,
        "steps": target_steps,
        "resumed_from_step": start_step,
        "root_c1_steps": config.c1_steps,
        "resumed_from": str(source) if source is not None else None,
        "source_config_sha256": source_run["config_sha256"] if source_run else None,
        "matched_vlm_tensors": matched,
        "parameter_groups": parameter_groups,
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        **expected,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    last_loss = math.nan
    with (output / "metrics.jsonl").open("w") as metrics:
        for step in range(start_step + 1, target_steps + 1):
            indices = _batch_indices(len(training_data), config.batch_size, config.seed, step)
            raw_batch = default_collate([training_data[index] for index in indices])
            batch = preprocessor(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"SmolVLA loss is not finite at step {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"SmolVLA gradient is not finite at step {step}")
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
    checkpoint = output / f"checkpoint-{target_steps:06d}"
    optimizer_state = {
        "step": target_steps,
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all()
        if config.device.startswith("cuda")
        else [],
    }
    manifest = save_learner_bundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        destination=checkpoint,
        task_alias=config.task_alias,
        configs_root=configs,
        vlm_dir=vlm,
        vlm_revision=revision,
        vlm_sha256=vlm_sha256,
        stage=stage,
        step=target_steps,
        parent_model_sha256=source_manifest["model_sha256"] if source_manifest else None,
        optimizer_state=optimizer_state,
    )
    (output / "result.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "steps": target_steps,
                "last_loss": last_loss,
                "elapsed_s": round(time.monotonic() - started, 3),
                "checkpoint": str(checkpoint),
                "model_sha256": manifest["model_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return SmolVLATrainResult(stage, target_steps, last_loss, checkpoint)


def main(argv: list[str] | None = None) -> None:
    """Train one declared learner stage from the terminal."""
    parser = argparse.ArgumentParser(description="Train the pinned SmolVLA learner")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train/smolvla_bowl_plate.yaml")
    )
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument(
        "--vlm-dir", type=Path, default=Path(".local-data/models/smolvlm2-500m-video-instruct")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("c1", "c2"), required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    result = train_smolvla_stage(
        config_path=args.config,
        configs_root=args.configs_root,
        data_root=args.data_root,
        vlm_dir=args.vlm_dir,
        output_dir=args.output,
        stage=args.stage,
        resume_from=args.resume_from,
    )
    print(
        json.dumps(
            {"stage": result.stage, "steps": result.steps, "checkpoint": str(result.checkpoint)}
        )
    )


if __name__ == "__main__":
    main()
