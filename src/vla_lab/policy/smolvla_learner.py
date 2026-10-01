"""Initialize a LIBERO action expert on the pinned SmolVLM2 backbone."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla import (
    SmolVLAConfig,
    SmolVLAPolicy,
    make_smolvla_pre_post_processors,
)
from lerobot.processor import PolicyProcessorPipeline, load_pretrained_policy_processors
from safetensors import safe_open

from vla_lab.data.stats import TrainingStats


def make_learner_config(vlm_dir: str | Path, *, device: str = "cuda") -> SmolVLAConfig:
    """Set LIBERO shapes and load the pretrained VLM for learner model A."""
    return SmolVLAConfig(
        device=device,
        push_to_hub=False,
        vlm_model_name=str(Path(vlm_dir).resolve()),
        load_vlm_weights=True,
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        n_action_steps=10,
        chunk_size=50,
        num_vlm_layers=16,
        expert_width_multiplier=0.75,
        prefix_length=0,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
    )


def make_learner_processors(
    config: SmolVLAConfig, task_alias: str, configs_root: str | Path
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Use only the frozen training split for state and action normalization."""
    root = Path(configs_root)
    split_manifest = yaml.safe_load((root / "data" / "episodes.yaml").read_text())
    try:
        split = split_manifest["tasks"][task_alias]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    payload = json.loads((root / "data" / "stats" / f"{task_alias}.json").read_text())
    if payload["split_sha256"] != split["split_sha256"]:
        raise ValueError(
            "Training statistics split hash differs from the frozen episode split hash"
        )
    if payload["dataset_revision"] != split_manifest["dataset_revision"]:
        raise ValueError("Training statistics dataset revision differs from the split revision")
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, dimension in (("observation.state", 8), ("action", 7)):
        verified = TrainingStats.from_dict(payload["modalities"][key])
        if verified.mean.shape != (dimension,):
            raise ValueError(f"Invalid training statistics shape for {key}")
        stats[key] = {
            "mean": torch.tensor(verified.mean, dtype=torch.float32),
            "std": torch.tensor(verified.std, dtype=torch.float32),
        }
    return make_smolvla_pre_post_processors(config, dataset_stats=stats)


def audit_loaded_vlm(model: torch.nn.Module, weights_path: str | Path) -> int:
    """Check every retained VLM tensor against the pinned source weights."""
    retained = model.state_dict()
    matched = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for name, value in retained.items():
            if name not in keys:
                raise ValueError(f"Retained VLM tensor is missing from source weights: {name}")
            original = source.get_tensor(name)
            if original.shape != value.shape or not torch.equal(
                value.cpu(), original.to(value.dtype)
            ):
                raise ValueError(f"Retained VLM tensor differs from source weights: {name}")
            matched += 1
    return matched


def audit_learner_parameter_groups(policy: SmolVLAPolicy) -> dict[str, dict[str, object]]:
    """Count each loaded or new parameter group and check its trainable flags."""
    prefixes = {
        "vlm_backbone": "model.vlm_with_expert.vlm.",
        "action_expert": "model.vlm_with_expert.lm_expert.",
        "state_projection": "model.state_proj.",
        "action_input_projection": "model.action_in_proj.",
        "action_output_projection": "model.action_out_proj.",
        "action_time_input": "model.action_time_mlp_in.",
        "action_time_output": "model.action_time_mlp_out.",
    }
    groups: dict[str, dict[str, object]] = {
        group: {
            "source": "loaded" if group == "vlm_backbone" else "initialized",
            "parameters": 0,
            "trainable_parameters": 0,
            "tensors": 0,
        }
        for group in prefixes
    }
    for name, parameter in policy.named_parameters():
        group = next((group for group, prefix in prefixes.items() if name.startswith(prefix)), None)
        if group is None:
            raise ValueError(f"Unclassified learner parameter: {name}")
        should_train = group != "vlm_backbone"
        if parameter.requires_grad != should_train:
            raise ValueError(f"Wrong trainable flag in {group}: {name}")
        counts = groups[group]
        counts["parameters"] += parameter.numel()
        counts["trainable_parameters"] += parameter.numel() if parameter.requires_grad else 0
        counts["tensors"] += 1
    return groups


def build_learner_policy(vlm_dir: str | Path, *, device: str = "cuda") -> tuple[SmolVLAPolicy, int]:
    """Build model A and check the retained backbone tensors."""
    root = Path(vlm_dir).resolve()
    weights = root / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Pinned VLM weights are missing: {weights}")
    policy = SmolVLAPolicy(make_learner_config(root, device=device)).to(device)
    matched = audit_loaded_vlm(policy.model.vlm_with_expert.vlm, weights)
    audit_learner_parameter_groups(policy)
    return policy, matched


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_learner_bundle(
    *,
    policy: SmolVLAPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    destination: str | Path,
    task_alias: str,
    configs_root: str | Path,
    vlm_dir: str | Path,
    vlm_revision: str,
    vlm_sha256: str,
    stage: str,
    step: int,
    parent_model_sha256: str | None = None,
) -> dict[str, object]:
    """Save learner weights and processors with their source identity."""
    if stage not in ("c1", "c2") or step < 1:
        raise ValueError("Learner stage must be c1 or c2 with a positive step")
    source = Path(vlm_dir).expanduser().resolve() / "model.safetensors"
    if _sha256(source) != vlm_sha256:
        raise ValueError("Pinned VLM weights differ from the declared source hash")
    configs = Path(configs_root)
    split_manifest = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    split = split_manifest["tasks"][task_alias]
    stats = json.loads((configs / "data" / "stats" / f"{task_alias}.json").read_text())
    if (
        stats["split_sha256"] != split["split_sha256"]
        or stats["dataset_revision"] != split_manifest["dataset_revision"]
    ):
        raise ValueError("Learner statistics do not match the training split")
    hashes = {
        key: TrainingStats.from_dict(stats["modalities"][key]).sha256
        for key in ("observation.state", "action")
    }
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Learner checkpoint directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(target)
    preprocessor.save_pretrained(target)
    postprocessor.save_pretrained(target)
    processor_files = sorted(
        path
        for path in target.iterdir()
        if path.is_file() and path.name.startswith(("policy_preprocessor", "policy_postprocessor"))
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "policy": "smolvla_learner",
        "stage": stage,
        "step": step,
        "task_alias": task_alias,
        "dataset_revision": split_manifest["dataset_revision"],
        "split_sha256": split["split_sha256"],
        "training_stats_sha256": hashes,
        "vlm_repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "vlm_revision": vlm_revision,
        "vlm_sha256": vlm_sha256,
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "model_sha256": _sha256(target / "model.safetensors"),
        "preprocessor_sha256": _sha256(target / "policy_preprocessor.json"),
        "postprocessor_sha256": _sha256(target / "policy_postprocessor.json"),
        "processor_files_sha256": {path.name: _sha256(path) for path in processor_files},
        "parent_model_sha256": parent_model_sha256,
    }
    temporary = target / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target / "manifest.json")
    return manifest


def load_learner_bundle(
    source: str | Path, vlm_dir: str | Path, *, device: str = "cuda"
) -> tuple[SmolVLAPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline, dict[str, object]]:
    """Load a trained learner with its own saved normalization."""
    root = Path(source).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("policy") != "smolvla_learner":
        raise ValueError("Unsupported learner checkpoint manifest")
    if _sha256(root / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Learner model hash differs from the saved manifest")
    for name, filename in (
        ("preprocessor_sha256", "policy_preprocessor.json"),
        ("postprocessor_sha256", "policy_postprocessor.json"),
    ):
        if _sha256(root / filename) != manifest[name]:
            raise ValueError(f"Learner {filename} hash differs from the saved manifest")
    for filename, expected_sha256 in manifest.get("processor_files_sha256", {}).items():
        if _sha256(root / filename) != expected_sha256:
            raise ValueError(f"Learner {filename} hash differs from the saved manifest")
    vlm = Path(vlm_dir).expanduser().resolve()
    if _sha256(vlm / "model.safetensors") != manifest["vlm_sha256"]:
        raise ValueError("Pinned VLM weights differ from the saved learner source")
    config = SmolVLAConfig.from_pretrained(str(root))
    config.vlm_model_name = str(vlm)
    config.load_vlm_weights = False
    config.device = device
    config.push_to_hub = False
    policy = SmolVLAPolicy.from_pretrained(
        str(root), config=config, strict=True, local_files_only=True
    ).to(device)
    audit_learner_parameter_groups(policy)
    preprocessor, postprocessor = load_pretrained_policy_processors(
        str(root),
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(vlm)}},
    )
    return policy, preprocessor, postprocessor, manifest
