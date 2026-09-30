"""Save and load a policy with its LeRobot processors and data identity."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from lerobot.policies.act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline, load_pretrained_policy_processors


def save_act_bundle(
    policy: ACTPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    destination: str | Path,
    task_alias: str,
    configs_root: str | Path,
) -> None:
    """Save one complete local ACT checkpoint; write its manifest last."""
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"ACT checkpoint directory is not empty: {target}")
    configs = Path(configs_root)
    split_manifest = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    split = split_manifest["tasks"][task_alias]
    stats = json.loads((configs / "data" / "stats" / f"{task_alias}.json").read_text())
    if stats["split_sha256"] != split["split_sha256"]:
        raise ValueError(
            "Training statistics split hash differs from the frozen episode split hash"
        )
    if stats["dataset_revision"] != split_manifest["dataset_revision"]:
        raise ValueError("Training statistics dataset revision differs from the split revision")

    target.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(target)
    preprocessor.save_pretrained(target)
    postprocessor.save_pretrained(target)
    manifest = {
        "schema_version": 1,
        "policy": "act",
        "task_alias": task_alias,
        "dataset_revision": split_manifest["dataset_revision"],
        "split_sha256": split["split_sha256"],
        "training_stats_sha256": {
            key: stats["modalities"][key]["sha256"] for key in ("observation.state", "action")
        },
        "n_action_steps": policy.config.n_action_steps,
        "chunk_size": policy.config.chunk_size,
    }
    temporary = target / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target / "manifest.json")


def load_act_bundle(
    source: str | Path,
) -> tuple[ACTPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline, dict[str, object]]:
    """Load saved weights, processors, and the manifest from one directory."""
    root = Path(source)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ACT checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("policy") != "act":
        raise ValueError("Unsupported ACT checkpoint manifest")
    policy = ACTPolicy.from_pretrained(root, local_files_only=True)
    preprocessor, postprocessor = load_pretrained_policy_processors(root)
    return policy, preprocessor, postprocessor, manifest
