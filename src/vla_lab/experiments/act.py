"""Run the fixed ACT data and action-horizon comparisons."""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from vla_lab.data.inspection import audit_task_split
from vla_lab.data.selection import prepare_act_selection
from vla_lab.evaluation.paired import wilson_interval
from vla_lab.evaluation.states import evaluate_act_states, selected_state_ids
from vla_lab.training.act import train_act_reference
from vla_lab.types import JSONValue


@dataclass(frozen=True)
class ActVariant:
    """One declared ACT training setting."""

    name: str
    episode_count: int
    chunk_size: int


@dataclass(frozen=True)
class ActSuiteConfig:
    """The declared task, training budget, and development states."""

    task_alias: str
    seed: int
    steps: int
    batch_size: int
    checkpoint_every: int
    device: str
    state_ids: tuple[int, ...]
    variants: tuple[ActVariant, ...]
    config_sha256: str


def load_act_suite_config(path: str | Path, configs_root: str | Path) -> ActSuiteConfig:
    """Read a local ACT suite and reject undeclared final states."""
    source = Path(path)
    content = source.read_bytes()
    payload = yaml.safe_load(content)
    if payload.get("schema_version") != 1 or payload.get("profile") != "local":
        raise ValueError("ACT suite needs schema version 1 and the local profile")
    task = str(payload["task_alias"])
    state_ids = tuple(
        selected_state_ids(configs_root, task, "development", payload["development_state_ids"])
    )
    split = yaml.safe_load((Path(configs_root) / "data" / "episodes.yaml").read_text())
    full_count = len(split["tasks"][task]["train_episode_ids"])
    rows = payload["variants"]
    if not rows:
        raise ValueError("ACT suite needs at least one variant")
    variants = []
    names = set()
    for row in rows:
        name = str(row["name"])
        count = int(row["episode_count"])
        chunk_size = int(row["chunk_size"])
        if (
            name in names
            or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None
            or not 1 <= count <= full_count
            or not 1 <= chunk_size <= 100
        ):
            raise ValueError("ACT suite has a duplicate or invalid variant")
        names.add(name)
        variants.append(ActVariant(name, count, chunk_size))
    steps = int(payload["steps"])
    batch_size = int(payload["batch_size"])
    checkpoint_every = int(payload["checkpoint_every"])
    if min(steps, batch_size, checkpoint_every) < 1:
        raise ValueError("ACT steps, batch size, and checkpoint period must be positive")
    return ActSuiteConfig(
        task_alias=task,
        seed=int(payload["seed"]),
        steps=steps,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        device=str(payload["device"]),
        state_ids=state_ids,
        variants=tuple(variants),
        config_sha256=hashlib.sha256(content).hexdigest(),
    )


def _write_json(path: Path, payload: dict[str, JSONValue]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_act_suite(
    *,
    config_path: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
) -> dict[str, JSONValue]:
    """Train and evaluate each ACT variant on the same development states."""

    source = Path(config_path).expanduser().resolve()
    configs = Path(configs_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    recipe = load_act_suite_config(source, configs)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"ACT suite output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "recipe.yaml").write_bytes(source.read_bytes())
    results = []
    for variant in recipe.variants:
        root = target / "variants" / variant.name
        resolved = prepare_act_selection(
            configs_root=configs,
            data_root=data,
            task_alias=recipe.task_alias,
            episode_count=variant.episode_count,
            output_dir=root / "resolved",
        )
        data_audit = audit_task_split(resolved, recipe.task_alias)
        trained = train_act_reference(
            task_alias=recipe.task_alias,
            configs_root=resolved,
            data_root=data,
            output_dir=root / "training",
            steps=recipe.steps,
            batch_size=recipe.batch_size,
            checkpoint_every=recipe.checkpoint_every,
            chunk_size=variant.chunk_size,
            seed=recipe.seed,
            device=recipe.device,
        )
        checkpoint = trained.output_dir / f"checkpoint-{recipe.steps:06d}"
        evaluation = evaluate_act_states(
            checkpoint=checkpoint,
            configs_root=configs,
            data_root=data,
            output_dir=root / "evaluation",
            state_ids=recipe.state_ids,
        )
        report = {
            "schema_version": 1,
            "variant": variant.name,
            "task_alias": recipe.task_alias,
            "train_episode_count": variant.episode_count,
            "train_frame_count": data_audit["training_frames"],
            "split_sha256": data_audit["split_sha256"],
            "chunk_size": variant.chunk_size,
            "n_action_steps": min(10, variant.chunk_size),
            "training_steps": trained.steps,
            "last_training_loss": trained.last_loss,
            "development_state_ids": evaluation["state_ids"],
            "development_successes": evaluation["successes"],
            "development_attempts": evaluation["attempts"],
            "development_wilson_95": wilson_interval(
                evaluation["successes"], evaluation["attempts"]
            ),
            "policy_errors": evaluation["policy_errors"],
            "infrastructure_errors": evaluation["infrastructure_errors"],
            "checkpoint": str(checkpoint),
            "config_sha256": recipe.config_sha256,
            "final_states_used": False,
        }
        _write_json(root / "report.json", report)
        results.append(report)
    summary = {
        "schema_version": 1,
        "task_alias": recipe.task_alias,
        "partition": "development",
        "config_sha256": recipe.config_sha256,
        "state_ids": list(recipe.state_ids),
        "variants": results,
        "valid_for_comparison": all(row["infrastructure_errors"] == 0 for row in results),
        "final_states_used": False,
    }
    _write_json(target / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the local ACT suite from a detached job or terminal."""
    parser = argparse.ArgumentParser(description="Train and evaluate fixed ACT variants")
    parser.add_argument("--config", type=Path, default=Path("configs/train/act_task.yaml"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = run_act_suite(
        config_path=arguments.config,
        configs_root=arguments.configs_root,
        data_root=arguments.data_root,
        output_dir=arguments.output,
    )
    print(json.dumps({"variants": len(summary["variants"]), "output": str(arguments.output)}))


if __name__ == "__main__":
    main()
