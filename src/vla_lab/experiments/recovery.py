"""Continue ACT on accepted recordings and keep assessment states separate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from vla_lab.data.recording import CONTROL_PERIOD_MS, RecordedEpisode
from vla_lab.teaching.interfaces import ResampleRowsFunction, TransitionRowsFunction, WindowFunction
from vla_lab.types import JSONValue

_ACT_CHUNK_SIZE = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timing_provenance(training_transition_hz: int) -> dict[str, JSONValue]:
    """Describe recording, training, chunk, and controller timing separately."""
    recording_hz = 1000 // CONTROL_PERIOD_MS
    if training_transition_hz < 1 or recording_hz % training_transition_hz:
        raise ValueError("training_transition_hz must be a positive divisor of the recording rate")
    return {
        "recorded_controller_period_ms": CONTROL_PERIOD_MS,
        "recorded_control_hz": recording_hz,
        "training_transition_hz": training_transition_hz,
        "training_resample_stride": recording_hz // training_transition_hz,
        "act_action_chunk_steps": _ACT_CHUNK_SIZE,
        "act_action_chunk_duration_s": _ACT_CHUNK_SIZE / training_transition_hz,
        "controller_action_hz": recording_hz,
        "controller_behavior": "apply one action per 20 Hz controller tick",
    }


def load_recovery_manifest(path: str | Path, configs_root: str | Path) -> dict[str, JSONValue]:
    """Check accepted recording identities and held-out development IDs."""
    source = Path(path)
    record = json.loads(source.read_text())
    task_alias = record.get("task_alias")
    if record.get("schema_version") != 1 or record.get("source_type") not in (
        "keyboard",
        "policy_generated",
    ):
        raise ValueError("Recovery manifest schema or source type is invalid")
    splits = yaml.safe_load((Path(configs_root) / "eval" / "splits.yaml").read_text())
    try:
        development = set(splits["tasks"][task_alias]["development_state_ids"])
    except (KeyError, TypeError) as error:
        raise ValueError("Recovery task is not in the frozen state split") from error
    heldout = record.get("heldout_development_state_ids")
    recordings = record.get("recordings")
    if (
        not isinstance(heldout, list)
        or not heldout
        or len(set(heldout)) != len(heldout)
        or not set(heldout) <= development
        or not isinstance(recordings, list)
        or not recordings
    ):
        raise ValueError("Recovery needs unique held-out development states and recordings")
    used: set[int] = set()
    episode_ids: set[str] = set()
    for item in recordings:
        state_id = item["state_id"]
        if state_id not in development:
            raise ValueError("Recovery recording is outside the development partition")
        if state_id in used:
            raise ValueError("Recovery state ID is repeated")
        used.add(state_id)
        if item.get("accepted") is not True:
            raise ValueError("Recovery training needs accepted recordings only")
        reset_seed = item["reset_seed"]
        if reset_seed != 20260930 + state_id:
            raise ValueError("Recovery reset seed differs from the frozen seed rule")
        recording_path = Path(item["path"])
        if _sha256(recording_path) != item["sha256"]:
            raise ValueError(f"Recovery recording hash differs: {recording_path}")
        episode = RecordedEpisode.load(recording_path)
        if (
            episode.task_id != task_alias
            or episode.reset_seed != reset_seed
            or episode.defect is not None
            or episode.episode_id in episode_ids
        ):
            raise ValueError("Recovery recording content differs from its manifest")
        episode_ids.add(episode.episode_id)
    if used & set(heldout):
        raise ValueError("Recovery recording and assessment state IDs overlap")
    return record


def continue_act_with_recovery(
    *,
    manifest_path: str | Path,
    base_checkpoint: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    steps: int = 200,
    seed: int = 20261001,
    transition_rows_fn: TransitionRowsFunction | None = None,
    resample_rows_fn: ResampleRowsFunction | None = None,
    training_transition_hz: int | None = None,
    window_fn: WindowFunction | None = None,
) -> dict[str, JSONValue]:
    """Train a bounded ACT continuation with a new optimizer and saved processor."""
    from vla_lab.data.recovery_dataset import RecoveryDataset
    from vla_lab.policy.bundle import load_act_bundle
    from vla_lab.training.act import train_act_reference

    if resample_rows_fn is not None and training_transition_hz is None:
        raise ValueError("Set training_transition_hz when resample_rows_fn changes the data rate")
    effective_training_hz = (
        1000 // CONTROL_PERIOD_MS if training_transition_hz is None else training_transition_hz
    )
    timing = _timing_provenance(effective_training_hz)

    manifest_file = Path(manifest_path)
    config_root = Path(configs_root)
    record = load_recovery_manifest(manifest_file, config_root)
    if not 5 <= len(record["recordings"]) <= 10:
        raise ValueError("The core recovery continuation needs 5 to 10 accepted recordings")
    if not 1 <= steps <= 500:
        raise ValueError("The core recovery continuation needs 1 to 500 steps")
    task_alias = record["task_alias"]
    base = Path(base_checkpoint)
    policy, _, _, bundle = load_act_bundle(base)
    split = yaml.safe_load((config_root / "data" / "episodes.yaml").read_text())["tasks"][
        task_alias
    ]
    if (
        bundle["task_alias"] != task_alias
        or bundle["split_sha256"] != split["split_sha256"]
        or bundle["chunk_size"] != 20
    ):
        raise ValueError("Base ACT checkpoint differs from the recovery task and split")
    paths = [Path(item["path"]) for item in record["recordings"]]
    dataset = RecoveryDataset(
        paths,
        task_id=task_alias,
        expected_reset_seeds={item["reset_seed"] for item in record["recordings"]},
        transition_rows_fn=transition_rows_fn,
        resample_rows_fn=resample_rows_fn,
        window_fn=window_fn,
    )
    result = train_act_reference(
        task_alias=task_alias,
        configs_root=config_root,
        data_root=data_root,
        output_dir=output_dir,
        steps=steps,
        batch_size=8,
        device="cuda",
        seed=seed,
        checkpoint_every=steps,
        chunk_size=20,
        dataset=dataset,
        policy=policy,
    )
    checkpoint = Path(output_dir) / f"checkpoint-{steps:06d}"
    provenance = {
        "schema_version": 1,
        "task_alias": task_alias,
        "source_type": record["source_type"],
        "recovery_manifest": str(manifest_file),
        "recovery_manifest_sha256": _sha256(manifest_file),
        "recording_count": len(paths),
        "recording_sample_count": len(dataset),
        **timing,
        "heldout_development_state_ids": record["heldout_development_state_ids"],
        "base_checkpoint": str(base),
        "base_model_sha256": _sha256(base / "model.safetensors"),
        "new_optimizer_steps": result.steps,
        "new_optimizer_seed": seed,
        "last_loss": result.last_loss,
        "checkpoint": str(checkpoint),
        "model_sha256": _sha256(checkpoint / "model.safetensors"),
        "optimizer_state": "new AdamW state; model weights start from the base checkpoint",
    }
    (Path(output_dir) / "recovery_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def main(argv: list[str] | None = None) -> None:
    """Run one local recovery continuation from a checked manifest."""
    parser = argparse.ArgumentParser(description="Continue ACT on accepted recovery recordings")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)
    result = continue_act_with_recovery(
        manifest_path=args.manifest,
        base_checkpoint=args.base_checkpoint,
        configs_root=args.configs_root,
        data_root=args.data_root,
        output_dir=args.output,
        steps=args.steps,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
