"""Make a checked training manifest from accepted keyboard recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from vla_lab.data.recording import RecordedEpisode
from vla_lab.types import JSONValue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_keyboard_manifest(
    recording_paths: Sequence[str | Path],
    *,
    task_alias: str,
    configs_root: str | Path,
    output: str | Path,
) -> dict[str, JSONValue]:
    """Treat selected files as accepted and reserve all unused development states."""
    if not 5 <= len(recording_paths) <= 10:
        raise ValueError("Keyboard recovery needs 5 to 10 accepted recordings")
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Recovery manifest already exists: {target}")
    splits = yaml.safe_load((Path(configs_root) / "eval" / "splits.yaml").read_text())
    try:
        development = list(splits["tasks"][task_alias]["development_state_ids"])
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    used: set[int] = set()
    episode_ids: set[str] = set()
    recordings: list[dict[str, JSONValue]] = []
    for value in recording_paths:
        path = Path(value)
        episode = RecordedEpisode.load(path)
        if episode.task_id != task_alias:
            raise ValueError(f"Recording task differs from {task_alias}: {path}")
        if episode.reset_seed is None:
            raise ValueError(f"Recording reset seed is missing: {path}")
        state_id = episode.reset_seed - 20260930
        if state_id not in development:
            raise ValueError(f"Recording state is outside the development partition: {path}")
        if state_id in used or episode.episode_id in episode_ids:
            raise ValueError(f"Recording state or episode ID is repeated: {path}")
        if episode.defect is not None:
            raise ValueError(f"Recording has a timing defect: {path}")
        used.add(state_id)
        episode_ids.add(episode.episode_id)
        recordings.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "state_id": state_id,
                "reset_seed": episode.reset_seed,
                "accepted": True,
                "action_count": len(episode.actions),
            }
        )
    record = {
        "schema_version": 1,
        "task_alias": task_alias,
        "source_type": "keyboard",
        "attempted_state_ids": sorted(used),
        "heldout_development_state_ids": [
            state_id for state_id in development if state_id not in used
        ],
        "recordings": recordings,
        "rejected_recordings": [],
        "provenance_limit": (
            "The operator selected these files as accepted. "
            "This command does not check task success or human control quality."
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv: list[str] | None = None) -> None:
    """Create one manifest for the bounded ACT recovery continuation."""
    parser = argparse.ArgumentParser(description="Check accepted keyboard recovery files")
    parser.add_argument("--recordings", nargs="+", type=Path, required=True)
    parser.add_argument("--task", default="bowl_plate")
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_keyboard_manifest(
        args.recordings,
        task_alias=args.task,
        configs_root=args.configs_root,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
