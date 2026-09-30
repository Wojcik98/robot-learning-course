"""Report paired simulator-state outcomes with exact small-sample checks."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a 95% Wilson interval for one binomial count."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Success count must fit a nonempty trial count")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar(a_only: int, b_only: int) -> float:
    """Return the two-sided exact McNemar probability for paired outcomes."""
    if a_only < 0 or b_only < 0:
        raise ValueError("Discordant counts must not be negative")
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def load_episode_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSONL evaluation result without dropping failed attempts."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Paired report needs nonempty episode rows")
    return rows


def paired_report(
    a_rows: Sequence[dict[str, Any]], b_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compare two policies on the same ordered simulator states."""
    if not a_rows or len(a_rows) != len(b_rows):
        raise ValueError("Paired report needs equal nonempty state lists")
    a_ids = [row["initial_state_id"] for row in a_rows]
    b_ids = [row["initial_state_id"] for row in b_rows]
    if a_ids != b_ids or len(set(a_ids)) != len(a_ids):
        raise ValueError("Paired report state IDs differ, repeat, or change order")
    for a, b in zip(a_rows, b_rows, strict=True):
        if a["task_id"] != b["task_id"] or a["partition"] != b["partition"]:
            raise ValueError("Paired report task or partition differs")
        if (
            a["task_id"] != a_rows[0]["task_id"]
            or a["partition"] != a_rows[0]["partition"]
            or a["policy_revision"] != a_rows[0]["policy_revision"]
            or b["policy_revision"] != b_rows[0]["policy_revision"]
        ):
            raise ValueError("Paired report mixes tasks, partitions, or policy revisions")
    n = len(a_rows)
    a_success = sum(bool(row["success"]) for row in a_rows)
    b_success = sum(bool(row["success"]) for row in b_rows)
    a_only = sum(
        bool(a["success"]) and not bool(b["success"]) for a, b in zip(a_rows, b_rows, strict=True)
    )
    b_only = sum(
        bool(b["success"]) and not bool(a["success"]) for a, b in zip(a_rows, b_rows, strict=True)
    )
    infrastructure_errors = sum(
        row["termination_reason"] == "infrastructure_error" for row in (*a_rows, *b_rows)
    )
    return {
        "schema_version": 1,
        "task_id": a_rows[0]["task_id"],
        "partition": a_rows[0]["partition"],
        "state_ids": a_ids,
        "paired_states": n,
        "a_policy_revision": a_rows[0]["policy_revision"],
        "b_policy_revision": b_rows[0]["policy_revision"],
        "a_successes": a_success,
        "b_successes": b_success,
        "a_success_rate": a_success / n,
        "b_success_rate": b_success / n,
        "a_wilson_95": wilson_interval(a_success, n),
        "b_wilson_95": wilson_interval(b_success, n),
        "paired_difference": (a_success - b_success) / n,
        "a_only": a_only,
        "b_only": b_only,
        "infrastructure_errors": infrastructure_errors,
        "valid_for_policy_comparison": infrastructure_errors == 0,
        "exact_mcnemar_p": exact_mcnemar(a_only, b_only) if infrastructure_errors == 0 else None,
    }


def write_paired_report(
    a_path: str | Path, b_path: str | Path, output: str | Path
) -> dict[str, Any]:
    """Save one comparison after checking both exact state lists."""
    report = paired_report(load_episode_rows(a_path), load_episode_rows(b_path))
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Paired report already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
