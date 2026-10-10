from dataclasses import replace

import pytest

from vla_lab.teaching.evaluation import (
    development_cases,
    final_cases,
    paired_summary,
    summarize_latency_samples,
    validate_case_results,
)
from vla_lab.teaching.records import CaseResult, Partition


def rows(successes):
    return [
        CaseResult(
            case_id=f"state-{index}-seed-0",
            state_id=index,
            reset_seed=0,
            task_id="bowl_plate",
            partition=Partition.DEVELOPMENT,
            prompt="put the bowl on the plate",
            action_steps=10,
            policy_revision="policy",
            success=success,
        )
        for index, success in enumerate(successes)
    ]


def test_state_groups_and_final_gate():
    assert development_cases("exploration") == list(range(10))
    assert development_cases("method") == list(range(10, 15))
    assert development_cases("intervention") == list(range(15, 20))
    with pytest.raises(ValueError, match="frozen"):
        final_cases(data_unit_complete=True, recipes_frozen=False)
    assert final_cases(data_unit_complete=True, recipes_frozen=True) == list(range(20, 50))


def test_paired_summary_keeps_failed_attempts_and_reports_disagreements():
    left, right = rows([True, False, False]), rows([False, False, True])
    right = [replace(row, policy_revision="other") for row in right]
    result = paired_summary(
        left, right, expected_case_ids=["state-0-seed-0", "state-1-seed-0", "state-2-seed-0"]
    )
    assert result.cases == 3
    assert result.left_successes == 1 and result.right_successes == 1
    assert result.left_only == 1 and result.right_only == 1
    assert result.failed_attempts_included is True


def test_case_validation_rejects_missing_or_duplicate_rows():
    with pytest.raises(ValueError, match="missing"):
        validate_case_results(rows([True]), expected_case_ids=["state-0-seed-0", "state-1-seed-0"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_case_results(
            [rows([True])[0], rows([False])[0]], expected_case_ids=["state-0-seed-0"]
        )


def test_paired_summary_rejects_seed_or_action_length_mismatch():
    left = rows([True])
    right = [replace(left[0], reset_seed=4, policy_revision="other")]
    with pytest.raises(ValueError, match="differ"):
        paired_summary(left, right, expected_case_ids=["state-0-seed-0"])


def test_methods_can_use_different_fixed_action_lengths():
    left = rows([True])
    right = [replace(left[0], action_steps=8, policy_revision="other", success=False)]
    result = paired_summary(left, right, expected_case_ids=["state-0-seed-0"])
    assert result.left_only == 1


def test_latency_summary_reports_tail_and_missing_values():
    assert summarize_latency_samples([]) == {
        "count": 0,
        "mean_s": None,
        "p95_s": None,
        "max_s": None,
    }
    summary = summarize_latency_samples([0.01, 0.02, 0.1])
    assert summary["count"] == 3 and summary["max_s"] == 0.1 and summary["p95_s"] >= 0.02
