"""Check paired counts, intervals, and infrastructure limits."""

import pytest

from vla_lab.evaluation.paired import exact_mcnemar, paired_report, wilson_interval


def _rows(values: list[bool], revision: str) -> list[dict]:
    return [
        {
            "task_id": "bowl_plate",
            "partition": "development",
            "initial_state_id": index,
            "policy_revision": revision,
            "success": success,
            "termination_reason": "success" if success else "timeout",
        }
        for index, success in enumerate(values)
    ]


def test_exact_paired_report_uses_one_row_per_state() -> None:
    report = paired_report(_rows([True, True, True], "a"), _rows([False, False, False], "b"))
    assert report["paired_difference"] == 1
    assert report["exact_mcnemar_p"] == pytest.approx(0.25)
    assert report["a_wilson_95"][0] < 1
    assert wilson_interval(0, 3)[1] < 1
    assert exact_mcnemar(0, 0) == 1


def test_infrastructure_failure_blocks_policy_inference() -> None:
    a = _rows([True, False], "a")
    b = _rows([False, False], "b")
    b[1]["termination_reason"] = "infrastructure_error"
    report = paired_report(a, b)
    assert report["infrastructure_errors"] == 1
    assert report["exact_mcnemar_p"] is None


def test_mismatched_or_repeated_states_are_rejected() -> None:
    a = _rows([True, False], "a")
    b = _rows([True, False], "b")
    b.reverse()
    with pytest.raises(ValueError, match="state IDs"):
        paired_report(a, b)
