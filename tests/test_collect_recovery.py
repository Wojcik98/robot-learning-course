"""Keep reference recovery collection on development states."""

import pytest

from vla_lab.experiments.collect_recovery import candidate_development_states


def test_recovery_candidates_reject_final_states() -> None:
    assert candidate_development_states("configs", "bowl_plate", [0, 1, 2]) == [0, 1, 2]
    with pytest.raises(ValueError, match="development"):
        candidate_development_states("configs", "bowl_plate", [0, 20])
