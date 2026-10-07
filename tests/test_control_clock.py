"""Keep dataset frame lookup separate from the simulator clock."""

import pytest

from vla_lab.timing import frame_offset_seconds, require_libero_rate


def test_history_and_action_offsets_use_adjacent_storage_frames():
    assert frame_offset_seconds([-1, 0, 1, 2]) == [-0.1, 0.0, 0.1, 0.2]
    require_libero_rate(20)
    with pytest.raises(ValueError, match="20 Hz"):
        require_libero_rate(10)


def test_lookup_rejects_zero_storage_rate():
    with pytest.raises(ValueError, match="positive"):
        frame_offset_seconds([0, 1], 0)
