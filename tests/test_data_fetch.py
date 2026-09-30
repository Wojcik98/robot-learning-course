"""Check that the course downloads only the selected LIBERO videos."""

import pytest

from vla_lab.data.fetch import dataset_patterns


def test_demo_download_uses_one_real_av1_sample() -> None:
    patterns = dataset_patterns("demo")
    assert patterns == ["videos/observation.images.image/chunk-000/file-000.mp4"]


def test_training_download_keeps_selected_camera_pairs() -> None:
    patterns = dataset_patterns("training")
    assert "data/**" in patterns
    assert "meta/**" in patterns
    for camera in ("image", "image2"):
        for index in (0, *range(10, 19)):
            assert f"videos/observation.images.{camera}/chunk-000/file-{index:03d}.mp4" in patterns
    assert not any("file-019.mp4" in name for name in patterns)


def test_unknown_download_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="demo or training"):
        dataset_patterns("all")
