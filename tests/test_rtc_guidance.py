"""Check the RTC mask and the model derivative in the guidance term."""

from types import SimpleNamespace

import pytest
import torch
from lerobot.policies.rtc.configuration_rtc import RTCConfig

from vla_lab.rtc.guidance import GuidedRTCProcessor, install_guided_rtc, prefix_weights


def test_soft_and_hard_masks_keep_delay_and_overlap_limits() -> None:
    soft = prefix_weights(6, overlap=4, delay=2, mode="soft")
    hard = prefix_weights(6, overlap=4, delay=2, mode="hard")

    assert soft.tolist()[:2] == [1.0, 1.0]
    assert 0.0 < soft[2] < 1.0
    assert 0.0 < soft[3] < soft[2]
    assert soft.tolist()[4:] == [0.0, 0.0]
    assert hard.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_guidance_uses_model_jacobian_under_no_grad() -> None:
    processor = GuidedRTCProcessor(RTCConfig(execution_horizon=2), mask_mode="soft")
    action = torch.ones((1, 6, 1))
    previous = torch.ones((1, 4, 1))

    with torch.no_grad():
        result = processor.denoise_step(
            x_t=action,
            prev_chunk_left_over=previous,
            inference_delay=2,
            time=0.5,
            original_denoise_step_partial=lambda x: 2 * x,
            execution_horizon=2,
        )

    assert torch.allclose(result, 2 * action)
    assert result.requires_grad is False


def test_guidance_does_not_change_padded_action_dimensions() -> None:
    processor = GuidedRTCProcessor(RTCConfig(execution_horizon=2), mask_mode="hard")
    action = torch.ones((1, 6, 2))
    previous = torch.full((1, 4, 1), 2.0)

    with torch.no_grad():
        result = processor.denoise_step(
            x_t=action,
            prev_chunk_left_over=previous,
            inference_delay=2,
            time=0.5,
            original_denoise_step_partial=torch.zeros_like,
            execution_horizon=2,
        )

    assert result[0, 0, 0].item() == pytest.approx(-2.0)
    assert result[0, 0, 1].item() == pytest.approx(0.0)
    assert torch.all(result[:, 2:] == 0)


def test_install_guidance_links_policy_and_action_model() -> None:
    policy = SimpleNamespace(config=SimpleNamespace(rtc_config=None), model=SimpleNamespace())
    processor = install_guided_rtc(policy, execution_horizon=10, mask_mode="soft")

    assert policy.config.rtc_config.enabled is True
    assert policy.rtc_processor is processor
    assert policy.model.rtc_processor is processor
