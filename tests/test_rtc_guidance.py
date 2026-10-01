"""Check the RTC mask and the model derivative in the guidance term."""

from types import SimpleNamespace

import pytest
import torch
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

from vla_lab.rtc.guidance import GuidedRTCProcessor, install_guided_rtc, prefix_weights


def test_soft_and_hard_masks_keep_delay_and_overlap_limits() -> None:
    soft = prefix_weights(6, overlap=4, delay=2, mode="soft")
    hard = prefix_weights(6, overlap=4, delay=2, mode="hard")

    assert soft.tolist()[:2] == [1.0, 1.0]
    assert 0.0 < soft[2] < 1.0
    assert 0.0 < soft[3] < soft[2]
    assert soft.tolist()[4:] == [0.0, 0.0]
    assert hard.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("delay,overlap", [(0, 4), (2, 4), (4, 4)])
def test_soft_mask_matches_pinned_upstream_schedule(delay, overlap) -> None:
    upstream = RTCProcessor(
        RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.EXP)
    ).get_prefix_weights(delay, overlap, 6)

    assert torch.allclose(prefix_weights(6, overlap=overlap, delay=delay, mode="soft"), upstream)


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


def test_installed_guidance_uses_declared_beta_five() -> None:
    policy = SimpleNamespace(config=SimpleNamespace(rtc_config=None), model=SimpleNamespace())

    processor = install_guided_rtc(policy, execution_horizon=10, mask_mode="hard", beta=5.0)

    assert processor.rtc_config.max_guidance_weight == 5.0


def test_first_guided_step_is_finite_when_weight_needs_clipping() -> None:
    processor = GuidedRTCProcessor(
        RTCConfig(execution_horizon=2, max_guidance_weight=5.0), mask_mode="soft"
    )

    with torch.no_grad():
        result = processor.denoise_step(
            x_t=torch.full((1, 6, 1), 0.25),
            prev_chunk_left_over=torch.ones((1, 4, 1)),
            inference_delay=2,
            time=1.0,
            original_denoise_step_partial=lambda x: 0.5 * x,
            execution_horizon=2,
        )

    assert torch.isfinite(result).all()
