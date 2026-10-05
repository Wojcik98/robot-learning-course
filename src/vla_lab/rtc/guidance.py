"""Apply the RTC v2 action mask and model derivative during sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.policies.smolvla import SmolVLAPolicy

import math
from collections.abc import Callable

import torch
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def prefix_weights(horizon: int, *, overlap: int, delay: int, mode: str) -> torch.Tensor:
    """Make the hard mask or the RTC v2 soft mask for one action chunk."""
    if mode not in ("hard", "soft"):
        raise ValueError(f"Unknown RTC mask mode: {mode}")
    if horizon < 1 or not 0 <= delay <= overlap <= horizon:
        raise ValueError("RTC delay, overlap, and horizon are not ordered")
    weights = torch.zeros(horizon, dtype=torch.float32)
    weights[:delay] = 1.0
    if mode == "soft":
        for index in range(delay, overlap):
            fraction = (overlap - index) / (overlap - delay + 1)
            weights[index] = fraction * math.expm1(fraction) / math.expm1(1.0)
    return weights


class GuidedRTCProcessor(RTCProcessor):
    """Compute the RTC vector-Jacobian correction before each Euler step."""

    def __init__(self, rtc_config: RTCConfig, *, mask_mode: str = "soft") -> None:
        super().__init__(rtc_config)
        if not rtc_config.enabled or rtc_config.mode != "guided":
            raise ValueError("Guided RTC needs an enabled guided configuration")
        if mask_mode not in ("hard", "soft"):
            raise ValueError(f"Unknown RTC mask mode: {mask_mode}")
        self.mask_mode = mask_mode

    def denoise_step(
        self,
        x_t: torch.Tensor,
        prev_chunk_left_over: torch.Tensor | None,
        inference_delay: int | None,
        time: float | torch.Tensor,
        original_denoise_step_partial: Callable[[torch.Tensor], torch.Tensor],
        execution_horizon: int | None = None,
    ) -> torch.Tensor:
        """Guide one denoising step with the previous action chunk."""
        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)
        if inference_delay is None:
            raise ValueError("RTC inference delay is required with a previous chunk")
        squeeze = x_t.ndim == 2
        current = x_t.unsqueeze(0) if squeeze else x_t
        previous = (
            prev_chunk_left_over.unsqueeze(0)
            if prev_chunk_left_over.ndim == 2
            else prev_chunk_left_over
        )
        if current.ndim != 3 or previous.ndim != 3 or previous.shape[0] != current.shape[0]:
            raise ValueError("RTC chunks must have matching batch dimensions")
        if previous.shape[2] > current.shape[2]:
            raise ValueError("Previous RTC action width exceeds the model action width")
        horizon = current.shape[1]
        execution = (
            self.rtc_config.execution_horizon if execution_horizon is None else execution_horizon
        )
        if not 1 <= execution <= horizon:
            raise ValueError("RTC execution horizon is outside the action chunk")
        overlap = min(horizon - execution, previous.shape[1])
        if not 0 <= inference_delay <= min(execution, overlap):
            raise ValueError("RTC delay exceeds the execution or overlap horizon")

        target = torch.zeros_like(current)
        target[:, :overlap, : previous.shape[2]] = previous[:, :overlap].to(current)
        mask = torch.zeros_like(current)
        weights = prefix_weights(
            horizon, overlap=overlap, delay=inference_delay, mode=self.mask_mode
        ).to(current)
        mask[:, :, : previous.shape[2]] = weights[None, :, None]
        clock = torch.as_tensor(time, dtype=current.dtype, device=current.device)
        remainder = 1.0 - clock
        denominator = (clock * remainder).clamp_min(1e-6)
        guidance_weight = ((clock.square() + remainder.square()) / denominator).clamp(
            max=self.rtc_config.max_guidance_weight
        )

        with torch.enable_grad():
            latent = current.detach().clone().requires_grad_(True)
            velocity = original_denoise_step_partial(latent)
            estimate = latent - clock * velocity
            error = (target - estimate).detach()
            correction = torch.autograd.grad(
                estimate, latent, grad_outputs=error * mask, retain_graph=False
            )[0]
            result = velocity - guidance_weight * correction

        self.track(
            time=time,
            x1_t=estimate.detach(),
            correction=correction.detach(),
            err=error,
            weights=mask,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution,
        )
        result = result.detach()
        return result.squeeze(0) if squeeze else result


def install_guided_rtc(
    policy: SmolVLAPolicy, *, execution_horizon: int, mask_mode: str = "soft", beta: float = 5.0
) -> GuidedRTCProcessor:
    """Connect the checked guidance path to a SmolVLA policy and its model."""
    config = RTCConfig(
        enabled=True,
        mode="guided",
        execution_horizon=execution_horizon,
        max_guidance_weight=beta,
    )
    processor = GuidedRTCProcessor(config, mask_mode=mask_mode)
    policy.config.rtc_config = config
    policy.rtc_processor = processor
    policy.model.rtc_processor = processor
    return processor
