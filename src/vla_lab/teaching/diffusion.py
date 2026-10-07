"""Diffusion Policy operations written step by step (course units 9, 10A, 10B and 11).

Conventions (the same as the pinned LeRobot diffusion policy and ``diffusers`` DDPM):

* ``alpha_bar[t]`` is the cumulative product of ``1 - beta`` up to and including step ``t``;
  timestep indices run ``0 .. T - 1``. A timestep tensor is ``(B,)`` and ``long``.
* Action tensors are ``(B, H, A)``. ``is_pad`` is ``(B, H)`` and ``True`` for padding.
* The model predicts the noise ``epsilon`` (``prediction_type="epsilon"``).
* Padded actions are copies of the last real action. They are excluded from the loss.

The functions named in the unit tables are the reference answers. Learner versions with the
same signature can be passed through the hooks (``ConditionalUnet1D(forward_fn=...)``,
``CallbackDiffusionPolicy(sampler=...)``, ...), so the real path calls the learner code.
"""

import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.processor import load_pretrained_policy_processors
from torch import Tensor, nn

from vla_lab.teaching.blocks import FilmForward, FilmParts, UNetForward, UNetParts
from vla_lab.teaching.interfaces import DiffusionTrainingModel, NoiseScheduler

if TYPE_CHECKING:
    from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
    from lerobot.policies.diffusion.modeling_diffusion import (
        DiffusionConditionalResidualBlock1d,
        DiffusionConditionalUnet1d,
    )
    from lerobot.processor import PolicyProcessorPipeline


# --- Forward process and training objective ---------------------------------------------


def cosine_betas(num_steps: int, max_beta: float = 0.999) -> Tensor:
    """The ``squaredcos_cap_v2`` schedule used by the pinned policy (Nichol and Dhariwal 2021).

    ``beta_i = min(1 - abar(t2) / abar(t1), max_beta)`` with ``abar(t) = cos((t + 0.008) / 1.008 * pi / 2) ** 2``,
    ``t1 = i / T`` and ``t2 = (i + 1) / T``. Supplied: it is a schedule choice, not the lesson.
    """

    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [
        min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
        for i in range(num_steps)
    ]
    return torch.tensor(betas, dtype=torch.float32)


def cumulative_alphas(betas: Tensor) -> Tensor:
    """``alpha_bar[t] = prod_{s <= t} (1 - beta[s])`` (DDPM paper, equation 4). Shape ``(T,)``."""
    return torch.cumprod(1.0 - betas, dim=0)


def forward_noise(x0: Tensor, noise: Tensor, timesteps: Tensor, alpha_bar: Tensor) -> Tensor:
    """Noise a clean sample in one step (DDPM equation 4).

    ``x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise``, with ``t`` chosen per
    batch row. ``x0`` and ``noise`` are ``(B, H, A)``, ``timesteps`` is ``(B,)``.
    """
    abar = alpha_bar.to(device=x0.device, dtype=x0.dtype)[timesteps].reshape(
        -1, *([1] * (x0.ndim - 1))
    )
    return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise


def sample_noise_and_timesteps(
    actions: Tensor, num_train_timesteps: int, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Draw ``noise ~ N(0, I)`` shaped like ``actions``, then one uniform timestep per row.

    Timesteps are integers in ``[0, T)`` (``T`` itself is excluded), dtype ``long``, on the
    device of ``actions``. Draw the noise first, then the timesteps, from the same generator.
    """
    noise = torch.randn(
        actions.shape, generator=generator, device=actions.device, dtype=actions.dtype
    )
    timesteps = torch.randint(
        0, num_train_timesteps, (actions.shape[0],), generator=generator, device=actions.device
    )
    return noise, timesteps


def masked_epsilon_loss(prediction: Tensor, target: Tensor, is_pad: Tensor) -> Tensor:
    """Mean squared error over valid ``(B, H, A)`` elements.

    ``sum((prediction - target) ** 2 * valid) / max(number_of_valid_steps * A, 1)`` with
    ``valid = ~is_pad`` broadcast over ``A``. Padded steps have zero gradient. A batch that is
    entirely padding returns zero.
    """
    valid = (~is_pad.to(torch.bool)).unsqueeze(-1).to(prediction.dtype)
    denominator = (valid.sum() * prediction.shape[-1]).clamp_min(1)
    return ((prediction - target).square() * valid).sum() / denominator


def epsilon_loss_fn(
    model: DiffusionTrainingModel,
    batch: Mapping[str, Tensor],
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Training objective callback: ``(loss, metrics)`` with the graph kept for ``backward``.

    ``model(noisy_actions, timesteps, batch)`` predicts the noise and ``model.schedule`` holds
    ``alpha_bar``. Use ``noise`` and ``time`` (timesteps) when given; draw whichever is missing.
    """
    actions = batch["action"]
    if noise is None:
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    if time is None:
        time = torch.randint(
            0,
            model.schedule.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
        )
    noisy = forward_noise(actions, noise, time, model.schedule.alpha_bar)
    loss = masked_epsilon_loss(model(noisy, time, batch), noise, batch["action_is_pad"])
    return loss, {"epsilon_mse": loss}


@dataclass(frozen=True)
class DDPMSchedule:
    """Beta, alpha and cumulative-alpha tables for ``T`` training steps."""

    betas: Tensor
    alpha_bar: Tensor

    @property
    def num_train_timesteps(self) -> int:
        return int(self.betas.shape[0])

    @classmethod
    def from_betas(
        cls, betas: Tensor, cumulative_fn: Callable[[Tensor], Tensor] = cumulative_alphas
    ) -> "DDPMSchedule":
        return cls(betas=betas, alpha_bar=cumulative_fn(betas))

    @classmethod
    def cosine(cls, num_steps: int = 100, **kwargs: Callable[[Tensor], Tensor]) -> "DDPMSchedule":
        return cls.from_betas(cosine_betas(num_steps), **kwargs)

    @classmethod
    def linear(
        cls,
        num_steps: int = 100,
        start: float = 1e-4,
        end: float = 0.02,
        **kwargs: Callable[[Tensor], Tensor],
    ) -> "DDPMSchedule":
        return cls.from_betas(torch.linspace(start, end, num_steps, dtype=torch.float32), **kwargs)

    @classmethod
    def from_scheduler(cls, scheduler: NoiseScheduler) -> "DDPMSchedule":
        """Read the tables of a ``diffusers`` ``DDPMScheduler`` (for comparisons)."""
        return cls(betas=scheduler.betas.clone(), alpha_bar=scheduler.alphas_cumprod.clone())


class DiffusionDenoiser(nn.Module):
    """A noise-prediction network plus its schedule and a condition function.

    ``condition_fn(batch) -> (B, cond_dim) or None`` turns observations into the global
    condition. ``forward(noisy_actions, timesteps, batch)`` returns ``epsilon`` ``(B, H, A)``.
    """

    def __init__(
        self,
        unet: nn.Module,
        schedule: DDPMSchedule,
        condition_fn: Callable[[Mapping[str, Tensor]], Tensor | None],
        extra: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.schedule = schedule
        self.condition_fn = condition_fn
        self.extra = extra  # holds trainable modules used by condition_fn, if any

    def forward(
        self, noisy_actions: Tensor, timesteps: Tensor, batch: Mapping[str, Tensor]
    ) -> Tensor:
        return self.unet(noisy_actions, timesteps, global_cond=self.condition_fn(batch))


def flat_state_condition(batch: Mapping[str, Tensor]) -> Tensor:
    """Condition on the observation states only: ``(B, n_obs, S)`` flattened to ``(B, n_obs * S)``."""
    return batch["observation.state"].flatten(1)


# --- Timestep embedding and FiLM residual block ---------------------------------------


def sinusoidal_timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Embed integer timesteps as sin/cos features, shape ``(B, dim)`` (Vaswani 3.5, DDPM B).

    ``half = dim // 2``; frequency ``f_k = exp(-k * log(10000) / (half - 1))`` for ``k = 0 .. half - 1``;
    output ``[sin(t * f), cos(t * f)]`` (all sines first, then all cosines).
    """
    half = dim // 2
    scale = math.log(10000) / (half - 1)
    frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(
        self, dim: int, fn: Callable[[Tensor, int], Tensor] = sinusoidal_timestep_embedding
    ) -> None:
        super().__init__()
        self.dim = dim
        self.fn = fn

    def forward(self, timesteps: Tensor) -> Tensor:
        return self.fn(timesteps, self.dim)


class Conv1dBlock(nn.Module):
    """Supplied: ``Conv1d -> GroupNorm -> Mish`` with "same" padding for odd kernels."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


def film_residual_forward(block: FilmParts, x: Tensor, cond: Tensor) -> Tensor:
    """FiLM-conditioned residual block (Perez et al. 2018, used in Diffusion Policy).

    ``x (B, C_in, T)``, ``cond (B, cond_dim)``, returns ``(B, C_out, T)``. Steps:
    ``out = conv1(x)``; ``e = cond_encoder(cond)`` reshaped to ``(B, K, 1)``. With scale
    modulation ``K = 2 * C_out``: ``scale = e[:, :C_out]``, ``bias = e[:, C_out:]`` and
    ``out = scale * out + bias``. Without it ``K = C_out`` and ``out = out + e``.
    Then ``out = conv2(out)`` and the result is ``out + residual_conv(x)``.
    Block attributes: ``conv1``, ``cond_encoder``, ``conv2``, ``residual_conv``,
    ``out_channels``, ``use_film_scale_modulation``.
    """
    out = block.conv1(x)
    embed = block.cond_encoder(cond).unsqueeze(-1)
    if block.use_film_scale_modulation:
        scale, bias = embed[:, : block.out_channels], embed[:, block.out_channels :]
        out = scale * out + bias
    else:
        out = out + embed
    out = block.conv2(out)
    return out + block.residual_conv(x)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
        forward_fn: FilmForward = film_residual_forward,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.use_film_scale_modulation = use_film_scale_modulation
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, out_channels * (2 if use_film_scale_modulation else 1))
        )
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.forward_fn = forward_fn

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.forward_fn(self, x, cond)


# --- Temporal U-Net -------------------------------------------------------------------


class UNetChannelPlan(TypedDict):
    down: list[tuple[int, int, bool]]
    mid: list[tuple[int, int]]
    up: list[tuple[int, int, bool]]


def unet_channel_plan(action_dim: int, down_dims: Sequence[int]) -> UNetChannelPlan:
    """Channel plan of the pinned conditional 1D U-Net.

    Returns ``{"down": [...], "mid": [...], "up": [...]}``. Each ``down`` entry is
    ``(in_channels, out_channels, downsample)``: two residual blocks ``in -> out`` and
    ``out -> out``, then a stride-2 convolution unless it is the last stage. ``mid`` is
    ``[(c, c), (c, c)]`` with ``c = down_dims[-1]``. Each ``up`` entry is
    ``(in_channels, out_channels, upsample)`` for the stages in reverse order, skipping the
    first ``(action_dim, down_dims[0])`` pair. Its first block takes ``2 * previous_width``
    channels because the matching encoder feature is concatenated. The pinned network
    upsamples after every ``up`` stage (the last included).
    """
    # Keep this function standalone when copied into notebook solution cells.
    pairs = [(action_dim, down_dims[0]), *zip(down_dims[:-1], down_dims[1:], strict=True)]  # noqa: RUF007
    down = [(i, o, index < len(pairs) - 1) for index, (i, o) in enumerate(pairs)]
    mid = [(down_dims[-1], down_dims[-1])] * 2
    up = [(width_in * 2, width_out, True) for width_out, width_in in reversed(pairs[1:])]
    return {"down": down, "mid": mid, "up": up}


def unet_forward(
    unet: UNetParts, x: Tensor, timestep: Tensor, global_cond: Tensor | None = None
) -> Tensor:
    """Conditional U-Net forward pass with skip connections.

    ``x (B, H, A)`` (time before channels), ``timestep (B,)``, optional ``global_cond (B, G)``;
    returns ``(B, H, A)``. Steps: move to ``(B, A, H)``; ``cond = time_mlp(timestep)``, with
    ``global_cond`` concatenated on the last axis; for each ``down_blocks`` entry
    ``(res1, res2, downsample)`` apply ``res1(x, cond)``, ``res2(x, cond)``, store the result
    as a skip feature, then ``downsample``; run ``mid_blocks``; for each ``up_blocks`` entry
    ``(res1, res2, upsample)`` concatenate the newest stored skip on the channel axis, apply
    the two blocks, then ``upsample``; finish with ``head`` and return to ``(B, H, A)``.
    """
    x = x.transpose(1, 2)
    cond = unet.time_mlp(timestep)
    if global_cond is not None:
        cond = torch.cat([cond, global_cond], dim=-1)
    skips = []
    for stage in unet.down_blocks:
        res1, res2, downsample = cast(nn.ModuleList, stage)
        x = res2(res1(x, cond), cond)
        skips.append(x)
        x = downsample(x)
    for block in unet.mid_blocks:
        x = block(x, cond)
    for stage in unet.up_blocks:
        res1, res2, upsample = cast(nn.ModuleList, stage)
        x = torch.cat((x, skips.pop()), dim=1)
        x = res2(res1(x, cond), cond)
        x = upsample(x)
    return unet.head(x).transpose(1, 2)


class ConditionalUnet1D(nn.Module):
    """1D temporal U-Net with FiLM conditioning on the timestep and a global condition."""

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        down_dims: Sequence[int] = (128, 256, 512),
        *,
        kernel_size: int = 5,
        n_groups: int = 8,
        step_embed_dim: int = 128,
        use_film_scale_modulation: bool = True,
        plan_fn: Callable[[int, Sequence[int]], UNetChannelPlan] = unet_channel_plan,
        forward_fn: UNetForward = unet_forward,
        block_forward_fn: FilmForward = film_residual_forward,
        embedding_fn: Callable[[Tensor, int], Tensor] = sinusoidal_timestep_embedding,
    ) -> None:
        super().__init__()
        self.action_dim, self.down_dims = action_dim, tuple(down_dims)
        self.forward_fn = forward_fn
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(step_embed_dim, embedding_fn),
            nn.Linear(step_embed_dim, step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(step_embed_dim * 4, step_embed_dim),
        )
        cond_dim = step_embed_dim + global_cond_dim

        def block(i: int, o: int) -> ConditionalResidualBlock1d:
            return ConditionalResidualBlock1d(
                i, o, cond_dim, kernel_size, n_groups, use_film_scale_modulation, block_forward_fn
            )

        plan = plan_fn(action_dim, self.down_dims)
        self.down_blocks = nn.ModuleList(
            nn.ModuleList(
                [block(i, o), block(o, o), nn.Conv1d(o, o, 3, 2, 1) if down else nn.Identity()]
            )
            for i, o, down in plan["down"]
        )
        self.mid_blocks = nn.ModuleList(block(i, o) for i, o in plan["mid"])
        self.up_blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    block(i, o),
                    block(o, o),
                    nn.ConvTranspose1d(o, o, 4, 2, 1) if up else nn.Identity(),
                ]
            )
            for i, o, up in plan["up"]
        )
        self.head = nn.Sequential(
            Conv1dBlock(self.down_dims[0], self.down_dims[0], kernel_size, n_groups),
            nn.Conv1d(self.down_dims[0], action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor, global_cond: Tensor | None = None) -> Tensor:
        return self.forward_fn(self, x, timestep, global_cond)


def convert_lerobot_unet_state_dict(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename the tensors of the pinned ``DiffusionConditionalUnet1d`` for ``ConditionalUnet1D``.

    Only the first name part differs: ``diffusion_step_encoder`` -> ``time_mlp``,
    ``down_modules`` -> ``down_blocks``, ``mid_modules`` -> ``mid_blocks``,
    ``up_modules`` -> ``up_blocks``, ``final_conv`` -> ``head``. Everything after it is unchanged.
    An unknown first name raises ``KeyError`` so a changed network cannot slip through. Returns a
    new dict; every source tensor appears exactly once (copied).
    """
    renames = {
        "diffusion_step_encoder": "time_mlp",
        "down_modules": "down_blocks",
        "mid_modules": "mid_blocks",
        "up_modules": "up_blocks",
        "final_conv": "head",
    }
    converted: dict[str, Tensor] = {}
    for name, value in source.items():
        top, _, rest = name.partition(".")
        if top not in renames:
            raise KeyError(f"no conversion rule for LeRobot U-Net tensor {name!r}")
        converted[f"{renames[top]}.{rest}"] = value.detach().clone()
    return converted


class UNetHooks(TypedDict, total=False):
    plan_fn: Callable[[int, Sequence[int]], UNetChannelPlan]
    forward_fn: UNetForward
    block_forward_fn: FilmForward
    embedding_fn: Callable[[Tensor, int], Tensor]


def unet_from_lerobot(
    lerobot_unet: "DiffusionConditionalUnet1d",
    *,
    converter: Callable[
        [Mapping[str, Tensor]], dict[str, Tensor]
    ] = convert_lerobot_unet_state_dict,
    **hooks: Unpack[UNetHooks],
) -> ConditionalUnet1D:
    """Build a learner U-Net with the LeRobot sizes and copy its weights (``strict=True``).

    ``converter`` turns the LeRobot state dict into this module's names (default: the package one).
    ``hooks`` are passed to ``ConditionalUnet1D`` (``plan_fn``, ``forward_fn``, ``block_forward_fn``, ...).
    """
    config = lerobot_unet.config
    first_stage = cast(nn.ModuleList, lerobot_unet.down_modules[0])
    first_block = cast("DiffusionConditionalResidualBlock1d", first_stage[0])
    conditioning = cast(nn.Linear, first_block.cond_encoder[1])
    feature = config.action_feature
    if feature is None:
        raise ValueError("Diffusion Policy requires an action feature")
    global_cond_dim = conditioning.in_features - config.diffusion_step_embed_dim
    learner = ConditionalUnet1D(
        action_dim=feature.shape[0],
        global_cond_dim=global_cond_dim,
        down_dims=config.down_dims,
        kernel_size=config.kernel_size,
        n_groups=config.n_groups,
        step_embed_dim=config.diffusion_step_embed_dim,
        use_film_scale_modulation=config.use_film_scale_modulation,
        **hooks,
    )
    learner.load_state_dict(converter(lerobot_unet.state_dict()), strict=True)
    return learner


@torch.no_grad()
def verify_unet_parity(
    learner: ConditionalUnet1D,
    lerobot_unet: "DiffusionConditionalUnet1d",
    x: Tensor,
    timestep: Tensor,
    global_cond: Tensor | None,
) -> float:
    """Largest absolute output difference in evaluation mode. Run this before any experiment."""
    learner.eval()
    lerobot_unet.eval()
    return float(
        (learner(x, timestep, global_cond) - lerobot_unet(x, timestep, global_cond)).abs().max()
    )


# --- Reverse process and sampling ------------------------------------------------------


def inference_timesteps(num_train_timesteps: int, num_inference_steps: int) -> list[int]:
    """Evenly spaced timesteps in descending order, with spacing ``T // n`` ("leading").

    ``[(n - 1) * r, ..., r, 0]`` with ``r = T // n``. The last timestep is always ``0``.
    """
    if not 1 <= num_inference_steps <= num_train_timesteps:
        raise ValueError("num_inference_steps must be between 1 and the training step count")
    ratio = num_train_timesteps // num_inference_steps
    return [index * ratio for index in reversed(range(num_inference_steps))]


def posterior_coefficients(
    abar_t: Tensor | float, abar_prev: Tensor | float
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean coefficients and variance of ``q(x_prev | x_t, x_0)`` (DDPM equations 6 and 7).

    For a jump from ``t`` to any earlier step ``prev`` use the ratio
    ``alpha_t = abar_t / abar_prev`` and ``beta_t = 1 - alpha_t``. (When ``prev = t - 1`` this is
    the ordinary ``1 - beta_t``; for skipped steps it is the product of the skipped alphas.)
    Returns ``(coef_x0, coef_xt, variance)``::

        coef_x0 = sqrt(abar_prev) * beta_t / (1 - abar_t)
        coef_xt = sqrt(alpha_t) * (1 - abar_prev) / (1 - abar_t)
        variance = (1 - abar_prev) / (1 - abar_t) * beta_t     (clamped to at least 1e-20)
    """
    abar_t = torch.as_tensor(abar_t, dtype=torch.float32)
    abar_prev = torch.as_tensor(abar_prev, dtype=torch.float32)
    alpha_t = abar_t / abar_prev
    beta_t = 1.0 - alpha_t
    coef_x0 = abar_prev.sqrt() * beta_t / (1.0 - abar_t)
    coef_xt = alpha_t.sqrt() * (1.0 - abar_prev) / (1.0 - abar_t)
    variance = ((1.0 - abar_prev) / (1.0 - abar_t) * beta_t).clamp_min(1e-20)
    return coef_x0, coef_xt, variance


def ddpm_reverse_step(
    x_t: Tensor,
    epsilon: Tensor,
    t: int,
    prev_t: int,
    alpha_bar: Tensor,
    noise: Tensor | None = None,
    clip_range: float | None = 1.0,
) -> Tensor:
    """One reverse step ``x_t -> x_prev`` of ancestral sampling (DDPM Algorithm 2, generalized).

    1. ``x0_hat = (x_t - sqrt(1 - abar_t) * epsilon) / sqrt(abar_t)``; clamp it to
       ``[-clip_range, clip_range]`` when ``clip_range`` is not ``None``.
    2. ``abar_prev = alpha_bar[prev_t]``, or exactly ``1`` when ``prev_t < 0`` (the final step).
    3. ``mean = coef_x0 * x0_hat + coef_xt * x_t`` from ``posterior_coefficients``.
    4. If ``prev_t >= 0`` return ``mean + sqrt(variance) * noise``. At the final step
       (``prev_t < 0``) return ``mean`` with no noise: that equals the clipped ``x0_hat``.
    ``noise`` is required unless this is the final step. ``t`` and ``prev_t`` are Python ints.
    """
    abar_t = alpha_bar[t].to(x_t.dtype)
    abar_prev = alpha_bar[prev_t].to(x_t.dtype) if prev_t >= 0 else torch.ones((), dtype=x_t.dtype)
    x0_hat = (x_t - (1.0 - abar_t).sqrt() * epsilon) / abar_t.sqrt()
    if clip_range is not None:
        x0_hat = x0_hat.clamp(-clip_range, clip_range)
    coef_x0, coef_xt, variance = posterior_coefficients(abar_t, abar_prev)
    mean = coef_x0.to(x_t.dtype) * x0_hat + coef_xt.to(x_t.dtype) * x_t
    if prev_t < 0:
        return mean
    if noise is None:
        raise ValueError("noise is required for every step except the final one")
    return mean + variance.sqrt().to(x_t.dtype) * noise


def make_step_noises(
    num_inference_steps: int,
    shape: Sequence[int],
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """One noise tensor for each non-final reverse step: ``num_inference_steps - 1`` of them."""
    return [
        torch.randn(tuple(shape), generator=generator, device=device, dtype=dtype)
        for _ in range(num_inference_steps - 1)
    ]


def sample_actions(
    denoise: Callable[[Tensor, Tensor], Tensor],
    initial_noise: Tensor,
    schedule: DDPMSchedule,
    num_inference_steps: int,
    step_noises: Sequence[Tensor],
    clip_range: float | None = 1.0,
) -> Tensor:
    """Run the full reverse process. Returns the final sample ``(B, H, A)``.

    ``denoise(x_t, t_batch)`` returns the predicted noise; ``t_batch`` is a ``long`` ``(B,)``
    tensor filled with the current timestep. Start from ``initial_noise``. For each timestep
    of ``inference_timesteps`` carry the result of one step into the next. The previous
    timestep of the last entry is ``-1``. ``step_noises[i]`` is used by step ``i``; the final
    step uses none. All randomness is passed in, so the same arguments give the same actions.
    """
    timesteps = inference_timesteps(schedule.num_train_timesteps, num_inference_steps)
    if len(step_noises) != len(timesteps) - 1:
        raise ValueError("need one noise tensor for every step except the last")
    x = initial_noise
    for index, t in enumerate(timesteps):
        prev_t = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        t_batch = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
        epsilon = denoise(x, t_batch)
        noise = step_noises[index] if prev_t >= 0 else None
        x = ddpm_reverse_step(x, epsilon, t, prev_t, schedule.alpha_bar, noise, clip_range)
    return x


def executed_action_slice(chunk: Tensor, n_obs_steps: int, n_action_steps: int) -> Tensor:
    """Pick the actions that are executed from a sampled window ``(B, H, A)``.

    The window starts ``n_obs_steps - 1`` steps before the current frame, so the current frame's
    action is at index ``n_obs_steps - 1``. Return indices
    ``n_obs_steps - 1 .. n_obs_steps - 1 + n_action_steps - 1`` (end exclusive at
    ``n_obs_steps - 1 + n_action_steps``). For ``n_obs_steps=2, n_action_steps=8`` that is 1 to 8.
    """
    start = n_obs_steps - 1
    end = start + n_action_steps
    if n_obs_steps < 1 or n_action_steps < 1 or end > chunk.shape[1]:
        raise ValueError("the executed slice must fit inside the sampled window")
    return chunk[:, start:end]


# --- Real data: state and action windows, no images ---------------------------------------------


def _episode_arrays(
    data_root: Path, episode_ids: Sequence[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read ``(state, action)`` arrays per episode from the local parquet files. Never downloads."""

    root = Path(data_root) / "hub" / "lerobot-libero"
    if not (root / "meta" / "episodes").is_dir():
        raise FileNotFoundError(
            f"Local LIBERO data not found under {root}. Install it once with `just data-training` "
            "(that command downloads data), then run this cell again."
        )
    wanted = pa.array(sorted(int(e) for e in episode_ids))
    index = pq.read_table(next((root / "meta" / "episodes").glob("*/*.parquet")))
    index = index.filter(pc.is_in(index["episode_index"], value_set=wanted)).to_pandas()
    found = {int(e) for e in index["episode_index"]}
    if found != {int(e) for e in episode_ids}:
        raise FileNotFoundError(
            f"episodes missing from the local metadata: {sorted(set(map(int, episode_ids)) - found)}"
        )
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for (chunk, file), rows in index.groupby(["data/chunk_index", "data/file_index"]):
        table = pq.read_table(root / f"data/chunk-{int(chunk):03d}/file-{int(file):03d}.parquet")
        episode = table["episode_index"].to_numpy()
        frame = table["frame_index"].to_numpy()
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        for episode_id in rows["episode_index"]:
            pick = np.flatnonzero(episode == int(episode_id))
            pick = pick[np.argsort(frame[pick])]
            result[int(episode_id)] = (state[pick], action[pick])
    return result


def load_state_action_windows(
    configs_root: str | Path,
    data_root: str | Path,
    task_alias: str,
    split: str,
    *,
    horizon: int = 16,
    n_obs_steps: int = 2,
    drop_n_last_frames: int | None = None,
    action_start: int | None = None,
) -> dict[str, Tensor]:
    """Real LIBERO state/action windows with the pinned diffusion alignment (no cameras).

    ``split`` is ``train`` or ``validation`` (whole episodes from ``configs/data/episodes.yaml``).
    For a current frame ``t`` the window holds states ``t - n_obs + 1 .. t`` and actions
    ``t - n_obs + 1 .. t - n_obs + horizon`` (index ``n_obs - 1`` is the action for frame ``t``).
    Indices outside the episode repeat the nearest frame and set ``action_is_pad``. As in the
    pinned recipe the last ``horizon - n_action_steps - n_obs_steps + 1`` frames (default 7 for
    16/8/2) are not used as current frames. Values are min-max scaled to ``[-1, 1]`` with the
    training extrema from ``configs/data/stats``. Source timestamps use a 10 FPS storage label; nothing is resampled
    to the 20 Hz simulator controller.

    ``action_start`` moves the first action of the window relative to the current frame. The
    default ``1 - n_obs_steps`` is the diffusion alignment above; ``0`` starts at the current frame.

    Returns ``observation.state (N, n_obs, 8)``, ``state_is_pad (N, n_obs)``,
    ``action (N, horizon, 7)``, ``action_is_pad (N, horizon)``, ``episode (N,)`` and ``frame (N,)``.
    """
    configs = Path(configs_root)
    manifest = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    ids = manifest["tasks"][task_alias][f"{split}_episode_ids"]
    stats = json.loads((configs / "data" / "stats" / f"{task_alias}.json").read_text())[
        "modalities"
    ]
    drop = 7 if drop_n_last_frames is None else drop_n_last_frames
    scale = {}
    for key in ("observation.state", "action"):
        low = np.asarray(stats[key]["min"], dtype=np.float32)
        high = np.asarray(stats[key]["max"], dtype=np.float32)
        scale[key] = (low, np.where(high > low, high - low, 1.0).astype(np.float32))
    first = 1 - n_obs_steps if action_start is None else action_start
    states, state_pads, actions, pads, episodes, frames = [], [], [], [], [], []
    for episode_id, (state, action) in _episode_arrays(Path(data_root), ids).items():
        length = len(action)
        state = (state - scale["observation.state"][0]) / scale["observation.state"][1] * 2 - 1
        action = (action - scale["action"][0]) / scale["action"][1] * 2 - 1
        for t in range(max(length - drop, 0)):
            action_index = np.arange(t + first, t + first + horizon)
            state_index = np.arange(t - n_obs_steps + 1, t + 1)
            states.append(state[np.clip(state_index, 0, length - 1)])
            state_pads.append(state_index < 0)
            actions.append(action[np.clip(action_index, 0, length - 1)])
            pads.append((action_index < 0) | (action_index >= length))
            episodes.append(episode_id)
            frames.append(t)
    return {
        "observation.state": torch.from_numpy(np.stack(states)),
        "state_is_pad": torch.from_numpy(np.stack(state_pads)),
        "action": torch.from_numpy(np.stack(actions)),
        "action_is_pad": torch.from_numpy(np.stack(pads)),
        "episode": torch.tensor(episodes),
        "frame": torch.tensor(frames),
    }


def take_batch(windows: Mapping[str, Tensor], indices: Sequence[int] | Tensor) -> dict[str, Tensor]:
    index = torch.as_tensor(
        list(indices) if not isinstance(indices, Tensor) else indices, dtype=torch.long
    )
    return {key: value[index] for key, value in windows.items()}


# --- Real policy integration -------------------------------------------------------------------


def load_prepared_diffusion(
    checkpoint: str | Path, *, device: str = "cpu"
) -> "tuple[DiffusionPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]":
    """Load a prepared diffusion bundle (policy and processors) on ``device``. Never downloads.

    The weights are checked against ``manifest.json`` (SHA-256 of ``model.safetensors``). The pinned
    ImageNet backbone file is not needed because the saved weights replace it. Returns
    ``(policy, preprocessor, postprocessor)``; the policy is in evaluation mode.
    """

    root = Path(checkpoint)
    manifest = json.loads((root / "manifest.json").read_text())
    digest = hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
    if digest != manifest["model_sha256"]:
        raise ValueError("model.safetensors differs from the manifest hash; do not use this bundle")
    config = DiffusionConfig.from_pretrained(root, local_files_only=True)
    config.device = device
    config.pretrained_backbone_weights = None
    policy = DiffusionPolicy.from_pretrained(
        root, config=config, strict=True, local_files_only=True
    ).eval()
    overrides = {"device_processor": {"device": device}}
    preprocessor, postprocessor = load_pretrained_policy_processors(
        root, preprocessor_overrides=overrides
    )
    return policy, preprocessor, postprocessor


class LeRobotDenoiser(nn.Module):
    """The pinned ``DiffusionPolicy`` as a noise predictor for ``epsilon_loss_fn``.

    ``forward(noisy_actions, timesteps, batch)`` encodes the observations with the policy's own
    image encoders and runs its U-Net (or a learner U-Net passed as ``unet``). The batch is a
    preprocessed training batch: states ``(B, n_obs, S)`` and one ``(B, n_obs, C, H, W)`` image
    per camera key. All policy parameters are visible to the optimizer through ``self.policy``.
    """

    def __init__(self, policy: "DiffusionPolicy", unet: nn.Module | None = None) -> None:
        super().__init__()
        self.policy = policy
        self.unet = unet if unet is not None else policy.diffusion.unet
        self.schedule = DDPMSchedule.from_scheduler(policy.diffusion.noise_scheduler)

    def global_condition(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.policy.diffusion._prepare_global_conditioning(
            stack_observation(self.policy, batch)
        )

    def forward(
        self, noisy_actions: Tensor, timesteps: Tensor, batch: Mapping[str, Tensor]
    ) -> Tensor:
        return self.unet(noisy_actions, timesteps, global_cond=self.global_condition(batch))


def stack_observation(policy: "DiffusionPolicy", batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Add ``observation.images`` ``(B, n_obs, n_cam, C, H, W)`` like ``DiffusionPolicy.forward``."""
    keys = list(policy.config.image_features)
    stacked = dict(batch)
    for key in keys:
        if policy.config.n_obs_steps == 1 and stacked[key].ndim == 4:
            stacked[key] = stacked[key].unsqueeze(1)
    if keys:
        stacked["observation.images"] = torch.stack([stacked[key] for key in keys], dim=-4)
    return stacked


class DiffusionSampler(Protocol):
    def __call__(
        self,
        denoise: Callable[[Tensor, Tensor], Tensor],
        initial_noise: Tensor,
        schedule: DDPMSchedule,
        num_inference_steps: int,
        step_noises: Sequence[Tensor],
        clip_range: float | None = 1.0,
    ) -> Tensor: ...


class CallbackDiffusionPolicy:
    """Run the pinned policy's observation encoders and U-Net with a *learner* sampler.

    It offers ``reset()`` and ``select_action(batch)`` so the existing simulator loop can use it.
    A new chunk is requested when its queue is empty. A chunk is made by
    ``sampler(denoise, initial_noise, schedule, num_inference_steps, step_noises, clip_range)``
    (``sample_actions`` has this signature) and cut by ``slicer`` (``executed_action_slice``).
    Noise is drawn from a generator that is re-seeded from ``seed`` and the chunk counter, so a
    run is repeatable and two runs with different step budgets can share the same initial noise.
    The pinned ``select_action`` and its ``noise_scheduler.step`` are not called.
    """

    def __init__(
        self,
        policy: "DiffusionPolicy",
        sampler: DiffusionSampler = sample_actions,
        slicer: Callable[[Tensor, int, int], Tensor] = executed_action_slice,
        *,
        num_inference_steps: int | None = None,
        seed: int = 0,
        unet: nn.Module | None = None,
    ) -> None:
        self.policy = policy
        self.sampler, self.slicer = sampler, slicer
        config = cast("DiffusionConfig", policy.config)
        self.num_inference_steps = (
            num_inference_steps or config.num_inference_steps or config.num_train_timesteps
        )
        self.seed = seed
        self.unet = unet if unet is not None else policy.diffusion.unet
        self.schedule = DDPMSchedule.from_scheduler(policy.diffusion.noise_scheduler)
        self.clip_range = config.clip_sample_range if config.clip_sample else None
        self.chunks_made = 0
        self.sample_seconds: list[float] = []
        self.reset()

    def reset(self) -> None:
        """Clear the observation history, the action queue and the chunk counter."""
        config = cast("DiffusionConfig", self.policy.config)
        self._observations: dict[str, deque] = {
            "observation.state": deque(maxlen=config.n_obs_steps),
            "observation.images": deque(maxlen=config.n_obs_steps),
        }
        self._actions: deque = deque(maxlen=config.n_action_steps)
        self.chunks_made = 0

    def _push(self, batch: Mapping[str, Tensor]) -> None:
        keys = list(self.policy.config.image_features)
        images = torch.stack([batch[key] for key in keys], dim=-4)
        for name, value in (
            ("observation.state", batch["observation.state"]),
            ("observation.images", images),
        ):
            queue = self._observations[name]
            queue.append(value)
            while (
                len(queue) < self.policy.config.n_obs_steps
            ):  # first step: repeat the first observation
                queue.append(value)

    @torch.no_grad()
    def predict_window(
        self,
        observation: Mapping[str, Tensor],
        initial_noise: Tensor,
        step_noises: Sequence[Tensor],
        *,
        num_inference_steps: int | None = None,
    ) -> Tensor:
        """Sample a full window ``(B, horizon, A)`` for stacked observations (history on dim 1)."""
        self.policy.eval()
        cond = self.policy.diffusion._prepare_global_conditioning(dict(observation))
        return self.sampler(
            lambda x, t: self.unet(x, t, global_cond=cond),
            initial_noise,
            self.schedule,
            num_inference_steps or self.num_inference_steps,
            step_noises,
            self.clip_range,
        )

    @torch.no_grad()
    def select_action(self, batch: Mapping[str, Tensor]) -> Tensor:

        config = cast("DiffusionConfig", self.policy.config)
        self._push(batch)
        if not self._actions:
            observation = {
                name: torch.stack(list(queue), dim=1) for name, queue in self._observations.items()
            }
            state = observation["observation.state"]
            generator = torch.Generator(device=state.device).manual_seed(
                self.seed + self.chunks_made
            )
            feature = config.action_feature
            if feature is None:
                raise ValueError("Diffusion Policy requires an action feature")
            shape = (state.shape[0], config.horizon, feature.shape[0])
            initial = torch.randn(
                shape, generator=generator, device=state.device, dtype=state.dtype
            )
            noises = make_step_noises(
                self.num_inference_steps, shape, generator, state.device, state.dtype
            )
            started = time.perf_counter()
            window = self.predict_window(observation, initial, noises)
            if state.device.type == "cuda":
                torch.cuda.synchronize()
            self.sample_seconds.append(time.perf_counter() - started)
            self.chunks_made += 1
            self._actions.extend(
                self.slicer(window, config.n_obs_steps, config.n_action_steps).transpose(0, 1)
            )
        return self._actions.popleft()
