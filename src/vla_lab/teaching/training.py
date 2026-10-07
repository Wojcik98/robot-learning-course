"""Small training operations used by the VLA course notebooks."""

from collections.abc import Iterable, Mapping
from math import isfinite
from time import monotonic

import torch
from torch import Tensor

from vla_lab.teaching.interfaces import ACTPredictionModel, LossCallback, ModelT


def masked_action_l1(prediction: Tensor, target: Tensor, is_pad: Tensor) -> Tensor:
    """Return mean absolute error over valid ``(batch, time, action)`` elements."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have the same (B, T, D) shape")
    if is_pad.shape != prediction.shape[:2]:
        raise ValueError("is_pad must have shape (B, T), with True for padding")
    valid = (~is_pad.to(torch.bool)).unsqueeze(-1).to(dtype=prediction.dtype)
    denominator = (valid.sum() * prediction.shape[-1]).clamp_min(1)
    return ((prediction - target).abs() * valid).sum() / denominator


def diagonal_gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL from a diagonal Gaussian to a standard normal, summed over latent width."""
    if mu.shape != logvar.shape or mu.ndim != 2:
        raise ValueError("mu and logvar must have the same (B, latent) shape")
    return (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(dim=-1).mean()


def act_loss_fn(
    model: ACTPredictionModel,
    batch: Mapping[str, Tensor],
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
    kl_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """ACT objective callback for a module that returns raw predictions and latent stats."""
    del noise, time
    if "action" not in batch or "action_is_pad" not in batch:
        raise KeyError("ACT batch requires action and action_is_pad")
    output = model(batch)
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise TypeError("ACT model must return (predicted_actions, (mu, logvar))")
    prediction, latent = output
    if not isinstance(latent, (tuple, list)) or len(latent) != 2:
        raise TypeError("ACT latent output must be (mu, logvar)")
    mu, logvar = latent
    if mu is None or logvar is None:
        raise ValueError("ACT training requires latent mean and variance")
    action_loss = masked_action_l1(prediction, batch["action"], batch["action_is_pad"])
    kl_loss = diagonal_gaussian_kl(mu, logvar)
    total = action_loss + float(kl_weight) * kl_loss
    return total, {"action_l1": action_loss, "kl": kl_loss}


def optimizer_update(
    model: ModelT,
    batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossCallback[ModelT],
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Run one callback-driven update and return detached scalar metrics."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = loss_fn(model, batch, noise=noise, time=time)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("loss_fn must return one finite scalar tensor")
    loss.backward()
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    result = {"loss": float(loss.detach().cpu())}
    for name, value in metrics.items():
        value_tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if value_tensor.numel() != 1:
            raise ValueError(f"metric {name!r} must be scalar")
        result[str(name)] = float(value_tensor.detach().cpu())
    return result


def train_loop(
    model: ModelT,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossCallback[ModelT],
    *,
    steps: int,
    max_grad_norm: float | None = None,
    noise: Tensor | list[Tensor] | tuple[Tensor, ...] | None = None,
    time: Tensor | list[Tensor] | tuple[Tensor, ...] | None = None,
    max_seconds: float | None = None,
) -> list[dict[str, float]]:
    """Run updates, cycling through batches and optional explicit random inputs.

    ``max_seconds`` is an optional positive wall-time budget in seconds. Check
    it before each update. Finish an update already in progress, then return
    the completed history so the caller can save the model. This cooperative
    limit can overrun by the duration of one update; it is not a process timer.
    """
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("batches must contain at least one batch")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if max_seconds is not None and (not isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    for name, values in (("noise", noise), ("time", time)):
        if isinstance(values, (list, tuple)) and not values:
            raise ValueError(f"{name} sequence must not be empty")

    def at_step(
        value: Tensor | list[Tensor] | tuple[Tensor, ...] | None, index: int
    ) -> Tensor | None:
        if isinstance(value, (list, tuple)):
            return value[index % len(value)]
        return value

    history = []
    deadline = None if max_seconds is None else monotonic() + max_seconds
    for index in range(steps):
        if deadline is not None and monotonic() >= deadline:
            break
        batch = batch_list[index % len(batch_list)]
        history.append(
            optimizer_update(
                model,
                batch,
                optimizer,
                loss_fn,
                max_grad_norm=max_grad_norm,
                noise=at_step(noise, index),
                time=at_step(time, index),
            )
        )
    return history
