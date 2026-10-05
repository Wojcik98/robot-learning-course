"""Compact conditional VAE components for the VLA course."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from vla_lab.teaching.training import diagonal_gaussian_kl, masked_action_l1


def reparameterize(mu: Tensor, logvar: Tensor, *, noise: Tensor | None = None) -> Tensor:
    """Draw a differentiable sample from a diagonal Gaussian."""
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have matching shapes")
    if noise is None:
        noise = torch.randn_like(mu)
    if noise.shape != mu.shape:
        raise ValueError("noise must have the same shape as mu")
    return mu + torch.exp(0.5 * logvar) * noise


class ConditionalEncoder(nn.Module):
    """Encode a condition and target into Gaussian latent parameters."""

    def __init__(
        self,
        condition_dim: int,
        target_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        mask_dim: int = 0,
    ) -> None:
        super().__init__()
        if min(condition_dim, target_dim, latent_dim, hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.body = nn.Sequential(
            nn.Linear(condition_dim + target_dim + mask_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, condition: Tensor, target: Tensor, target_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if condition.shape[:-1] != target.shape[:-1]:
            raise ValueError("condition and target leading dimensions must match")
        pieces = [condition, target]
        if target_mask is not None:
            pieces.append(target_mask.to(dtype=target.dtype))
        hidden = self.body(torch.cat(pieces, dim=-1))
        return self.mu(hidden), self.logvar(hidden).clamp(-20.0, 10.0)


class ConditionalDecoder(nn.Module):
    """Predict a target from its condition and latent code."""

    def __init__(
        self, condition_dim: int, latent_dim: int, target_dim: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        if min(condition_dim, target_dim, latent_dim, hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(condition_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, condition: Tensor, latent: Tensor) -> Tensor:
        if condition.shape[:-1] != latent.shape[:-1]:
            raise ValueError("condition and latent leading dimensions must match")
        return self.network(torch.cat((condition, latent), dim=-1))


class ConditionalVAE(nn.Module):
    """Conditional VAE for vectors or fixed-size action chunks.

    For action chunks, set ``target_shape=(chunk_size, action_dim)``. Padding
    values are zeroed before encoding and the time mask is included in the
    posterior input.
    """

    def __init__(
        self,
        condition_dim: int,
        target_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        *,
        target_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        self.target_width = target_dim
        self.mask_width = 0
        if target_shape is not None:
            self.target_width = 1
            for width in target_shape:
                self.target_width *= width
            if self.target_width != target_dim:
                raise ValueError("target_dim must equal the product of target_shape")
            if len(target_shape) == 2:
                self.mask_width = target_shape[0]
        self.encoder = ConditionalEncoder(
            condition_dim, self.target_width, latent_dim, hidden_dim, mask_dim=self.mask_width
        )
        self.decoder = ConditionalDecoder(condition_dim, latent_dim, self.target_width, hidden_dim)

    def forward(
        self,
        condition: Tensor,
        target: Tensor,
        *,
        noise: Tensor | None = None,
        is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if target.shape[:-1] != condition.shape[:-1] and target.shape[0] != condition.shape[0]:
            raise ValueError("condition and target must have the same batch size")
        original_shape = target.shape
        target_flat = target.reshape(target.shape[0], -1)
        target_mask = None
        if self.target_shape is not None and tuple(target.shape[1:]) != self.target_shape:
            raise ValueError(f"target must have shape (B, {self.target_shape})")
        if is_pad is not None:
            if target.ndim != 3 or is_pad.shape != target.shape[:2]:
                raise ValueError("is_pad must have shape (B,T) for action chunks")
            valid = (~is_pad.bool()).unsqueeze(-1).to(target.dtype)
            target_flat = (target * valid).reshape(target.shape[0], -1)
            target_mask = is_pad.to(target.dtype)
        elif self.mask_width:
            target_mask = torch.zeros(
                (target.shape[0], self.mask_width), device=target.device, dtype=target.dtype
            )
        mu, logvar = self.encoder(condition, target_flat, target_mask)
        latent = reparameterize(mu, logvar, noise=noise)
        prediction = self.decoder(condition, latent)
        if self.target_shape is not None:
            prediction = prediction.reshape(target.shape[0], *self.target_shape)
        elif len(original_shape) > 2:
            prediction = prediction.reshape(original_shape)
        return prediction, mu, logvar

    def infer(
        self,
        condition: Tensor,
        *,
        latent: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Decode a supplied prior latent or a standard-normal sample."""
        expected = (*condition.shape[:-1], self.latent_dim)
        if latent is not None and noise is not None:
            raise ValueError("provide latent or noise, not both")
        if latent is None:
            latent = (
                noise
                if noise is not None
                else torch.randn(expected, device=condition.device, dtype=condition.dtype)
            )
        if latent.shape != expected:
            raise ValueError(f"latent must have shape {expected}")
        prediction = self.decoder(condition, latent)
        if self.target_shape is not None:
            prediction = prediction.reshape(condition.shape[0], *self.target_shape)
        return prediction


def cvae_loss_fn(
    model: ConditionalVAE,
    batch: Mapping[str, Tensor],
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
    beta: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Callback-compatible CVAE loss for vector targets or padded action chunks."""
    del time
    if "condition" not in batch or "target" not in batch:
        raise KeyError("CVAE batch requires condition and target")
    reconstruction, mu, logvar = model(
        batch["condition"], batch["target"], noise=noise, is_pad=batch.get("is_pad")
    )
    if reconstruction.ndim == 3:
        if "is_pad" not in batch:
            raise KeyError("chunk targets require is_pad")
        reconstruction_loss = masked_action_l1(reconstruction, batch["target"], batch["is_pad"])
    else:
        reconstruction_loss = (reconstruction - batch["target"]).abs().mean()
    kl = diagonal_gaussian_kl(mu, logvar)
    return reconstruction_loss + float(beta) * kl, {"reconstruction": reconstruction_loss, "kl": kl}
