"""Behaviour checks for the Lesson 15 exercise.

The notebook checks one small configuration. These checks build prepared U-Nets with
other widths, depths, action sizes and FiLM settings, and test that the learner copy is
independent of the prepared network and can be trained.
Read a failing assertion message first: it states the expected behaviour.
"""

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d
from torch import nn


def _done(message: str) -> None:
    print(f"✓ {message}")


def _prepared(
    action_dim: int, down_dims: tuple[int, ...], cond_dim: int, film_scale: bool, seed: int
):
    torch.manual_seed(seed)
    config = DiffusionConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (8,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=down_dims,
        diffusion_step_embed_dim=16,
        use_film_scale_modulation=film_scale,
        pretrained_backbone_weights=None,
        push_to_hub=False,
        device="cpu",
    )
    unet = DiffusionConditionalUnet1d(config, global_cond_dim=cond_dim)
    for parameter in unet.parameters():  # distinct values make a wrong copy visible
        if parameter.dim() == 1:
            nn.init.normal_(parameter, std=0.1)
    return unet.eval()


def check_make_finetune_unet(make_finetune_unet) -> None:
    """Exercise 15.1: same function as the prepared U-Net, for several configurations."""
    configs = [
        (7, (8, 16, 32), 10, True),
        (3, (8, 16), 4, False),
        (2, (16,), 5, True),
    ]
    for index, (action_dim, down_dims, cond_dim, film_scale) in enumerate(configs):
        prepared = _prepared(action_dim, down_dims, cond_dim, film_scale, seed=70 + index)
        learner = make_finetune_unet(prepared)
        assert isinstance(learner, nn.Module), "return an nn.Module"
        learner.eval()
        x = torch.randn(3, 16, action_dim)
        t = torch.tensor([0, 42, 99])
        cond = torch.randn(3, cond_dim)
        with torch.no_grad():
            gap = (
                (prepared(x, t, global_cond=cond) - learner(x, t, global_cond=cond))
                .abs()
                .max()
                .item()
            )
        label = f"action_dim={action_dim}, down_dims={down_dims}, cond_dim={cond_dim}, FiLM scale={film_scale}"
        assert gap < 1e-4, f"{label}: learner output differs from the prepared U-Net by {gap:.2e}"
        prepared_count = sum(p.numel() for p in prepared.parameters())
        learner_count = sum(p.numel() for p in learner.parameters())
        assert learner_count == prepared_count, (
            f"{label}: the learner has {learner_count} parameters, the prepared U-Net {prepared_count}"
        )
    _done("matches the prepared U-Net for three depths, two action sizes, and both FiLM modes")

    prepared = _prepared(7, (8, 16), 6, True, seed=80)
    snapshot = {name: value.clone() for name, value in prepared.state_dict().items()}
    learner = make_finetune_unet(prepared).train()
    prepared_storages = {p.data_ptr() for p in prepared.parameters()}
    assert not any(p.data_ptr() in prepared_storages for p in learner.parameters()), (
        "copy the weights: the learner must not share parameter tensors with the prepared U-Net"
    )
    assert all(p.requires_grad for p in learner.parameters()), (
        "every learner weight must be trainable"
    )
    optimizer = torch.optim.SGD(learner.parameters(), lr=0.1)
    x, t, cond = torch.randn(2, 16, 7), torch.tensor([3, 50]), torch.randn(2, 6)
    learner(x, t, global_cond=cond).square().mean().backward()
    optimizer.step()
    for name, value in prepared.state_dict().items():
        assert torch.equal(value, snapshot[name]), (
            f"training the learner changed the prepared weight {name}"
        )
    _done(
        "the learner holds its own trainable copy; updating it leaves the prepared U-Net unchanged"
    )
