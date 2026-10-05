"""Check the Diffusion Policy operations against diffusers' DDPM and the pinned LeRobot network."""

from pathlib import Path

import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalResidualBlock1d,
    DiffusionConditionalUnet1d,
    DiffusionSinusoidalPosEmb,
)

from vla_lab.teaching.diffusion import (
    CallbackDiffusionPolicy,
    ConditionalResidualBlock1d,
    ConditionalUnet1D,
    DDPMSchedule,
    DiffusionDenoiser,
    LeRobotDenoiser,
    convert_lerobot_unet_state_dict,
    cosine_betas,
    cumulative_alphas,
    ddpm_reverse_step,
    epsilon_loss_fn,
    executed_action_slice,
    flat_state_condition,
    forward_noise,
    inference_timesteps,
    load_state_action_windows,
    make_step_noises,
    masked_epsilon_loss,
    posterior_coefficients,
    sample_actions,
    sample_noise_and_timesteps,
    sinusoidal_timestep_embedding,
    stack_observation,
    take_batch,
    unet_channel_plan,
    unet_from_lerobot,
    verify_unet_parity,
)
from vla_lab.teaching.training import optimizer_update, train_loop

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ("observation.images.image", "observation.images.image2")
H, A, S, N_OBS = 16, 7, 8, 2


def ddpm(prediction_type="epsilon", clip=True):
    return DDPMScheduler(
        num_train_timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=clip,
        clip_sample_range=1.0,
        prediction_type=prediction_type,
    )


def tiny_config(**changes):
    values = {
        "input_features": {
            IMAGES[0]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            IMAGES[1]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (S,)),
        },
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (A,))},
        "n_obs_steps": N_OBS,
        "horizon": H,
        "n_action_steps": 8,
        "drop_n_last_frames": 7,
        "down_dims": (16, 32, 64),
        "spatial_softmax_num_keypoints": 8,
        "diffusion_step_embed_dim": 32,
        "pretrained_backbone_weights": None,
        "crop_is_random": False,
        "num_train_timesteps": 100,
        "num_inference_steps": 20,
        "do_mask_loss_for_padding": True,
        "push_to_hub": False,
        "device": "cpu",
    }
    values.update(changes)
    return DiffusionConfig(**values)


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(5)
    model = DiffusionPolicy(tiny_config())
    for parameter in model.parameters():
        if parameter.dim() == 1:
            torch.nn.init.normal_(parameter, std=0.05)
    return model.eval()


def training_batch(batch=3, seed=1):
    generator = torch.Generator().manual_seed(seed)
    pad = torch.zeros(batch, H, dtype=torch.bool)
    pad[1, 12:] = True
    pad[2, 3:] = True
    return {
        "observation.state": torch.randn(batch, N_OBS, S, generator=generator),
        IMAGES[0]: torch.rand(batch, N_OBS, 3, 64, 64, generator=generator),
        IMAGES[1]: torch.rand(batch, N_OBS, 3, 64, 64, generator=generator),
        "action": torch.rand(batch, H, A, generator=generator) * 2 - 1,
        "action_is_pad": pad,
    }


# --- unit 9 ---------------------------------------------------------------------------------


def test_schedule_tables_match_diffusers():
    scheduler = ddpm()
    assert torch.allclose(cosine_betas(100), scheduler.betas, atol=0, rtol=0)
    assert torch.allclose(
        cumulative_alphas(scheduler.betas), scheduler.alphas_cumprod, atol=0, rtol=0
    )
    schedule = DDPMSchedule.cosine(100)
    assert schedule.num_train_timesteps == 100
    assert (schedule.alpha_bar[1:] < schedule.alpha_bar[:-1]).all()


def test_forward_noise_matches_diffusers_and_endpoints():
    scheduler = ddpm()
    x0, noise = torch.randn(4, H, A), torch.randn(4, H, A)
    t = torch.tensor([0, 17, 50, 99])
    assert torch.allclose(
        forward_noise(x0, noise, t, scheduler.alphas_cumprod),
        scheduler.add_noise(x0, noise, t),
        atol=1e-6,
    )
    low = forward_noise(x0, noise, torch.zeros(4, dtype=torch.long), scheduler.alphas_cumprod)
    assert (low - x0).abs().max() < 0.2  # step 0 is almost clean
    high = forward_noise(x0, noise, torch.full((4,), 99), scheduler.alphas_cumprod)
    assert (high - noise).abs().max() < 0.2 + (
        x0.abs().max() * 0.05
    )  # step 99 is almost pure noise


def test_noise_and_timestep_draws_follow_the_pinned_order_and_range():
    actions = torch.zeros(500, H, A)
    generator = torch.Generator().manual_seed(3)
    noise, t = sample_noise_and_timesteps(actions, 100, generator)
    assert noise.shape == actions.shape and t.shape == (500,) and t.dtype == torch.long
    assert t.min() == 0 and t.max() == 99
    again = sample_noise_and_timesteps(actions, 100, torch.Generator().manual_seed(3))
    assert torch.equal(noise, again[0]) and torch.equal(t, again[1])
    torch.manual_seed(9)
    expected_noise = torch.randn(actions.shape)
    expected_t = torch.randint(0, 100, (500,))
    torch.manual_seed(9)
    drawn = sample_noise_and_timesteps(actions, 100)
    assert torch.equal(drawn[0], expected_noise) and torch.equal(drawn[1], expected_t)


def test_masked_epsilon_loss_values_gradient_and_all_padding():
    pred = torch.zeros(2, 3, 2, requires_grad=True)
    target = torch.ones(2, 3, 2)
    pad = torch.tensor([[False, False, True], [False, True, True]])
    loss = masked_epsilon_loss(pred, target, pad)
    assert torch.allclose(loss, torch.tensor(1.0))
    loss.backward()
    assert torch.count_nonzero(pred.grad[pad]) == 0
    changed = pred.detach().clone()
    changed[pad] = 99.0
    assert torch.allclose(masked_epsilon_loss(changed, target, pad), loss.detach())
    everything = masked_epsilon_loss(pred.detach(), target, torch.ones_like(pad))
    assert everything == 0.0


def test_epsilon_loss_fn_equals_the_pinned_loss_for_the_same_random_numbers(policy):
    batch = training_batch()
    torch.manual_seed(21)
    expected = policy.diffusion.compute_loss(stack_observation(policy, batch))
    torch.manual_seed(21)
    loss, metrics = epsilon_loss_fn(LeRobotDenoiser(policy), batch)
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.allclose(metrics["epsilon_mse"], loss)


def test_epsilon_loss_fn_uses_explicit_noise_and_time_and_keeps_the_graph(policy):
    denoiser = LeRobotDenoiser(policy)
    denoiser.train()
    batch = training_batch()
    noise = torch.randn_like(batch["action"])
    time = torch.tensor([3, 40, 90])
    a, _ = epsilon_loss_fn(denoiser, batch, noise=noise, time=time)
    b, _ = epsilon_loss_fn(denoiser, batch, noise=noise, time=time)
    assert torch.allclose(a, b, atol=1e-6)
    a.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in denoiser.parameters())
    denoiser.zero_grad()
    denoiser.eval()


def test_epsilon_loss_fn_draws_only_missing_random_inputs():
    from types import SimpleNamespace

    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))
            self.schedule = SimpleNamespace(
                num_train_timesteps=11,
                alpha_bar=torch.linspace(0.99, 0.1, 11),
            )

        def forward(self, noisy_actions, timesteps, batch):
            del timesteps, batch
            return noisy_actions * self.scale

    batch = {
        "action": torch.zeros(2, 3, 2),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }
    model = TinyDenoiser()
    supplied_noise = torch.ones_like(batch["action"])
    supplied_time = torch.tensor([2, 7])

    torch.manual_seed(31)
    initial_state = torch.get_rng_state()
    epsilon_loss_fn(model, batch, noise=supplied_noise, time=supplied_time)
    assert torch.equal(torch.get_rng_state(), initial_state), (
        "Explicit inputs must not advance RNG state"
    )

    seed = 37
    torch.manual_seed(seed)
    torch.randn(batch["action"].shape)
    expected_after_noise = torch.get_rng_state()
    torch.manual_seed(seed)
    epsilon_loss_fn(model, batch, time=supplied_time)
    assert torch.equal(torch.get_rng_state(), expected_after_noise), (
        "Only missing noise must be drawn"
    )

    torch.manual_seed(seed)
    torch.randint(0, model.schedule.num_train_timesteps, (batch["action"].shape[0],))
    expected_after_time = torch.get_rng_state()
    torch.manual_seed(seed)
    epsilon_loss_fn(model, batch, noise=supplied_noise)
    assert torch.equal(torch.get_rng_state(), expected_after_time), (
        "Only missing time must be drawn"
    )


def test_one_update_changes_unet_weights_through_the_training_callback(policy):
    import copy

    model = LeRobotDenoiser(copy.deepcopy(policy)).train()
    before = model.unet.final_conv[1].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = training_batch()
    noise, time = sample_noise_and_timesteps(batch["action"], 100, torch.Generator().manual_seed(0))
    metrics = optimizer_update(model, batch, optimizer, epsilon_loss_fn, noise=noise, time=time)
    assert metrics["loss"] == metrics["epsilon_mse"] and not torch.equal(
        before, model.unet.final_conv[1].weight
    )


# --- unit 10 --------------------------------------------------------------------------------


def test_timestep_embedding_matches_lerobot():
    t = torch.tensor([0, 1, 7, 99])
    for dim in (16, 32, 128):
        assert torch.allclose(
            sinusoidal_timestep_embedding(t, dim), DiffusionSinusoidalPosEmb(dim)(t), atol=1e-6
        )


@pytest.mark.parametrize("scale", [True, False])
def test_film_block_matches_lerobot_block(scale):
    torch.manual_seed(2)
    reference = DiffusionConditionalResidualBlock1d(
        8, 16, 10, kernel_size=5, n_groups=8, use_film_scale_modulation=scale
    )
    mine = ConditionalResidualBlock1d(8, 16, 10, 5, 8, scale)
    mine.load_state_dict(reference.state_dict(), strict=True)
    x, cond = torch.randn(3, 8, 16, requires_grad=True), torch.randn(3, 10)
    out = mine(x, cond)
    assert torch.allclose(out, reference(x, cond), atol=1e-6)
    out.square().sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    reference(x2, cond).square().sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-5)


def reference_unet(down_dims=(16, 32, 64), cond=6):
    torch.manual_seed(4)
    config = tiny_config(down_dims=down_dims)
    return DiffusionConditionalUnet1d(config, global_cond_dim=cond).eval()


@pytest.mark.parametrize("down_dims", [(16, 32), (16, 32, 64), (8, 16, 24, 32)])
def test_channel_plan_matches_the_pinned_network(down_dims):
    unet = reference_unet(down_dims)
    plan = unet_channel_plan(A, down_dims)
    for (i, o, down), (res1, res2, downsample) in zip(plan["down"], unet.down_modules, strict=True):
        assert (res1.conv1.block[0].in_channels, res1.conv1.block[0].out_channels) == (i, o)
        assert res2.conv2.block[0].out_channels == o
        assert isinstance(downsample, torch.nn.Conv1d) == down
    for (i, o, up), (res1, res2, upsample) in zip(plan["up"], unet.up_modules, strict=True):
        assert (res1.conv1.block[0].in_channels, res1.conv1.block[0].out_channels) == (i, o)
        assert isinstance(upsample, torch.nn.ConvTranspose1d) == up
    assert len(plan["mid"]) == len(unet.mid_modules)


@pytest.mark.parametrize("down_dims", [(16, 32), (16, 32, 64)])
def test_unet_with_converted_weights_matches_the_pinned_network(down_dims):
    unet = reference_unet(down_dims)
    for p in unet.parameters():
        if p.dim() == 1:
            torch.nn.init.normal_(p, std=0.1)
    learner = unet_from_lerobot(unet)
    x, t, cond = torch.randn(3, H, A), torch.tensor([0, 10, 99]), torch.randn(3, 6)
    assert verify_unet_parity(learner, unet, x, t, cond) < 1e-5
    x1 = x.clone().requires_grad_(True)
    learner(x1, t, cond).square().sum().backward()
    x2 = x.clone().requires_grad_(True)
    unet(x2, t, cond).square().sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4)


def test_unet_converter_is_strict():
    unet = reference_unet()
    converted = convert_lerobot_unet_state_dict(unet.state_dict())
    assert set(converted) == set(
        ConditionalUnet1D(A, 6, (16, 32, 64), step_embed_dim=32).state_dict()
    )
    with pytest.raises(KeyError):
        convert_lerobot_unet_state_dict({"mystery.weight": torch.zeros(1)})
    with pytest.raises(RuntimeError):
        ConditionalUnet1D(A, 6, (16, 32), step_embed_dim=32).load_state_dict(converted, strict=True)


def test_unet_hooks_are_used_by_the_real_forward_path():
    calls = []

    def forward(unet, x, timestep, cond=None):
        from vla_lab.teaching.diffusion import unet_forward

        calls.append("unet")
        return unet_forward(unet, x, timestep, cond)

    def block(block_, x, cond):
        from vla_lab.teaching.diffusion import film_residual_forward

        calls.append("block")
        return film_residual_forward(block_, x, cond)

    unet = ConditionalUnet1D(
        A, 0, (16, 32), step_embed_dim=32, forward_fn=forward, block_forward_fn=block
    )
    unet(torch.randn(2, H, A), torch.tensor([1, 2]))
    assert calls[0] == "unet" and calls.count("block") == 4 + 2 + 2


def test_state_conditioned_denoiser_trains_with_the_callback_loop():
    torch.manual_seed(0)
    schedule = DDPMSchedule.cosine(100)
    unet = ConditionalUnet1D(A, N_OBS * S, (16, 32), step_embed_dim=32)
    model = DiffusionDenoiser(unet, schedule, flat_state_condition)
    batch = {
        "observation.state": torch.rand(8, N_OBS, S) * 2 - 1,
        "action": torch.rand(8, H, A) * 2 - 1,
        "action_is_pad": torch.zeros(8, H, dtype=torch.bool),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    history = train_loop(model, [batch], optimizer, epsilon_loss_fn, steps=30)
    assert history[-1]["loss"] < history[0]["loss"]


# --- unit 11 --------------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [1, 5, 10, 20, 50, 100])
def test_inference_timesteps_match_diffusers(steps):
    scheduler = ddpm()
    scheduler.set_timesteps(steps)
    assert inference_timesteps(100, steps) == scheduler.timesteps.tolist()
    assert inference_timesteps(100, steps)[-1] == 0


def test_inference_timesteps_reject_bad_budgets():
    for bad in (0, 101):
        with pytest.raises(ValueError):
            inference_timesteps(100, bad)


def test_posterior_coefficients_for_adjacent_and_skipped_steps():
    scheduler = ddpm()
    abar = scheduler.alphas_cumprod
    for t, prev in ((50, 49), (50, 45), (99, 94), (5, 0)):
        coef_x0, coef_xt, var = posterior_coefficients(abar[t], abar[prev])
        alpha = abar[t] / abar[prev]
        assert torch.allclose(alpha, torch.prod(1 - scheduler.betas[prev + 1 : t + 1]), atol=1e-6)
        assert torch.allclose(coef_x0, abar[prev].sqrt() * (1 - alpha) / (1 - abar[t]))
        assert var > 0
    # final jump to abar_prev = 1: x_prev is the clean sample estimate itself.
    coef_x0, coef_xt, var = posterior_coefficients(abar[0], torch.tensor(1.0))
    assert torch.allclose(coef_x0, torch.tensor(1.0), atol=1e-6) and torch.allclose(
        coef_xt, torch.tensor(0.0), atol=1e-6
    )


@pytest.mark.parametrize("steps", [5, 10, 20, 100])
def test_reverse_step_matches_scheduler_for_every_timestep_including_final(steps):
    scheduler = ddpm()
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps.tolist()
    abar = scheduler.alphas_cumprod
    generator = torch.Generator().manual_seed(0)
    for index, t in enumerate(timesteps):
        prev = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        x = torch.randn(2, H, A, generator=generator) * 1.5
        eps = torch.randn(2, H, A, generator=generator)
        noise = torch.randn(2, H, A, generator=generator)
        # Same noise for the scheduler: a generator that yields exactly `noise`.
        seed_generator = torch.Generator().manual_seed(100 + index)
        scheduler_noise = torch.randn(2, H, A, generator=torch.Generator().manual_seed(100 + index))
        expected = scheduler.step(eps, t, x, generator=seed_generator).prev_sample
        mine = ddpm_reverse_step(x, eps, t, prev, abar, scheduler_noise if prev >= 0 else None, 1.0)
        assert torch.allclose(mine, expected, atol=1e-5), (steps, t, prev)
        if prev < 0:  # the final step is deterministic and returns the clipped x0 estimate
            x0 = ((x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()).clamp(-1, 1)
            assert torch.allclose(mine, x0, atol=1e-5)
            assert torch.equal(mine, ddpm_reverse_step(x, eps, t, prev, abar, noise, 1.0))


def test_non_final_step_needs_noise():
    abar = DDPMSchedule.cosine(100).alpha_bar
    with pytest.raises(ValueError):
        ddpm_reverse_step(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), 10, 5, abar, None)


def _toy_denoiser():
    torch.manual_seed(8)
    net = torch.nn.Linear(A, A)
    return lambda x, t: net(x) * (t.float().reshape(-1, 1, 1) / 100.0)


@pytest.mark.parametrize("steps", [3, 20, 100])
def test_sample_actions_matches_the_diffusers_loop_with_the_same_noise(steps):
    scheduler = ddpm()
    scheduler.set_timesteps(steps)
    denoise = _toy_denoiser()
    shape = (2, H, A)
    g = torch.Generator().manual_seed(1)
    initial = torch.randn(shape, generator=g)
    expected = initial
    for t in scheduler.timesteps:
        eps = denoise(expected, torch.full((2,), int(t), dtype=torch.long))
        expected = scheduler.step(eps, int(t), expected, generator=g).prev_sample
    g2 = torch.Generator().manual_seed(1)
    initial2 = torch.randn(shape, generator=g2)
    noises = make_step_noises(steps, shape, g2)
    mine = sample_actions(
        denoise, initial2, DDPMSchedule.from_scheduler(scheduler), steps, noises, 1.0
    )
    assert torch.allclose(mine, expected, atol=1e-4)
    assert torch.allclose(mine.abs().max(), expected.abs().max(), atol=1e-4)


def test_sample_actions_is_repeatable_and_checks_noise_count():
    denoise = _toy_denoiser()
    schedule = DDPMSchedule.cosine(100)
    shape = (2, H, A)
    initial = torch.randn(shape)
    noises = make_step_noises(10, shape, torch.Generator().manual_seed(2))
    a = sample_actions(denoise, initial, schedule, 10, noises)
    b = sample_actions(denoise, initial, schedule, 10, noises)
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        sample_actions(denoise, initial, schedule, 10, noises[:-1])
    assert len(make_step_noises(1, shape)) == 0


def test_sample_actions_carries_each_sample_into_the_next_step_and_passes_batch_times():
    seen = []

    def denoise(x, t):
        seen.append((x.clone(), t.clone()))
        return torch.zeros_like(x)

    schedule = DDPMSchedule.cosine(100)
    initial = torch.ones(2, 4, 2) * 0.5
    sample_actions(
        denoise,
        initial,
        schedule,
        4,
        make_step_noises(4, initial.shape, torch.Generator().manual_seed(0)),
    )
    assert [int(t[0]) for _, t in seen] == [75, 50, 25, 0]
    assert all(t.dtype == torch.long and t.shape == (2,) for _, t in seen)
    assert torch.equal(seen[0][0], initial) and not torch.equal(seen[1][0], initial)


def test_executed_slice_aligns_with_the_history():
    chunk = torch.arange(16).reshape(1, 16, 1).float()
    taken = executed_action_slice(chunk, 2, 8)
    assert taken.flatten().tolist() == [float(i) for i in range(1, 9)]
    assert executed_action_slice(chunk, 1, 4).flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        executed_action_slice(chunk, 2, 16)


# --- integration with the real policy objects ---------------------------------------------------


def _observation(batch=2, seed=3):
    g = torch.Generator().manual_seed(seed)
    return {
        "observation.state": torch.randn(batch, S, generator=g),
        IMAGES[0]: torch.rand(batch, 3, 64, 64, generator=g),
        IMAGES[1]: torch.rand(batch, 3, 64, 64, generator=g),
    }


def test_callback_policy_window_matches_pinned_conditional_sample(policy):
    obs = _observation()
    wrapper = CallbackDiffusionPolicy(policy, seed=7)
    wrapper._push(obs)
    observation = {k: torch.stack(list(q), dim=1) for k, q in wrapper._observations.items()}
    generator = torch.Generator().manual_seed(7)
    shape = (2, H, A)
    initial = torch.randn(shape, generator=generator)
    noises = make_step_noises(20, shape, generator)
    mine = wrapper.predict_window(observation, initial, noises)
    cond = policy.diffusion._prepare_global_conditioning(observation)
    with torch.no_grad():
        expected = policy.diffusion.conditional_sample(
            2, global_cond=cond, generator=torch.Generator().manual_seed(7)
        )
    assert torch.allclose(mine, expected, atol=1e-4)


def test_callback_policy_select_action_uses_the_learner_sampler_slices_and_resets(policy):
    calls = []

    def sampler(*args, **kwargs):
        calls.append(args[3])
        return sample_actions(*args, **kwargs)

    slices = []

    def slicer(chunk, n_obs, n_act):
        slices.append((n_obs, n_act))
        return executed_action_slice(chunk, n_obs, n_act)

    wrapper = CallbackDiffusionPolicy(policy, sampler, slicer, num_inference_steps=5, seed=1)
    outputs = [wrapper.select_action(_observation(seed=i)) for i in range(9)]
    assert all(o.shape == (2, A) for o in outputs)
    assert calls == [5, 5] and slices == [(2, 8), (2, 8)]  # 8 actions per chunk, so 9 calls need 2
    wrapper.select_action(_observation(seed=0))
    assert len(wrapper.sample_seconds) == 2 and wrapper.chunks_made == 2
    wrapper.reset()
    assert wrapper.chunks_made == 0 and len(wrapper._actions) == 0
    assert all(len(queue) == 0 for queue in wrapper._observations.values())
    wrapper.select_action(_observation(seed=0))
    assert wrapper.chunks_made == 1 and len(calls) == 3


def test_callback_policy_noise_is_shared_across_step_budgets(policy):
    initial = []

    def sampler(denoise, initial_noise, *args, **kwargs):
        initial.append(initial_noise.clone())
        return sample_actions(denoise, initial_noise, *args, **kwargs)

    for steps in (5, 20):
        CallbackDiffusionPolicy(policy, sampler, num_inference_steps=steps, seed=3).select_action(
            _observation()
        )
    assert torch.equal(initial[0], initial[1])


# --- real data ---------------------------------------------------------------------------------

DATA = ROOT / ".local-data/hub/lerobot-libero/meta/episodes"


@pytest.mark.skipif(not DATA.exists(), reason="local LIBERO metadata missing")
def test_real_windows_follow_the_pinned_alignment_and_split():
    train = load_state_action_windows(ROOT / "configs", ROOT / ".local-data", "bowl_plate", "train")
    valid = load_state_action_windows(
        ROOT / "configs", ROOT / ".local-data", "bowl_plate", "validation"
    )
    assert train["action"].shape[1:] == (16, 7) and train["observation.state"].shape[1:] == (2, 8)
    assert set(train["episode"].tolist()).isdisjoint(valid["episode"].tolist())
    assert (
        train["action"].abs().max() <= 1.0 + 1e-4
    )  # training extrema scale the training split into [-1, 1]
    first = train["frame"] == 0
    # frame 0: the first window slot is the step before the episode, so it is padding and a copy of step 0
    assert (
        train["action_is_pad"][first][:, 0].all() and not train["action_is_pad"][first][:, 1].any()
    )
    assert torch.equal(train["action"][first][:, 0], train["action"][first][:, 1])
    assert torch.equal(
        train["observation.state"][first][:, 0], train["observation.state"][first][:, 1]
    )
    last_of_episode = {}
    for index, (e, f) in enumerate(zip(train["episode"].tolist(), train["frame"].tolist())):
        last_of_episode[e] = max(last_of_episode.get(e, -1), f)
    index = next(
        i
        for i, (e, f) in enumerate(zip(train["episode"].tolist(), train["frame"].tolist()))
        if f == last_of_episode[e]
    )
    assert train["action_is_pad"][index][-1], "the last kept frame still has padded future actions"
    batch = take_batch(train, [0, 1, 2])
    assert batch["action"].shape == (3, 16, 7)


CHECKPOINT = ROOT / ".local-data/runs/diffusion-bowl-plate-reference-2000/checkpoint-002000"


@pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(), reason="local diffusion checkpoint missing"
)
def test_prepared_checkpoint_unet_and_sampler_reproduce_in_the_learner():
    config = DiffusionConfig.from_pretrained(CHECKPOINT, local_files_only=True)
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    checkpoint_policy = DiffusionPolicy.from_pretrained(
        CHECKPOINT, config=config, strict=True, local_files_only=True
    ).eval()
    unet = checkpoint_policy.diffusion.unet
    learner = unet_from_lerobot(unet)
    g = torch.Generator().manual_seed(0)
    cond_dim = unet.down_modules[0][0].cond_encoder[1].in_features - config.diffusion_step_embed_dim
    x, t, cond = (
        torch.randn(2, 16, 7, generator=g),
        torch.tensor([3, 80]),
        torch.randn(2, cond_dim, generator=g),
    )
    assert verify_unet_parity(learner, unet, x, t, cond) < 1e-4
    schedule = DDPMSchedule.from_scheduler(checkpoint_policy.diffusion.noise_scheduler)
    shape = (2, 16, 7)
    generator = torch.Generator().manual_seed(4)
    initial = torch.randn(shape, generator=generator)
    noises = make_step_noises(20, shape, generator)
    mine = sample_actions(lambda a, b: learner(a, b, cond), initial, schedule, 20, noises, 1.0)
    with torch.no_grad():
        expected = checkpoint_policy.diffusion.conditional_sample(
            2, global_cond=cond, generator=torch.Generator().manual_seed(4)
        )
    assert torch.allclose(mine, expected, atol=1e-3)


@pytest.mark.skipif(not DATA.exists(), reason="local LIBERO metadata missing")
def test_real_windows_equal_the_lerobot_dataset_windows_after_undoing_the_scaling():
    import json
    import os

    import yaml

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = yaml.safe_load((ROOT / "configs/data/episodes.yaml").read_text())
    stats = json.loads((ROOT / "configs/data/stats/bowl_plate.json").read_text())["modalities"]
    episodes = manifest["tasks"]["bowl_plate"]["validation_episode_ids"][:2]
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=ROOT / ".local-data/hub/lerobot-libero",
        episodes=episodes,
        revision=manifest["dataset_revision"],
        video_backend="torchcodec",
        delta_timestamps={
            "observation.state": [-0.1, 0.0],
            "action": [i / 10 for i in range(-1, 15)],
        },
    )
    windows = load_state_action_windows(
        ROOT / "configs", ROOT / ".local-data", "bowl_plate", "validation"
    )
    low = torch.tensor(stats["action"]["min"])
    span = torch.tensor(stats["action"]["max"]) - low
    slow = torch.tensor(stats["observation.state"]["min"])
    sspan = torch.tensor(stats["observation.state"]["max"]) - slow
    checked = 0
    for index in range(len(windows["frame"])):
        if windows["episode"][index] != episodes[0] or int(windows["frame"][index]) not in (
            0,
            1,
            40,
        ):
            continue
        item = dataset[int(windows["frame"][index])]
        action = (windows["action"][index] + 1) / 2 * span + low
        state = (windows["observation.state"][index] + 1) / 2 * sspan + slow
        assert torch.allclose(action, item["action"], atol=1e-4)
        assert torch.equal(windows["action_is_pad"][index], item["action_is_pad"])
        assert torch.allclose(state, item["observation.state"], atol=1e-4)
        checked += 1
    assert checked == 3


@pytest.mark.skipif(not DATA.exists(), reason="local LIBERO metadata missing")
def test_real_windows_with_a_longer_history_and_actions_from_the_current_frame():
    windows = load_state_action_windows(
        ROOT / "configs",
        ROOT / ".local-data",
        "bowl_plate",
        "validation",
        horizon=16,
        n_obs_steps=8,
        action_start=0,
        drop_n_last_frames=0,
    )
    assert windows["observation.state"].shape[1] == 8 and windows["state_is_pad"].shape[1] == 8
    start = windows["frame"] == 0
    assert (
        windows["state_is_pad"][start][:, :7].all()
        and not windows["state_is_pad"][start][:, 7].any()
    )
    assert not windows["action_is_pad"][start][:, 0].any()  # actions begin at the current frame


@pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(), reason="local diffusion checkpoint missing"
)
def test_prepared_bundle_loads_on_cpu_and_the_learner_loss_equals_the_pinned_loss(tmp_path):
    from vla_lab.teaching.diffusion import load_prepared_diffusion

    policy, _preprocessor, _ = load_prepared_diffusion(CHECKPOINT, device="cpu")
    assert next(policy.parameters()).device.type == "cpu" and not policy.training
    g = torch.Generator().manual_seed(0)
    batch = {
        "observation.state": torch.rand(2, 2, 8, generator=g) * 2 - 1,
        IMAGES[0]: torch.rand(2, 2, 3, 256, 256, generator=g),
        IMAGES[1]: torch.rand(2, 2, 3, 256, 256, generator=g),
        "action": torch.rand(2, 16, 7, generator=g) * 2 - 1,
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }
    batch["action_is_pad"][1, 12:] = True
    torch.manual_seed(2)
    expected, _ = policy(dict(batch))
    torch.manual_seed(2)
    loss, _ = epsilon_loss_fn(LeRobotDenoiser(policy), batch)
    assert torch.allclose(loss, expected, atol=1e-6)
    broken = tmp_path / "bundle"
    broken.mkdir()
    for name in ("manifest.json", "config.json"):
        (broken / name).write_bytes((CHECKPOINT / name).read_bytes())
    (broken / "model.safetensors").write_bytes(b"not the model")
    with pytest.raises(ValueError):
        load_prepared_diffusion(broken)
