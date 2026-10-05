"""Check SmolVLA input preparation and flow matching on a small real SmolVLA module.

The small model is the pinned ``SmolVLAPolicy`` class with a tiny VLM configuration and the
real tokenizer files copied from the local SmolVLM2 directory. Nothing is downloaded.
"""

import copy
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy

from vla_lab.teaching.smolvla import (
    CallbackSmolVLAPolicy,
    SmolVLAAdapter,
    euler_sample,
    flow_loss_fn,
    flow_training_pair,
    masked_velocity_loss,
    pad_to_width,
    prepare_vla_inputs,
    resize_pad_normalize_image,
    select_trainable_parameters,
    tokenize_prompts,
    train_mode_with_frozen_vlm,
)
from vla_lab.teaching.training import optimizer_update

ROOT = Path(__file__).resolve().parents[1]
VLM_DIR = ROOT / ".local-data/models/smolvlm2-500m-video-instruct"

IMAGES = ("observation.images.image", "observation.images.image2")
B, T, D, ACT_DIM, SIZE = 2, 6, 16, 7, 64


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    if not (VLM_DIR / "tokenizer.json").is_file():
        pytest.skip("local SmolVLM2 tokenizer files are missing")
    target = tmp_path_factory.mktemp("tiny_vlm")
    for path in VLM_DIR.iterdir():
        if path.is_file() and path.name != "model.safetensors":
            shutil.copy(path, target / path.name)
    config = json.loads((VLM_DIR / "config.json").read_text())
    config["text_config"].update(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
    )
    config["vision_config"].update(
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=64,
        image_size=SIZE,
        patch_size=16,
        max_image_size={"longest_edge": SIZE},
        size={"longest_edge": SIZE},
    )
    config["scale_factor"] = 2
    (target / "config.json").write_text(json.dumps(config))
    return target


def make_policy(tiny_dir):
    config = SmolVLAConfig(
        device="cpu",
        push_to_hub=False,
        vlm_model_name=str(tiny_dir),
        load_vlm_weights=False,
        input_features={
            IMAGES[0]: PolicyFeature(FeatureType.VISUAL, (3, SIZE, SIZE)),
            IMAGES[1]: PolicyFeature(FeatureType.VISUAL, (3, SIZE, SIZE)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
        chunk_size=T,
        n_action_steps=3,
        num_vlm_layers=2,
        resize_imgs_with_padding=(SIZE, SIZE),
        max_state_dim=D,
        max_action_dim=D,
        prefix_length=0,
        num_steps=4,
        expert_width_multiplier=0.5,
        tokenizer_max_length=12,
    )
    torch.manual_seed(1)
    return SmolVLAPolicy(config)


@pytest.fixture()
def policy(tiny_dir):
    return make_policy(tiny_dir)


@pytest.fixture(scope="module")
def tokenizer():
    if not (VLM_DIR / "tokenizer.json").is_file():
        pytest.skip("local SmolVLM2 tokenizer files are missing")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(VLM_DIR))


def preprocessed_batch(tokenizer, with_actions=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens, mask = tokenize_prompts(tokenizer, ["put the bowl on the plate", "open it"], 12)
    batch = {
        IMAGES[0]: torch.rand(B, 3, SIZE, SIZE, generator=g),
        IMAGES[1]: torch.rand(B, 3, SIZE, SIZE, generator=g),
        "observation.state": torch.randn(B, 8, generator=g),
        "observation.language.tokens": tokens,
        "observation.language.attention_mask": mask,
    }
    if with_actions:
        pad = torch.zeros(B, T, dtype=torch.bool)
        pad[1, 4:] = True
        batch["action"] = torch.randn(B, T, ACT_DIM, generator=g)
        batch["action_is_pad"] = pad
    return batch


# --- 12A -----------------------------------------------------------------------------------------


def test_image_preparation_matches_the_pinned_policy_for_wide_square_and_tall_images(policy):
    for h, w in ((SIZE, SIZE), (48, 64), (64, 32), (128, 96)):
        image = torch.rand(2, 3, h, w)
        mine = resize_pad_normalize_image(image, SIZE, SIZE)
        expected, _ = policy.prepare_images({IMAGES[0]: image, IMAGES[1]: image})
        assert mine.shape == (2, 3, SIZE, SIZE)
        assert torch.allclose(mine, expected[0], atol=1e-6)
    wide = resize_pad_normalize_image(torch.ones(1, 3, 32, 64), SIZE, SIZE)
    assert (wide[:, :, :32] == -1).all() and (wide[:, :, 32:] == 1).all()  # padding is on the top


def test_pad_to_width_pads_with_zeros_and_rejects_wider_input():
    out = pad_to_width(torch.ones(2, 3), 5)
    assert out.shape == (2, 5) and out[:, :3].eq(1).all() and out[:, 3:].eq(0).all()
    assert torch.equal(pad_to_width(torch.ones(2, 5), 5), torch.ones(2, 5))
    with pytest.raises(ValueError):
        pad_to_width(torch.ones(2, 6), 5)


def test_tokenizer_output_matches_the_pinned_processor_and_padding_is_masked(tokenizer):
    from lerobot.processor import NewLineTaskProcessorStep, TokenizerProcessorStep

    tokens, mask = tokenize_prompts(tokenizer, ["put the bowl on the plate", "open it\n"], 48)
    assert tokens.dtype == torch.long and mask.dtype == torch.bool and tokens.shape == mask.shape
    assert mask[0].all() and mask[1].sum() < mask[0].sum()  # the short prompt is right padded
    assert not mask[1][int(mask[1].sum()) :].any()
    single, single_mask = tokenize_prompts(tokenizer, "put the bowl on the plate", 48)
    assert torch.equal(single[0], tokens[0]) and single_mask.all()
    step = TokenizerProcessorStep(
        tokenizer=tokenizer, max_length=48, padding="longest", padding_side="right"
    )
    newline = NewLineTaskProcessorStep()
    pipeline_task = newline.complementary_data({"task": "put the bowl on the plate"})["task"]
    assert pipeline_task == "put the bowl on the plate\n"
    expected = step._tokenize_text([pipeline_task])
    assert torch.equal(single, expected["input_ids"])
    long_tokens, _ = tokenize_prompts(tokenizer, "word " * 100, 12)
    assert long_tokens.shape == (1, 12)


def test_prepare_vla_inputs_equals_the_pinned_preparation(policy, tokenizer):
    batch = preprocessed_batch(tokenizer, with_actions=False)
    adapter = SmolVLAAdapter(policy)
    expected = adapter.prepare_inputs(batch)
    observation = {k: v for k, v in batch.items() if "language" not in k}
    mine = prepare_vla_inputs(
        observation, ["put the bowl on the plate", "open it"], tokenizer, policy.config
    )
    assert len(mine["images"]) == 2 and len(mine["img_masks"]) == 2
    for a, b in zip(mine["images"], expected["images"], strict=True):
        assert torch.allclose(a, b, atol=1e-6)
    assert all(m.dtype == torch.bool and m.all() for m in mine["img_masks"])
    assert torch.equal(mine["lang_tokens"], expected["lang_tokens"])
    assert torch.equal(mine["lang_masks"], expected["lang_masks"])
    assert torch.allclose(mine["state"], expected["state"]) and mine["state"].shape == (B, D)
    one = prepare_vla_inputs(observation, "put the bowl on the plate", tokenizer, policy.config)
    assert one["lang_tokens"].shape[0] == B


def test_select_trainable_parameters_freezes_the_vlm_only(policy):
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    trainable = select_trainable_parameters(policy)
    assert len(trainable) == len({id(p) for p in trainable})
    names = {id(p): n for n, p in policy.named_parameters()}
    assert not any(names[id(p)].startswith("model.vlm_with_expert.vlm.") for p in trainable)
    assert all(
        not p.requires_grad
        for n, p in policy.named_parameters()
        if n.startswith("model.vlm_with_expert.vlm.")
    )
    assert any(
        n.startswith("model.vlm_with_expert.lm_expert.") for n in (names[id(p)] for p in trainable)
    )
    assert any(n.startswith("model.action_out_proj") for n in (names[id(p)] for p in trainable))
    policy.model.state_proj.requires_grad_(False)  # a deliberate freeze must be kept
    assert not any(
        names[id(p)].startswith("model.state_proj") for p in select_trainable_parameters(policy)
    )


def test_train_mode_keeps_the_frozen_vlm_in_eval_mode(policy):
    policy.model.vlm_with_expert.vlm.train()  # something switched the backbone on
    train_mode_with_frozen_vlm(policy)
    assert policy.training and policy.model.action_out_proj.training
    assert not any(m.training for m in policy.model.vlm_with_expert.vlm.modules())
    assert policy.model.vlm_with_expert.lm_expert.training


# --- 12B -----------------------------------------------------------------------------------------


def test_flow_pair_direction_and_endpoints():
    actions, noise = torch.randn(3, T, D), torch.randn(3, T, D)
    x0, u = flow_training_pair(actions, noise, torch.zeros(3))
    assert torch.equal(x0, actions) and torch.equal(u, noise - actions)
    x1, _ = flow_training_pair(actions, noise, torch.ones(3))
    assert torch.allclose(x1, noise)
    half, _ = flow_training_pair(actions, noise, torch.tensor([0.25, 0.5, 0.75]))
    assert torch.allclose(half[0], 0.25 * noise[0] + 0.75 * actions[0])


def test_masked_velocity_loss_ignores_padded_steps_and_padded_columns_with_zero_gradient():
    pred = torch.zeros(2, 3, 5, requires_grad=True)
    target = torch.ones(2, 3, 5)
    pad = torch.tensor([[False, False, True], [False, True, True]])
    loss = masked_velocity_loss(pred, target, pad, 3)
    assert torch.allclose(loss, torch.tensor(1.0))
    loss.backward()
    assert torch.count_nonzero(pred.grad[pad]) == 0 and torch.count_nonzero(pred.grad[..., 3:]) == 0
    other = pred.detach().clone()
    other[..., 3:] = 50.0
    other[pad] = 50.0
    assert torch.allclose(masked_velocity_loss(other, target, pad, 3), loss.detach())
    assert masked_velocity_loss(pred.detach(), target, torch.ones_like(pad), 3) == 0


def test_euler_matches_the_pinned_integrator_and_follows_time_from_one_to_zero():
    noise = torch.randn(2, T, D)
    field = torch.randn(2, T, D)
    seen = []

    def velocity(x, t):
        seen.append(t.clone())
        return field + 0.1 * x

    mine = euler_sample(velocity, noise, 5)
    assert [round(float(t[0]), 6) for t in seen] == [1.0, 0.8, 0.6, 0.4, 0.2]
    assert all(t.dtype == torch.float32 and t.shape == (2,) for t in seen)
    assert torch.allclose(mine, euler_integrate(lambda x, t: field + 0.1 * x, noise, 5), atol=1e-6)
    actions = torch.randn(2, T, D)
    exact = euler_sample(lambda x, t: noise - actions, noise, 7)  # the straight-line field is exact
    assert torch.allclose(exact, actions, atol=1e-5)


def test_velocity_adapter_reproduces_the_pinned_forward_losses(policy, tokenizer):
    batch = preprocessed_batch(tokenizer)
    adapter = SmolVLAAdapter(policy)
    inputs = adapter.prepare_inputs(batch)
    noise = torch.randn_like(inputs["actions"])
    time = torch.tensor([0.2, 0.9])
    x_t, target = flow_training_pair(inputs["actions"], noise, time)
    velocity = adapter.velocity(inputs, x_t, time)
    assert velocity.shape == (B, T, D)
    ours = (target - velocity).square()
    assert torch.allclose(ours, adapter.pinned_per_element_loss(inputs, noise, time), atol=1e-6)


def test_flow_loss_fn_equals_pinned_policy_loss_for_same_noise_and_time_and_keeps_gradient(
    policy, tokenizer
):
    batch = preprocessed_batch(tokenizer)
    adapter = SmolVLAAdapter(policy)
    noise, time = torch.randn(B, T, D), torch.tensor([0.3, 0.6])
    expected, _ = policy(copy.deepcopy(batch), noise=noise, time=time)
    loss, metrics = flow_loss_fn(adapter, batch, noise=noise, time=time)
    assert torch.allclose(loss, expected, atol=1e-6) and torch.allclose(metrics["flow_mse"], loss)
    loss.backward()
    graded = [
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.model.vlm_with_expert.lm_expert.parameters()
    ]
    assert any(graded)
    assert all(p.grad is None for p in policy.model.vlm_with_expert.vlm.parameters())
    with pytest.raises(ValueError):
        flow_loss_fn(adapter, batch, noise=torch.randn(B, T, ACT_DIM), time=time)


def test_all_padding_gives_zero_loss_and_finite_gradients(policy, tokenizer):
    batch = preprocessed_batch(tokenizer)
    batch["action_is_pad"] = torch.ones(B, T, dtype=torch.bool)
    loss, _ = flow_loss_fn(SmolVLAAdapter(policy), batch)
    loss.backward()
    assert loss == 0 and all(
        torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None
    )


def test_update_with_the_callback_changes_the_expert_and_keeps_the_vlm_fixed(policy, tokenizer):
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    trainable = select_trainable_parameters(policy)
    calls = []

    def train_mode(model):
        calls.append(1)
        return train_mode_with_frozen_vlm(model)

    adapter = SmolVLAAdapter(policy, train_mode_fn=train_mode)
    vlm_before = {
        n: p.detach().clone() for n, p in policy.model.vlm_with_expert.vlm.named_parameters()
    }
    expert_before = policy.model.action_out_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    batch = preprocessed_batch(tokenizer)
    noise, time = torch.randn(B, T, D), torch.tensor([0.4, 0.7])
    first = optimizer_update(adapter, batch, optimizer, flow_loss_fn, noise=noise, time=time)
    for _ in range(4):
        last = optimizer_update(adapter, batch, optimizer, flow_loss_fn, noise=noise, time=time)
    assert last["loss"] < first["loss"]
    assert not torch.equal(expert_before, policy.model.action_out_proj.weight)
    assert all(
        torch.equal(vlm_before[n], p)
        for n, p in policy.model.vlm_with_expert.vlm.named_parameters()
    )
    assert len(calls) == 5  # optimizer_update calls model.train() once per update
    assert not any(m.training for m in policy.model.vlm_with_expert.vlm.modules())


def test_sample_chunk_with_the_learner_sampler_matches_pinned_sample_actions(policy, tokenizer):
    policy.eval()
    batch = preprocessed_batch(tokenizer, with_actions=False)
    adapter = SmolVLAAdapter(policy)
    inputs = adapter.prepare_inputs(batch)
    noise = torch.randn(B, T, D)
    mine = adapter.sample_chunk(inputs, euler_sample, noise, 4)
    with torch.no_grad():
        expected = policy.model.sample_actions(
            inputs["images"],
            inputs["img_masks"],
            inputs["lang_tokens"],
            inputs["lang_masks"],
            inputs["state"],
            noise=noise,
        )[:, :, :ACT_DIM]
    assert mine.shape == (B, T, ACT_DIM) and torch.allclose(mine, expected, atol=1e-5)


def test_callback_sampling_treats_none_action_like_an_omitted_action(policy, tokenizer):
    batch = preprocessed_batch(tokenizer, with_actions=False)
    batch_with_none_action = {**batch, "action": None}
    without_action = CallbackSmolVLAPolicy(policy, num_steps=2, seed=14).select_action(batch)
    with_none_action = CallbackSmolVLAPolicy(policy, num_steps=2, seed=14).select_action(
        batch_with_none_action
    )

    assert torch.allclose(with_none_action, without_action, atol=1e-6)


def test_callback_policy_calls_learner_input_and_sampler_hooks_and_queues_actions(
    policy, tokenizer
):
    seen = {"sampler": 0, "input": 0}

    def sampler(velocity_fn, noise, steps):
        seen["sampler"] += 1
        return euler_sample(velocity_fn, noise, steps)

    def input_fn(adapter, batch):
        seen["input"] += 1
        return adapter.prepare_inputs(batch)

    wrapper = CallbackSmolVLAPolicy(policy, sampler, input_fn, num_steps=2, seed=5)
    batch = preprocessed_batch(tokenizer, with_actions=False)
    actions = [wrapper.select_action(batch) for _ in range(4)]
    assert all(a.shape == (B, ACT_DIM) for a in actions)
    assert seen == {"sampler": 2, "input": 2}  # n_action_steps = 3: calls 1 and 4 sample a chunk
    wrapper.reset()
    assert wrapper.chunks_made == 0 and len(wrapper._queue) == 0
    wrapper.select_action(batch)
    assert wrapper.chunks_made == 1 and len(wrapper._queue) == 2


def test_noise_seed_gives_repeatable_actions(policy, tokenizer):
    batch = preprocessed_batch(tokenizer, with_actions=False)
    a = CallbackSmolVLAPolicy(policy, num_steps=3, seed=9).select_action(batch)
    b = CallbackSmolVLAPolicy(policy, num_steps=3, seed=9).select_action(batch)
    c = CallbackSmolVLAPolicy(policy, num_steps=3, seed=10).select_action(batch)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_input_hook_replaces_the_pinned_input_path_in_the_loss_and_the_sampler(policy, tokenizer):
    batch = preprocessed_batch(tokenizer)
    calls = []

    def input_fn(adapter, raw):
        calls.append(1)
        observation = {
            k: v
            for k, v in raw.items()
            if "language" not in k and k not in ("action", "action_is_pad")
        }
        inputs = prepare_vla_inputs(
            observation, ["put the bowl on the plate", "open it"], tokenizer, policy.config
        )
        if "action" in raw:
            inputs["actions"] = pad_to_width(raw["action"], policy.config.max_action_dim)
        return inputs

    pinned = SmolVLAAdapter(policy)
    hooked = SmolVLAAdapter(policy, input_fn=input_fn)
    noise, time = torch.randn(B, T, D), torch.tensor([0.3, 0.6])
    expected, _ = flow_loss_fn(pinned, batch, noise=noise, time=time)
    loss, _ = flow_loss_fn(hooked, batch, noise=noise, time=time)
    assert torch.allclose(loss, expected, atol=1e-6) and calls == [1]
    wrapper = CallbackSmolVLAPolicy(policy, input_fn=input_fn, num_steps=2)
    wrapper.select_action({k: v for k, v in batch.items() if k not in ("action", "action_is_pad")})
    assert len(calls) == 2
