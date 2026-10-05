"""The learner ACT must reproduce the pinned LeRobot ACT module when it holds the same weights."""

import copy

import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTSinusoidalPositionEmbedding2d

from vla_lab.teaching.act_model import (
    ACTHooks,
    ACTModel,
    ACTSettings,
    act_from_lerobot,
    build_encoder_tokens,
    camera_position_embedding_2d,
    convert_lerobot_act_state_dict,
    decode_actions,
    encode_posterior,
    lerobot_batch,
    load_lerobot_act,
    make_act_backbone,
    select_latent,
    verify_act_parity,
)
from vla_lab.teaching.training import act_loss_fn, train_loop

IMAGES = ("observation.images.image", "observation.images.image2")
B, T, SIZE = 3, 5, 64


def lerobot_config(**changes):
    values = {
        "input_features": {
            IMAGES[0]: PolicyFeature(FeatureType.VISUAL, (3, SIZE, SIZE)),
            IMAGES[1]: PolicyFeature(FeatureType.VISUAL, (3, SIZE, SIZE)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (7,))},
        "chunk_size": T,
        "n_action_steps": T,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 2,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 2,
        "latent_dim": 6,
        "pretrained_backbone_weights": None,
        "push_to_hub": False,
        "device": "cpu",
    }
    values.update(changes)
    return ACTConfig(**values)


def make_batch(seed=0):
    generator = torch.Generator().manual_seed(seed)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[1, 3:] = True
    pad[2, 1:] = True
    return {
        "observation.state": torch.randn(B, 8, generator=generator),
        IMAGES[0]: torch.rand(B, 3, SIZE, SIZE, generator=generator),
        IMAGES[1]: torch.rand(B, 3, SIZE, SIZE, generator=generator),
        "action": torch.randn(B, T, 7, generator=generator),
        "action_is_pad": pad,
    }


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(11)
    model = ACT(lerobot_config())
    # Random LeRobot weights are close to symmetric; make every parameter distinct.
    for parameter in model.parameters():
        if parameter.dim() == 1:
            torch.nn.init.normal_(parameter, std=0.1)
    return model


@pytest.fixture(scope="module")
def learner(reference):
    return act_from_lerobot(reference)


def test_camera_position_embedding_matches_lerobot_for_several_shapes():
    for height, width, dim in ((2, 2, 32), (3, 5, 32), (15, 20, 64)):
        expected = ACTSinusoidalPositionEmbedding2d(dim // 2)(torch.zeros(1, 4, height, width))
        assert torch.allclose(camera_position_embedding_2d(height, width, dim), expected, atol=1e-6)


def test_converter_uses_every_tensor_once_and_reports_unknown_names(reference):
    converted = convert_lerobot_act_state_dict(reference.state_dict())
    packed = [k for k in reference.state_dict() if "in_proj" in k]
    assert len(converted) == len(reference.state_dict()) + 2 * len(packed)
    assert not any("in_proj" in key or "multihead_attn" in key for key in converted)
    with pytest.raises(KeyError):
        convert_lerobot_act_state_dict({"surprise.weight": torch.zeros(1)})


def test_load_is_strict(reference):
    settings = ACTSettings.from_lerobot(reference.config)
    bigger = ACTSettings(**{**settings.__dict__, "dim_model": settings.dim_model * 2})
    other = ACTModel.build(bigger)
    with pytest.raises((RuntimeError, ValueError)):
        load_lerobot_act(other, reference)


def test_identical_weights_give_identical_outputs(reference, learner):
    diffs = verify_act_parity(learner, reference, make_batch())
    assert diffs["inference_actions"] < 1e-4
    assert diffs["training_actions"] < 1e-4
    assert diffs["mu"] < 1e-5 and diffs["logvar"] < 1e-5


def test_verify_does_not_change_the_models(reference, learner):
    before = copy.deepcopy(learner.state_dict())
    verify_act_parity(learner, reference, make_batch())
    assert learner.training
    assert all(torch.equal(before[k], v) for k, v in learner.state_dict().items())
    assert all(
        m.p == learner.settings.dropout
        for m in learner.modules()
        if isinstance(m, torch.nn.Dropout)
    )


def test_inference_uses_zero_latent_and_returns_no_posterior(learner):
    learner.eval()
    batch = make_batch()
    actions, (mu, logvar) = learner(batch)
    assert actions.shape == (B, T, 7) and mu is None and logvar is None
    latent, _, _ = select_latent(learner, batch)
    assert not latent.any()
    learner.train()


def test_posterior_ignores_padded_actions(learner):
    batch = make_batch()
    changed = batch["action"].clone()
    changed[batch["action_is_pad"]] = 50.0
    learner.eval()
    mu, logvar = encode_posterior(
        learner, batch["observation.state"], batch["action"], batch["action_is_pad"]
    )
    mu2, logvar2 = encode_posterior(
        learner, batch["observation.state"], changed, batch["action_is_pad"]
    )
    assert torch.allclose(mu, mu2, atol=1e-5) and torch.allclose(logvar, logvar2, atol=1e-5)
    mu3, _ = encode_posterior(
        learner, batch["observation.state"], changed, torch.zeros_like(batch["action_is_pad"])
    )
    assert not torch.allclose(mu, mu3, atol=1e-3)
    learner.train()


def test_posterior_fully_padded_tail_keeps_gradient_finite_and_zero_for_padded_actions(learner):
    model = copy.deepcopy(learner).train()
    batch = make_batch()
    batch["action"].requires_grad_(True)
    batch["action_is_pad"][:, 2:] = True
    loss, _ = act_loss_fn(model, batch)
    loss.backward()
    assert torch.isfinite(batch["action"].grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_encoder_tokens_shape_order_and_positions(learner):
    batch = make_batch()
    maps = [learner.backbone(batch[key])["feature_map"] for key in IMAGES]
    latent = torch.randn(B, 6)
    tokens, pos = build_encoder_tokens(learner, latent, batch["observation.state"], maps)
    cells = sum(m.shape[-2] * m.shape[-1] for m in maps)
    assert tokens.shape == (B, 2 + cells, 32) and pos.shape == (1, 2 + cells, 32)
    assert torch.allclose(tokens[:, 0], learner.latent_proj(latent))
    assert torch.allclose(pos[0, :2], learner.pos_1d.weight)


def test_encoder_camera_positions_follow_feature_device():
    from types import SimpleNamespace

    model = SimpleNamespace(
        settings=SimpleNamespace(dim_model=32),
        latent_proj=torch.nn.Linear(6, 32, device="meta"),
        state_proj=torch.nn.Linear(8, 32, device="meta"),
        camera_proj=torch.nn.Conv2d(4, 32, kernel_size=1, device="meta"),
        pos_1d=torch.nn.Embedding(2, 32, device="meta"),
        hooks=ACTHooks(),
    )
    latent = torch.empty(2, 6, device="meta")
    state = torch.empty(2, 8, device="meta")
    feature_maps = [torch.empty(2, 4, 2, 3, device="meta")]

    tokens, positions = build_encoder_tokens(model, latent, state, feature_maps)

    assert tokens.device.type == "meta"
    assert positions.device.type == "meta"
    assert tokens.shape == (2, 8, 32)
    assert positions.shape == (1, 8, 32)


def test_decode_actions_depends_on_query_rows_and_encoder_output(learner):
    encoder_out = torch.randn(B, 9, 32)
    pos = torch.randn(1, 9, 32)
    out = decode_actions(learner, encoder_out, pos)
    assert out.shape == (B, T, 7)
    other = decode_actions(learner, encoder_out + 1.0, pos)
    assert not torch.allclose(out, other)


def test_forward_is_deterministic_in_eval_and_differs_across_latent_noise_in_train(learner):
    batch = make_batch()
    model = copy.deepcopy(learner)
    model.eval()
    assert torch.equal(model(batch)[0], model(batch)[0])
    model.train()
    a, _ = model(batch, noise=torch.zeros(B, 6))
    b, _ = model(batch, noise=torch.ones(B, 6))
    assert not torch.allclose(a, b)


def test_hooks_are_called_by_the_real_forward_path(learner):
    calls = []

    def positions(h, w, d):
        calls.append("positions")
        return camera_position_embedding_2d(h, w, d)

    def posterior(*args):
        calls.append("posterior")
        return encode_posterior(*args)

    def decode(*args):
        calls.append("decode")
        return decode_actions(*args)

    model = copy.deepcopy(learner)
    model.hooks = ACTHooks(camera_positions=positions, posterior=posterior, decode=decode)
    model.train()
    model(make_batch())
    assert calls == ["posterior", "positions", "positions", "decode"]


def test_loss_callback_updates_the_real_assembly(learner):
    model = copy.deepcopy(learner)
    before = model.action_head.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    history = train_loop(model, [make_batch()], optimizer, act_loss_fn, steps=3)
    assert len(history) == 3 and all(torch.isfinite(torch.tensor(h["loss"])) for h in history)
    assert not torch.equal(before, model.action_head.weight)
    assert history[-1]["loss"] < history[0]["loss"]


def test_backbone_is_the_pinned_construction(reference):
    backbone, channels = make_act_backbone("resnet18")
    assert channels == 512
    assert set(backbone.state_dict()) == set(reference.backbone.state_dict())
    out = backbone(torch.zeros(1, 3, SIZE, SIZE))["feature_map"]
    assert out.shape == (1, 512, SIZE // 32, SIZE // 32)


def test_lerobot_batch_adds_image_list(learner):
    batch = lerobot_batch(make_batch(), learner.settings)
    assert len(batch["observation.images"]) == 2


CHECKPOINT = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / ".local-data/runs/act-bowl-plate-probe-200/checkpoint-000200"
)


@pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(), reason="local ACT checkpoint missing"
)
def test_prepared_checkpoint_weights_reproduce_in_the_learner():
    from lerobot.policies.act import ACTPolicy

    config = ACTConfig.from_pretrained(CHECKPOINT, local_files_only=True)
    config.device = "cpu"
    policy = ACTPolicy.from_pretrained(
        CHECKPOINT, config=config, strict=True, local_files_only=True
    )
    learner = act_from_lerobot(policy.model)
    generator = torch.Generator().manual_seed(3)
    chunk = config.chunk_size
    batch = {
        "observation.state": torch.randn(2, 8, generator=generator),
        IMAGES[0]: torch.rand(2, 3, 256, 256, generator=generator),
        IMAGES[1]: torch.rand(2, 3, 256, 256, generator=generator),
        "action": torch.randn(2, chunk, 7, generator=generator),
        "action_is_pad": torch.zeros(2, chunk, dtype=torch.bool),
    }
    diffs = verify_act_parity(learner, policy.model, batch)
    assert max(diffs.values()) < 5e-4, diffs


@pytest.mark.skipif(
    not (CHECKPOINT.parents[2] / "hub/lerobot-libero/meta/episodes").exists(),
    reason="local LIBERO data missing",
)
def test_real_samples_have_the_training_contract_and_feed_the_learner():
    from vla_lab.teaching.act_model import load_act_samples

    root = CHECKPOINT.parents[3]
    samples = load_act_samples(
        root / "configs",
        root / ".local-data",
        "bowl_plate",
        "validation",
        count=4,
        chunk_size=20,
        image_size=64,
    )
    assert samples["observation.images.image"].shape == (4, 3, 64, 64)
    assert samples["action"].shape == (4, 20, 7) and samples["action_is_pad"].dtype == torch.bool
    assert (
        samples["observation.images.image"].min() >= 0
        and samples["observation.images.image"].max() <= 1
    )
    model = ACTModel.build(
        ACTSettings(
            state_dim=8,
            action_dim=7,
            image_keys=IMAGES,
            chunk_size=20,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            latent_dim=6,
            n_encoder_layers=1,
            n_vae_encoder_layers=1,
        )
    )
    loss, metrics = act_loss_fn(model, samples)
    assert torch.isfinite(loss) and set(metrics) == {"action_l1", "kl"}
