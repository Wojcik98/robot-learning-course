"""An Action Chunking Transformer assembled from the hand-written blocks.

Contract (same as the raw pinned LeRobot ``ACT`` module, not its policy wrapper):
``model(batch) -> (actions (B, T, A), (mu, logvar))``. In training mode with
``batch["action"]`` present, ``mu`` and ``logvar`` are ``(B, L)`` posterior parameters.
Otherwise the latent is zero and both are ``None``. The model never computes a loss;
callers use ``vla_lab.teaching.training.act_loss_fn``.

Batch keys (all already normalized): ``observation.state (B, S)``, one ``(B, C, H, W)``
image per entry of ``settings.image_keys``, and for training ``action (B, T, A)`` and
``action_is_pad (B, T)`` (True for padding).

Three kinds of weights must stay distinct:

* *reference weights in the learner*: ``load_lerobot_act`` copies LeRobot weights here;
* *learner trained weights*: weights produced by your own updates;
* *prepared policy results*: numbers saved from the LeRobot policy, not from this module.

The hooks in ``ACTHooks`` are the five functions of course unit 8B. Each default is the
reference answer. A learner function with the same signature can replace it.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from vla_lab.teaching.attention import (
    MultiHeadAttention,
    TransformerDecoder,
    TransformerEncoder,
    sinusoidal_positions,
)
from vla_lab.teaching.cvae import reparameterize

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
ACTION_KEY = "action"
PAD_KEY = "action_is_pad"


@dataclass(frozen=True)
class ACTSettings:
    """Sizes of the model. ``from_lerobot`` copies them from a pinned ``ACTConfig``."""

    state_dim: int
    action_dim: int
    image_keys: tuple[str, ...]
    chunk_size: int = 20
    dim_model: int = 256
    n_heads: int = 4
    dim_feedforward: int = 1024
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 2
    latent_dim: int = 32
    dropout: float = 0.1
    pre_norm: bool = False
    activation: str = "relu"
    vision_backbone: str = "resnet18"

    @classmethod
    def from_lerobot(cls, config) -> ACTSettings:
        if not config.use_vae:
            raise ValueError("the learner ACT always has a posterior encoder (use_vae=True)")
        if config.env_state_feature is not None or config.robot_state_feature is None:
            raise ValueError("the learner ACT needs a robot state and no environment state")
        return cls(
            state_dim=config.robot_state_feature.shape[0],
            action_dim=config.action_feature.shape[0],
            image_keys=tuple(config.image_features),
            chunk_size=config.chunk_size,
            dim_model=config.dim_model,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            n_vae_encoder_layers=config.n_vae_encoder_layers,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
            pre_norm=config.pre_norm,
            activation=config.feedforward_activation,
            vision_backbone=config.vision_backbone,
        )


# --- Supplied CNN backbone ---------------------------------------------------------------------


def make_act_backbone(
    name: str = "resnet18",
    *,
    weights: str | None = None,
    dilate_last_stage: bool = False,
) -> tuple[nn.Module, int]:
    """Return ``(backbone, channels)``. ``backbone(image)["feature_map"]`` is ``(B, C, h, w)``.

    This is the supplied CNN, built exactly as the pinned ACT builds it (frozen batch norm,
    ``layer4`` output) so its weights load without any renaming. ``weights=None`` gives random
    weights; a torchvision weight name needs its file in the local torch cache (no download is
    started here unless torchvision itself decides to fetch it, so prefer ``None``).
    """
    model = getattr(torchvision.models, name)(
        replace_stride_with_dilation=[False, False, dilate_last_stage],
        weights=weights,
        norm_layer=FrozenBatchNorm2d,
    )
    return IntermediateLayerGetter(
        model, return_layers={"layer4": "feature_map"}
    ), model.fc.in_features


# --- Unit 8B reference functions ----------------------------------------------------------------


def camera_position_embedding_2d(height: int, width: int, dim: int) -> Tensor:
    """2D sinusoidal position map of shape ``(1, dim, height, width)`` (ACT, after DETR).

    Half of the channels encode the row (first) and half the column (second). With
    ``d = dim // 2`` per axis: ``index`` runs ``1..H`` (or ``1..W``), is scaled to
    ``index / (last + 1e-6) * 2*pi``, then divided by ``10000 ** (2 * (j // 2) / d)`` for
    channel ``j``. Even channels use sin and odd channels use cos, interleaved.
    """
    per_axis = dim // 2
    rows = torch.arange(1, height + 1, dtype=torch.float32)
    columns = torch.arange(1, width + 1, dtype=torch.float32)
    rows = rows / (rows[-1] + 1e-6) * 2 * math.pi
    columns = columns / (columns[-1] + 1e-6) * 2 * math.pi
    inverse_frequency = 10000 ** (2 * (torch.arange(per_axis, dtype=torch.float32) // 2) / per_axis)

    def encode(positions: Tensor) -> Tensor:
        angles = positions[:, None] / inverse_frequency
        return torch.stack((angles[:, 0::2].sin(), angles[:, 1::2].cos()), dim=-1).flatten(1)

    row_code = encode(rows)[:, None, :].expand(height, width, per_axis)
    column_code = encode(columns)[None, :, :].expand(height, width, per_axis)
    return torch.cat((row_code, column_code), dim=-1).permute(2, 0, 1).unsqueeze(0)


def encode_posterior(
    model: nn.Module, state: Tensor, actions: Tensor, action_is_pad: Tensor
) -> tuple[Tensor, Tensor]:
    """BERT-style posterior ``q(z | state, actions)``. Returns ``(mu, logvar)``, each ``(B, L)``.

    Tokens are ``[cls, state, action_1 .. action_T]``, width ``D``. Add the fixed table
    ``model.vae_pos_embed (1, T + 2, D)`` to the query and key (the encoder does this when you
    pass it as ``pos_embed``). Mask padded actions: the key padding mask is
    ``[False, False, *action_is_pad]``. Read the output of the cls token (position 0), apply
    ``model.vae_latent_proj`` and split its ``2 * L`` outputs into ``mu`` then ``logvar``.
    Modules: ``vae_cls_embed`` (Embedding with 1 row), ``vae_state_proj``, ``vae_action_proj``,
    ``vae_encoder``, ``vae_latent_proj``. ``logvar`` is ``log(sigma ** 2)``.
    """
    batch = state.shape[0]
    cls_token = model.vae_cls_embed.weight[None].expand(batch, -1, -1)
    tokens = torch.cat(
        [cls_token, model.vae_state_proj(state)[:, None], model.vae_action_proj(actions)], dim=1
    )
    no_pad = torch.zeros(batch, 2, dtype=torch.bool, device=state.device)
    mask = torch.cat([no_pad, action_is_pad.to(torch.bool)], dim=1)
    cls_out = model.vae_encoder(tokens, model.vae_pos_embed, mask)[:, 0]
    mu, logvar = model.vae_latent_proj(cls_out).chunk(2, dim=-1)
    return mu, logvar


def select_latent(
    model: nn.Module, batch: Mapping[str, Tensor], noise: Tensor | None = None
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Choose the latent. Returns ``(latent (B, L), mu, logvar)``.

    Training with ``batch["action"]`` present: encode the posterior and sample
    ``mu + exp(logvar / 2) * noise`` (``noise`` defaults to ``randn_like(mu)``).
    Otherwise (``model.training`` is False, or there are no actions): the latent is all zeros
    on the state's device and dtype, and ``mu`` and ``logvar`` are ``None``.
    """
    state = batch["observation.state"]
    if model.training and "action" in batch:
        mu, logvar = model.hooks.posterior(model, state, batch["action"], batch["action_is_pad"])
        return reparameterize(mu, logvar, noise=noise), mu, logvar
    zeros = torch.zeros(
        state.shape[0], model.settings.latent_dim, dtype=state.dtype, device=state.device
    )
    return zeros, None, None


def build_encoder_tokens(
    model: nn.Module, latent: Tensor, state: Tensor, feature_maps: list[Tensor]
) -> tuple[Tensor, Tensor]:
    """Tokens and positions for the Transformer encoder.

    Returns ``tokens (B, N, D)`` and ``pos (1, N, D)`` with ``N = 2 + sum(h * w)``.
    Order: latent token, state token, then each camera in order, row by row.
    ``latent_proj(latent)`` and ``state_proj(state)`` use the two rows of the learned table
    ``model.pos_1d.weight (2, D)``. A feature map ``(B, C, h, w)`` goes through the 1x1
    convolution ``model.camera_proj``, is flattened to ``(B, h * w, D)``, and uses
    ``model.hooks.camera_positions(h, w, D)`` (``(1, D, h, w)``) flattened the same way.
    """
    tokens = [model.latent_proj(latent)[:, None], model.state_proj(state)[:, None]]
    positions = [model.pos_1d.weight[None]]
    for features in feature_maps:
        height, width = features.shape[-2:]
        tokens.append(model.camera_proj(features).flatten(2).transpose(1, 2))
        camera_pos = model.hooks.camera_positions(height, width, model.settings.dim_model)
        positions.append(
            camera_pos.to(device=features.device, dtype=features.dtype).flatten(2).transpose(1, 2)
        )
    return torch.cat(tokens, dim=1), torch.cat(positions, dim=1)


def decode_actions(model: nn.Module, encoder_out: Tensor, pos: Tensor) -> Tensor:
    """Action queries cross-attend to the encoder output. Returns ``(B, T, A)``.

    The decoder input is ``T = settings.chunk_size`` zero vectors of width ``D``. Their only
    information is ``query_pos = model.query_embed.weight (T, D)``, one learned row per future
    step. The encoder tokens' ``pos`` is the memory position. Finish with ``model.action_head``.
    """
    batch = encoder_out.shape[0]
    queries = torch.zeros(
        batch,
        model.settings.chunk_size,
        model.settings.dim_model,
        dtype=encoder_out.dtype,
        device=encoder_out.device,
    )
    out = model.decoder(queries, encoder_out, model.query_embed.weight[None], pos)
    return model.action_head(out)


@dataclass
class ACTHooks:
    """The five replaceable functions of unit 8B."""

    camera_positions: Callable[[int, int, int], Tensor] = camera_position_embedding_2d
    posterior: Callable[..., tuple[Tensor, Tensor]] = encode_posterior
    latent: Callable[..., tuple[Tensor, Tensor | None, Tensor | None]] = select_latent
    encoder_tokens: Callable[..., tuple[Tensor, Tensor]] = build_encoder_tokens
    decode: Callable[..., Tensor] = decode_actions


class ACTModel(nn.Module):
    """ACT with the pinned structure: posterior encoder, encoder, decoder, action head."""

    def __init__(
        self,
        settings: ACTSettings,
        backbone: nn.Module,
        backbone_channels: int,
        hooks: ACTHooks | None = None,
        **block_hooks: Callable,
    ):
        super().__init__()
        s = settings
        self.settings = s
        self.hooks = hooks or ACTHooks()
        shape = {
            "d_model": s.dim_model,
            "n_heads": s.n_heads,
            "d_ff": s.dim_feedforward,
            "dropout": s.dropout,
            "activation": s.activation,
            "pre_norm": s.pre_norm,
        }
        self.vae_encoder = TransformerEncoder(s.n_vae_encoder_layers, **shape, **block_hooks)
        self.vae_cls_embed = nn.Embedding(1, s.dim_model)
        self.vae_state_proj = nn.Linear(s.state_dim, s.dim_model)
        self.vae_action_proj = nn.Linear(s.action_dim, s.dim_model)
        self.vae_latent_proj = nn.Linear(s.dim_model, s.latent_dim * 2)
        self.register_buffer(
            "vae_pos_embed", sinusoidal_positions(s.chunk_size + 2, s.dim_model)[None]
        )
        self.backbone = backbone
        self.encoder = TransformerEncoder(s.n_encoder_layers, **shape, **block_hooks)
        self.decoder = TransformerDecoder(s.n_decoder_layers, **shape, **block_hooks)
        self.latent_proj = nn.Linear(s.latent_dim, s.dim_model)
        self.state_proj = nn.Linear(s.state_dim, s.dim_model)
        self.camera_proj = nn.Conv2d(backbone_channels, s.dim_model, kernel_size=1)
        self.pos_1d = nn.Embedding(2, s.dim_model)
        self.query_embed = nn.Embedding(s.chunk_size, s.dim_model)
        self.action_head = nn.Linear(s.dim_model, s.action_dim)

    @classmethod
    def build(
        cls,
        settings: ACTSettings,
        *,
        backbone_weights: str | None = None,
        hooks: ACTHooks | None = None,
        **block_hooks: Callable,
    ) -> ACTModel:
        backbone, channels = make_act_backbone(settings.vision_backbone, weights=backbone_weights)
        return cls(settings, backbone, channels, hooks, **block_hooks)

    def images(self, batch: Mapping[str, Tensor]) -> list[Tensor]:
        if "observation.images" in batch:  # raw LeRobot layout: a list of (B, C, H, W)
            return list(batch["observation.images"])
        return [batch[key] for key in self.settings.image_keys]

    def forward(
        self, batch: Mapping[str, Tensor], *, noise: Tensor | None = None
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        if self.training and ACTION_KEY in batch and PAD_KEY not in batch:
            raise KeyError("training batches need action_is_pad beside action")
        latent, mu, logvar = self.hooks.latent(self, batch, noise)
        feature_maps = [self.backbone(image)["feature_map"] for image in self.images(batch)]
        tokens, pos = self.hooks.encoder_tokens(self, latent, batch[STATE_KEY], feature_maps)
        encoded = self.encoder(tokens, pos)
        return self.hooks.decode(self, encoded, pos), (mu, logvar)


# --- Supplied real-data reader (images, state, action chunk) ------------------------------------


def open_libero_dataset(
    configs_root: str | Path,
    data_root: str | Path,
    task_alias: str,
    split: str,
    delta_timestamps: Mapping[str, list[float]],
    *,
    episodes: Sequence[int] | None = None,
):
    """Open whole episodes of the frozen ``train`` or ``validation`` split from local files.

    Never downloads (offline mode is forced). ``delta_timestamps`` are in seconds at the dataset's
    10 Hz; the 20 Hz simulator controller is a separate rate and nothing is resampled here.
    """
    import os

    import yaml

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    manifest = yaml.safe_load((Path(configs_root) / "data" / "episodes.yaml").read_text())
    ids = manifest["tasks"][task_alias][f"{split}_episode_ids"]
    if episodes is not None:
        if not set(episodes) <= set(ids):
            raise ValueError(f"episodes must come from the {split} split")
        ids = list(episodes)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        "lerobot/libero",
        root=Path(data_root) / "hub" / "lerobot-libero",
        episodes=list(ids),
        revision=manifest["dataset_revision"],
        delta_timestamps={key: list(value) for key, value in delta_timestamps.items()},
        video_backend="torchcodec",
    )


def load_act_samples(
    configs_root: str | Path,
    data_root: str | Path,
    task_alias: str,
    split: str,
    *,
    count: int,
    chunk_size: int = 20,
    image_size: int | None = None,
) -> dict[str, Tensor]:
    """Real ACT training samples, evenly spread over the split's frames.

    Each sample has both camera images ``(3, H, W)`` in ``[0, 1]`` (box-filtered to
    ``image_size`` when given), the 8-value state and the next ``chunk_size`` actions, both
    standardized with the *training* mean and standard deviation (as the pinned ACT recipe
    does), and ``action_is_pad`` for chunks that run past the episode end.
    """
    import json

    import torch.nn.functional as F

    from vla_lab.data.stats import TrainingStats

    dataset = open_libero_dataset(
        configs_root,
        data_root,
        task_alias,
        split,
        {"action": [i / 10 for i in range(chunk_size)]},
    )
    stats = json.loads((Path(configs_root) / "data" / "stats" / f"{task_alias}.json").read_text())[
        "modalities"
    ]
    scale = {key: TrainingStats.from_dict(stats[key]) for key in (STATE_KEY, ACTION_KEY)}
    picks = torch.linspace(0, len(dataset) - 1, count).round().long().tolist()
    rows: dict[str, list[Tensor]] = {
        key: [] for key in (*IMAGE_KEYS, STATE_KEY, ACTION_KEY, PAD_KEY)
    }
    for index in picks:
        item = dataset[index]
        for key in IMAGE_KEYS:
            image = item[key]
            if image_size is not None:
                image = F.interpolate(image[None], size=(image_size, image_size), mode="area")[0]
            rows[key].append(image)
        rows[STATE_KEY].append(
            (item[STATE_KEY].reshape(-1) - torch.as_tensor(scale[STATE_KEY].mean))
            / torch.as_tensor(scale[STATE_KEY].std)
        )
        rows[ACTION_KEY].append(
            (item[ACTION_KEY] - torch.as_tensor(scale[ACTION_KEY].mean))
            / torch.as_tensor(scale[ACTION_KEY].std)
        )
        rows[PAD_KEY].append(item[PAD_KEY])
    return {
        key: torch.stack(value).float() if key != PAD_KEY else torch.stack(value)
        for key, value in rows.items()
    }


# --- Explicit converter from the pinned LeRobot ACT ---------------------------------------------

_RENAMES: tuple[tuple[str, str], ...] = (
    (r"^vae_encoder_cls_embed\.", "vae_cls_embed."),
    (r"^vae_encoder_robot_state_input_proj\.", "vae_state_proj."),
    (r"^vae_encoder_action_input_proj\.", "vae_action_proj."),
    (r"^vae_encoder_latent_output_proj\.", "vae_latent_proj."),
    (r"^vae_encoder_pos_enc$", "vae_pos_embed"),
    (r"^encoder_latent_input_proj\.", "latent_proj."),
    (r"^encoder_robot_state_input_proj\.", "state_proj."),
    (r"^encoder_img_feat_input_proj\.", "camera_proj."),
    (r"^encoder_1d_feature_pos_embed\.", "pos_1d."),
    (r"^decoder_pos_embed\.", "query_embed."),
    (r"^(decoder\.layers\.\d+)\.multihead_attn\.", r"\1.cross_attn."),
)
_IN_PROJ = re.compile(r"^(.*\.(?:self_attn|multihead_attn))\.in_proj_(weight|bias)$")


def convert_lerobot_act_state_dict(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename and split LeRobot ``ACT`` tensors into this module's names.

    Every source key must be used exactly once; an unknown key raises ``KeyError``. Only
    ``nn.MultiheadAttention``'s packed ``in_proj_*`` tensors change shape: rows
    ``[0:D]``, ``[D:2D]``, ``[2D:3D]`` become the query, key and value layers.
    """
    converted: dict[str, Tensor] = {}
    known = (
        r"^(backbone|vae_encoder|encoder|decoder)\.",
        r"^action_head\.",
    )
    for name, value in source.items():
        packed = _IN_PROJ.match(name)
        if packed:
            prefix, kind = packed.groups()
            for part, chunk in zip(
                ("q_proj", "k_proj", "v_proj"), value.detach().chunk(3, dim=0), strict=True
            ):
                target = _rename(f"{prefix}.{part}.{kind}")
                converted[target] = chunk.clone()
            continue
        target = _rename(name)
        if target == name and not any(re.match(pattern, name) for pattern in known):
            raise KeyError(f"no conversion rule for LeRobot ACT tensor {name!r}")
        converted[target] = value.detach().clone()
    return converted


def _rename(name: str) -> str:
    for pattern, replacement in _RENAMES:
        updated = re.sub(pattern, replacement, name)
        if updated != name:
            return updated
    return name


def load_lerobot_act(learner: ACTModel, lerobot_act: nn.Module) -> dict[str, int]:
    """Copy the weights of a LeRobot ``ACT`` into ``learner`` and return tensor counts.

    Uses ``strict=True``: a missing or extra tensor raises. After the call the learner holds
    *reference weights*; it is not a trained learner.
    """
    converted = convert_lerobot_act_state_dict(lerobot_act.state_dict())
    learner.load_state_dict(converted, strict=True)
    return {"tensors": len(converted), "parameters": sum(p.numel() for p in learner.parameters())}


def act_from_lerobot(lerobot_act: nn.Module, **block_hooks) -> ACTModel:
    """Build a learner with the LeRobot model's sizes and copy its weights."""
    settings = ACTSettings.from_lerobot(lerobot_act.config)
    backbone, channels = make_act_backbone(settings.vision_backbone)
    learner = ACTModel(settings, backbone, channels, **block_hooks)
    load_lerobot_act(learner, lerobot_act)
    return learner


def lerobot_batch(learner_batch: Mapping[str, Tensor], settings: ACTSettings) -> dict:
    """Add the image list the raw LeRobot ``ACT.forward`` expects."""
    batch = dict(learner_batch)
    batch["observation.images"] = [learner_batch[key] for key in settings.image_keys]
    return batch


def _zero_dropout(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = 0.0
        if isinstance(child, (nn.MultiheadAttention, MultiHeadAttention)):
            child.dropout = 0.0


@torch.no_grad()
def verify_act_parity(
    learner: ACTModel, lerobot_act: nn.Module, batch: Mapping[str, Tensor], *, seed: int = 0
) -> dict[str, float]:
    """Largest absolute differences between the learner and LeRobot ``ACT`` on one batch.

    ``batch`` needs the state, images, ``action`` and ``action_is_pad``. Neither model is
    changed: the check runs on copies. Two cases run. Inference: evaluation mode, zero latent.
    Training: the posterior path, the same random seed for both models, dropout set to zero on
    the copies so both paths are deterministic. Returns ``inference_actions``,
    ``training_actions``, ``mu`` and ``logvar`` as maximum absolute differences.
    """
    ours_model, theirs_model = copy.deepcopy(learner), copy.deepcopy(lerobot_act)
    reference_batch = lerobot_batch(batch, learner.settings)
    ours_model.eval()
    theirs_model.eval()
    ours, _ = ours_model(batch)
    theirs, _ = theirs_model(reference_batch)
    results = {"inference_actions": float((ours - theirs).abs().max())}
    for model in (ours_model, theirs_model):
        model.train()
        _zero_dropout(model)
    torch.manual_seed(seed)
    ours, (mu, logvar) = ours_model(batch)
    torch.manual_seed(seed)
    theirs, (ref_mu, ref_logvar) = theirs_model(reference_batch)
    results["training_actions"] = float((ours - theirs).abs().max())
    results["mu"] = float((mu - ref_mu).abs().max())
    results["logvar"] = float((logvar - ref_logvar).abs().max())
    return results
