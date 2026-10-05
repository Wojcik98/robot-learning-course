"""Required fields and callback signatures for learner model blocks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, TypedDict

from torch import Tensor, nn

if TYPE_CHECKING:
    from vla_lab.teaching.act_model import ACTHooks, ACTSettings
    from vla_lab.teaching.attention import TransformerDecoder, TransformerEncoder


class ScaledAttention(Protocol):
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> tuple[Tensor, Tensor]: ...


class AttentionOperation(Protocol):
    def __call__(
        self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor | None = None
    ) -> Tensor: ...


class ProjectedAttention(Protocol):
    @property
    def q_proj(self) -> nn.Linear: ...

    @property
    def k_proj(self) -> nn.Linear: ...

    @property
    def v_proj(self) -> nn.Linear: ...

    @property
    def out_proj(self) -> nn.Linear: ...

    @property
    def n_heads(self) -> int: ...

    @property
    def dropout(self) -> float: ...

    @property
    def training(self) -> bool: ...


class EncoderParts(Protocol):
    @property
    def self_attn(self) -> AttentionOperation: ...

    @property
    def pre_norm(self) -> bool: ...

    @property
    def norm1(self) -> nn.LayerNorm: ...

    @property
    def norm2(self) -> nn.LayerNorm: ...

    @property
    def linear1(self) -> nn.Linear: ...

    @property
    def linear2(self) -> nn.Linear: ...

    @property
    def dropout(self) -> nn.Dropout: ...

    @property
    def dropout1(self) -> nn.Dropout: ...

    @property
    def dropout2(self) -> nn.Dropout: ...

    @property
    def activation(self) -> Callable[[Tensor], Tensor]: ...


class DecoderParts(EncoderParts, Protocol):
    @property
    def cross_attn(self) -> AttentionOperation: ...

    @property
    def norm3(self) -> nn.LayerNorm: ...

    @property
    def dropout3(self) -> nn.Dropout: ...


class ACTPosteriorParts(Protocol):
    @property
    def vae_cls_embed(self) -> nn.Embedding: ...

    @property
    def vae_state_proj(self) -> nn.Linear: ...

    @property
    def vae_action_proj(self) -> nn.Linear: ...

    @property
    def vae_encoder(self) -> TransformerEncoder: ...

    @property
    def vae_pos_embed(self) -> Tensor: ...

    @property
    def vae_latent_proj(self) -> nn.Linear: ...


class ACTLatentParts(ACTPosteriorParts, Protocol):
    @property
    def training(self) -> bool: ...

    @property
    def settings(self) -> ACTSettings: ...

    @property
    def hooks(self) -> ACTHooks: ...


class ACTEncoderParts(Protocol):
    @property
    def latent_proj(self) -> nn.Linear: ...

    @property
    def state_proj(self) -> nn.Linear: ...

    @property
    def camera_proj(self) -> nn.Conv2d: ...

    @property
    def pos_1d(self) -> nn.Embedding: ...

    @property
    def settings(self) -> ACTSettings: ...

    @property
    def hooks(self) -> ACTHooks: ...


class ACTDecoderParts(Protocol):
    @property
    def settings(self) -> ACTSettings: ...

    @property
    def decoder(self) -> TransformerDecoder: ...

    @property
    def query_embed(self) -> nn.Embedding: ...

    @property
    def action_head(self) -> nn.Linear: ...


class MultiHeadForward(Protocol):
    def __call__(
        self,
        attention: ProjectedAttention,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attend: ScaledAttention = ...,
    ) -> tuple[Tensor, Tensor]: ...


class EncoderForward(Protocol):
    def __call__(
        self,
        layer: EncoderParts,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor: ...


class DecoderForward(Protocol):
    def __call__(
        self,
        layer: DecoderParts,
        x: Tensor,
        memory: Tensor,
        query_pos: Tensor | None = None,
        memory_pos: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor: ...


class SharedAttentionHooks(TypedDict, total=False):
    attend_fn: ScaledAttention
    attention_forward_fn: MultiHeadForward


class EncoderHooks(SharedAttentionHooks, total=False):
    forward_fn: EncoderForward


class DecoderHooks(SharedAttentionHooks, total=False):
    forward_fn: DecoderForward


class PosteriorFunction(Protocol):
    def __call__(
        self, model: ACTPosteriorParts, state: Tensor, actions: Tensor, action_is_pad: Tensor
    ) -> tuple[Tensor, Tensor]: ...


class LatentFunction(Protocol):
    def __call__(
        self, model: ACTLatentParts, batch: Mapping[str, Tensor], noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor | None]: ...


class EncoderTokensFunction(Protocol):
    def __call__(
        self, model: ACTEncoderParts, latent: Tensor, state: Tensor, feature_maps: list[Tensor]
    ) -> tuple[Tensor, Tensor]: ...


class DecodeFunction(Protocol):
    def __call__(self, model: ACTDecoderParts, encoder_out: Tensor, pos: Tensor) -> Tensor: ...


class FilmParts(Protocol):
    @property
    def conv1(self) -> nn.Module: ...

    @property
    def conv2(self) -> nn.Module: ...

    @property
    def cond_encoder(self) -> nn.Sequential: ...

    @property
    def out_channels(self) -> int: ...

    @property
    def use_film_scale_modulation(self) -> bool: ...

    @property
    def residual_conv(self) -> nn.Module: ...


class UNetParts(Protocol):
    @property
    def time_mlp(self) -> nn.Sequential: ...

    @property
    def down_blocks(self) -> nn.ModuleList: ...

    @property
    def mid_blocks(self) -> nn.ModuleList: ...

    @property
    def up_blocks(self) -> nn.ModuleList: ...

    @property
    def head(self) -> nn.Sequential: ...


class FilmForward(Protocol):
    def __call__(self, block: FilmParts, x: Tensor, cond: Tensor) -> Tensor: ...


class UNetForward(Protocol):
    def __call__(
        self, unet: UNetParts, x: Tensor, timestep: Tensor, global_cond: Tensor | None = None
    ) -> Tensor: ...
