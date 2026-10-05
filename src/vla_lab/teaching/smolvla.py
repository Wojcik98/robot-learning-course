"""SmolVLA input preparation and flow-matching operations (course units 12A and 12B).

The pinned model (``lerobot.policies.smolvla``) is a vision-language model (VLM) with a
separate action expert. The VLM and the expert blocks are supplied; this module exposes
their raw tensors so a learner can write the pieces around them.

Flow convention (the same as the pinned implementation, openpi style):

* time ``t`` is a ``float32`` tensor ``(B,)``; ``t = 0`` is the clean action chunk and ``t = 1`` is noise;
* ``x_t = t * noise + (1 - t) * actions`` and the regression target is ``u_t = noise - actions``;
* sampling is forward Euler from ``t = 1`` to ``t = 0``: ``x <- x + dt * v`` with ``dt = -1 / n``;
* actions are padded with zeros to ``max_action_dim`` (32); the loss uses only the first
  ``action_dim`` (7) columns and only time steps with ``action_is_pad == False``.

This is *not* the ACT Transformer: the SmolVLA expert is a small language-model-style stack
whose attention reads the VLM's cached keys and values. Do not treat the two as the same block.
"""

from __future__ import annotations

import time as _time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_lab.teaching.interfaces import (
    FlowSampler,
    ModelT,
    PromptTokenizer,
    VLAInputConfig,
    VLAInputs,
)

if TYPE_CHECKING:
    from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from transformers.cache_utils import DynamicCache


STATE_KEY = "observation.state"
ACTION_KEY = "action"
PAD_KEY = "action_is_pad"
TOKENS_KEY = "observation.language.tokens"
TOKEN_MASK_KEY = "observation.language.attention_mask"
VLM_PREFIX = "model.vlm_with_expert.vlm."

# --- Unit 12A: input preparation ---------------------------------------------------------------


def resize_pad_normalize_image(image: Tensor, height: int, width: int) -> Tensor:
    """Fit a camera image into ``(height, width)`` without distortion and scale it to ``[-1, 1]``.

    ``image (B, 3, h, w)`` has float values in ``[0, 1]``. Let
    ``ratio = max(w / width, h / height)``; resize bilinearly (``align_corners=False``) to
    ``(int(h / ratio), int(w / ratio))`` unless the size already matches; pad zeros on the top and
    left to reach ``(height, width)``; then map ``v -> 2 * v - 1``. Padding is therefore ``-1``.
    """
    current_height, current_width = image.shape[-2:]
    if (current_height, current_width) != (height, width):
        ratio = max(current_width / width, current_height / height)
        resized = (int(current_height / ratio), int(current_width / ratio))
        image = F.interpolate(image, size=resized, mode="bilinear", align_corners=False)
        image = F.pad(image, (width - resized[1], 0, height - resized[0], 0), value=0.0)
    return image * 2.0 - 1.0


def pad_to_width(values: Tensor, width: int) -> Tensor:
    """Append zeros to the last axis until it has ``width`` entries. Wider input raises ``ValueError``."""
    missing = width - values.shape[-1]
    if missing < 0:
        raise ValueError(f"last axis has {values.shape[-1]} entries, more than width {width}")
    return F.pad(values, (0, missing))


def tokenize_prompts(
    tokenizer: PromptTokenizer, prompts: str | Sequence[str], max_length: int
) -> tuple[Tensor, Tensor]:
    """Turn prompts into ``(tokens (B, L) long, mask (B, L) bool)``; ``True`` marks a real token.

    Append ``"\\n"`` to each prompt that does not end with one. Call the tokenizer with
    ``max_length``, ``truncation=True``, ``padding="longest"``, ``padding_side="right"`` and
    ``return_tensors="pt"``. So ``L`` is the longest prompt in the batch, at most ``max_length``.
    """
    texts = [prompts] if isinstance(prompts, str) else list(prompts)
    texts = [text if text.endswith("\n") else f"{text}\n" for text in texts]
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="longest",
        padding_side="right",
        return_tensors="pt",
    )
    return encoded["input_ids"].long(), encoded["attention_mask"].to(torch.bool)


def prepare_vla_inputs(
    observation: Mapping[str, Tensor],
    prompts: str | Sequence[str],
    tokenizer: PromptTokenizer,
    config: VLAInputConfig,
) -> VLAInputs:
    """Assemble the model inputs from a preprocessed observation and a prompt.

    ``observation`` holds ``observation.state (B, S)`` (already normalized) and one
    ``(B, 3, h, w)`` image in ``[0, 1]`` for each key of ``config.image_features`` (all must be
    present). Returns a dict with ``images`` (list of ``(B, 3, H, W)`` in ``[-1, 1]``, camera order
    as in ``config.image_features``; ``(H, W) = (config.resize_imgs_with_padding[1],
    config.resize_imgs_with_padding[0])``), ``img_masks`` (list of ``(B,)`` bool, all ``True``),
    ``lang_tokens``, ``lang_masks`` (from ``tokenize_prompts`` with ``config.tokenizer_max_length``)
    and ``state`` (``(B, config.max_state_dim)``, zero padded).
    """
    size = config.resize_imgs_with_padding
    if size is None:
        raise ValueError("VLA inputs require an image resize size")
    width, height = size
    images = [
        resize_pad_normalize_image(observation[key], height, width) for key in config.image_features
    ]
    batch = images[0].shape[0]
    tokens, token_mask = tokenize_prompts(tokenizer, prompts, config.tokenizer_max_length)
    if tokens.shape[0] == 1 and batch > 1:
        tokens, token_mask = tokens.expand(batch, -1), token_mask.expand(batch, -1)
    return {
        "images": images,
        "img_masks": [torch.ones(batch, dtype=torch.bool, device=images[0].device) for _ in images],
        "lang_tokens": tokens.to(images[0].device),
        "lang_masks": token_mask.to(images[0].device),
        "state": pad_to_width(observation["observation.state"], config.max_state_dim),
    }


def select_trainable_parameters(
    policy: ModelT, frozen_prefix: str = "model.vlm_with_expert.vlm."
) -> list[nn.Parameter]:
    """Freeze the VLM and return the parameters an optimizer should update.

    Set ``requires_grad = False`` for every parameter whose name starts with ``frozen_prefix``.
    Do not change any other flag: the action expert and projections keep what they have (so a
    parameter that was already frozen on purpose stays frozen). Return the parameters that have
    ``requires_grad=True``, once each, in ``policy.named_parameters()`` order.
    """
    for name, parameter in policy.named_parameters():
        if name.startswith(frozen_prefix):
            parameter.requires_grad_(False)
    return [parameter for _, parameter in policy.named_parameters() if parameter.requires_grad]


def train_mode_with_frozen_vlm(
    policy: ModelT, vlm_name: str = "model.vlm_with_expert.vlm"
) -> ModelT:
    """Put ``policy`` in training mode but keep the frozen VLM in evaluation mode.

    ``policy.train()`` switches every submodule on. A frozen backbone should not use dropout or
    update running statistics, so call ``.eval()`` on the submodule named ``vlm_name`` afterwards.
    Return ``policy``.
    """
    policy.train()
    policy.get_submodule(vlm_name).eval()
    return policy


# --- Unit 12B: flow matching --------------------------------------------------------------------


def flow_training_pair(actions: Tensor, noise: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """Noisy actions and velocity target for flow matching.

    ``actions``, ``noise``: ``(B, T, D)``; ``time``: ``(B,)`` in ``[0, 1]``. Returns
    ``x_t = t * noise + (1 - t) * actions`` and ``u_t = noise - actions``, both ``(B, T, D)``.
    """
    t = time.reshape(-1, 1, 1)
    return t * noise + (1.0 - t) * actions, noise - actions


def masked_velocity_loss(
    prediction: Tensor, target: Tensor, is_pad: Tensor, action_dim: int
) -> Tensor:
    """Mean squared velocity error over real time steps and real action columns.

    ``prediction`` and ``target``: ``(B, T, D_max)``; ``is_pad``: ``(B, T)``, ``True`` for padding;
    ``action_dim <= D_max`` is the number of real action columns (the rest are zero padding).
    ``sum(error ** 2 over [:, :, :action_dim] * ~is_pad) / max(valid_steps * action_dim, 1)``.
    """
    squared = (prediction[..., :action_dim] - target[..., :action_dim]).square()
    valid = (~is_pad.to(torch.bool)).unsqueeze(-1).to(squared.dtype)
    return (squared * valid).sum() / (valid.sum() * action_dim).clamp_min(1)


def euler_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor], noise: Tensor, num_steps: int
) -> Tensor:
    """Integrate the learned velocity from noise (``t = 1``) to actions (``t = 0``).

    ``velocity_fn(x_t, t)`` takes ``x_t (B, T, D)`` and ``t`` as a ``float32`` ``(B,)`` tensor and
    returns ``v`` of the same shape as ``x_t``. With ``dt = -1 / num_steps`` repeat
    ``num_steps`` times: ``t = 1 + step * dt``; ``x = x + dt * velocity_fn(x, t)``.
    The last velocity is evaluated at ``t = 1 / num_steps``, not at ``0``.
    """
    dt = -1.0 / num_steps
    x = noise
    for step in range(num_steps):
        t = torch.full((x.shape[0],), 1.0 + step * dt, dtype=torch.float32, device=x.device)
        x = x + dt * velocity_fn(x, t)
    return x


class SmolVLAAdapter(nn.Module):
    """Raw access to the pinned ``SmolVLAPolicy``: velocity, prefix cache, and time sampling.

    ``policy`` is a ``SmolVLAPolicy`` (published, learner, or a small real one). Nothing here
    computes a loss. ``velocity`` re-implements ``VLAFlowMatching.forward`` up to the velocity
    output so that a learner loss can sit on top of it. ``denoise`` is the pinned inference
    step with the cached image/text prefix.
    """

    def __init__(
        self,
        policy: SmolVLAPolicy,
        train_mode_fn: Callable[[SmolVLAPolicy], SmolVLAPolicy] = train_mode_with_frozen_vlm,
        input_fn: Callable[[SmolVLAAdapter, Mapping[str, Tensor]], VLAInputs] | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.train_mode_fn = train_mode_fn
        self.input_fn = input_fn
        self.config = cast("SmolVLAConfig", policy.config)
        feature = self.config.action_feature
        if feature is None:
            raise ValueError("SmolVLA requires an action feature")
        self.action_dim = feature.shape[0]
        self.max_action_dim = self.config.max_action_dim
        self.chunk_size = self.config.chunk_size

    def train(self, mode: bool = True) -> SmolVLAAdapter:
        """``adapter.train()`` (called by every update) uses ``train_mode_fn`` to keep the VLM frozen."""
        super().train(mode)
        if mode:
            self.train_mode_fn(self.policy)
        return self

    # --- inputs ---
    def prepare_inputs(self, batch: Mapping[str, Tensor]) -> VLAInputs:
        """Model inputs for ``batch``: ``input_fn(self, batch)`` if one was given, else the pinned path.

        A learner ``input_fn`` must return the keys of ``pinned_inputs`` (``images``, ``img_masks``,
        ``lang_tokens``, ``lang_masks``, ``state`` and, in training, ``actions`` padded to
        ``max_action_dim``). It must not call ``prepare_inputs`` itself.
        """
        if self.input_fn is not None:
            return self.input_fn(self, batch)
        return self.pinned_inputs(batch)

    def pinned_inputs(self, batch: Mapping[str, Tensor]) -> VLAInputs:
        """Inputs from a batch produced by the pinned preprocessor (images, state, tokens)."""
        images, img_masks = self.policy.prepare_images(batch)
        inputs: VLAInputs = {
            "images": images,
            "img_masks": img_masks,
            "lang_tokens": batch[TOKENS_KEY],
            "lang_masks": batch[TOKEN_MASK_KEY],
            "state": self.policy.prepare_state(batch),
        }
        if batch.get(ACTION_KEY) is not None:
            inputs["actions"] = self.policy.prepare_action(batch)
        return inputs

    def sample_time(self, batch_size: int, device: torch.device | str) -> Tensor:
        """Training times from the pinned prior ``0.001 + 0.999 * Beta(1.5, 1)``."""
        return self.policy.model.sample_time(batch_size, device)

    # --- raw velocity (training path) ---
    def velocity(self, inputs: VLAInputs, x_t: Tensor, time: Tensor) -> Tensor:
        """Predicted velocity ``v (B, T, max_action_dim)`` for noisy actions ``x_t`` at ``time``."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        model = self.policy.model
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
            inputs["images"],
            inputs["img_masks"],
            inputs["lang_tokens"],
            inputs["lang_masks"],
            state=inputs["state"],
        )
        suffix_embs, suffix_pad, suffix_att = model.embed_suffix(x_t, time)
        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        (_, suffix_out), _ = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad_masks, att_masks),
            position_ids=cast("torch.LongTensor", torch.cumsum(pad_masks, dim=1) - 1),
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
        )
        suffix_out = suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)

    def pinned_per_element_loss(self, inputs: VLAInputs, noise: Tensor, time: Tensor) -> Tensor:
        """The pinned ``VLAFlowMatching.forward`` result ``(B, T, max_action_dim)``, for comparison only."""
        return self.policy.model.forward(
            inputs["images"],
            inputs["img_masks"],
            inputs["lang_tokens"],
            inputs["lang_masks"],
            inputs["state"],
            inputs["actions"],
            noise,
            time,
        )

    # --- inference path ---
    @torch.no_grad()
    def prefix_cache(self, inputs: VLAInputs) -> tuple[Tensor, DynamicCache | None]:
        """Run the image/text/state prefix once. Returns ``(prefix_pad_masks, key_value_cache)``."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        model = self.policy.model
        embs, pad, att = model.embed_prefix(
            inputs["images"],
            inputs["img_masks"],
            inputs["lang_tokens"],
            inputs["lang_masks"],
            state=inputs["state"],
        )
        _, cache = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad, att),
            position_ids=cast("torch.LongTensor", torch.cumsum(pad, dim=1) - 1),
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=model.config.use_cache,
        )
        return pad, cache

    @torch.no_grad()
    def denoise(
        self, cache: tuple[Tensor, DynamicCache | None], x_t: Tensor, time: Tensor
    ) -> Tensor:
        """The pinned ``denoise_step``: velocity for ``x_t`` using the cached prefix."""
        pad, past = cache
        return self.policy.model.denoise_step(pad, past, x_t, time)

    @torch.no_grad()
    def sample_chunk(
        self,
        inputs: VLAInputs,
        sampler: FlowSampler = euler_sample,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Sample ``(B, chunk_size, action_dim)`` actions with a learner sampler.

        ``sampler(velocity_fn, noise, num_steps)`` is ``euler_sample``-shaped. ``noise`` defaults
        to standard normal ``(B, chunk_size, max_action_dim)``. The first ``action_dim`` columns are
        returned (the zero padding columns are dropped).
        """
        state = inputs["state"]
        if noise is None:
            noise = torch.randn(
                state.shape[0],
                self.chunk_size,
                self.max_action_dim,
                device=state.device,
                dtype=torch.float32,
            )
        cache = self.prefix_cache(inputs)
        steps = num_steps or self.config.num_steps
        actions = sampler(lambda x, t: self.denoise(cache, x, t), noise, steps)
        return actions[:, :, : self.action_dim]


def flow_loss_fn(
    model: SmolVLAAdapter,
    batch: Mapping[str, Tensor],
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Loss callback ``(scalar, metrics)`` that keeps the gradient path through the real model.

    ``model`` is a ``SmolVLAAdapter``. Build the inputs with ``model.prepare_inputs(batch)``
    (actions come back zero padded to ``max_action_dim``). Use ``noise`` ``(B, T, max_action_dim)`` and
    ``time`` ``(B,)`` when given; otherwise draw standard normal noise and ``model.sample_time``.
    Make ``(x_t, u_t)`` with ``flow_training_pair``, get ``v = model.velocity(inputs, x_t, time)`` and return
    ``masked_velocity_loss(v, u_t, batch["action_is_pad"], model.action_dim)``.
    """
    inputs = model.prepare_inputs(batch)
    actions = inputs["actions"]
    if noise is None:
        noise = torch.randn_like(actions)
    if noise.shape != actions.shape:
        raise ValueError(f"noise must have the padded action shape {tuple(actions.shape)}")
    if time is None:
        time = model.sample_time(actions.shape[0], actions.device)
    x_t, target = flow_training_pair(actions, noise, time)
    velocity = model.velocity(inputs, x_t, time)
    loss = masked_velocity_loss(velocity, target, batch["action_is_pad"], model.action_dim)
    return loss, {"flow_mse": loss}


class CallbackSmolVLAPolicy:
    """Run the pinned policy's VLM and expert with a *learner* input function and sampler.

    It offers ``reset()`` and ``select_action(batch)`` for the simulator loop. ``batch`` comes from
    the pinned preprocessor (tokens included). ``input_fn(adapter, batch) -> inputs`` defaults to the
    pinned preparation; pass a learner function to use your own. ``sampler`` is ``euler_sample``-shaped.
    A chunk is sampled when the queue is empty and ``n_action_steps`` actions are queued. The pinned
    ``select_action`` and ``sample_actions`` are not called.
    """

    def __init__(
        self,
        policy: SmolVLAPolicy,
        sampler: FlowSampler = euler_sample,
        input_fn: Callable[[SmolVLAAdapter, Mapping[str, Tensor]], VLAInputs] | None = None,
        *,
        num_steps: int | None = None,
        seed: int = 0,
    ) -> None:
        self.adapter = SmolVLAAdapter(policy)
        self.policy = policy
        self.sampler = sampler
        self.input_fn = input_fn or (lambda adapter, batch: adapter.prepare_inputs(batch))
        self.num_steps = num_steps or self.adapter.config.num_steps
        self.seed = seed
        self.sample_seconds: list[float] = []
        self.reset()

    def reset(self) -> None:
        self._queue: deque = deque(maxlen=self.adapter.config.n_action_steps)
        self.chunks_made = 0

    @torch.no_grad()
    def select_action(self, batch: Mapping[str, Tensor]) -> Tensor:
        if not self._queue:
            self.policy.eval()
            inputs = self.input_fn(self.adapter, dict(batch))
            state = inputs["state"]
            generator = torch.Generator(device=state.device).manual_seed(
                self.seed + self.chunks_made
            )
            noise = torch.randn(
                state.shape[0],
                self.adapter.chunk_size,
                self.adapter.max_action_dim,
                generator=generator,
                device=state.device,
                dtype=torch.float32,
            )
            started = _time.perf_counter()
            chunk = self.adapter.sample_chunk(inputs, self.sampler, noise, self.num_steps)
            if state.device.type == "cuda":
                torch.cuda.synchronize()
            self.sample_seconds.append(_time.perf_counter() - started)
            self.chunks_made += 1
            self._queue.extend(chunk[:, : self.adapter.config.n_action_steps].transpose(0, 1))
        return self._queue.popleft()


def load_prepared_smolvla(
    kind: str, data_root: str | Path, *, device: str = "cpu", n_action_steps: int = 10
) -> tuple[SmolVLAPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load a prepared SmolVLA policy with its processors on ``device`` from local files only.

    ``kind="published"`` is the published ``smolvla-libero`` policy; ``kind="learner"`` is the
    project's trained learner (``smolvla-c2-5000-development``). Both need the local SmolVLM2
    directory for the tokenizer and backbone. Returns ``(policy, preprocessor, postprocessor)``;
    the policy is in evaluation mode. One call loads one 450M-parameter model; unload it
    (``del policy``) before loading another on a small GPU.
    """
    from lerobot.processor import load_pretrained_policy_processors

    data = Path(data_root)
    vlm_dir = data / "models" / "smolvlm2-500m-video-instruct"
    if kind == "published":
        from vla_lab.policy.smolvla import load_published_smolvla

        source = data / "models" / "smolvla-libero"
        policy, _, _ = load_published_smolvla(
            source, vlm_dir, device=device, n_action_steps=n_action_steps
        )
    elif kind == "learner":
        from vla_lab.policy.smolvla_learner import load_learner_bundle

        source = data / "runs" / "smolvla-c2-5000-development" / "checkpoint-005000"
        policy, _, _, _ = load_learner_bundle(source, vlm_dir, device=device)
    else:
        raise ValueError("kind must be 'published' or 'learner'")
    preprocessor, postprocessor = load_pretrained_policy_processors(
        str(source),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(vlm_dir)},
            "device_processor": {"device": device},
        },
    )
    return policy.eval(), preprocessor, postprocessor
