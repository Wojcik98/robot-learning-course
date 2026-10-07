"""Notebook support for Diffusion Policy adaptation."""

from collections.abc import Mapping

from torch import Tensor

_LEARNER_TO_LEROBOT = {
    "time_mlp": "diffusion_step_encoder",
    "down_blocks": "down_modules",
    "mid_blocks": "mid_modules",
    "up_blocks": "up_modules",
    "head": "final_conv",
}


def learner_unet_state_for_lerobot(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Copy learner U-Net tensors under the native LeRobot module names.

    The learner U-Net uses the same layer layout as LeRobot. Only five root names differ.
    Strict loading by the policy checks that the selected model sizes and tensor shapes match.
    """
    converted: dict[str, Tensor] = {}
    for name, value in source.items():
        root, separator, rest = name.partition(".")
        if root not in _LEARNER_TO_LEROBOT or not separator or not rest:
            raise KeyError(f"unknown learner U-Net key {name!r}")
        native_name = f"{_LEARNER_TO_LEROBOT[root]}.{rest}"
        if native_name in converted:
            raise ValueError(f"duplicate native U-Net key {native_name!r}")
        converted[native_name] = value.detach().clone()
    return converted
