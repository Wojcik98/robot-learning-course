"""Static checks for the policy types used by the learner and real ACT paths."""

from typing import assert_type

from lerobot.policies.act.modeling_act import ACTPolicy
from torch import Tensor

from vla_lab.teaching.act_model import ACTModel
from vla_lab.teaching.interfaces import (
    ActionChunkPolicy,
    ACTOutput,
    ACTPredictionModel,
    ResettablePolicy,
    TensorBatch,
)


class SmallPolicy(ActionChunkPolicy):
    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: TensorBatch) -> Tensor:
        return batch["action"]


def same_interface(real: ACTPolicy, small: SmallPolicy) -> tuple[ActionChunkPolicy, ...]:
    resetters: tuple[ResettablePolicy, ...] = (real, small)
    for policy in resetters:
        policy.reset()
    return real, small


def real_output(real: ACTPolicy, batch: TensorBatch) -> None:
    assert_type(real.predict_action_chunk(batch), Tensor)


def raw_output(real: ACTModel, batch: TensorBatch) -> None:
    model: ACTPredictionModel = real
    assert_type(model(batch), ACTOutput)
