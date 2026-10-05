"""These calls must fail the static type check."""

from torch import Tensor

from vla_lab.teaching.interfaces import ActionChunkPolicy, TensorBatch
from vla_lab.teaching.runtime import reset_environment


class MissingReset:
    def predict_action_chunk(self, batch: TensorBatch) -> Tensor:
        return batch["action"]


class WrongBatch:
    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict[str, int]) -> str:
        return str(batch["action"])


def require_policy(policy: ActionChunkPolicy) -> None:
    policy.reset()


def wrong_calls(policy: ActionChunkPolicy) -> None:
    reset_environment(object(), object())
    require_policy(MissingReset())
    require_policy(WrongBatch())
    policy.predict_action_chunk({"action": 3})
