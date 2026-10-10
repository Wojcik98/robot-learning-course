"""Behaviour checks for the Lesson 17 exercise.

The notebook checks one action value of one example. These checks use a stand-in adapter
with two examples, different flow times, several real action columns plus zero-padded
expert columns, and a recording ``velocity``. They show which inputs reached the expert
and which values entered the loss. Read a failing assertion message first.
"""

import torch
from torch import nn


def _done(message: str) -> None:
    print(f"✓ {message}")


def _close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-5) -> None:
    actual = torch.as_tensor(actual)
    assert actual.shape == expected.shape, (
        f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    )
    worst = (actual.detach() - expected.detach()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


class _Adapter:
    """2 real action values padded to an expert width of 4; velocity = scale * f(x_t, t)."""

    action_dim = 2

    def __init__(self, predict) -> None:
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.predict = predict
        self.prepared: list = []
        self.calls: list = []

    def prepare_inputs(self, batch):
        actions = torch.zeros(*batch["action"].shape[:2], 4)
        actions[..., :2] = batch["action"]
        inputs = {"actions": actions, "state": batch["state"]}
        self.prepared.append(inputs)
        return inputs

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device) * 0.8 + 0.1

    def velocity(self, inputs, x_t, time):
        self.calls.append((inputs, x_t.detach().clone(), time))
        return self.predict(inputs, x_t, time) * self.scale


def check_learner_flow_loss(learner_flow_loss) -> None:
    """Exercise 17.1: x_t, target u = noise - actions, masked MSE over real steps and columns."""
    generator = torch.Generator().manual_seed(100)
    batch = {
        "action": torch.randn(2, 3, 2, generator=generator),
        "action_is_pad": torch.tensor([[False, False, True], [False, True, True]]),
        "state": torch.randn(2, 5, generator=generator),
    }
    noise = torch.randn(2, 3, 4, generator=generator)
    time = torch.tensor([0.2, 0.7])

    def wrong_on_padding(inputs, x_t, t):
        out = torch.zeros_like(x_t)
        out[..., 2:] = 50.0  # padded expert columns: must not enter the loss
        return out

    adapter = _Adapter(wrong_on_padding)
    loss, metrics = learner_flow_loss(adapter, batch, noise, time)
    actions = adapter.prepared[0]["actions"]
    assert len(adapter.calls) == 1, "call adapter.velocity once"
    inputs, x_t, got_time = adapter.calls[0]
    assert inputs is adapter.prepared[0], "pass the prepared inputs to adapter.velocity"
    assert torch.equal(got_time, time), "pass the flow time to adapter.velocity"
    t = time.view(2, 1, 1)
    _close(
        x_t,
        (1 - t) * actions + t * noise,
        "x_t must be (1 - t) * actions + t * noise, with each example's own t",
    )
    target = noise - actions
    valid = ~batch["action_is_pad"]
    expected = target[..., :2].square()[valid].sum() / (
        int(valid.sum()) * 2
    )  # prediction 0 on real columns
    _close(
        loss,
        expected,
        "loss = mean of (v - (noise - actions))^2 over real steps and the first action_dim columns only",
    )
    assert set(metrics) == {"flow_mse"} and torch.equal(
        torch.as_tensor(metrics["flow_mse"]), loss
    ), "metrics must be {'flow_mse': loss}"
    try:
        loss.backward()
    except RuntimeError as error:
        raise AssertionError(
            "the loss has no gradient path: do not detach the velocity or the loss"
        ) from error
    assert adapter.scale.grad is not None, (
        "do not detach the velocity: gradients must reach the expert"
    )
    _done("x_t per example; target noise - actions; padded steps and padded columns are excluded")

    changed = dict(batch, action=batch["action"].clone())
    changed["action"][batch["action_is_pad"]] = 1e3
    changed_loss, _ = learner_flow_loss(_Adapter(wrong_on_padding), changed, noise, time)
    _close(changed_loss, loss, "padded action steps must not change the loss")
    _done("padded action steps do not change the loss")

    # Without noise/time: draw them; a velocity that recovers the drawn path has zero loss.
    def recover_velocity(inputs, x_t, t):
        return (x_t - inputs["actions"]) / t.view(-1, 1, 1)  # = noise - actions on the path

    adapter = _Adapter(recover_velocity)
    torch.manual_seed(101)
    loss, _ = learner_flow_loss(adapter, batch)
    _, x_t, drawn_time = adapter.calls[0]
    assert drawn_time.shape == (2,), "draw one flow time per example, shape (B,)"
    assert float(drawn_time.min()) >= 0 and float(drawn_time.max()) <= 1, "flow times lie in [0, 1]"
    assert x_t.shape == (2, 3, 4), "noise has the shape of inputs['actions'] (the padded width)"
    assert loss.item() < 1e-8, "the target must use the same drawn noise as x_t"
    _done("draws noise shaped like inputs['actions'] and one time per example when none are given")
