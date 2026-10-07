import torch
from torch import nn

from vla_lab.teaching import training
from vla_lab.teaching.training import (
    act_loss_fn,
    diagonal_gaussian_kl,
    masked_action_l1,
    optimizer_update,
    train_loop,
)


def test_masked_action_l1_ignores_padding_and_has_zero_padded_gradient():
    prediction = torch.tensor([[[0.0, 2.0], [100.0, -100.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    pad = torch.tensor([[False, True]])
    loss = masked_action_l1(prediction, target, pad)
    assert loss.item() == 1.5
    loss.backward()
    assert torch.equal(prediction.grad[0, 1], torch.zeros(2))
    assert masked_action_l1(prediction.detach(), target, torch.ones_like(pad)).item() == 0


def test_gaussian_kl_reduction_is_latent_sum_then_batch_mean():
    mu = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    logvar = torch.tensor([[0.0, 0.0], [0.0, torch.log(torch.tensor(4.0))]])
    expected = (0.5 + 0.5 * (4 - 1 - torch.log(torch.tensor(4.0)))) / 2
    torch.testing.assert_close(diagonal_gaussian_kl(mu, logvar), expected)
    zero = torch.zeros(3, 4)
    assert diagonal_gaussian_kl(zero, zero).item() == 0


class TinyACT(nn.Module):
    def __init__(self):
        super().__init__()
        self.actions = nn.Parameter(torch.zeros(2))

    def forward(self, batch):
        pred = self.actions[None, None, :].expand_as(batch["action"])
        mu = self.actions[None, :].expand(batch["action"].shape[0], -1)
        return pred, (mu, torch.zeros_like(mu))


def batch():
    return {
        "action": torch.ones(2, 3, 2),
        "action_is_pad": torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.bool),
    }


def test_callback_drives_two_updates_and_clears_gradients():
    model = TinyACT()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = train_loop(model, [batch()], optimizer, act_loss_fn, steps=2)
    assert len(history) == 2
    assert history[0]["loss"] > history[-1]["loss"]
    assert torch.all(model.actions > 0)


def test_optimizer_update_forwards_noise_and_time_without_detaching_loss():
    model = nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    seen = {}
    noise, time = torch.tensor([2.0]), torch.tensor([0.25])

    def callback(current_model, data, *, noise=None, time=None):
        seen.update(noise=noise, time=time)
        value = current_model(data["x"]).square().mean()
        return value, {"same": value}

    metrics = optimizer_update(
        model, {"x": torch.ones(1, 1)}, opt, callback, noise=noise, time=time
    )
    assert seen["noise"] is noise and seen["time"] is time
    assert metrics["loss"] == metrics["same"]
    assert not torch.equal(model.weight.detach(), torch.tensor([[0.5]]))


def test_empty_batches_and_bad_callback_loss_are_rejected():
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def callback(current_model, data, *, noise=None, time=None):
        return torch.tensor(0.0), {}

    try:
        train_loop(model, [], opt, callback, steps=1)
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty batch list must fail")


def test_train_loop_cycles_explicit_noise_and_time_into_callback():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    received = []

    def callback(current_model, data, *, noise=None, time=None):
        received.append((noise.item(), time.item()))
        loss = current_model(data["x"]).sum() * 0
        return loss, {}

    train_loop(
        model,
        [{"x": torch.ones(1, 1)}],
        optimizer,
        callback,
        steps=3,
        noise=[torch.tensor(1.0), torch.tensor(2.0)],
        time=[torch.tensor(0.1), torch.tensor(0.2)],
    )
    assert [row[0] for row in received] == [1.0, 2.0, 1.0]
    torch.testing.assert_close(
        torch.tensor([row[1] for row in received]), torch.tensor([0.1, 0.2, 0.1])
    )


def test_optimizer_update_clips_gradients_before_step():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    def callback(current_model, data, *, noise=None, time=None):
        loss = current_model(data["x"]).sum() * 100
        return loss, {}

    optimizer_update(model, {"x": torch.ones(1, 1)}, optimizer, callback, max_grad_norm=0.25)
    torch.testing.assert_close(model.weight, torch.tensor([[-0.25]]))


def test_training_deadline_stops_before_another_update(monkeypatch):
    model = TinyACT()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    clock = iter([10.0, 10.0, 10.5, 11.0])
    monkeypatch.setattr(training, "monotonic", lambda: next(clock), raising=False)
    history = train_loop(model, [batch()], optimizer, act_loss_fn, steps=100, max_seconds=1.0)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in history)
