import torch

from vla_lab.teaching.cvae import ConditionalVAE, cvae_loss_fn, reparameterize
from vla_lab.teaching.training import optimizer_update


def test_reparameterize_is_deterministic_when_noise_is_given_and_has_gradients():
    mu = torch.tensor([[1.0, -1.0]], requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    sample = reparameterize(mu, logvar, noise=torch.tensor([[2.0, 0.5]]))
    torch.testing.assert_close(sample, torch.tensor([[3.0, -0.5]]))
    sample.sum().backward()
    assert torch.equal(mu.grad, torch.ones_like(mu))
    assert torch.equal(logvar.grad, torch.tensor([[1.0, 0.25]]))


def test_cvae_posterior_training_and_prior_inference_paths():
    torch.manual_seed(7)
    model = ConditionalVAE(condition_dim=3, target_dim=2, latent_dim=4, hidden_dim=8)
    condition = torch.randn(5, 3)
    target = torch.randn(5, 2)
    reconstruction, mu, logvar = model(condition, target, noise=torch.zeros(5, 4))
    assert reconstruction.shape == target.shape
    assert mu.shape == logvar.shape == (5, 4)
    prior = model.infer(condition, noise=torch.ones(5, 4))
    assert prior.shape == target.shape
    assert not torch.equal(reconstruction, prior)


def test_callback_supports_padded_action_chunks_and_updates_model():
    torch.manual_seed(11)
    model = ConditionalVAE(3, 10, 2, hidden_dim=8, target_shape=(5, 2))
    condition = torch.randn(4, 3)
    target = torch.randn(4, 5, 2)
    data = {
        "condition": condition,
        "target": target,
        "is_pad": torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]], dtype=torch.bool
        ),
    }
    noise = torch.zeros(4, 2)
    loss, _metrics = cvae_loss_fn(model, data, noise=noise, beta=0.1)
    changed_pad = dict(data, target=data["target"].clone())
    changed_pad["target"][data["is_pad"]] = 10000
    same_loss, _ = cvae_loss_fn(model, changed_pad, noise=noise, beta=0.1)
    torch.testing.assert_close(
        loss, same_loss, msg="padded target values must not affect posterior or loss"
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    before = next(model.parameters()).detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    row = optimizer_update(
        model,
        data,
        opt,
        lambda m, b, *, noise=None, time=None: cvae_loss_fn(m, b, noise=noise, beta=0.1),
        noise=noise,
    )
    assert torch.isfinite(torch.tensor(row["reconstruction"]))
    assert not torch.equal(before, next(model.parameters()).detach())


def test_infer_rejects_ambiguous_or_mis_shaped_latent():
    model = ConditionalVAE(2, 1, 3, hidden_dim=4)
    condition = torch.zeros(2, 2)
    try:
        model.infer(condition, latent=torch.zeros(2, 3), noise=torch.zeros(2, 3))
    except ValueError as exc:
        assert "latent or noise" in str(exc)
    else:
        raise AssertionError("ambiguous latent input must fail")
