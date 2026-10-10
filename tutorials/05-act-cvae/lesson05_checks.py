"""Behaviour checks for the Lesson 5 exercises.

The modules have random weights, so these checks test properties that any correct
implementation has: which inputs an output depends on, that examples in a batch do not
mix, that leading dimensions are kept, and how the training and inference paths relate.
Read a failing assertion message first: it states the expected behaviour.
"""

import torch
from torch import nn

C, Y, L, HIDDEN = 3, 4, 2, 16


def _same(actual: object, expected: object) -> bool:
    """True when ``actual`` is ``expected`` or holds equal values (tensors, lists, dicts)."""
    if actual is expected:
        return True
    if isinstance(expected, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor)
            and actual.shape == expected.shape
            and torch.equal(actual.cpu(), expected.cpu())
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(_same(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same(actual[k], expected[k]) for k in expected)
        )
    return actual == expected


def _done(message: str) -> None:
    print(f"✓ {message}")


def _expect_error(call, errors: tuple[type[Exception], ...], message: str) -> None:
    try:
        call()
    except errors:
        return
    except Exception as error:  # noqa: BLE001 - report the unexpected type
        raise AssertionError(f"{message}; got {type(error).__name__}: {error}") from error
    raise AssertionError(message)


def _differs(a: torch.Tensor, b: torch.Tensor) -> bool:
    return not torch.allclose(a, b, atol=1e-6)


def check_conditional_encoder(ConditionalEncoder) -> None:  # noqa: N803 - receives a class
    """Exercise 1: (condition, target[, mask]) → (mu, logvar), one example at a time."""
    torch.manual_seed(0)
    encoder = ConditionalEncoder(C, Y, L, HIDDEN)
    assert isinstance(encoder, nn.Module), "ConditionalEncoder must be an nn.Module"
    condition, target = torch.randn(6, C), torch.randn(6, Y)
    mu, logvar = encoder(condition, target)
    assert mu.shape == logvar.shape == (6, L), "mu and logvar must have shape (B, latent_dim)"
    assert _differs(mu, logvar), "mu and logvar must come from separate heads"
    _done("returns mu and logvar of shape (B, latent_dim) from separate heads")

    changed_target = target.clone()
    changed_target[2] += 1.0
    mu_t, _ = encoder(condition, changed_target)
    changed_condition = condition.clone()
    changed_condition[2] += 1.0
    mu_c, _ = encoder(changed_condition, target)
    assert _differs(mu_t[2], mu[2]), "the posterior must depend on the target"
    assert _differs(mu_c[2], mu[2]), "the posterior must depend on the condition"
    keep = [0, 1, 3, 4, 5]
    torch.testing.assert_close(
        mu_t[keep], mu[keep], msg="changing example 2 must not change other examples"
    )
    torch.testing.assert_close(mu_c[keep], mu[keep])
    _done("depends on both condition and target, separately for each example")

    single_mu, single_logvar = encoder(condition[3:4], target[3:4])
    torch.testing.assert_close(single_mu, mu[3:4], msg="batch size must not change the result")
    torch.testing.assert_close(single_logvar, logvar[3:4])
    nested_mu, nested_logvar = encoder(condition.reshape(2, 3, C), target.reshape(2, 3, Y))
    assert nested_mu.shape == nested_logvar.shape == (2, 3, L), "keep all leading dimensions"
    torch.testing.assert_close(nested_mu.reshape(6, L), mu, msg="join features on the last axis")
    _done("keeps leading dimensions: (2, 3, C) inputs give (2, 3, latent_dim)")

    big_mu, big_logvar = encoder(condition * 1e5, target * 1e5)
    assert torch.isfinite(big_logvar).all() and torch.isfinite(big_logvar.exp()).all(), (
        "clamp logvar so exp(logvar) stays finite even for extreme inputs"
    )
    assert torch.isfinite(big_mu).all()
    _done("logvar is clamped to a stable range")

    (mu.sum() + logvar.sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()), (
        "the outputs must keep their gradient path to the encoder weights"
    )

    masked_encoder = ConditionalEncoder(C, Y, L, HIDDEN, mask_dim=3)
    mask_a = torch.zeros(6, 3)
    mask_b = mask_a.clone()
    mask_b[:, -1] = 1.0
    out_a, _ = masked_encoder(condition, target, mask_a)
    out_b, _ = masked_encoder(condition, target, mask_b)
    assert _differs(out_a, out_b), "with mask_dim > 0, target_mask must be an encoder input"
    _done("with mask_dim, the target mask is an extra encoder input")


def check_reparameterize(reparameterize) -> None:
    """Exercise 2: z = mu + exp(logvar / 2) * noise, differentiable in mu and logvar."""
    mu = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
    logvar = torch.tensor(
        [[0.0, 2 * torch.log(torch.tensor(3.0)).item(), -2.0]], requires_grad=True
    )
    noise = torch.tensor([[0.5, -1.0, 2.0]])
    z = reparameterize(mu, logvar, noise=noise)
    expected = torch.tensor([[1.5, -5.0, 0.5 + torch.exp(torch.tensor(-1.0)).item() * 2.0]])
    torch.testing.assert_close(
        z, expected, msg="z = mu + exp(0.5 * logvar) * noise (logvar = log 9 means std 3)"
    )
    z.sum().backward()
    assert mu.grad is not None and logvar.grad is not None, "keep mu and logvar in the graph"
    torch.testing.assert_close(mu.grad, torch.ones_like(mu))
    torch.testing.assert_close(
        logvar.grad,
        0.5 * torch.exp(0.5 * logvar.detach()) * noise,
        msg="d z / d logvar must be 0.5 * exp(0.5 * logvar) * noise",
    )
    _done("uses the supplied noise; gradients reach mu and logvar")

    torch.manual_seed(0)
    mu = torch.full((20000, 2), 3.0)
    logvar = torch.log(torch.tensor([[0.25, 4.0]])).expand(20000, 2)
    first = reparameterize(mu, logvar)
    second = reparameterize(mu, logvar)
    assert first.shape == mu.shape, "a sample has the same shape as mu"
    assert _differs(first, second), "without noise, draw new random values on every call"
    torch.testing.assert_close(first.mean(0), torch.tensor([3.0, 3.0]), atol=0.05, rtol=0)
    torch.testing.assert_close(
        first.std(0),
        torch.tensor([0.5, 2.0]),
        atol=0.05,
        rtol=0,
        msg="without noise, samples must have mean mu and standard deviation exp(logvar / 2)",
    )
    _done("without noise, samples follow N(mu, exp(logvar))")

    _expect_error(
        lambda: reparameterize(torch.zeros(2, 3), torch.zeros(2, 3), noise=torch.zeros(3, 2)),
        (ValueError, RuntimeError),
        "noise of the wrong shape must be rejected",
    )
    _done("noise of the wrong shape is rejected")


def check_conditional_decoder(ConditionalDecoder) -> None:  # noqa: N803 - receives a class
    """Exercise 3: (condition, latent) → prediction of width target_dim, per example."""
    torch.manual_seed(1)
    decoder = ConditionalDecoder(C, L, Y, HIDDEN)
    assert isinstance(decoder, nn.Module), "ConditionalDecoder must be an nn.Module"
    condition, latent = torch.randn(5, C), torch.randn(5, L)
    prediction = decoder(condition, latent)
    assert prediction.shape == (5, Y), "the output width is target_dim"
    changed_latent = latent.clone()
    changed_latent[1] += 1.0
    changed_condition = condition.clone()
    changed_condition[1] += 1.0
    by_latent = decoder(condition, changed_latent)
    by_condition = decoder(changed_condition, latent)
    assert _differs(by_latent[1], prediction[1]), "the prediction must depend on the latent"
    assert _differs(by_condition[1], prediction[1]), "the prediction must depend on the condition"
    torch.testing.assert_close(by_latent[[0, 2, 3, 4]], prediction[[0, 2, 3, 4]])
    torch.testing.assert_close(by_condition[[0, 2, 3, 4]], prediction[[0, 2, 3, 4]])
    _done("depends on condition and latent, separately for each example")

    nested = decoder(condition[:4].reshape(2, 2, C), latent[:4].reshape(2, 2, L))
    assert nested.shape == (2, 2, Y), "keep all leading dimensions"
    torch.testing.assert_close(nested.reshape(4, Y), prediction[:4])
    _done("keeps leading dimensions")


def check_conditional_vae(ConditionalVAE) -> None:  # noqa: N803 - receives a class
    """Exercise 4: posterior training path and prior inference path."""
    torch.manual_seed(2)
    model = ConditionalVAE(C, Y, L, HIDDEN)
    assert model.latent_dim == L, "store latent_dim on the model"
    condition, target = torch.randn(5, C), torch.randn(5, Y)
    noise = torch.randn(5, L)
    prediction, mu, logvar = model(condition, target, noise=noise)
    assert prediction.shape == (5, Y) and mu.shape == logvar.shape == (5, L)
    # The training path decodes z = mu + exp(logvar / 2) * noise; infer decodes a given z.
    z = mu + torch.exp(0.5 * logvar) * noise
    torch.testing.assert_close(
        model.infer(condition, noise=z),
        prediction,
        msg="forward must decode z = mu + exp(logvar/2) * noise with the same decoder that infer uses",
    )
    _done("forward = decode(condition, mu + exp(logvar / 2) * noise)")

    other_target = target + 1.0
    _, mu_other, _ = model(condition, other_target, noise=noise)
    assert _differs(mu_other, mu), "the posterior (mu) must depend on the target"
    zeros = torch.zeros(5, L)
    torch.testing.assert_close(
        model.infer(condition, noise=zeros), model.infer(condition, noise=zeros)
    )
    random_a, random_b = model.infer(condition), model.infer(condition)
    assert random_a.shape == (5, Y), "infer returns a prediction of the target shape"
    assert _differs(random_a, random_b), "without noise, infer draws a new prior sample"
    _expect_error(
        lambda: model.infer(condition, noise=torch.zeros(5, L + 1)),
        (ValueError, RuntimeError),
        "infer must reject a latent of the wrong width",
    )
    _done("infer decodes a prior sample without a target; supplied noise makes it repeatable")

    # Action chunks: target_shape=(T, D), is_pad masks padded steps.
    chunk_model = ConditionalVAE(C, 5 * 2, L, HIDDEN, target_shape=(5, 2))
    chunk_target = torch.randn(4, 5, 2)
    is_pad = torch.tensor(
        [[False] * 5, [False, False, True, True, True], [False] * 4 + [True], [False] + [True] * 4]
    )
    chunk_prediction, mu, logvar = chunk_model(
        condition[:4], chunk_target, noise=zeros[:4], is_pad=is_pad
    )
    assert chunk_prediction.shape == (4, 5, 2), "chunk predictions have shape (B, T, D)"
    assert chunk_model.infer(condition[:4], noise=zeros[:4]).shape == (4, 5, 2)
    changed = chunk_target.clone()
    changed[is_pad] = 500.0
    _, mu_changed, logvar_changed = chunk_model(
        condition[:4], changed, noise=zeros[:4], is_pad=is_pad
    )
    torch.testing.assert_close(
        mu_changed, mu, msg="padded target values must not reach the encoder"
    )
    torch.testing.assert_close(logvar_changed, logvar)
    _done("chunk targets: padded values do not change mu or logvar")

    # The mask itself is an encoder input: a zero target that is padded differs from a real zero.
    zero_target = chunk_target.clone()
    zero_target[:, -1] = 0.0
    unpadded = torch.zeros(4, 5, dtype=torch.bool)
    last_padded = unpadded.clone()
    last_padded[:, -1] = True
    _, mu_real_zero, _ = chunk_model(condition[:4], zero_target, noise=zeros[:4], is_pad=unpadded)
    _, mu_padded, _ = chunk_model(condition[:4], zero_target, noise=zeros[:4], is_pad=last_padded)
    assert _differs(mu_real_zero, mu_padded), (
        "pass the time mask to the encoder: a padded slot must differ from a real zero action"
    )
    _done("chunk targets: the time mask is passed to the encoder")


class _FixedCVAE(nn.Module):
    """A stand-in model that returns fixed outputs and records its call arguments."""

    def __init__(self, prediction: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> None:
        super().__init__()
        self.prediction = nn.Parameter(prediction.clone())
        self.mu, self.logvar = nn.Parameter(mu.clone()), nn.Parameter(logvar.clone())
        self.calls: list[dict] = []

    def forward(self, condition, target, *, noise=None, is_pad=None):
        self.calls.append(
            {"condition": condition, "target": target, "noise": noise, "is_pad": is_pad}
        )
        return self.prediction, self.mu, self.logvar


def check_cvae_loss_fn(cvae_loss_fn) -> None:
    """Callback: masked reconstruction L1 + beta * KL, with both parts as metrics."""
    generator = torch.Generator().manual_seed(3)
    mu, logvar = torch.randn(3, L, generator=generator), torch.randn(3, L, generator=generator)
    kl = (0.5 * (logvar.exp() + mu**2 - 1 - logvar)).sum(-1).mean()

    prediction, target = (
        torch.randn(3, Y, generator=generator),
        torch.randn(3, Y, generator=generator),
    )
    model = _FixedCVAE(prediction, mu, logvar)
    noise = torch.randn(3, L, generator=generator)
    batch = {"condition": torch.randn(3, C, generator=generator), "target": target}
    loss, metrics = cvae_loss_fn(model, batch, noise=noise, beta=0.25)
    reconstruction = (prediction - target).abs().mean()
    assert model.calls and _same(model.calls[-1]["noise"], noise), "pass noise to the model"
    assert _same(model.calls[-1]["condition"], batch["condition"]), (
        "pass the condition to the model"
    )
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "return a scalar loss tensor"
    torch.testing.assert_close(
        loss, reconstruction + 0.25 * kl, msg="loss = L1 reconstruction + beta * KL"
    )
    assert set(metrics) >= {"reconstruction", "kl"}, (
        "metrics must include 'reconstruction' and 'kl'"
    )
    torch.testing.assert_close(torch.as_tensor(metrics["reconstruction"]), reconstruction)
    torch.testing.assert_close(
        torch.as_tensor(metrics["kl"]), kl, msg="kl must be the Gaussian KL in nats"
    )
    loss.backward()
    assert model.prediction.grad is not None and model.mu.grad is not None, (
        "the loss must keep gradients"
    )
    _done("vector targets: loss = mean |prediction - target| + beta * KL")

    loss_beta0, _ = cvae_loss_fn(_FixedCVAE(prediction, mu, logvar), batch, noise=noise, beta=0.0)
    torch.testing.assert_close(
        loss_beta0, reconstruction, msg="beta = 0 leaves only the reconstruction"
    )
    loss_default, _ = cvae_loss_fn(_FixedCVAE(prediction, mu, logvar), batch, noise=noise)
    torch.testing.assert_close(loss_default, reconstruction + kl, msg="the default beta is 1")
    _done("beta scales only the KL term")

    chunk_prediction = torch.randn(3, 4, 2, generator=generator)
    chunk_target = torch.randn(3, 4, 2, generator=generator)
    is_pad = torch.tensor([[False] * 4, [False, False, True, True], [False, True, True, True]])
    chunk_model = _FixedCVAE(chunk_prediction, mu, logvar)
    chunk_batch = {"condition": batch["condition"], "target": chunk_target, "is_pad": is_pad}
    loss, metrics = cvae_loss_fn(chunk_model, chunk_batch, noise=noise, beta=0.5)
    valid = ~is_pad
    masked = (chunk_prediction - chunk_target).abs()[valid].mean()
    assert _same(chunk_model.calls[-1]["is_pad"], is_pad), "pass is_pad from the batch to the model"
    torch.testing.assert_close(
        torch.as_tensor(metrics["reconstruction"]),
        masked,
        msg="chunk targets: reconstruction must be the masked L1 over valid steps only",
    )
    torch.testing.assert_close(loss, masked + 0.5 * kl)
    _done("chunk targets: reconstruction ignores padded steps; is_pad reaches the model")
