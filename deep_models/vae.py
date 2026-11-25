"""Configurable convolutional VAE / conditional VAE (CVAE) for (bands, C, C) connectivity matrices, plain PyTorch.

The defaults reproduce the hand-written 5 x 17 x 17 model: 17 -> 9 -> 5 -> 1 (k3/s2, k3/s2, k5/s1), ReLU,
no normalisation, sigmoid output, off-diagonal mask, log-variance clamped to [-10, 2].
Any other architecture: change hidden / kernel / strides / padding / batchnorm / latent.
(The old configurable defaults were: latent=16, hidden=(16, 32), kernel=3, strides=2, padding=1,
batchnorm=True, out_act="none", mask_diag=False, fc_act=False, logvar_clamp=None.)

Conditioning (cond_dim > 0): the condition vector c of an exam (e.g. [age, female, site one-hot]) is
concatenated to the flattened encoder features (if cond_in_encoder) and to z before the decoder, i.e.
q(z | x, c) and p(x | z, c). All windows of an exam share the exam's c. cond_dim = 0 is the plain VAE.

SparseConvVAE (below) is the same network with a spike-and-slab latent (Variational Sparse Coding,
Tonolini et al., UAI 2020): every latent dimension is either exactly 0 ("spike") or Gaussian ("slab").

Interface shared by both models (used by vae_train.py):
    encode(x, cond)           -> posterior parameters: (mu, logvar) or, sparse, (mu, logvar, gamma)
    sample_z(*posterior)      -> z (sampled in training, deterministic in eval)
    decode(z, cond)           -> reconstruction
    forward(x, cond)          -> (reconstruction, *posterior)
    kl_per_sample(*posterior) -> (N,) KL(q || prior) in nats
    embed(x, cond)            -> deterministic code for downstream use (mu; sparse: gated mu with exact zeros)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from . import config as c


def _per_layer(value, n_layers, name):
    """int -> same value for every layer; sequence -> must have one entry per layer."""
    if isinstance(value, int):
        return (value,) * n_layers
    value = tuple(value)
    if len(value) != n_layers:
        raise ValueError(f"{name} must be an int or have {n_layers} entries (got {len(value)}).")
    return value


class ConvVAE(nn.Module):
    sparse = False

    def __init__(self, latent=64, hidden=(16, 32, 64), kernel=(3, 3, 5), strides=(2, 2, 1),
                 padding=(1, 1, 0), batchnorm=False, in_ch=None, size=None,
                 out_act="sigmoid", mask_diag=True, fc_act=True, logvar_clamp=(-10.0, 2.0),
                 cond_dim=0, cond_in_encoder=True):
        super().__init__()
        in_ch = len(c.freq_bands) if in_ch is None else in_ch
        size = c.N_channels if size is None else size

        if not hidden:
            raise ValueError("hidden must be non-empty.")
        if out_act not in ("sigmoid", "none"):
            raise ValueError(f"out_act must be 'sigmoid' or 'none' (got {out_act!r}).")
        if cond_dim < 0:
            raise ValueError("cond_dim must be >= 0.")
        n_layers = len(hidden)
        kernel = _per_layer(kernel, n_layers, "kernel")
        strides = _per_layer(strides, n_layers, "strides")
        padding = _per_layer(padding, n_layers, "padding")
        if any(k < 1 for k in kernel):
            raise ValueError("All kernels must be >= 1.")
        if any(s < 1 for s in strides):
            raise ValueError("All strides must be >= 1.")
        if any(p < 0 for p in padding):
            raise ValueError("All paddings must be >= 0.")

        conv_out = lambda n, k, s, p: (n + 2 * p - k) // s + 1
        convT_out = lambda n, k, s, p: (n - 1) * s - 2 * p + k
        sizes = [size]
        for k, s, p in zip(kernel, strides, padding):
            sizes.append(conv_out(sizes[-1], k, s, p))
        if min(sizes) < 1:
            raise ValueError(f"Convolutions produce an invalid spatial size: {sizes}")

        def norm_act(ch):
            return ([nn.BatchNorm2d(ch)] if batchnorm else []) + [nn.ReLU()]

        chans = (in_ch, *hidden)
        enc, dec_blocks = [], []
        for i, (k, s, p) in enumerate(zip(kernel, strides, padding)):
            enc += [nn.Conv2d(chans[i], chans[i + 1], k, s, p), *norm_act(chans[i + 1])]
            out_pad = sizes[i] - convT_out(sizes[i + 1], k, s, p)
            if not 0 <= out_pad < s:
                raise ValueError(
                    f"layer {i}: kernel={k}, stride={s}, padding={p} do not reconstruct "
                    f"{sizes[i]} from {sizes[i + 1]} (output_padding={out_pad})."
                )
            block = [nn.ConvTranspose2d(chans[i + 1], chans[i], k, s, p, output_padding=out_pad)]
            block += norm_act(chans[i]) if i > 0 else []   # last decoder layer: raw output
            dec_blocks.append(block)

        self.latent, self.cond_dim = latent, cond_dim
        self.out_act, self.mask_diag, self.fc_act = out_act, mask_diag, fc_act
        self.logvar_clamp = tuple(logvar_clamp) if logvar_clamp is not None else None
        self.cond_in_encoder = bool(cond_in_encoder) and cond_dim > 0
        self.bottleneck_shape = (hidden[-1], sizes[-1], sizes[-1])
        flat = hidden[-1] * sizes[-1] ** 2
        enc_dim = flat + (cond_dim if self.cond_in_encoder else 0)
        self.enc_dim = enc_dim

        self.encoder = nn.Sequential(*enc, nn.Flatten())
        self.fc_mu = nn.Linear(enc_dim, latent)
        self.fc_logvar = nn.Linear(enc_dim, latent)
        self.fc = nn.Linear(latent + cond_dim, flat)
        self.decoder = nn.Sequential(*[layer for block in reversed(dec_blocks) for layer in block])
        if mask_diag:
            self.register_buffer("offdiag_mask", 1.0 - torch.eye(size))

        self.config = dict(latent=latent, hidden=tuple(hidden), kernel=kernel, strides=strides,
                           padding=padding, batchnorm=batchnorm, in_ch=in_ch, size=size,
                           out_act=out_act, mask_diag=mask_diag, fc_act=fc_act,
                           logvar_clamp=self.logvar_clamp, cond_dim=cond_dim,
                           cond_in_encoder=cond_in_encoder)
        self.reset_parameters()

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def reset_parameters(self):
        """Xavier everywhere, He (ReLU gain) for layers followed by ReLU; mu/logvar heads and the
        output layer stay Xavier. Resets BatchNorm stats/affine params to prevent cross-fold leakage."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)  # type: ignore
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
        for m in self.encoder:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if self.fc_act:
            nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        convts = [m for m in self.decoder if isinstance(m, nn.ConvTranspose2d)]
        for m in convts[:-1]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @torch.no_grad()
    def set_output_bias(self, band_means):
        """Start the output near each band's typical value (logit of the mean for a sigmoid output)."""
        last = [m for m in self.decoder if isinstance(m, nn.ConvTranspose2d)][-1]
        p = band_means.to(device=last.bias.device, dtype=last.bias.dtype)  # type: ignore
        last.bias.copy_(torch.logit(p.clamp(1e-6, 1 - 1e-6)) if self.out_act == "sigmoid" else p)  # type: ignore

    # ---------------------------------------------------------------- forward
    def _cond(self, cond, n):
        if cond is None:
            raise ValueError(f"model was built with cond_dim={self.cond_dim}: "
                             f"pass cond of shape (N, {self.cond_dim}).")
        if cond.shape != (n, self.cond_dim):
            raise ValueError(f"cond must have shape ({n}, {self.cond_dim}), got {tuple(cond.shape)}.")
        return cond

    def _features(self, x, cond=None):
        if self.mask_diag:
            x = x * self.get_buffer("offdiag_mask")
        h = self.encoder(x)
        if self.cond_in_encoder:
            h = torch.cat([h, self._cond(cond, x.shape[0])], dim=1)
        return h

    def _mu_logvar(self, h):
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        if self.logvar_clamp is not None:
            logvar = torch.clamp(logvar, *self.logvar_clamp)
        return mu, logvar

    def encode(self, x, cond=None):
        """x: (N, bands, C, C), cond: (N, cond_dim) -> mu, logvar: (N, latent)."""
        return self._mu_logvar(self._features(x, cond))

    def sample_z(self, mu, logvar):
        """Reparameterization trick while training, deterministic z = mu in eval / test / predict."""
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    @staticmethod
    def kl_per_sample(mu, logvar):
        """KL(q(z|x) || N(0, I)) in nats, shape (N,)."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)

    def embed(self, x, cond=None):
        """Deterministic code of x for downstream use (feature extraction), shape (N, latent)."""
        return self.encode(x, cond)[0]

    def decode(self, z, cond=None):
        """z: (N, latent), cond: (N, cond_dim) -> (N, bands, C, C). Pass another cond to swap the condition."""
        if self.cond_dim:
            z = torch.cat([z, self._cond(cond, z.shape[0])], dim=1)
        h = self.fc(z)
        if self.fc_act:
            h = F.relu(h)
        x = self.decoder(h.unflatten(1, self.bottleneck_shape))
        if self.out_act == "sigmoid":
            x = torch.sigmoid(x)
        if self.mask_diag:
            x = x * self.get_buffer("offdiag_mask")
        return x

    def forward(self, x, cond=None):
        post = self.encode(x, cond)                    # (mu, logvar) or (mu, logvar, gamma)
        return (self.decode(self.sample_z(*post), cond), *post)


# ======================================================================================
# Sparse VAE: spike-and-slab latent (Variational Sparse Coding)
# ======================================================================================
class SparseConvVAE(ConvVAE):
    """Variational Sparse Coding (Tonolini, Jensen & Murray-Smith, UAI 2020), conv architecture of ConvVAE.

        prior      p(z_j)    = alpha * N(0, 1) + (1 - alpha) * delta_0
        posterior  q(z_j|x)  = gamma_j * N(mu_j, sigma_j^2) + (1 - gamma_j) * delta_0

    gamma, mu, logvar come from three linear heads on the encoder features (gamma = sigmoid(logit)).
    KL (closed form, per sample, summed over latent dimensions):
        sum_j  gamma_j * KL(N(mu_j, sigma_j^2) || N(0, 1))  +  KL(Bernoulli(gamma_j) || Bernoulli(alpha))

    Training: z = eta * (mu + sigma * eps) with eta ~ Bernoulli(gamma) drawn as 1[gamma + rho - 1 > 0],
    rho ~ U(0, 1); the backward pass uses a steep sigmoid of that step (straight-through), so spikes are
    exact zeros during training too. Eval: z = 1[gamma > 0.5] * mu (deterministic, exact zeros).
    The ConvVAE options (architecture, conditioning, output bias, init) carry over unchanged.

    alpha          prior probability that a dimension is active (the main sparsity knob; sweep it)
    gate_steepness steepness of the sigmoid used for the spike gate's backward pass
    init_gamma     initial gamma (bias of the gamma head); 0.5 lets every unit start half-active
    """
    sparse = True

    def __init__(self, *, alpha=0.1, gate_steepness=50.0, init_gamma=0.5, **kw):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1) (got {alpha}).")
        if gate_steepness <= 0:
            raise ValueError("gate_steepness must be > 0.")
        if not 0.0 < init_gamma < 1.0:
            raise ValueError("init_gamma must be in (0, 1).")
        super().__init__(**kw)
        self.alpha, self.gate_steepness, self.init_gamma = float(alpha), float(gate_steepness), float(init_gamma)
        self.fc_gamma = nn.Linear(self.enc_dim, self.latent)
        self.config.update(alpha=self.alpha, gate_steepness=self.gate_steepness, init_gamma=self.init_gamma)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()                      # Xavier for every Linear, incl. fc_gamma once it exists
        fc_gamma = getattr(self, "fc_gamma", None)      # (not there yet when ConvVAE.__init__ calls this)
        if fc_gamma is not None:
            nn.init.constant_(fc_gamma.bias, math.log(self.init_gamma / (1.0 - self.init_gamma)))

    def encode(self, x, cond=None):
        """-> mu, logvar, gamma: each (N, latent); gamma in (0, 1) is the probability that the unit is active."""
        h = self._features(x, cond)
        mu, logvar = self._mu_logvar(h)
        return mu, logvar, torch.sigmoid(self.fc_gamma(h))

    @staticmethod
    def code(mu, gamma):
        """Deterministic sparse code: mu where gamma > 0.5, exactly 0 elsewhere."""
        return (gamma > 0.5).to(mu.dtype) * mu

    def sample_z(self, mu, logvar, gamma):
        if not self.training:
            return self.code(mu, gamma)
        slab = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        u = gamma + torch.rand_like(gamma) - 1.0                 # u > 0 with probability gamma
        soft = torch.sigmoid(self.gate_steepness * u)
        eta = (u > 0).to(soft.dtype) + (soft - soft.detach())    # hard 0/1 forward, sigmoid backward
        return eta * slab

    def kl_per_sample(self, mu, logvar, gamma):
        g = gamma.clamp(1e-6, 1.0 - 1e-6)
        slab = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)   # KL(N(mu, sigma^2) || N(0, 1))
        bern = (g * (g.log() - math.log(self.alpha))
                + (1.0 - g) * ((1.0 - g).log() - math.log(1.0 - self.alpha)))
        return (g * slab + bern).sum(-1)

    def embed(self, x, cond=None):
        mu, _, gamma = self.encode(x, cond)
        return self.code(mu, gamma)


# ======================================================================================
# Latent diagnostics (mu, logvar: (exams, windows, latent) numpy arrays from vae_train.encode_bags)
# ======================================================================================
def active_units(mu, logvar, thr=0.01):
    """mu, logvar: (exams, windows, latent) numpy arrays."""
    mu = mu.reshape(-1, mu.shape[-1])
    logvar = logvar.reshape(-1, logvar.shape[-1])
    var_mu = mu.var(axis=0)                           # Burda et al.: active if Var_x[mu(x)] > thr
    kl_dim = (-0.5 * (1 + logvar - mu ** 2 - np.exp(logvar))).mean(axis=0)   # nats per dimension
    return int((var_mu > thr).sum()), var_mu, kl_dim


def plot_latent_usage(mu, logvar, thr=0.01):
    """Per-dimension KL, Var_x[mu] and mean posterior std, all sorted by KL (largest first).
    mu, logvar: (exams, windows, latent) numpy arrays, e.g. from encode_bags on the validation set.
    Returns (fig, order, n_active); order[:n] are the dimension indices, most informative first."""
    n_active, var_mu, kl_dim = active_units(mu, logvar, thr)
    std = np.exp(0.5 * logvar).mean(axis=(0, 1))      # mean posterior std per dimension
    order = np.argsort(-kl_dim)
    x = np.arange(len(order))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(x, kl_dim[order])
    ax[0].set_title(f"KL per dimension (sum = {kl_dim.sum():.1f} nats)")
    ax[0].set_ylabel("nats per window")

    ax[1].bar(x, var_mu[order])
    ax[1].axhline(thr, ls="--", c="k", label=f"threshold {thr}")
    ax[1].set_yscale("log")
    ax[1].set_ylim(bottom=min(thr / 10, 1e-5))
    ax[1].set_title(f"Var[mu] per dimension ({n_active} active)")
    ax[1].legend()

    ax[2].bar(x, std[order])
    ax[2].axhline(1.0, ls=":", c="k", label="prior std = 1")
    ax[2].set_title("mean posterior std per dimension")
    ax[2].legend()

    for a in ax:
        a.set_xlabel("dimension (sorted by KL)")
    fig.tight_layout()
    print(f"active units: {n_active}/{len(order)}   total KL: {kl_dim.sum():.1f} nats")
    return fig, order, n_active


# ======================================================================================
# Sparse-latent diagnostics (z, gamma: (exams, windows, latent) numpy arrays from vae_train.encode_bags_sparse)
# ======================================================================================
def sparse_usage(z, gamma):
    """Usage statistics of a sparse code. A unit is 'on' in a window when gamma > 0.5 (the eval gate).
    Returns (on_freq, var_z, l0): on_freq (latent,) = fraction of windows where each unit is on,
    var_z (latent,) = Var[z] per unit, l0 (exams, windows) = number of units on per window."""
    on = gamma > 0.5
    on_freq = on.reshape(-1, on.shape[-1]).mean(axis=0)
    var_z = z.reshape(-1, z.shape[-1]).var(axis=0)
    return on_freq, var_z, on.sum(-1)


def plot_sparse_latent_usage(z, gamma, alpha=None, thr=0.01):
    """On-frequency and Var[z] per dimension (sorted by on-frequency, most used first) and the histogram of
    active units per window. alpha (optional) draws the prior sparsity level for comparison.
    A dimension counts as used if it is on in more than `thr` of the windows.
    Returns (fig, order, n_used); order[:n] are the dimension indices, most used first."""
    on_freq, var_z, l0 = sparse_usage(z, gamma)
    order = np.argsort(-on_freq)
    x = np.arange(len(order))
    n_used = int((on_freq > thr).sum())

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(x, on_freq[order])
    if alpha is not None:
        ax[0].axhline(alpha, ls=":", c="k", label=f"prior alpha = {alpha}")
        ax[0].legend()
    ax[0].set_title(f"fraction of windows each unit is on ({n_used} used)")
    ax[0].set_ylim(0, 1)

    ax[1].bar(x, var_z[order])
    ax[1].set_yscale("log")
    ax[1].set_title("Var[z] per dimension (same order)")

    ax[2].hist(l0.ravel(), bins=np.arange(-0.5, len(order) + 1.5, 1.0))
    ax[2].axvline(l0.mean(), ls="--", c="k", label=f"mean = {l0.mean():.1f}")
    ax[2].set_title("active units per window (L0)")
    ax[2].set_xlabel("number of units on")
    ax[2].legend()

    for a in ax[:2]:
        a.set_xlabel("dimension (sorted by on-frequency)")
    fig.tight_layout()
    print(f"used units: {n_used}/{len(order)}   mean L0 per window: {l0.mean():.1f}")
    return fig, order, n_used