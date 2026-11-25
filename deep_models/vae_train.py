"""Training, evaluation and embedding extraction for ConvVAE / SparseConvVAE / conditional ConvVAE (CVAE).

Loss convention = notebook convention: per-sample SUM of squared errors (off-diagonal only) + beta * KL,
so beta = 2 * sigma^2 of a Gaussian decoder.

Batches are (x, y) or (x, y, cond) with x = (exams, windows, bands, C, C). A conditional model
(model.cond_dim > 0) needs a dataset built with return_cond=True; all windows of an exam share its cond.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from . import config as c
from .viz import heatmap_bands, compare_heatmaps_model
from .train_utils import FoldR2, mean_offdiag_per_band, unpack_batch


# ======================================================================================
# 1. Initialisation
# ======================================================================================
def init_model(model, loader, n_batches=None):
    """Xavier/He weights, output bias at the typical value of each band (needs the training loader)."""
    model.reset_parameters()
    model.set_output_bias(mean_offdiag_per_band(loader, n_batches, offdiag=model.mask_diag))
    return model


# ======================================================================================
# 2. Loss, beta schedule, optimiser
# ======================================================================================
def _offdiag(x):
    mask = ~torch.eye(x.shape[-1], dtype=torch.bool, device=x.device)
    return x[..., mask]


def _kl_per_sample(model, posterior):
    if hasattr(model, "kl_per_sample"):
        return model.kl_per_sample(*posterior)
    mu, log_var = posterior
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)


def loss_function(model, x_rec, x, posterior, beta):
    n = x.shape[0]
    recon = F.mse_loss(_offdiag(x_rec), _offdiag(x), reduction="sum") / n   # off-diag SSE per sample
    kld = _kl_per_sample(model, posterior).mean()                          # nats per sample
    return recon + beta * kld, recon, kld


def beta_at(epoch, beta_max, anneal_start, anneal_epochs):
    """0 before anneal_start, linear ramp over anneal_epochs, then beta_max."""
    if anneal_epochs <= 0:
        return beta_max
    t = (epoch - anneal_start) / anneal_epochs
    return beta_max * min(max(t, 0.0), 1.0)


def make_optimizer(model, lr=1e-3, weight_decay=1e-5):
    """AdamW; weight decay only on conv/linear weights, not on biases, norm layers or latent heads."""
    head_modules = [model.fc_mu, model.fc_logvar]
    if getattr(model, "fc_gamma", None) is not None:
        head_modules.append(model.fc_gamma)
    heads = {id(p) for m in head_modules for p in m.parameters()}

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 and id(p) not in heads else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


# ======================================================================================
# 3. Train / evaluate
# ======================================================================================
def train_one_epoch(model, loader, optimizer, device, beta):
    model.train()
    tot = rec = kl = 0.0
    n = 0
    for batch in loader:
        x, cond = unpack_batch(batch, device, model.cond_dim)
        x_rec, *posterior = model(x, cond)
        loss, r, k = loss_function(model, x_rec, x, posterior, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = x.shape[0]
        tot += loss.item() * bs
        rec += r.item() * bs
        kl += k.item() * bs
        n += bs
    return tot / n, rec / n, kl / n


@torch.no_grad()
def evaluate(model, loader, device, beta, offdiag_r2=True):
    """Deterministic (z = mu, or gated mu for SparseConvVAE). Accumulated over the whole loader."""
    model.eval()
    n, numel, sse_all, kl_sum = 0, 0, 0.0, 0.0
    r2 = FoldR2(offdiag=offdiag_r2)
    for batch in loader:
        x, cond = unpack_batch(batch, device, model.cond_dim)
        x_rec, *posterior = model(x, cond)
        x_off, rec_off = _offdiag(x), _offdiag(x_rec)
        n += x.shape[0]
        numel += x_off.numel()
        sse_all += F.mse_loss(rec_off, x_off, reduction="sum").item()
        kl_sum += _kl_per_sample(model, posterior).sum().item()
        r2.update(x, x_rec)
    recon, mse, kld = sse_all / n, sse_all / numel, kl_sum / n
    return {"total": recon + beta * kld, "recon": recon, "mse": mse, "kld": kld, "r2": r2.compute()}


def fit(model, trainloader, valloader, device, epochs=150, lr=1e-3, weight_decay=1e-5,
        beta_max=5e-4, anneal_start=50, anneal_epochs=50, use_plateau=True,
        ckpt_path="best_vae.pt"):
    optimizer = make_optimizer(model, lr, weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
                 if use_plateau else None)
    anneal_end = anneal_start + anneal_epochs
    if epochs <= anneal_end:
        print(f"warning: epochs ({epochs}) <= end of annealing ({anneal_end}); "
              f"no best checkpoint will be saved, the last epoch is kept")

    keys = ["beta", "train_total", "train_recon", "train_kld",
            "val_total", "val_recon", "val_mse", "val_kld", "val_r2"]
    history = {k: [] for k in keys}
    best_val, best_state = float("inf"), None

    for epoch in range(epochs):
        beta = beta_at(epoch, beta_max, anneal_start, anneal_epochs)
        tr = train_one_epoch(model, trainloader, optimizer, device, beta)
        va = evaluate(model, valloader, device, beta)

        for k, v in zip(keys, (beta, *tr, va["total"], va["recon"], va["mse"], va["kld"], va["r2"])):
            history[k].append(v)

        if epoch >= anneal_end:
            if va["total"] < best_val:
                best_val = va["total"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt_path)
            if scheduler is not None:
                scheduler.step(va["total"])

        print(f"epoch {epoch + 1:3d}/{epochs}  beta {beta:.2e}  "
              f"train {tr[0]:.4f}  val {va['total']:.4f}  "
              f"recon {va['recon']:.4f}  MSE {va['mse']:.5f}  "
              f"KL {va['kld']:.2f}  R2 {va['r2']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ======================================================================================
# 4. Plots and embeddings for the MIL step
# ======================================================================================
def plot_history(history, skip=1):
    h = {k: np.asarray(v)[skip:] for k, v in history.items()}
    ep = np.arange(skip, skip + len(h["beta"]))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    ax[0, 0].plot(ep, h["train_total"], label="loss (train)")
    ax[0, 0].plot(ep, h["val_total"], label="loss (val)")
    ax[0, 0].set_title("beta-weighted loss: recon + beta*KL")

    ax[0, 1].plot(ep, h["train_recon"], label="reconstruction (train)")
    ax[0, 1].plot(ep, h["val_recon"], label="reconstruction (val)")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("reconstruction (off-diag SSE per sample)")

    ax[1, 0].plot(ep, h["train_kld"], label="KL (train)")
    ax[1, 0].plot(ep, h["val_kld"], label="KL (val)")
    ax[1, 0].set_title("KL divergence (nats per window)")
    twin = ax[1, 0].twinx()
    twin.plot(ep, h["beta"], "k:", label="beta")
    twin.set_ylabel("beta")

    ax[1, 1].plot(ep, h["val_r2"], label="R2 (val)")
    ax[1, 1].set_title("R2 vs. predicting the mean matrix")

    for a in ax.ravel():
        a.set_xlabel("epoch")
        if a is not ax[1, 0]:
            a.legend()

    lines = ax[1, 0].get_lines() + twin.get_lines()
    ax[1, 0].legend(lines, [l.get_label() for l in lines])

    fig.tight_layout()
    return fig


@torch.no_grad()
def encode_bags(model, loader, device):
    """Posterior means/log-vars per window, shaped (exams, windows, latent), plus exam labels.
    Needs the same number of windows in every bag (e.g. 150) so batches can be concatenated.
    For a conditional model the loader must also return the condition (return_cond=True).
    For SparseConvVAE, use encode_bags_sparse (for diagnostics) or embed_bags (for gated MIL features)."""
    model.eval()
    mus, logvars, ys = [], [], []
    for batch in loader:
        e, w = batch[0].shape[:2]
        x, cond = unpack_batch(batch, device, model.cond_dim)
        post = model.encode(x, cond)
        mu, log_var = post[0], post[1]
        mus.append(mu.view(e, w, -1).cpu())
        logvars.append(log_var.view(e, w, -1).cpu())
        ys.append(torch.as_tensor(batch[1]))
    return torch.cat(mus).numpy(), torch.cat(logvars).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def encode_bags_sparse(model, loader, device):
    """Gated sparse codes z and gate probabilities gamma, shaped (exams, windows, latent), plus exam labels."""
    model.eval()
    zs, gammas, ys = [], [], []
    for batch in loader:
        e, w = batch[0].shape[:2]
        x, cond = unpack_batch(batch, device, model.cond_dim)
        mu, _, gamma = model.encode(x, cond)
        z = model.code(mu, gamma)
        zs.append(z.view(e, w, -1).cpu())
        gammas.append(gamma.view(e, w, -1).cpu())
        ys.append(torch.as_tensor(batch[1]))
    return torch.cat(zs).numpy(), torch.cat(gammas).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def embed_bags(model, loader, device):
    """Deterministic embeddings per window for downstream MIL: (exams, windows, latent), plus exam labels.
    Returns mu for ConvVAE and exact-zero gated mu for SparseConvVAE."""
    model.eval()
    zs, ys = [], []
    for batch in loader:
        e, w = batch[0].shape[:2]
        x, cond = unpack_batch(batch, device, model.cond_dim)
        z = model.embed(x, cond) if hasattr(model, "embed") else model.encode(x, cond)[0]
        zs.append(z.view(e, w, -1).cpu())
        ys.append(torch.as_tensor(batch[1]))
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def plot_reconstructions(model, loader, device, exam_idx=0, window_indices=(0, 1, 2)):
    """Original vs reconstruction vs squared error using viz.compare_heatmaps_model."""
    model.eval()
    batch = next(iter(loader))
    single_batch = [torch.as_tensor(b)[exam_idx : exam_idx + 1] for b in batch]
    x, cond = unpack_batch(single_batch, device, model.cond_dim)
    x_rec, *_ = model(x, cond)

    # Restore (1, windows, bands, C, C) for compare_heatmaps_model
    x_5d = x.unsqueeze(0)
    rec_5d = x_rec.unsqueeze(0)

    for wi in window_indices:
        compare_heatmaps_model(x_5d, rec_5d, recon_i=0, window_i=wi)


def plot_tsne(val_mu, val_y, random_state=0):
    """t-SNE of all validation window embeddings (val_mu or gated val_z, plus val_y)."""
    _, n_windows, latent_dim = val_mu.shape
    z_flat = val_mu.reshape(-1, latent_dim)
    instance_labels = np.repeat(val_y, n_windows)

    z_tsne = TSNE(n_components=2, random_state=random_state).fit_transform(z_flat)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=instance_labels, cmap="coolwarm", s=5)
    fig.colorbar(sc, ax=ax)
    ax.set(title="t-SNE of Latent Space", xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
    plt.show()
    return fig


@torch.no_grad()
def plot_prior_samples(model, device, num_samples=7, cond=None):
    """Sample from the prior (N(0, I), or spike-and-slab alpha * N(0, I) for SparseConvVAE) and decode."""
    model.eval()
    z = torch.randn((num_samples, model.latent), device=device)
    if getattr(model, "sparse", False):
        z = z * (torch.rand_like(z) < model.alpha).to(z.dtype)
    generated = model.decode(z, cond=cond)

    for i in range(num_samples):
        heatmap_bands(generated[i], input_type="reconstructed", title=f"Prior sample {i}")
        plt.show()