"""Training, evaluation and embedding extraction for ConvVQVAE (plain PyTorch).

Loss convention: MSE(x_rec, x) + vq_loss (vq_loss = codebook_loss + commitment_cost * commitment_loss),
matching the tutorial VQ-VAE relative weighting while also tracking SSE per sample for direct
comparison with ConvVAE. Batches are (x, y[, cond]) with x = (exams, windows, bands, C, C).
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .viz import heatmap_bands, compare_heatmaps_model
from .train_utils import FoldR2, mean_offdiag_per_band, unpack_batch


# ======================================================================================
# 1. Initialisation
# ======================================================================================
def init_model(model, loader, n_batches=None):
    """He/Xavier weights, output bias at the typical value of each band (needs the training loader)."""
    model.reset_parameters()
    model.set_output_bias(mean_offdiag_per_band(loader, n_batches, offdiag=model.mask_diag))
    return model


# ======================================================================================
# 2. Loss & optimiser
# ======================================================================================
def loss_function(x_rec, x, vq_loss):
    """Tutorial weighting: MSE + vq_loss. Also returns SSE per sample to compare with ConvVAE."""
    n = x.shape[0]
    mse = F.mse_loss(x_rec, x)
    sse_per_sample = F.mse_loss(x_rec, x, reduction="sum") / n
    return mse + vq_loss, mse, vq_loss, sse_per_sample


def make_optimizer(model, lr=1e-3, weight_decay=1e-5):
    """Adam; weight decay only on conv weights, not on biases, GroupNorms, or the codebook."""
    codebook = {id(p) for p in model.vq.parameters()}
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 and id(p) not in codebook else no_decay).append(p)
    return torch.optim.Adam(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


# ======================================================================================
# 3. Train / evaluate
# ======================================================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    tot = mse_sum = vq_sum = sse_sum = perp_sum = 0.0
    nb = 0
    K = model.vq.code_book_size
    for batch in loader:
        x, _ = unpack_batch(batch, device)
        x_rec, vq_loss, idx = model(x)
        loss, mse, vq, sse = loss_function(x_rec, x, vq_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = F.one_hot(idx.flatten(), K).float().mean(0)
            perp = torch.exp(-(probs * (probs + 1e-10).log()).sum()).item()

        tot += loss.item()
        mse_sum += mse.item()
        vq_sum += vq.item()
        sse_sum += sse.item()
        perp_sum += perp
        nb += 1
    return tot / nb, mse_sum / nb, vq_sum / nb, sse_sum / nb, perp_sum / nb


@torch.no_grad()
def evaluate(model, loader, device, offdiag_r2=True):
    """Everything is accumulated over the WHOLE loader, so R^2 and perplexity are fold-wide
    values, not a mean of per-batch values."""
    model.eval()
    n, n_el, sse_all, vq_weighted = 0, 0, 0.0, 0.0
    r2 = FoldR2(offdiag=offdiag_r2)
    K = model.vq.code_book_size
    code_counts = torch.zeros(K, dtype=torch.float64, device=device)

    for batch in loader:
        x, _ = unpack_batch(batch, device)
        x_rec, vq_loss, idx = model(x)
        bs = x.shape[0]
        n += bs
        n_el = x[0].numel()
        sse_all += F.mse_loss(x_rec, x, reduction="sum").item()
        vq_weighted += vq_loss.item() * bs
        code_counts += torch.bincount(idx.flatten(), minlength=K).double()
        r2.update(x, x_rec)

    recon_sse = sse_all / n
    mse = sse_all / (n * n_el)
    vq = vq_weighted / n
    probs = code_counts / code_counts.sum()
    perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum()).item()
    return {
        "total": mse + vq,
        "mse": mse,
        "recon": recon_sse,
        "vq": vq,
        "perplexity": perplexity,
        "r2": r2.compute(),
    }


def fit(model, trainloader, valloader, device, epochs=150, lr=1e-3, weight_decay=1e-5,
        use_plateau=True, ckpt_path="best_vqvae.pt"):
    optimizer = make_optimizer(model, lr, weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
                 if use_plateau else None)

    keys = ["train_total", "train_mse", "train_vq", "train_recon", "train_perplexity",
            "val_total", "val_mse", "val_vq", "val_recon", "val_perplexity", "val_r2"]
    history = {k: [] for k in keys}
    best_val, best_state = float("inf"), None

    for epoch in range(epochs):
        tr = train_one_epoch(model, trainloader, optimizer, device)
        va = evaluate(model, valloader, device)

        for k, v in zip(keys, (*tr, va["total"], va["mse"], va["vq"],
                               va["recon"], va["perplexity"], va["r2"])):
            history[k].append(v)

        if va["total"] < best_val:
            best_val = va["total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
        if scheduler is not None:
            scheduler.step(va["total"])

        print(f"epoch {epoch + 1:3d}/{epochs}  "
              f"train {tr[0]:.5f}  val {va['total']:.5f}  "
              f"recon(SSE) {va['recon']:.4f}  VQ {va['vq']:.5f}  "
              f"perp {va['perplexity']:.1f}  R2 {va['r2']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ======================================================================================
# 4. Plots and embeddings for the MIL step
# ======================================================================================
def plot_history(history, skip=1):
    h = {k: np.asarray(v)[skip:] for k, v in history.items()}
    ep = np.arange(skip, skip + len(h["train_total"]))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    ax[0, 0].plot(ep, h["train_total"], label="loss (train)")
    ax[0, 0].plot(ep, h["val_total"], label="loss (val)")
    ax[0, 0].set_title("total loss: MSE + VQ")

    ax[0, 1].plot(ep, h["train_recon"], label="reconstruction (train)")
    ax[0, 1].plot(ep, h["val_recon"], label="reconstruction (val)")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("reconstruction (SSE per sample)")

    ax[1, 0].plot(ep, h["train_vq"], label="VQ loss (train)")
    ax[1, 0].plot(ep, h["val_vq"], label="VQ loss (val)")
    ax[1, 0].set_title("VQ loss & codebook perplexity")
    twin = ax[1, 0].twinx()
    twin.plot(ep, h["val_perplexity"], "k:", label="perplexity (val)")
    twin.set_ylabel("perplexity")

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
    """Quantized embeddings and discrete code indices per window, shaped:
      z_q:     (exams, windows, latent_channels * grid * grid)  -- continuous features for MIL
      indices: (exams, windows, grid * grid)                    -- discrete code indices in [0, K)
      ys:      (exams,)
    Needs the same number of windows in every bag (e.g. 150) so batches can be concatenated."""
    model.eval()
    z_qs, idxs, ys = [], [], []
    for batch in loader:
        e, w = batch[0].shape[:2]
        x, _ = unpack_batch(batch, device)
        z_e = model.encoder(x)
        _, quantized, idx = model.vq(z_e)
        z_qs.append(quantized.view(e, w, -1).cpu())
        idxs.append(idx.view(e, w, -1).cpu())
        ys.append(torch.as_tensor(batch[1]))
    return torch.cat(z_qs).numpy(), torch.cat(idxs).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def plot_reconstructions(model, loader, device, exam_idx=0, window_indices=(0, 1, 2)):
    """Original vs reconstruction vs squared error using viz.compare_heatmaps_model."""
    model.eval()
    batch = next(iter(loader))
    single_batch = [torch.as_tensor(b)[exam_idx : exam_idx + 1] for b in batch]
    x, _ = unpack_batch(single_batch, device)
    x_rec, _, _ = model(x)

    # Restore (1, windows, bands, C, C) for compare_heatmaps_model
    x_5d = x.unsqueeze(0)
    rec_5d = x_rec.unsqueeze(0)

    for wi in window_indices:
        compare_heatmaps_model(x_5d, rec_5d, recon_i=0, window_i=wi)


def plot_tsne(val_z_q, val_y, random_state=0):
    """t-SNE of all validation window embeddings (val_z_q, val_y from encode_bags)."""
    _, n_windows, latent_dim = val_z_q.shape
    z_flat = val_z_q.reshape(-1, latent_dim)
    instance_labels = np.repeat(val_y, n_windows)

    z_tsne = TSNE(n_components=2, random_state=random_state).fit_transform(z_flat)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=instance_labels, cmap="coolwarm", s=5)
    fig.colorbar(sc, ax=ax)
    ax.set(title="t-SNE of Quantized Latent Space", xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
    plt.show()
    return fig


@torch.no_grad()
def plot_prior_samples(model, device, num_samples=7, probs=None):
    """Sample a (grid x grid) codebook index map (uniform, or from marginal `probs`) and decode."""
    model.eval()
    K = model.vq.code_book_size
    shape = (num_samples, model.grid, model.grid)
    if probs is None:
        idx = torch.randint(0, K, shape, device=device)
    else:
        p = torch.as_tensor(probs, device=device, dtype=torch.float32)
        idx = torch.multinomial(p, num_samples * model.grid * model.grid, replacement=True).view(*shape)

    quantized = model.vq.embedding(idx).permute(0, 3, 1, 2).contiguous()
    generated = model.decoder(quantized)

    for i in range(num_samples):
        heatmap_bands(generated[i], input_type="reconstructed", title=f"Prior sample {i}")
        plt.show()