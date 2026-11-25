import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------------------------
# VQ-VAE Specific Extractors & Helpers
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def vq_encode_raw_bags(vqvae, raw_bags, batch=256, C=5, H=17, W=17):
    """
    Returns both the continuous quantized embeddings (z_q) and the discrete codes (idx).
    z_qs: list of (n_windows, latent_channels * grid * grid) arrays.
    idxs: list of (n_windows, grid * grid) arrays.
    """
    vqvae.eval()
    z_qs, idxs = [], []
    for x in raw_bags:
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1, C, H, W)
        z_q_bag, idx_bag = [], []
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(DEVICE)
            z_e = vqvae.encoder(xb)
            _, quantized, idx = vqvae.vq(z_e)
            
            # Flatten the spatial grids for MIL processing
            z_q_bag.append(quantized.view(xb.shape[0], -1).cpu())
            idx_bag.append(idx.view(xb.shape[0], -1).cpu())
            
        z_qs.append(torch.cat(z_q_bag).numpy())
        idxs.append(torch.cat(idx_bag).numpy())
    return z_qs, idxs

def make_code_histograms(idx_bags, code_book_size=64):
    """
    Converts lists of spatial discrete VQ codes into normalized histograms.
    Input:  list of (n_windows, H*W) discrete index arrays
    Output: list of (n_windows, code_book_size) continuous histogram arrays
    """
    hists = []
    for b in idx_bags:
        # np.eye creates one-hot of shape (n_windows, H*W, K)
        one_hot = np.eye(code_book_size, dtype=np.float32)[b]
        # Average over the spatial locations (H*W) to get code frequency per window
        hists.append(one_hot.mean(axis=1))
    return hists

def active_code_mask(hist_bags, thr=1e-4):
    """
    Finds codes that are actually used across the dataset.
    hist_bags: list of (n_windows, K) arrays.
    """
    flat = np.concatenate(hist_bags)  # (total_windows, K)
    return flat.mean(axis=0) > thr

def extract_raw_from_loader(loader):
    """Extracts raw bags (exams, windows, 5, 17, 17) and labels (exams,) from a DataLoader."""
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy())
        ys.append(np.asarray(y))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def flatten_offdiag_bags(raw_bags, eeg_channels=17):
    """(exams, windows, 5, 17, 17) or list of (w_i, 5, 17, 17) -> list of (w_i, 5*272) off-diagonal features."""
    offdiag = ~np.eye(eeg_channels, dtype=bool)
    return [np.asarray(b)[..., offdiag].reshape(len(b), -1).astype(np.float32) for b in raw_bags]

def pca_bags(raw_flat_splits, n_components=64, seed=0):
    """Fits StandardScaler + PCA on train windows ONLY, returns {split: [(w_i, n_components), ...]}."""
    scaler = StandardScaler().fit(np.concatenate(raw_flat_splits["train"]))
    pca = PCA(n_components=n_components, random_state=seed).fit(
        scaler.transform(np.concatenate(raw_flat_splits["train"]))
    )
    return {
        s: [pca.transform(scaler.transform(b)).astype(np.float32) for b in bl]
        for s, bl in raw_flat_splits.items()
    }

def _vae_encode(vae, x):
    """Supports both the new ConvVAE (encoder -> (mu, log_var)) and legacy VAEs."""
    out = vae.encoder(x)
    if isinstance(out, tuple):
        return out
    fc_lv = getattr(vae, "fc_logvar", getattr(vae, "fc_log_var", None))
    return vae.fc_mu(out), fc_lv(out) #type:ignore


def mu_encoder(vae):
    return lambda x: _vae_encode(vae, x)[0]


def logvar_encoder(vae):
    return lambda x: _vae_encode(vae, x)[1]


@torch.no_grad()
def encode_raw_bags(vae, raw_bags, batch=256, C=5, H=17, W=17, device=None, stack=True):
    """Encodes raw bags using ConvVAE.
    - Moves batches to the model's device automatically.
    - Returns 3D numpy arrays (exams, windows, latent) if all bags have equal length and stack=True,
      otherwise returns lists of 2D numpy arrays [(windows_i, latent), ...].
    """
    vae.eval()
    if device is None:
        device = next(vae.parameters()).device
    mus, lvs = [], []
    for x in raw_bags:
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1, C, H, W)
        m, l = [], []
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            mu, lv = _vae_encode(vae, xb)
            m.append(mu.cpu())
            l.append(lv.cpu())
        mus.append(torch.cat(m).numpy())
        lvs.append(torch.cat(l).numpy())

    if stack and len({len(b) for b in mus}) == 1:
        return np.stack(mus), np.stack(lvs)
    return mus, lvs


def active_mask(bag_list, thr=0.01):
    """Works on either a 3D array (exams, windows, latent) or a list of 2D (windows_i, latent) arrays."""
    flat = np.concatenate([np.asarray(b) for b in bag_list], axis=0)   # (exams*windows, latent)
    return flat.var(axis=0) > thr


def select_dims(rep_bags, mask):
    """rep_bags: {split: [bag (windows, latent), ...] or (exams, windows, latent)}
    -> restricted to mask columns."""
    return {s: [np.asarray(b)[:, mask] for b in bl] for s, bl in rep_bags.items()}


class SoftmaxProbe(nn.Module):
    def __init__(self, d, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.fc(x)

def extract_raw_from_loader(loader):
    """Extracts raw bags (exams, windows, 5, 17, 17), labels (exams,), and cond (exams, cond_dim) if present."""
    xs, ys, cs = [], [], []
    for batch in loader:
        xs.append(batch[0].numpy())
        ys.append(np.asarray(batch[1]))
        if len(batch) > 2:
            cs.append(batch[2].numpy())
    if cs:
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(cs, axis=0)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


@torch.no_grad()
def embed_raw_bags(vae, raw_bags, cond_bags=None, batch=256, C=5, H=17, W=17, device=None, stack=True):
    """Deterministic embeddings for downstream MIL via vae.embed(x, cond):
      - ConvVAE:       mu
      - SparseConvVAE: gated mu = (gamma > 0.5) * mu (exact zeros on inactive spike units)
    Works for both unconditional (cond_dim=0) and conditional (cond_dim>0) models.
    """
    vae.eval()
    if device is None:
        device = next(vae.parameters()).device
    zs = []
    for idx, x in enumerate(raw_bags):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1, C, H, W)
        c_bag = None
        if cond_bags is not None:
            c_bag = torch.as_tensor(np.asarray(cond_bags[idx]), dtype=torch.float32).reshape(1, -1).expand(len(x), -1)

        z_bag = []
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            cb = c_bag[i:i + batch].to(device) if c_bag is not None else None
            z = vae.embed(xb, cb) if hasattr(vae, "embed") else _vae_encode(vae, xb)[0]
            z_bag.append(z.cpu())
        zs.append(torch.cat(z_bag).numpy())

    if stack and len({len(b) for b in zs}) == 1:
        return np.stack(zs)
    return zs

# ----------------------------------------------------------------------------------------------
# Attention-based MIL probe (gated attention, Ilse et al. 2018)
# ----------------------------------------------------------------------------------------------
class ABMIL(nn.Module):
    """Bag (B, N, d_in) + mask (B, N) -> (logits (B, n_classes), attention (B, N)).
    hidden=None: attention directly over the input features (minimal head);
    hidden=int : a Linear+ReLU+Dropout projection to `hidden` dims first."""

    def __init__(self, d_in, hidden=64, attn_dim=32, dropout=0.25, n_classes=2):
        super().__init__()
        if hidden:
            self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        else:
            self.proj = nn.Identity()
        d = hidden or d_in
        self.V = nn.Linear(d, attn_dim)
        self.U = nn.Linear(d, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, mask=None):
        h = self.proj(x)                                                    # (B, N, d)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)   # (B, N)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))                         # padded windows get zero weight
        a = torch.softmax(a, dim=1)
        z = self.drop((a.unsqueeze(-1) * h).sum(1))                         # (B, d)
        return self.head(z), a


def _pad_bags(bags):
    """List of (n_i, d) arrays -> zero-padded CPU tensor (B, N_max, d) and bool mask (B, N_max)."""
    sizes = np.array([len(b) for b in bags])
    assert sizes.min() > 0
    X = torch.zeros(len(bags), int(sizes.max()), bags[0].shape[1])
    M = torch.zeros(len(bags), int(sizes.max()), dtype=torch.bool)
    for i, b in enumerate(bags):
        X[i, :len(b)] = torch.as_tensor(b, dtype=torch.float32)
        M[i, :len(b)] = True
    return X, M


def abmil_scorer(bags, chunk=64):
    """score(model) -> P(class 1 = pathology) per bag. Same contract as pooled_scorer / bag_scorer."""
    X, M = _pad_bags(bags)

    @torch.no_grad()
    def score(model):
        model.eval()
        out = [F.softmax(model(X[i:i + chunk].to(DEVICE), M[i:i + chunk].to(DEVICE))[0], dim=-1)[:, 1].cpu()
               for i in range(0, len(X), chunk)]
        return torch.cat(out).numpy()

    return score


def fit_abmil(W_train, y_train, val_score, y_val, hidden=64, attn_dim=32, dropout=0.25,
              lr=1e-3, weight_decay=1e-2, batch_size=16, max_epochs=200, patience=25, seed=0):
    torch.manual_seed(seed)
    X, M = _pad_bags(W_train)
    y = torch.as_tensor(y_train, dtype=torch.long)

    model = ABMIL(X.shape[-1], hidden=hidden, attn_dim=attn_dim, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch_size):
            idx = perm[i:i + batch_size]
            xb, mb, yb = X[idx].to(DEVICE), M[idx].to(DEVICE), y[idx].to(DEVICE)
            loss = F.cross_entropy(model(xb, mb)[0], yb)        # one bag = one sample, no class weights
            opt.zero_grad()
            loss.backward()
            opt.step()

        auc = roc_auc_score(y_val, val_score(model))
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)  # type:ignore
    return model, best_auc, best_epoch


@torch.no_grad()
def abmil_attention(model, bags, chunk=64):
    """Per-bag attention weights as a list of 1-D arrays (padding removed)."""
    model.eval()
    X, M = _pad_bags(bags)
    A = torch.cat([model(X[i:i + chunk].to(DEVICE), M[i:i + chunk].to(DEVICE))[1].cpu()
                   for i in range(0, len(X), chunk)])
    return [A[i, :len(b)].numpy() for i, b in enumerate(bags)]


def attention_entropy(att_list):
    """Normalised entropy per bag: 1 = uniform (equivalent to mean pooling), 0 = one window takes all."""
    out = []
    for a in att_list:
        if len(a) < 2:
            out.append(1.0)
            continue
        a = np.clip(a, 1e-12, 1.0)
        out.append(float(-(a * np.log(a)).sum() / np.log(len(a))))
    return np.array(out)


# ----------------------------------------------------------------------------------------------
# Linear probes
# ----------------------------------------------------------------------------------------------
def pooled_scorer(P):
    """score(model) -> P(class 1 = pathology) for each mean-pooled bag."""
    X = torch.as_tensor(P, dtype=torch.float32, device=DEVICE)

    @torch.no_grad()
    def score(model):
        model.eval()
        return F.softmax(model(X), dim=-1)[:, 1].cpu().numpy()

    return score


def _bag_layout(bags):
    sizes = np.array([len(b) for b in bags])
    assert sizes.min() > 0
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    X = torch.as_tensor(np.concatenate(bags), dtype=torch.float32)
    return sizes, offsets, X


def bag_scorer(bags):
    sizes, offsets, X = _bag_layout(bags)

    def score(model):
        z = batched_logits(model, X)
        margin = (z[:, 1] - z[:, 0]).numpy().astype(np.float64)
        return 1.0 / (1.0 + np.exp(-np.add.reduceat(margin, offsets) / sizes))

    return score


@torch.no_grad()
def batched_logits(model, X, chunk=32768):
    """X is a CPU tensor; logits are computed on DEVICE chunk by chunk and returned on CPU."""
    model.eval()
    return torch.cat([model(X[i:i + chunk].to(DEVICE)).cpu() for i in range(0, len(X), chunk)])


@torch.no_grad()
def window_votes(model, X, threshold=0.5):
    """Per-window P(class 1) and hard vote. X stays on the CPU; the model runs on DEVICE in chunks."""
    X = torch.as_tensor(X, dtype=torch.float32)
    p1 = F.softmax(batched_logits(model, X), dim=-1)[:, 1].numpy()
    return p1, (p1 > threshold).astype(int)


def vote_scorer(bags, threshold=0.5):
    """HARD readout: score(model) -> fraction of windows with P(class 1) > threshold, per bag."""
    sizes, offsets, X = _bag_layout(bags)

    def score(model):
        _, vote = window_votes(model, X, threshold)
        return np.add.reduceat(vote.astype(np.float64), offsets) / sizes

    return score


def fit_probe(X, y, val_score, y_val, sample_w=None, lr=1e-2, batch_size=128, max_epochs=300, patience=30, seed=0):
    torch.manual_seed(seed)
    # data stays on the CPU; only minibatches go to the GPU
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    w = torch.ones(len(y)) if sample_w is None else torch.as_tensor(sample_w, dtype=torch.float32)

    model = SoftmaxProbe(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(y))                        # CPU permutation, matching the CPU data
        for i in range(0, len(y), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb, wb = X[idx].to(DEVICE), y[idx].to(DEVICE), w[idx].to(DEVICE)
            per_sample = F.cross_entropy(model(xb), yb, reduction="none")
            loss = (per_sample * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        auc = roc_auc_score(y_val, val_score(model))
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)  # type:ignore
    return model, best_auc, best_epoch


def report(rep, method, model, val_auc, best_epoch, test_score, y_test):
    auc_test = roc_auc_score(y_test, test_score(model))
    print(f"[{rep}] {method}  val AUC = {val_auc:.3f}  epoch = {best_epoch}  test AUC = {auc_test:.3f}")
    return {
        "Representation": rep,
        "Method": method,
        "Val AUC": val_auc,
        "Epoch": best_epoch,
        "Test AUC": auc_test}


DEFAULT_ABMIL_VARIANTS = {"ABMIL_linear": None, "ABMIL_h64": 64}   # name -> hidden size (None = no projection)


def run_nn_probes(bags, representations, y_train, y_val, y_test, return_artifacts=False,
                  abmil_variants=DEFAULT_ABMIL_VARIANTS, abmil_kw=None, abmil_seeds=(0,), **fit_kw):
    """fit_kw goes to the linear probes (fit_probe); abmil_kw goes to fit_abmil.
    Pass abmil_variants={} to skip the attention probes."""
    abmil_kw = abmil_kw or {}
    rows, artifacts = [], {}
    for rep in representations:
        W_train, W_val, W_test = (
            [np.asarray(b, dtype=np.float32) for b in bags[rep][s]]
            for s in ("train", "val", "test")
        )

        # PROBE1: nn.Linear + softmax on the mean latent embedding
        pool = lambda split: np.stack([b.mean(0) for b in split])
        P_train, P_val, P_test = pool(W_train), pool(W_val), pool(W_test)

        val_score, test_score = pooled_scorer(P_val), pooled_scorer(P_test)
        model, val_auc, ep = fit_probe(P_train, y_train, val_score, y_val, **fit_kw)
        rows.append(report(rep, "LinearBagProbe", model, val_auc, ep, test_score, y_test))

        # PROBE2: nn.Linear + softmax on instances with copied bag labels = weak labels (one trained model, two readouts)
        sizes = np.array([len(b) for b in W_train])
        X_windows = np.concatenate(W_train)
        y_windows = np.repeat(y_train, sizes)  # every window gets the label of the bag it came from
        w_windows = np.repeat(1.0 / sizes, sizes)  # each bag contributes equally, no class weights

        # model selection (early stopping) is on the SOFT val AUC for both readouts
        val_score = bag_scorer(W_val)
        model, val_auc, ep = fit_probe(X_windows, y_windows, val_score, y_val, sample_w=w_windows, **fit_kw)
        vote_val = roc_auc_score(y_val, vote_scorer(W_val)(model))
        rows.append(report(rep, "LinearInstanceProbeBagLabels_soft", model, val_auc, ep, bag_scorer(W_test), y_test))
        rows.append(report(rep, "LinearInstanceProbeBagLabels_vote", model, vote_val, ep, vote_scorer(W_test), y_test))
        artifacts[rep] = {"model": model, "X_val": np.concatenate(W_val)}

        # PROBE3: gated-attention MIL on bags, bag labels only. Selected on val AUC like the others;
        # one row per seed, the best-val seed's model is kept in artifacts[rep][variant_name].
        abmil_val, abmil_test = abmil_scorer(W_val), abmil_scorer(W_test)
        for tag, hid in abmil_variants.items():
            best = None
            for seed in abmil_seeds:
                m, v_auc, ep = fit_abmil(W_train, y_train, abmil_val, y_val, hidden=hid, seed=seed, **abmil_kw)
                row = report(rep, tag, m, v_auc, ep, abmil_test, y_test)
                row["Seed"] = seed
                rows.append(row)
                if best is None or v_auc > best[0]:
                    best = (v_auc, m)
            artifacts[rep][tag] = best[1]  # type:ignore
    return (rows, artifacts) if return_artifacts else rows


def plot_embedding_votes(embeddings, artifacts, exam_ids, window_labels, rep_of,
                         mode="prob", threshold=0.5, size=(6, 5)):
    """Rows = embeddings. Columns = by exam, by weak label, by instance-probe output.
    mode="prob": unthresholded P(class 1);  mode="vote": hard vote at `threshold`."""
    assert mode in ("prob", "vote")
    C0, C1 = plt.cm.coolwarm(0.0), plt.cm.coolwarm(1.0)  # type:ignore

    fig, axes = plt.subplots(len(embeddings), 3, figsize=(size[0] * 3, size[1] * len(embeddings)),
                             constrained_layout=True, squeeze=False)

    for row, (name, emb) in enumerate(embeddings.items()):
        art = artifacts[rep_of[row]]
        assert len(emb) == len(art["X_val"]) == len(exam_ids) == len(window_labels), \
            f"{name}: rows are not aligned (embedding / probe input / exam_ids / labels)"
        p1, vote = window_votes(art["model"], art["X_val"], threshold)

        ax = axes[row, 0]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=exam_ids, cmap="tab20", s=1, alpha=0.7, rasterized=True)
        ax.set_title(f"{name}: by bags")
        fig.colorbar(sc, ax=ax, label="Exam ID")

        ax = axes[row, 1]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=window_labels, cmap="coolwarm", vmin=0, vmax=1,
                        s=8, alpha=0.5, rasterized=True)
        ax.set_title(f"{name}: by exam's (weak) label")
        fig.colorbar(sc, ax=ax, label="Pathology")

        ax = axes[row, 2]
        if mode == "prob":
            order = np.argsort(np.abs(p1 - 0.5))      # confident windows drawn on top
            sc = ax.scatter(emb[order, 0], emb[order, 1], c=p1[order], cmap="coolwarm",
                            vmin=0, vmax=1, s=8, alpha=0.6, rasterized=True)
            ax.set_title(f"{name}: instance probe P(class 1)")
            fig.colorbar(sc, ax=ax, label="P(class 1)")
        else:
            ax.scatter(emb[:, 0], emb[:, 1], c=vote, cmap=ListedColormap([C0, C1]), vmin=0, vmax=1,
                       s=8, alpha=0.5, rasterized=True)
            ax.set_title(f"{name}: by predicted class (vote)")
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C0, label="vote: class 0"),
                               Line2D([], [], marker="o", ls="", color=C1, label="vote: class 1")],
                      loc="best")

        for ax in axes[row]:
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

    return fig, axes

