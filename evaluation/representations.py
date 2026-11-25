"""Turn raw / encoded bags into the representations the probes are trained on."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_bags(raw_flat_splits, n_components=64, seed=0):
    """Fits StandardScaler + PCA on train windows ONLY, returns {split: [(w_i, n_components), ...]}."""
    train = np.concatenate(raw_flat_splits["train"])
    scaler = StandardScaler().fit(train)
    pca = PCA(n_components=n_components, random_state=seed).fit(scaler.transform(train))
    return {
        s: [pca.transform(scaler.transform(b)).astype(np.float32) for b in bl]
        for s, bl in raw_flat_splits.items()
    }


# ---- continuous latents (VAE) ---------------------------------------------------------------------
def active_mask(bag_list, thr=0.01):
    """Units whose variance across windows exceeds thr. Takes a 3D array (exams, windows, latent) or a list
    of 2D (windows_i, latent) arrays; the same rule works for sparse codes (inactive units stay at 0)."""
    flat = np.concatenate([np.asarray(b) for b in bag_list], axis=0)   # (exams*windows, latent)
    return flat.var(axis=0) > thr


def select_dims(rep_bags, mask):
    """rep_bags: {split: [bag (windows, latent), ...] or (exams, windows, latent)} -> restricted to mask columns."""
    return {s: [np.asarray(b)[:, mask] for b in bl] for s, bl in rep_bags.items()}


# ---- discrete codes (VQ-VAE) ----------------------------------------------------------------------
def make_code_histograms(idx_bags, code_book_size=64):
    """List of (n_windows, H*W) discrete code indices -> list of (n_windows, code_book_size) histograms
    (fraction of spatial positions that use each code, per window)."""
    hists = []
    for b in idx_bags:
        one_hot = np.eye(code_book_size, dtype=np.float32)[b]      # (n_windows, H*W, K)
        hists.append(one_hot.mean(axis=1))
    return hists


def active_code_mask(hist_bags, thr=1e-4):
    """Codes that are actually used across the dataset. hist_bags: list of (n_windows, K) arrays."""
    flat = np.concatenate(hist_bags)  # (total_windows, K)
    return flat.mean(axis=0) > thr
