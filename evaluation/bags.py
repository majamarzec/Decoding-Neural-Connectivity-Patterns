"""Bag handling. A bag = all windows of one exam, an array (n_windows, ...); a split = a list of bags."""
import numpy as np
import torch


def extract_raw_from_loader(loader):
    """
    Raw bags (exams, windows, bands, C, C) and labels (exams,) from a DataLoader of (bag, label[, cond]).
    Returns (x, y), or (x, y, cond) with cond (exams, cond_dim) when the loader yields a condition.
    Needs the same number of windows in every bag (for now set for 150).
    """
    xs, ys, cs = [], [], []
    for batch in loader:
        xs.append(batch[0].numpy())
        ys.append(np.asarray(batch[1]))
        if len(batch) > 2:
            cs.append(batch[2].numpy())
    if cs:
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(cs, axis=0)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def flatten_offdiag_bags(raw_bags, eeg_channels=None):
    """(exams, windows, bands, C, C) or a list of (w_i, bands, C, C) -> list of (w_i, bands * C * (C - 1))
    off-diagonal features. C is read from the data unless eeg_channels is given."""
    bags = [np.asarray(b) for b in raw_bags]
    if not bags:
        return []
    offdiag = ~np.eye(bags[0].shape[-1] if eeg_channels is None else eeg_channels, dtype=bool)
    return [b[..., offdiag].reshape(len(b), -1).astype(np.float32) for b in bags]


def mean_pool_bags(bags):
    """List of (w_i, d) bags -> (n_bags, d) mean embedding per bag."""
    return np.stack([np.asarray(b).mean(0) for b in bags])


def bag_layout(bags):
    """All windows stacked into one tensor, plus how to find each bag in it:
    sizes (n_bags,), offsets (n_bags,), X (total_windows, d)."""
    sizes = np.array([len(b) for b in bags])
    assert sizes.min() > 0
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    X = torch.as_tensor(np.concatenate(bags), dtype=torch.float32)
    return sizes, offsets, X


def pad_bags(bags):
    """List of (n_i, d) arrays -> zero-padded CPU tensor (B, N_max, d) and bool mask (B, N_max)."""
    sizes = np.array([len(b) for b in bags])
    assert sizes.min() > 0
    X = torch.zeros(len(bags), int(sizes.max()), bags[0].shape[1])
    M = torch.zeros(len(bags), int(sizes.max()), dtype=torch.bool)
    for i, b in enumerate(bags):
        X[i, :len(b)] = torch.as_tensor(b, dtype=torch.float32)
        M[i, :len(b)] = True
    return X, M
