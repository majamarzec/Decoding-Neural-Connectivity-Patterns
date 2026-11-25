"""Fitting the probes: one training loop, model selection on validation AUC with early stopping."""
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .bags import pad_bags
from .device import resolve_device
from .probes import ABMIL, SoftmaxProbe


def _fit_early_stopping(model, train_epoch, val_score, y_val, max_epochs, patience):
    """Call train_epoch() once per epoch; after each epoch score the validation bags. Keeps the weights of the
    best validation AUC and stops after `patience` epochs without improvement. Returns (model, best_auc, best_epoch)."""
    best_auc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_epoch()
        auc = roc_auc_score(y_val, val_score(model))
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)  # type:ignore
    return model, best_auc, best_epoch


def fit_probe(X, y, val_score, y_val, sample_w=None, lr=1e-2, batch_size=128, max_epochs=300, patience=30,
              seed=0, device=None):
    """Linear softmax probe on rows X (n, d) with labels y, optional per-row weights.
    val_score(model) -> P(class 1) per validation bag (see scoring.py); selection is on its AUC."""
    torch.manual_seed(seed)
    device = resolve_device(device)
    # data stays on the CPU; only minibatches go to the device
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    w = torch.ones(len(y)) if sample_w is None else torch.as_tensor(sample_w, dtype=torch.float32)

    model = SoftmaxProbe(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch():
        perm = torch.randperm(len(y))                        # CPU permutation, matching the CPU data
        for i in range(0, len(y), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb, wb = X[idx].to(device), y[idx].to(device), w[idx].to(device)
            per_sample = F.cross_entropy(model(xb), yb, reduction="none")
            loss = (per_sample * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return _fit_early_stopping(model, train_epoch, val_score, y_val, max_epochs, patience)


def fit_abmil(W_train, y_train, val_score, y_val, hidden=64, attn_dim=32, dropout=0.25,
              lr=1e-3, weight_decay=1e-2, batch_size=16, max_epochs=200, patience=25, seed=0, device=None):
    """Gated-attention MIL probe on bags (list of (n_i, d) arrays), bag labels only."""
    torch.manual_seed(seed)
    device = resolve_device(device)
    X, M = pad_bags(W_train)
    y = torch.as_tensor(y_train, dtype=torch.long)

    model = ABMIL(X.shape[-1], hidden=hidden, attn_dim=attn_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch():
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch_size):
            idx = perm[i:i + batch_size]
            xb, mb, yb = X[idx].to(device), M[idx].to(device), y[idx].to(device)
            loss = F.cross_entropy(model(xb, mb)[0], yb)        # one bag = one sample, no class weights
            opt.zero_grad()
            loss.backward()
            opt.step()

    return _fit_early_stopping(model, train_epoch, val_score, y_val, max_epochs, patience)
