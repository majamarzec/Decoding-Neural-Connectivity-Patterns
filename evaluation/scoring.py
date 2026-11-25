"""Turning a fitted probe into a score per bag (the readouts), plus attention analysis.

A *scorer* is built once from a list of bags and called as score(model) -> P(class 1 = pathology) per bag.
All scorers run the model on its own device, so they need no device argument."""
import numpy as np
import torch
import torch.nn.functional as F

from .bags import bag_layout, pad_bags
from .device import model_device


# ---- window-level outputs of a SoftmaxProbe -------------------------------------------------------
@torch.no_grad()
def batched_logits(model, X, chunk=32768):
    """X is a CPU tensor; logits are computed on the model's device chunk by chunk and returned on CPU."""
    model.eval()
    device = model_device(model)
    return torch.cat([model(X[i:i + chunk].to(device)).cpu() for i in range(0, len(X), chunk)])


@torch.no_grad()
def window_votes(model, X, threshold=0.5):
    """Per-window P(class 1) and hard vote. X stays on the CPU; the model runs in chunks."""
    X = torch.as_tensor(X, dtype=torch.float32)
    p1 = F.softmax(batched_logits(model, X), dim=-1)[:, 1].numpy()
    return p1, (p1 > threshold).astype(int)


# ---- bag-level scorers ----------------------------------------------------------------------------
def pooled_scorer(P):
    """Linear probe on mean-pooled bags P (n_bags, d)."""
    X = torch.as_tensor(P, dtype=torch.float32)

    @torch.no_grad()
    def score(model):
        model.eval()
        return F.softmax(model(X.to(model_device(model))), dim=-1)[:, 1].cpu().numpy()

    return score


def bag_scorer(bags):
    """SOFT readout of an instance probe: sigmoid of the mean logit margin over the bag's windows."""
    sizes, offsets, X = bag_layout(bags)

    def score(model):
        z = batched_logits(model, X)
        margin = (z[:, 1] - z[:, 0]).numpy().astype(np.float64)
        return 1.0 / (1.0 + np.exp(-np.add.reduceat(margin, offsets) / sizes))

    return score


def vote_scorer(bags, threshold=0.5):
    """HARD readout of an instance probe: fraction of windows with P(class 1) > threshold, per bag."""
    sizes, offsets, X = bag_layout(bags)

    def score(model):
        _, vote = window_votes(model, X, threshold)
        return np.add.reduceat(vote.astype(np.float64), offsets) / sizes

    return score


def abmil_scorer(bags, chunk=64):
    """Gated-attention MIL probe on bags (padded once)."""
    X, M = pad_bags(bags)

    @torch.no_grad()
    def score(model):
        model.eval()
        device = model_device(model)
        out = [F.softmax(model(X[i:i + chunk].to(device), M[i:i + chunk].to(device))[0], dim=-1)[:, 1].cpu()
               for i in range(0, len(X), chunk)]
        return torch.cat(out).numpy()

    return score


# ---- attention analysis ---------------------------------------------------------------------------
@torch.no_grad()
def abmil_attention(model, bags, chunk=64):
    """Per-bag attention weights as a list of 1-D arrays (padding removed)."""
    model.eval()
    device = model_device(model)
    X, M = pad_bags(bags)
    A = torch.cat([model(X[i:i + chunk].to(device), M[i:i + chunk].to(device))[1].cpu()
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
