"""Run trained VAE / VQ-VAE encoders over raw bags (lists of (w_i, bands, C, C) arrays), one bag at a time.
Batches are moved to the model's own device; the window shape is read from the data."""
import numpy as np
import torch

from .device import model_device


def _posterior(vae, x, cond=None):
    """(mu, logvar) of a batch. 
    Uses vae.encode (ConvVAE / SparseConvVAE, with the condition if it has one)
    """
    if hasattr(vae, "encode"):
        out = vae.encode(x, cond)
        return out[0], out[1]
    out = vae.encoder(x)
    if isinstance(out, tuple):                       
        return out
    fc_lv = getattr(vae, "fc_logvar", getattr(vae, "fc_log_var", None))
    return vae.fc_mu(out), fc_lv(out)                


@torch.no_grad()
def _map_bags(raw_bags, fn, device, batch, cond_bags=None):
    """Apply fn(x, cond) -> tuple of tensors to every bag, in chunks of `batch` windows.
    cond_bags[i] (cond_dim,) is the exam's condition, shared by all of its windows.
    Returns one list per output of fn, each holding one (w_i, ...) numpy array per bag."""
    outs = None
    for i, bag in enumerate(raw_bags):
        x = torch.as_tensor(np.asarray(bag), dtype=torch.float32)
        if x.ndim < 3:
            raise ValueError(f"raw bags must be (windows, bands, C, C); bag {i} has shape {tuple(x.shape)}")
        x = x.reshape(-1, *x.shape[-3:])
        cond = None
        if cond_bags is not None:
            cond = torch.as_tensor(np.asarray(cond_bags[i]), dtype=torch.float32).reshape(1, -1).expand(len(x), -1)
        chunks = []
        for s in range(0, len(x), batch):
            xb = x[s:s + batch].to(device)
            cb = cond[s:s + batch].to(device) if cond is not None else None
            chunks.append([o.cpu() for o in fn(xb, cb)])
        if outs is None:
            outs = [[] for _ in chunks[0]]
        for k in range(len(outs)):
            outs[k].append(torch.cat([c[k] for c in chunks]).numpy())
    if outs is None:
        raise ValueError("raw_bags is empty")
    return outs


def _stack_if_equal(lists, stack):
    """Stack each list of (w_i, ...) arrays into (exams, w, ...) if all bags have the same length."""
    if stack and len({len(b) for b in lists[0]}) == 1:
        return tuple(np.stack(l) for l in lists)
    return tuple(lists)


def mu_encoder(vae):
    """Callable x (, cond) -> mu, e.g. for features.extract."""
    return lambda x, cond=None: _posterior(vae, x, cond)[0]


def logvar_encoder(vae):
    """Callable x (, cond) -> log-variance."""
    return lambda x, cond=None: _posterior(vae, x, cond)[1]


def encode_raw_bags(vae, raw_bags, batch=256, device=None, stack=True, cond_bags=None):
    """
    Posterior (mu, logvar) per window. 
    3D numpy arrays (exams, windows, latent) if all bags have equal length and stack=True, 
    otherwise lists of (windows_i, latent) arrays. cond_bags for conditional models."""

    vae.eval()
    device = model_device(vae) if device is None else device
    mus, lvs = _map_bags(raw_bags, lambda x, c: _posterior(vae, x, c), device, batch, cond_bags)
    return _stack_if_equal([mus, lvs], stack)


def embed_raw_bags(vae, raw_bags, cond_bags=None, batch=256, device=None, stack=True):
    """Deterministic embedding per window via vae.embed(x, cond), for the downstream MIL step:
      ConvVAE       -> mu
      SparseConvVAE -> (gamma > 0.5) * mu, exact zeros on inactive units
    Works for unconditional and conditional (cond_bags) models; falls back to mu for models without embed."""

    vae.eval()
    device = model_device(vae) if device is None else device
    embed = vae.embed if hasattr(vae, "embed") else (lambda x, c: _posterior(vae, x, c)[0])
    (zs,) = _map_bags(raw_bags, lambda x, c: (embed(x, c),), device, batch, cond_bags)
    return _stack_if_equal([zs], stack)[0]


def vq_encode_raw_bags(vqvae, raw_bags, batch=256, device=None, stack=False):
    """Quantized embeddings and discrete codes per window:
      z_qs: (w_i, latent_channels * grid * grid)   idxs: (w_i, grid * grid)
    Lists of per-bag arrays (stack=True gives 3D arrays when all bags have equal length)."""
    
    vqvae.eval()
    device = model_device(vqvae) if device is None else device

    def encode(x, _):
        _, quantized, idx = vqvae.vq(vqvae.encoder(x))
        return quantized.flatten(1), idx.flatten(1)

    z_qs, idxs = _map_bags(raw_bags, encode, device, batch)
    return _stack_if_equal([z_qs, idxs], stack)
