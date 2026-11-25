"""Turn a trained encoder into per-window feature bags for linear probing / MIL heads."""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from deep_models.data import S00MILDataset


def read_from_summary(out_dir, split_k):
    with open(f"{out_dir}/summary.json") as f:
        summary = json.load(f)
        hp = summary["hparams"]
        ckpt = {r["split"]: r["best_model_checkpoint"] for r in summary["splits"]}[split_k]
        return hp, ckpt


@torch.no_grad()
def extract(exam_idx, encoder=None, max_instances=None, bs=10, dataset_cls=S00MILDataset, use_cond=False):
    """Bags of per-window features.
       encoder=None -> raw flattened ffDTF windows.
       dataset_cls  -> S00MILDataset (default) or MILDataset (all sites).
       use_cond=True -> the dataset also returns each exam's condition and the encoder is called as
                        encoder(x, cond), e.g. a conditional VAE: `model.embed` (works for ConvVAE and SparseConvVAE)
                        (the encoder must be in eval mode and on the same device as the data)."""
    if use_cond and encoder is None:
        raise ValueError("use_cond=True needs an encoder")
    ds = dataset_cls(exam_idx, max_instances=max_instances, train=False, return_cond=use_cond)  # deterministic window subsampling
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    bags, ys = [], []
    for batch in dl:
        x, y = batch[0], batch[1]
        B, W = x.shape[:2]
        flat = x.flatten(0, 1)
        if use_cond:
            z = encoder(flat, batch[2].repeat_interleave(W, dim=0))   # each exam's cond for all its windows
        else:
            z = flat.flatten(1) if encoder is None else encoder(flat)
        bags += [b.numpy() for b in z.reshape(B, W, -1)]
        ys.append(y.numpy())
    return bags, np.concatenate(ys).astype(int)