"""
Linear probing of the representations learned by the BagVariationalEncoder
(run train_cv_vae.py first; it writes the checkpoints and summary.json read here).
   8 folds train -> 1 fold val (probe early stopping) -> 1 fold test (holdout)
"""

import numpy as np
import pandas as pd
import os

import torch

import pytorch_lightning as L

from deep_models.lib_models import ConvVAE, BagVariationalEncoder, make_splits, read_from_summary, extract
from lib_linear_probing_lightning import run_nn_probes, mu_encoder
from sklearn.decomposition import PCA
import deep_models.hyperparameters as h


seed = 42  # for reproducibility


###
hparams = h.hyper_set_v3

fmt = lambda v: "-".join(map(str, np.atleast_1d(v)))
run_name = (f"convvae"
            f"_l{hparams['latent']}"
            f"_k{fmt(hparams['kernel'])}s{fmt(hparams['strides'])}p{hparams['padding']}"
            f"_h{fmt(hparams['hidden'])}"
            f"_bn{int(hparams['batchnorm'])}"  # type:ignore
            f"_bs{hparams['batch_size']}"
            f"_mi{hparams['max_instances']}"
            f"_beta{hparams['beta']}_ann{hparams['anneal_epochs']}")  # type:ignore

dev_path = "splits/s00_developmental.npz"
out_dir = "results_s00/" + run_name
os.makedirs(out_dir, exist_ok=True)

with np.load(dev_path) as s:  # type: ignore
    dev_folds = [s[f"fold_{k}"] for k in range(len(s.files))]


def build_ae(hp):
    return ConvVAE(latent=hp["latent"], kernel=hp["kernel"], strides=hp["strides"],
                   padding=hp["padding"], batchnorm=hp["batchnorm"], hidden=hp["hidden"])

representations = ["trained_ae", "untrained_ae", "pca", "raw"]

results = []
for k, train_idx, val_idx, test_idx in make_splits(dev_folds):
    print(f"\n=== Split {k}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} ===")
    L.seed_everything(seed + k, workers=True)

    hp, ckpt = read_from_summary(out_dir, k)
    splits = [("train", train_idx), ("val", val_idx), ("test", test_idx)]
    raw = {n: extract(i) for n, i in splits}  # (bags, labels), raw flattened windows
    y_train, y_val, y_test = raw["train"][1], raw["val"][1], raw["test"][1]
    print("pathology rate  train/val/test:", y_train.mean().round(3), y_val.mean().round(3), y_test.mean().round(3))

    trained_vae = BagVariationalEncoder.load_from_checkpoint(ckpt, ae=build_ae(hp), map_location="cpu").eval()

    torch.manual_seed(seed)
    random_vae = build_ae(hp).eval()

    bags = {"raw": {n: raw[n][0] for n in raw}}
    for name, encoder in [("trained_ae", mu_encoder(trained_vae.ae)),
                          ("untrained_ae", mu_encoder(random_vae))]:
        bags[name] = {n: extract(i, encoder)[0] for n, i in splits}

    pca = PCA(n_components=hp["latent"], random_state=seed).fit(np.concatenate(bags["raw"]["train"]))
    bags["pca"] = {n: [pca.transform(b) for b in bl] for n, bl in bags["raw"].items()}

    rows = run_nn_probes(bags, representations, y_train, y_val, y_test,
                         lr=1e-3,
                         batch_size=100,
                         max_epochs=200)
    for r in rows:
        r["split"] = k #type:ignore
    results.extend(rows)
    pd.DataFrame(results).to_csv(f"{out_dir}/probe_results.csv", index=False)

# mean/std across splits
df = pd.DataFrame(results)
summary = df.groupby(["Representation", "Method"])[["Val AUC", "Test AUC"]].agg(["mean", "std"])
summary.to_csv(f"{out_dir}/probe_summary.csv")
print(summary.round(3))