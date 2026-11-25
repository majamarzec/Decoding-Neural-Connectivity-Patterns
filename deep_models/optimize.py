from pathlib import Path
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from . import data 
from data import load_folds, make_splits, S00MILDataset
from . import vae 
from vae import ConvVAE, active_units
from . import vae_train
from vae_train import init_model, fit, encode_bags

dev_path = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/split_data/one_site/splits/s00_developmental.npz"
TUNE_SPLITS = (0, 3, 7)
ANNEAL_START = 50
ANNEAL_EPOCHS = 50
ANNEAL_END = ANNEAL_START + ANNEAL_EPOCHS

ckpt_dir = Path("optuna_ckpts")
ckpt_dir.mkdir(exist_ok=True)
device = torch.device("cuda")

# Pre-build loaders once for all tuning splits
developmental_folds = load_folds(dev_path)
all_splits = {k: (tr, va, te) for k, tr, va, te in make_splits(developmental_folds)}

split_loaders = {}
for k in TUNE_SPLITS:
    train_idx, val_idx, test_idx = all_splits[k]
    print(f"split {k}: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    train_ds = S00MILDataset(train_idx, max_instances=150, train=True)
    val_ds   = S00MILDataset(val_idx,   max_instances=150, train=False)
    split_loaders[k] = (
        DataLoader(train_ds, batch_size=10, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(val_ds,   batch_size=10, shuffle=False, num_workers=2, pin_memory=True),
    )


def objective(trial):
    latent = trial.suggest_categorical("latent", [16, 32, 48, 64])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batchnorm = trial.suggest_categorical("batchnorm", [False, True])
    fc_act = trial.suggest_categorical("fc_act", [False, True])
    beta_max = 5e-4

    totals, r2s, mses, actives = [], [], [], []

    for step, k in enumerate(TUNE_SPLITS):
        train_loader, val_loader = split_loaders[k]

        model = ConvVAE(
            latent=latent,
            batchnorm=batchnorm,
            fc_act=fc_act,
            cond_dim=0,
        ).to(device)
        init_model(model, train_loader)

        history = fit(
            model,
            train_loader,
            val_loader,
            device,
            epochs=150,
            lr=lr,
            weight_decay=weight_decay,
            beta_max=beta_max,
            anneal_start=ANNEAL_START,
            anneal_epochs=ANNEAL_EPOCHS,
            ckpt_path=str(ckpt_dir / f"trial_{trial.number}_split{k}.pt"))

        val_mu, val_logvar, _ = encode_bags(model, val_loader, device)
        n_active, _, _ = active_units(val_mu, val_logvar, thr=0.01)

        post_total = history["val_total"][ANNEAL_END:]
        best_idx = int(np.argmin(post_total))

        totals.append(post_total[best_idx])
        r2s.append(history["val_r2"][ANNEAL_END + best_idx])
        mses.append(history["val_mse"][ANNEAL_END + best_idx])
        actives.append(n_active)

        # Report running mean after each fold so Optuna can prune poor configs early
        trial.report(float(np.mean(totals)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("val_total_std", float(np.std(totals)))
    trial.set_user_attr("val_r2_mean",   float(np.mean(r2s)))
    trial.set_user_attr("val_mse_mean",  float(np.mean(mses)))
    trial.set_user_attr("n_active_mean", float(np.mean(actives)))

    return float(np.mean(totals))


pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=25)

bt = study.best_trial
print("Best params:", study.best_params)
print(f"Best val_total: {study.best_value:.4f} ± {bt.user_attrs['val_total_std']:.4f}")
print(f"Mean active units: {bt.user_attrs['n_active_mean']:.1f}")
print(f"Mean val R2:  {bt.user_attrs['val_r2_mean']:.3f}")
print(f"Mean val MSE: {bt.user_attrs['val_mse_mean']:.5f}")