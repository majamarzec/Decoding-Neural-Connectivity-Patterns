"""Run the probes on every representation and collect the results table."""
import numpy as np
from sklearn.metrics import roc_auc_score

from .bags import mean_pool_bags
from .scoring import abmil_scorer, bag_scorer, pooled_scorer, vote_scorer
from .training import fit_abmil, fit_probe

DEFAULT_ABMIL_VARIANTS = {"ABMIL_linear": None, "ABMIL_h64": 64}   # name -> hidden size (None = no projection)


def report(rep, method, model, val_auc, best_epoch, test_score, y_test):
    """One results row (also printed): selected on validation AUC, evaluated once on test."""
    auc_test = roc_auc_score(y_test, test_score(model))
    print(f"[{rep}] {method}  val AUC = {val_auc:.3f}  epoch = {best_epoch}  test AUC = {auc_test:.3f}")
    return {
        "Representation": rep,
        "Method": method,
        "Val AUC": val_auc,
        "Epoch": best_epoch,
        "Test AUC": auc_test}


def _probe_pooled(rep, W, y, device, fit_kw):
    """PROBE 1: nn.Linear + softmax on the mean latent embedding of each bag."""
    P_train, P_val, P_test = (mean_pool_bags(W[s]) for s in ("train", "val", "test"))
    model, val_auc, ep = fit_probe(P_train, y["train"], pooled_scorer(P_val), y["val"], device=device, **fit_kw)
    return [report(rep, "LinearBagProbe", model, val_auc, ep, pooled_scorer(P_test), y["test"])]


def _probe_instances(rep, W, y, device, fit_kw):
    """PROBE 2: nn.Linear + softmax on windows with the bag label copied to each window (weak labels).
    One trained model, two readouts (soft = mean margin, vote = fraction of windows). Early stopping is on the
    SOFT validation AUC for both. Returns (rows, artifacts)."""
    sizes = np.array([len(b) for b in W["train"]])
    X_windows = np.concatenate(W["train"])
    y_windows = np.repeat(y["train"], sizes)       # every window gets the label of the bag it came from
    w_windows = np.repeat(1.0 / sizes, sizes)      # each bag contributes equally, no class weights

    model, val_auc, ep = fit_probe(X_windows, y_windows, bag_scorer(W["val"]), y["val"],
                                   sample_w=w_windows, device=device, **fit_kw)
    vote_val = roc_auc_score(y["val"], vote_scorer(W["val"])(model))
    rows = [report(rep, "LinearInstanceProbeBagLabels_soft", model, val_auc, ep, bag_scorer(W["test"]), y["test"]),
            report(rep, "LinearInstanceProbeBagLabels_vote", model, vote_val, ep, vote_scorer(W["test"]), y["test"])]
    return rows, {"model": model, "X_val": np.concatenate(W["val"])}


def _probe_abmil(rep, W, y, device, variants, seeds, abmil_kw):
    """PROBE 3: gated-attention MIL on bags, bag labels only. One row per (variant, seed); the best-validation
    seed's model is returned per variant. Returns (rows, {variant: model})."""
    val_score, test_score = abmil_scorer(W["val"]), abmil_scorer(W["test"])
    rows, models = [], {}
    for tag, hidden in variants.items():
        best = None
        for seed in seeds:
            m, v_auc, ep = fit_abmil(W["train"], y["train"], val_score, y["val"],
                                     hidden=hidden, seed=seed, device=device, **abmil_kw)
            row = report(rep, tag, m, v_auc, ep, test_score, y["test"])
            row["Seed"] = seed
            rows.append(row)
            if best is None or v_auc > best[0]:
                best = (v_auc, m)
        models[tag] = best[1]  # type:ignore
    return rows, models


def run_nn_probes(bags, representations, y_train, y_val, y_test, return_artifacts=False,
                  abmil_variants=DEFAULT_ABMIL_VARIANTS, abmil_kw=None, abmil_seeds=(0,), device=None, **fit_kw):
    """bags[rep][split] = list of (n_windows, d) arrays, split in train / val / test.
    fit_kw goes to the linear probes (training.fit_probe); abmil_kw goes to training.fit_abmil.
    Pass abmil_variants={} to skip the attention probes. device=None -> device.get_device().
    Returns rows (list of dicts; one per probe/readout, ABMIL also per seed), plus artifacts
    {rep: {"model", "X_val", <abmil variant>: model}} if return_artifacts."""
    abmil_kw = abmil_kw or {}
    y = {"train": y_train, "val": y_val, "test": y_test}
    rows, artifacts = [], {}
    for rep in representations:
        W = {s: [np.asarray(b, dtype=np.float32) for b in bags[rep][s]] for s in ("train", "val", "test")}

        rows += _probe_pooled(rep, W, y, device, fit_kw)
        inst_rows, artifacts[rep] = _probe_instances(rep, W, y, device, fit_kw)
        rows += inst_rows
        abmil_rows, abmil_models = _probe_abmil(rep, W, y, device, abmil_variants, abmil_seeds, abmil_kw)
        rows += abmil_rows
        artifacts[rep].update(abmil_models)
    return (rows, artifacts) if return_artifacts else rows