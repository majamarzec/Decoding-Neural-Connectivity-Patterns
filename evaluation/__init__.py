"""linear_probing: do the learned representations carry the pathology label?

Pipeline, in the order of the modules:
    bags.py             raw bags from a DataLoader; flatten / pool / pad bags
    encoders.py         trained VAE / VQ-VAE  ->  embeddings per window            (needs a model)
    representations.py  PCA, active-unit masks, code histograms                    (builds probe inputs)
    probes.py           SoftmaxProbe, ABMIL                                        (the probe models)
    training.py         fit_probe, fit_abmil (early stopping on validation AUC)
    scoring.py          scorers (bag -> P(class 1)), window votes, attention analysis
    experiment.py       run_nn_probes: every probe on every representation -> results rows
    plots.py            embedding scatter plots
    device.py           get_device / set_device (only needed where a model is created)
features.py (separate) builds bags straight from a dataset with an encoder callable (extract).

Dependencies point one way: device <- bags <- encoders / representations / probes <- training <- scoring
<- experiment <- plots. Import the names below from `linear_probing`, or the modules for the rest.
"""
from .bags import (bag_layout, extract_raw_from_loader, flatten_offdiag_bags, mean_pool_bags, pad_bags)
from .device import get_device, model_device, resolve_device, set_device
from .encoders import (embed_raw_bags, encode_raw_bags, logvar_encoder, mu_encoder, vq_encode_raw_bags)
from .experiment import DEFAULT_ABMIL_VARIANTS, report, run_nn_probes
from .plots import plot_embedding_votes
from .probes import ABMIL, SoftmaxProbe
from .representations import (active_code_mask, active_mask, make_code_histograms, pca_bags, select_dims)
from .scoring import (abmil_attention, abmil_scorer, attention_entropy, bag_scorer, batched_logits, pooled_scorer,
                      vote_scorer, window_votes)
from .training import fit_abmil, fit_probe

__all__ = [
    "bag_layout", "extract_raw_from_loader", "flatten_offdiag_bags", "mean_pool_bags", "pad_bags",
    "get_device", "model_device", "resolve_device", "set_device",
    "embed_raw_bags", "encode_raw_bags", "logvar_encoder", "mu_encoder", "vq_encode_raw_bags",
    "DEFAULT_ABMIL_VARIANTS", "report", "run_nn_probes",
    "plot_embedding_votes",
    "ABMIL", "SoftmaxProbe",
    "active_code_mask", "active_mask", "make_code_histograms", "pca_bags", "select_dims",
    "abmil_attention", "abmil_scorer", "attention_entropy", "bag_scorer", "batched_logits", "pooled_scorer",
    "vote_scorer", "window_votes",
    "fit_abmil", "fit_probe",
]
