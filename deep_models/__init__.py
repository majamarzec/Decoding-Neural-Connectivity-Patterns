"""deep_models: datasets, architectures and training code for the connectivity MIL pipeline.

Stage 1 (encoders, plain PyTorch):  vae.py + vae_train.py, vqvae.py + vqvae_train.py
Stage 2 (MIL classifier, Lightning): mil.py + lit_models.py
Old Lightning stage-1 code lives in deep_models.legacy.

The two training modules define the same names (fit, evaluate, ...), so they are imported as modules:
    from deep_models import vae_train;  vae_train.fit(...)
"""
from . import config, train_utils, vae_train, vqvae_train, viz
from .data import MILDataset, S00MILDataset, load_conn_memmap, load_s00_conn_memmap, read_metadata
from .lit_models import BagClassifier
from .mil import AttnMILHead
from .vae_BrainNetCNN import BrainNetCNNVAE
from .vae import (
    ConvVAE,
    SparseConvVAE,
    active_units,
    plot_latent_usage,
    sparse_usage,
    plot_sparse_latent_usage,
)
from .vqvae import ConvVQVAE, active_codes, plot_codebook_usage

__all__ = [
    "config", "train_utils", "vae_train", "vqvae_train", "viz",
    "MILDataset", "S00MILDataset", "load_conn_memmap", "load_s00_conn_memmap", "read_metadata",
    "BagClassifier", "AttnMILHead",
    "BrainNetCNNVAE",
    "ConvVAE", "SparseConvVAE", "active_units", "plot_latent_usage", "sparse_usage", "plot_sparse_latent_usage",
    "ConvVQVAE", "active_codes", "plot_codebook_usage",
]