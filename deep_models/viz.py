"""Connectivity heatmaps and reconstruction comparisons."""
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from . import config as c


def heatmap_bands(mats, cmap='viridis', input_type='original', title=None, vmin=None, vmax=None, axes=None):
    """
    input_type='original'      : mats is (C, C, B)
    input_type='reconstructed' : mats is (B, C, C), torch tensor or numpy array
    """
    if isinstance(mats, torch.Tensor):
        mats = mats.detach().cpu().numpy()
    mats = np.asarray(mats)

    if input_type == 'reconstructed':
        mats = np.moveaxis(mats, 0, -1)  # (B,C,C) -> (C,C,B)
    elif input_type != 'original':
        raise ValueError(f"unknown input_type: {input_type!r}")

    band_names = list(c.freq_bands.keys())
    vmin = mats.min() if vmin is None else vmin
    vmax = mats.max() if vmax is None else vmax

    if axes is None:
        fig, axes = plt.subplots(1, mats.shape[-1], figsize=(15, 5), squeeze=False)
        axes = axes.ravel()
        add_colorbar = True
    else:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure
        add_colorbar = False

    for i, ax in enumerate(axes):
        sns.heatmap(mats[:, :, i].T,
                    ax=ax,
                    xticklabels=c.channels,
                    yticklabels=c.channels,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax if vmax > vmin else vmin + 1e-8,
                    square=True,
                    cbar=False)
        ax.set_title(band_names[i])
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    if add_colorbar:
        sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap) #type:ignore
        fig.colorbar(sm, ax=axes, shrink=0.65, pad=0.02, label='Connectivity strength')

    if title:
        fig.suptitle(title, fontweight='bold', fontsize=15)

    return fig


def compare_heatmaps_orig(val_ds, connectivity_mm, reconstructed, recon_i, window_i):
    val_exam_ids = val_ds.exam_idx
    heatmap_bands(connectivity_mm[val_exam_ids[recon_i]][window_i], input_type='original')
    heatmap_bands(reconstructed[recon_i][window_i], input_type='reconstructed')
    plt.show()


def compare_heatmaps_model(x, reconstructed, recon_i, window_i):
    """
    Compare original input, reconstruction, and squared error.

    x, reconstructed: (N, W, B, C, C)
    recon_i: index of the exam within the batch
    window_i: index of the window within the exam
    """

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    original = np.asarray(x[recon_i, window_i])  # (B, C, C)
    recon = np.asarray(reconstructed[recon_i, window_i])

    if original.shape != recon.shape:
        raise ValueError(f"Shape mismatch: original {original.shape}, reconstruction {recon.shape}")

    squared_error = (original - recon) ** 2
    offdiag = ~np.eye(original.shape[-1], dtype=bool)
    mse = squared_error[..., offdiag].mean()

    band_names = list(c.freq_bands.keys())
    n_bands = original.shape[0]

    fig, axes = plt.subplots(n_bands, 3, figsize=(15, 4 * n_bands), squeeze=False)
    vmin = min(original.min(), recon.min())
    vmax = max(original.max(), recon.max())
    error_max = squared_error.max()

    for b in range(n_bands):
        matrices = [
            (original[b], "original", "viridis", vmin, vmax),
            (recon[b], "reconstruction", "viridis", vmin, vmax),
            (squared_error[b], "squared error", "magma", 0, error_max),
        ]

        for j, (matrix, label, cmap, lo, hi) in enumerate(matrices):
            heatmap_bands(matrix[np.newaxis, ...],
                          input_type="reconstructed",
                          title=None,
                          cmap=cmap,
                          vmin=lo,
                          vmax=hi if hi > lo else lo + 1e-8,
                          axes=axes[b, j:j + 1],
                          )
            axes[b, j].set_title(f"{label} — {band_names[b]}")

    fig.suptitle(
        f"Connectivity reconstruction | Exam index {recon_i}, "
        f"window {window_i} | MSE = {mse:.5g}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # type: ignore
    plt.show()

    return fig, mse