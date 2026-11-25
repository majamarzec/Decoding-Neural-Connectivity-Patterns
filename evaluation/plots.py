"""Embedding scatter plots coloured by exam, by (weak) label and by the instance probe's output."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from .scoring import window_votes


def _by_exam(ax, fig, emb, name, exam_ids):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=exam_ids, cmap="tab20", s=1, alpha=0.7, rasterized=True)
    ax.set_title(f"{name}: by bags")
    fig.colorbar(sc, ax=ax, label="Exam ID")


def _by_label(ax, fig, emb, name, window_labels):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=window_labels, cmap="coolwarm", vmin=0, vmax=1,
                    s=8, alpha=0.5, rasterized=True)
    ax.set_title(f"{name}: by exam's (weak) label")
    fig.colorbar(sc, ax=ax, label="Pathology")


def _by_probe_prob(ax, fig, emb, name, p1):
    order = np.argsort(np.abs(p1 - 0.5))      # confident windows drawn on top
    sc = ax.scatter(emb[order, 0], emb[order, 1], c=p1[order], cmap="coolwarm",
                    vmin=0, vmax=1, s=8, alpha=0.6, rasterized=True)
    ax.set_title(f"{name}: instance probe P(class 1)")
    fig.colorbar(sc, ax=ax, label="P(class 1)")


def _by_probe_vote(ax, emb, name, vote):
    c0, c1 = plt.cm.coolwarm(0.0), plt.cm.coolwarm(1.0)  # type:ignore
    ax.scatter(emb[:, 0], emb[:, 1], c=vote, cmap=ListedColormap([c0, c1]), vmin=0, vmax=1,
               s=8, alpha=0.5, rasterized=True)
    ax.set_title(f"{name}: by predicted class (vote)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c0, label="vote: class 0"),
                       Line2D([], [], marker="o", ls="", color=c1, label="vote: class 1")],
              loc="best")


def plot_embedding_votes(embeddings, artifacts, exam_ids, window_labels, rep_of,
                         mode="prob", threshold=0.5, size=(6, 5)):
    """Rows = embeddings {name: (n_windows, >=2) array}. Columns = by exam, by weak label, by instance-probe output.
    rep_of[row] names the representation whose instance probe (artifacts[rep]) scores that row's windows.
    mode="prob": unthresholded P(class 1);  mode="vote": hard vote at `threshold`."""
    assert mode in ("prob", "vote")
    fig, axes = plt.subplots(len(embeddings), 3, figsize=(size[0] * 3, size[1] * len(embeddings)),
                             constrained_layout=True, squeeze=False)

    for row, (name, emb) in enumerate(embeddings.items()):
        art = artifacts[rep_of[row]]
        assert len(emb) == len(art["X_val"]) == len(exam_ids) == len(window_labels), \
            f"{name}: rows are not aligned (embedding / probe input / exam_ids / labels)"
        p1, vote = window_votes(art["model"], art["X_val"], threshold)

        _by_exam(axes[row, 0], fig, emb, name, exam_ids)
        _by_label(axes[row, 1], fig, emb, name, window_labels)
        if mode == "prob":
            _by_probe_prob(axes[row, 2], fig, emb, name, p1)
        else:
            _by_probe_vote(axes[row, 2], emb, name, vote)

        for ax in axes[row]:
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

    return fig, axes