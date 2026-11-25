from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="ticks", context="paper", font_scale=1.2)


def _assert_disjoint(groups: dict, unit: str) -> None:
    """groups: name -> array of group ids (patients or sites). Raise if any two share an id."""
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert set(groups[a]).isdisjoint(groups[b]), f"{a}/{b}: {unit} leakage"


def _check_group_by(group_by: str, sites) -> None:
    if group_by not in ("patient", "site"):
        raise ValueError("group_by must be 'patient' or 'site'.")
    if group_by == "site" and sites is None:
        raise ValueError("group_by='site' requires `sites`.")


def build_split_frame(labels, patients, partitions: dict[str, np.ndarray], sites=None, group_by: str = "patient") -> pd.DataFrame:
    """
    Long-format table (one row per exam). Fails if partitions leak.

    group_by="patient": partitions must be patient-disjoint.
    group_by="site":    partitions must be site-disjoint (and therefore patient-disjoint).
    """
    _check_group_by(group_by, sites)
    labels, patients = np.asarray(labels), np.asarray(patients)
    sites = None if sites is None else np.asarray(sites)

    names = list(partitions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.intersect1d(partitions[a], partitions[b]).size == 0, f"{a}/{b}: index overlap"
    _assert_disjoint({n: patients[idx] for n, idx in partitions.items()}, "patient")
    if group_by == "site":
        _assert_disjoint({n: sites[idx] for n, idx in partitions.items()}, "site") #type:ignore

    frames = []
    for name, idx in partitions.items():
        cols = {"label": labels[idx], "patient": patients[idx], "partition": name}
        if sites is not None:
            cols["site"] = sites[idx]
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def plot_class_proportions(df, ax, palette):
    data = (df.groupby("partition")["label"]
              .value_counts(normalize=True)
              .rename("proportion")
              .reset_index())

    sns.barplot(data=data,
                x="label",
                y="proportion",
                hue="partition",
                hue_order=list(palette),
                palette=palette,
                edgecolor="black",
                linewidth=0.6,
                ax=ax)

    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=3, fontsize=9, rotation=90)

    ax.set(xlabel="Pathology", ylabel="Fraction of exams",
           title="Label balance")
    ax.set_ylim(0, data["proportion"].max() * 1.3) 
    ax.get_legend().remove()


def plot_sizes(df, ax, palette, unit="patient"):
    agg = {"Exams": ("label", "size"), f"{unit.title()}s": ("patient", "nunique")}
    if "site" in df and unit != "site":
        agg["Sites"] = ("site", "nunique")
    data = (df.groupby("partition")
              .agg(**agg)
              .reset_index()
              .melt(id_vars="partition", var_name="unit", value_name="count"))

    sns.barplot(data=data,
                x="unit",
                y="count",
                hue="partition",
                hue_order=list(palette),
                palette=palette,
                edgecolor="black",
                linewidth=0.6,
                ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%d", padding=2, fontsize=9)
    ax.set(xlabel="", ylabel="Count", title="After split volume")
    top, bottom = data["count"].max(), max(data["count"].min(), 1)
    if top / bottom > 30:  
        ax.set_yscale("log")
        ax.set_ylim(0.5, top * 3)
    else:
        ax.set_ylim(0, top * 1.15)
    ax.get_legend().remove()


def plot_exams_per_patient(df, ax, palette, unit="patient"):
    per_patient = df.groupby(["partition", "patient"]).size().rename("n_exams").reset_index()
    per_patient["n_exams"] = per_patient["n_exams"].clip(upper=4).map(
        lambda n: "4+" if n == 4 else str(n))
    data = (per_patient.groupby("partition")["n_exams"]
              .value_counts(normalize=True)
              .rename("proportion")
              .reset_index())
    sns.barplot(data=data, x="n_exams", y="proportion", hue="partition",
                order=["1", "2", "3", "4+"], hue_order=list(palette), palette=palette,
                edgecolor="black", linewidth=0.6, ax=ax)
    ax.set(xlabel=f"Exams per {unit}", ylabel=f"Fraction of {unit}s",
           title=f"{unit.title()} contribution")
    ax.get_legend().remove()


def plot_exams_per_site(df, ax, palette, col="patient"):
    """One bar per site (sorted by size), coloured by partition. `col` is the column holding site ids."""
    counts = (df.groupby(["partition", col]).size().rename("n_exams").reset_index()
                .sort_values("n_exams", ascending=False))
    counts[col] = counts[col].astype(str)
    sns.barplot(data=counts, x="n_exams", y=col, order=list(counts[col]),
                hue="partition", hue_order=list(palette), palette=palette, dodge=False,
                edgecolor="black", linewidth=0.6, ax=ax)
    ax.set(xlabel="Exams", ylabel="Site", title="Exams per site")
    if len(counts) > 40: 
        ax.set_yticklabels([])
    ax.get_legend().remove()


def plot_site_composition(df, ax, by: str, order=None, col: str = "site"):
    """Heatmap of exam counts per site for each partition / fold (`by`).
    Site-disjoint splits show up as a block pattern: each site column is filled in one row only."""
    table = pd.crosstab(df[by], df[col])
    if order is not None:
        table = table.loc[list(order)]
    annot = table.shape[1] <= 12  # skip cell numbers when there are too many sites
    sns.heatmap(table, annot=annot, fmt="d", cmap="Blues", cbar=not annot, cbar_kws=dict(label="Count of exams"), linewidths=0.5, ax=ax)
    ax.set(xlabel="Site", ylabel=by.capitalize(), title="Site contribution")


def plot_split_overview(labels, patients, partitions, path_stem="split_overview", sites=None, group_by: str = "patient", unit: str = "patient"):
    """`unit` only changes the plot wording. Pass unit="site" when the grouping array
    given as `patients` actually holds site ids."""
    df = build_split_frame(labels, patients, partitions, sites=sites, group_by=group_by)
    palette = dict(zip(partitions, sns.color_palette("colorblind", len(partitions))))

    n = 4 if "site" in df else 3
    if n == 4:  # 2 x 2 grid when the site panel is present
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    plot_class_proportions(df, axes[0], palette)
    plot_sizes(df, axes[1], palette, unit=unit)
    if unit == "site":  # exams-per-patient buckets (1, 2, 3, 4+) mean nothing for sites
        plot_exams_per_site(df, axes[2], palette)
    else:
        plot_exams_per_patient(df, axes[2], palette, unit=unit)
    if n == 4:
        plot_site_composition(df, axes[3], by="partition", order=list(partitions))

    for ax, letter in zip(axes, "ABCD"):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
        sns.despine(ax=ax)

    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.6) for c in palette.values()]
    fig.legend(handles, palette.keys(), loc="outside lower center", ncol=len(palette), frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    return fig, df


# ----------------------------------------------------------------------------
# Cross-validation folds
# ----------------------------------------------------------------------------
def check_folds(folds, dev_set, test_set, patients, sites=None, group_by: str = "patient") -> None:
    """Raise if the folds are not a clean partition of dev_set.

    group_by="patient": folds (and the hold-out) must be patient-disjoint.
    group_by="site":    folds (and the hold-out) must be site-disjoint as well.
    """
    _check_group_by(group_by, sites)
    patients = np.asarray(patients)
    sites = None if sites is None else np.asarray(sites)

    all_idx = np.concatenate(folds)
    assert len(all_idx) == len(np.unique(all_idx)), "a sample appears in several folds"
    assert set(all_idx) == set(dev_set), "folds do not cover exactly the development set"
    assert np.intersect1d(all_idx, test_set).size == 0, "hold-out samples inside folds"

    assert set(patients[all_idx]).isdisjoint(patients[test_set]), "patient shared with test set"
    _assert_disjoint({f"fold {k}": patients[f] for k, f in enumerate(folds)}, "patient")

    if group_by == "site":
        assert set(sites[all_idx]).isdisjoint(sites[test_set]), "site shared with test set" #type:ignore
        _assert_disjoint({f"fold {k}": sites[f] for k, f in enumerate(folds)}, "site") #type:ignore

    print(f"OK: {len(folds)} folds, {len(all_idx)} samples, {group_by}-disjoint, test set untouched")


def build_fold_frame(folds, labels, patients, sites=None) -> pd.DataFrame:
    labels, patients = np.asarray(labels), np.asarray(patients)
    sites = None if sites is None else np.asarray(sites)
    frames = []
    for k, idx in enumerate(folds):
        cols = {"fold": k, "label": labels[idx], "patient": patients[idx]}
        if sites is not None:
            cols["site"] = sites[idx]
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def plot_class_per_fold(df, ax):
    data = (df.groupby("fold")["label"].value_counts(normalize=True)
              .rename("proportion").reset_index())
    sns.barplot(data=data, x="fold", y="proportion", hue="label",
                palette="colorblind", edgecolor="black", linewidth=0.6, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=3, fontsize=8, rotation=90)
    ax.set(xlabel="Fold", ylabel="Fraction of samples",
           title="Label balance")
    ax.set_ylim(0, data["proportion"].max() * 1.4)  # headroom for rotated labels + legend
    ax.legend(title="Pathology", frameon=False, ncol=2, loc="upper center")


def plot_sizes_per_fold(df, ax, unit="patient"):
    agg = {"Samples": ("label", "size"), f"{unit.title()}s": ("patient", "nunique")}
    if "site" in df and unit != "site":
        agg["Sites"] = ("site", "nunique")
    data = (df.groupby("fold")
              .agg(**agg)
              .reset_index()
              .melt(id_vars="fold", var_name="unit", value_name="count"))
    sns.barplot(data=data, x="fold", y="count", hue="unit",
                palette="colorblind", edgecolor="black", linewidth=0.6, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%d", padding=3, fontsize=8, rotation=90)
    ax.set(xlabel="Fold", ylabel="Count", title="Fold volume")
    top, bottom = data["count"].max(), max(data["count"].min(), 1)
    if top / bottom > 30:  # samples vs. a few sites: log axis keeps every bar and label readable
        ax.set_yscale("log")
        ax.set_ylim(0.5, top * 15)  # bottom < 1 so bars of height 1 stay visible; headroom for labels + legend
    else:
        ax.set_ylim(0, top * 1.25)
    ax.legend(title=None, frameon=False, ncol=len(agg), loc="upper center")


def plot_exams_per_patient_f(df, ax):
    n = df.groupby(["fold", "patient"]).size().clip(upper=3).rename("n_exams").reset_index()
    table = (pd.crosstab(n["fold"], n["n_exams"], normalize="index")
               .rename(columns={1: "1", 2: "2", 3: "3+"}))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                linewidths=0.5, ax=ax)
    ax.set(xlabel="Exams per patient", ylabel="Fold", title="Patient contribution")


def plot_fold_overview(folds, labels, patients, path_stem="dev_folds_overview",
                       sites=None, unit: str = "patient"):
    """`unit` only changes the plot wording. Pass unit="site" when the array given as
    `patients` actually holds site ids."""
    df = build_fold_frame(folds, labels, patients, sites=sites)

    n = 4 if "site" in df else 3
    if n == 4:  # 2 x 2 grid when the site panel is present
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    plot_class_per_fold(df, axes[0])
    plot_sizes_per_fold(df, axes[1], unit=unit)
    if unit == "site":  # exams-per-patient buckets mean nothing for sites: show exams per site per fold
        plot_site_composition(df, axes[2], by="fold", col="patient")
    else:
        plot_exams_per_patient_f(df, axes[2])
    if n == 4:
        plot_site_composition(df, axes[3], by="fold")

    for ax, letter in zip(axes, "ABCD"):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
        sns.despine(ax=ax)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    return fig, df