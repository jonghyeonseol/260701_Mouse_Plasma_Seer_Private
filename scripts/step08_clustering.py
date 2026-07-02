"""Step 08 — hierarchical clustering analysis.

Provides an explicit hierarchical-clustering section (distinct from the DEP
volcano/heatmap views):

Sample-level:
  - sample_correlation_clustermap.{png,svg} : 80×80 Pearson-correlation
    clustermap (Ward linkage) annotated by genotype + timepoint + group.
  - sample_dendrogram.{png,svg}             : sample dendrogram, leaves coloured
    by group, with genotype/timepoint colour strips.
  - sample_cluster_assignments.tsv          : cluster labels at k=2/4/8.
  - cluster_label_agreement.txt             : Adjusted Rand Index + purity of the
    hierarchical clusters vs genotype (k=2) and timepoint (k=4).

Feature-level:
  - feature_clustermap_topvar.{png,svg}     : top-variance protein clustermap
    (z-scored), samples annotated by group → protein co-abundance modules.
  - heatmap_union_deps.{png,svg}            : union of DEPs across all contrasts,
    hierarchically clustered (rows = proteins, columns = samples).
"""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

import config
import utils


LINK_METHOD = "ward"


def _mirror(png, fig_mirror):
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


def _purity(clusters, labels) -> float:
    """Fraction of samples in the majority label of their assigned cluster."""
    df = pd.DataFrame({"c": clusters, "l": labels})
    correct = df.groupby("c")["l"].apply(lambda s: s.value_counts().iloc[0]).sum()
    return float(correct / len(df))


def _sample_correlation_clustermap(X, sample_ids, bio, out, fig_mirror, logger,
                                   annotation_axes):
    """Pearson-correlation clustermap across samples."""
    corr = np.corrcoef(X)              # samples × samples
    col_colors = utils.heatmap_col_colors(bio, sample_ids, which=annotation_axes)
    link = linkage(pdist(X, metric="euclidean"), method=LINK_METHOD)
    g = sns.clustermap(
        pd.DataFrame(corr, index=sample_ids, columns=sample_ids),
        row_linkage=link, col_linkage=link,
        cmap="magma", figsize=(15.5, 13),
        row_colors=col_colors, col_colors=col_colors,
        xticklabels=False, yticklabels=(len(sample_ids) <= 90),
        cbar_kws=dict(label=""), dendrogram_ratio=(0.12, 0.12),
        colors_ratio=0.018, linewidths=0)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
    t = utils.set_clustermap_title(g, "Sample–sample correlation (hierarchical clustering, Ward)")
    # Row tick-labels shown → push legends further right to avoid overlap.
    legs = utils.add_heatmap_annotation_legends(g, which=annotation_axes, x=1.16,
                                                cbar_title="Pearson r")
    png = utils.save_clustermap(g, out, "sample_correlation_clustermap",
                                extra_artists=legs + [t])
    plt.close(g.fig)
    _mirror(png, fig_mirror)
    return link


def _sample_dendrogram(link, sample_ids, bio, out, fig_mirror, group_order):
    m = bio.set_index("sample_id")
    utils.apply_style()
    fig, ax = plt.subplots(figsize=(max(12, 0.16 * len(sample_ids) + 3), 6.5))
    dn = dendrogram(link, labels=sample_ids, ax=ax, color_threshold=0,
                    above_threshold_color="#555555", leaf_font_size=6)
    ax.set_ylabel(f"Ward distance")
    ax.set_title("Sample hierarchical clustering dendrogram")
    # Colour leaf labels by group.
    for lbl in ax.get_xticklabels():
        s = lbl.get_text()
        lbl.set_color(config.GROUP_COLORS.get(m.loc[s, "group"], "#333"))
        lbl.set_rotation(90)
    handles = [Patch(color=config.GROUP_COLORS[g], label=config.GROUP_DISPLAY[g])
               for g in group_order]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=7,
              frameon=False, title="Group (leaf colour)")
    png = utils.save_fig(fig, out, "sample_dendrogram", tight=True)
    plt.close(fig)
    _mirror(png, fig_mirror)
    return dn


def _cluster_agreement(link, bio, sample_ids, out, logger, group_order):
    """ARI/purity of hierarchical clusters vs design labels.  Adapts to the
    dataset: the genotype axis is only scored when >1 genotype is present
    (branch projects hold a single genotype)."""
    m = bio.set_index("sample_id")
    genotype = [m.loc[s, "genotype"] for s in sample_ids]
    timepoint = [int(m.loc[s, "timepoint"]) for s in sample_ids]
    group = [m.loc[s, "group"] for s in sample_ids]

    # Factors to score (skip genotype if constant).
    factors = {"timepoint": timepoint, "group": group}
    if len(set(genotype)) > 1:
        factors["genotype"] = genotype

    n_tp = len(set(timepoint))
    n_grp = len(group_order)
    ks = sorted(set([2, n_tp, n_grp]))

    rows = {"sample_id": sample_ids, "genotype": genotype,
            "timepoint": timepoint, "group": group}
    lines = ["Hierarchical clustering vs known design labels (Ward linkage)\n"]
    for k in ks:
        cl = fcluster(link, t=k, criterion="maxclust")
        rows[f"cluster_k{k}"] = cl
        parts = [f"ARI({name})={adjusted_rand_score(lab, cl):.3f}"
                 for name, lab in factors.items()]
        parts += [f"purity({name})={_purity(cl, lab):.3f}"
                  for name, lab in factors.items()]
        lines.append(f"k={k}: " + ", ".join(parts))
    assign = pd.DataFrame(rows)
    utils.save_tsv(assign, out, "sample_cluster_assignments")
    with open(os.path.join(out, "cluster_label_agreement.txt"), "w") as f:
        f.write("\n".join(lines) + "\n\n")
        kt = n_tp if f"cluster_k{n_tp}" in assign.columns else ks[-1]
        f.write(f"Cross-tab k={kt} vs timepoint:\n")
        f.write(pd.crosstab(assign[f"cluster_k{kt}"], assign["timepoint"]).to_string() + "\n")
        if "genotype" in factors:
            f.write("\nCross-tab k=2 vs genotype:\n")
            f.write(pd.crosstab(assign["cluster_k2"], assign["genotype"]).to_string() + "\n")
    logger.info("[step08] " + " | ".join(lines[1:]))
    return assign


def _feature_clustermap_topvar(matrix, bio, sample_ids, sample_cols, out, fig_mirror,
                               annotation_axes, top_n=500):
    var = matrix[sample_cols].var(axis=1).values
    idx = np.argsort(-var)[:min(top_n, len(var))]
    sub = matrix.iloc[idx]
    mat = sub[sample_cols].values.astype(float)
    z = np.clip((mat - mat.mean(axis=1, keepdims=True)) /
                (mat.std(axis=1, keepdims=True, ddof=0) + 1e-9), -3, 3)
    col_colors = utils.heatmap_col_colors(bio, sample_ids, which=annotation_axes)
    row_link = linkage(pdist(z, metric="correlation"), method="average")
    col_link = linkage(pdist(z.T, metric="euclidean"), method=LINK_METHOD)
    g = sns.clustermap(
        pd.DataFrame(z, columns=sample_ids),
        row_linkage=row_link, col_linkage=col_link,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        col_colors=col_colors, figsize=(15.5, 12),
        xticklabels=False, yticklabels=False,
        cbar_kws=dict(label=""), dendrogram_ratio=(0.10, 0.12),
        colors_ratio=0.02, linewidths=0)
    t = utils.set_clustermap_title(g, f"Top-{len(idx)} variable proteins — hierarchical clustering")
    legs = utils.add_heatmap_annotation_legends(g, which=annotation_axes, x=1.04)
    png = utils.save_clustermap(g, out, "feature_clustermap_topvar", extra_artists=legs + [t])
    plt.close(g.fig)
    _mirror(png, fig_mirror)


def _union_dep_heatmap(matrix, meta, deps_by_comparison, out, fig_mirror, logger,
                       group_order, annotation_axes):
    seen, rows = set(), []
    for name, res in deps_by_comparison.items():
        for _, r in res[res["Significant"]].iterrows():
            pid = r[config.ID_COL]
            if pid not in seen:
                seen.add(pid)
                rows.append({config.ID_COL: pid, "label": r["label"]})
    if not rows:
        logger.warning("[step08] no DEPs to build union heatmap.")
        return
    union = pd.DataFrame(rows)
    utils.save_tsv(union, out, "union_dep_table")

    bio = meta[~meta["is_ref"]]
    ordered_cols, ordered_meta = utils.ordered_bio_cols(meta, matrix.columns.tolist(),
                                                        group_order)
    sub = matrix[matrix[config.ID_COL].isin(union[config.ID_COL])]
    mat = sub[ordered_cols].values.astype(float)
    z = np.clip((mat - mat.mean(axis=1, keepdims=True)) /
                (mat.std(axis=1, keepdims=True, ddof=0) + 1e-9), -3, 3)
    sample_ids = ordered_meta["sample_id"].tolist()
    col_colors = utils.heatmap_col_colors(bio, sample_ids, which=annotation_axes)
    row_link = linkage(pdist(z, metric="correlation"), method="average")
    g = sns.clustermap(
        pd.DataFrame(z, columns=sample_ids),
        row_linkage=row_link, col_cluster=True,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        col_colors=col_colors, figsize=(15.5, 12),
        xticklabels=False, yticklabels=False,
        cbar_kws=dict(label=""), dendrogram_ratio=(0.10, 0.12),
        colors_ratio=0.02, linewidths=0)
    t = utils.set_clustermap_title(
        g, f"Union DEPs across all contrasts (n={len(union)}) — hierarchical clustering")
    legs = utils.add_heatmap_annotation_legends(g, which=annotation_axes, x=1.04)
    png = utils.save_clustermap(g, out, "heatmap_union_deps", extra_artists=legs + [t])
    plt.close(g.fig)
    _mirror(png, fig_mirror)
    logger.info(f"[step08] union DEP heatmap: {len(union)} proteins.")


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    group_order = cfg["group_order"]
    annotation_axes = cfg.get("annotation_axes", ["Genotype", "Timepoint", "Group"])
    out = utils.step_path(dataset, "08_clustering")
    fig_mirror = utils.figures_dir(dataset)
    utils.apply_style()

    matrix = state["data_df"]
    meta = state["meta"]
    bio = meta[~meta["is_ref"]].copy()
    bio = bio[bio["group"].isin(group_order)].reset_index(drop=True)
    sample_cols = bio["column_name"].tolist()
    sample_ids = bio["sample_id"].tolist()
    X = matrix[sample_cols].values.T.astype(float)   # samples × proteins

    logger.info(f"[step08] {dataset}: hierarchical clustering of {len(sample_ids)} "
                f"samples × {matrix.shape[0]} proteins (Ward).")

    link = _sample_correlation_clustermap(X, sample_ids, bio, out, fig_mirror, logger,
                                          annotation_axes)
    _sample_dendrogram(link, sample_ids, bio, out, fig_mirror, group_order)
    _cluster_agreement(link, bio, sample_ids, out, logger, group_order)
    _feature_clustermap_topvar(matrix, bio, sample_ids, sample_cols, out, fig_mirror,
                               annotation_axes)
    _union_dep_heatmap(matrix, meta, state.get("deps_by_comparison", {}), out,
                       fig_mirror, logger, group_order, annotation_axes)

    return state


if __name__ == "__main__":
    pass
