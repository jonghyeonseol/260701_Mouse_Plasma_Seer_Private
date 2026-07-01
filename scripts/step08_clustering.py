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


def _sample_correlation_clustermap(X, sample_ids, bio, out, fig_mirror, logger):
    """Pearson-correlation clustermap across samples."""
    corr = np.corrcoef(X)              # samples × samples
    col_colors = utils.heatmap_col_colors(bio, sample_ids)
    link = linkage(pdist(X, metric="euclidean"), method=LINK_METHOD)
    g = sns.clustermap(
        pd.DataFrame(corr, index=sample_ids, columns=sample_ids),
        row_linkage=link, col_linkage=link,
        cmap="magma", figsize=(15.5, 13),
        row_colors=col_colors, col_colors=col_colors,
        xticklabels=False, yticklabels=(len(sample_ids) <= 90),
        cbar_kws=dict(label="Pearson r"), dendrogram_ratio=(0.12, 0.12),
        colors_ratio=0.018, linewidths=0)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
    g.fig.suptitle("Sample–sample correlation (hierarchical clustering, Ward)",
                   y=1.02, fontsize=config.TITLE_SIZE)
    # Row tick-labels shown → push legends further right to avoid overlap.
    legs = utils.add_heatmap_annotation_legends(g, x=1.16, cbar_title="Pearson r")
    png = utils.save_clustermap(g, out, "sample_correlation_clustermap", extra_artists=legs)
    plt.close(g.fig)
    _mirror(png, fig_mirror)
    return link


def _sample_dendrogram(link, sample_ids, bio, out, fig_mirror):
    m = bio.set_index("sample_id")
    utils.apply_style()
    fig, ax = plt.subplots(figsize=(max(12, 0.16 * len(sample_ids) + 3), 6.5))
    dn = dendrogram(link, labels=sample_ids, ax=ax, color_threshold=0,
                    above_threshold_color="#555555", leaf_font_size=6)
    ax.set_ylabel(f"Ward distance")
    ax.set_title("Sample hierarchical clustering dendrogram")
    # Colour leaf labels by group; add genotype/timepoint strip below.
    for lbl in ax.get_xticklabels():
        s = lbl.get_text()
        lbl.set_color(config.GROUP_COLORS.get(m.loc[s, "group"], "#333"))
        lbl.set_rotation(90)
    handles = [Patch(color=config.GROUP_COLORS[g], label=config.GROUP_DISPLAY[g])
               for g in config.PLASMA_GROUP_ORDER]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=7,
              frameon=False, title="Group (leaf colour)")
    png = utils.save_fig(fig, out, "sample_dendrogram", tight=True)
    plt.close(fig)
    _mirror(png, fig_mirror)
    return dn


def _cluster_agreement(link, bio, sample_ids, out, logger):
    m = bio.set_index("sample_id")
    genotype = [m.loc[s, "genotype"] for s in sample_ids]
    timepoint = [int(m.loc[s, "timepoint"]) for s in sample_ids]
    group = [m.loc[s, "group"] for s in sample_ids]

    rows = {"sample_id": sample_ids, "genotype": genotype,
            "timepoint": timepoint, "group": group}
    lines = ["Hierarchical clustering vs known design labels (Ward linkage)\n"]
    for k in (2, 4, 8):
        cl = fcluster(link, t=k, criterion="maxclust")
        rows[f"cluster_k{k}"] = cl
        ari_g = adjusted_rand_score(genotype, cl)
        ari_t = adjusted_rand_score(timepoint, cl)
        ari_grp = adjusted_rand_score(group, cl)
        pur_g = _purity(cl, genotype)
        pur_t = _purity(cl, timepoint)
        lines.append(f"k={k}: ARI(genotype)={ari_g:.3f}, ARI(timepoint)={ari_t:.3f}, "
                     f"ARI(group)={ari_grp:.3f} | purity(genotype)={pur_g:.3f}, "
                     f"purity(timepoint)={pur_t:.3f}")
    assign = pd.DataFrame(rows)
    utils.save_tsv(assign, out, "sample_cluster_assignments")
    with open(os.path.join(out, "cluster_label_agreement.txt"), "w") as f:
        f.write("\n".join(lines) + "\n\n")
        # Cross-tabs at the natural k for each factor.
        cl2 = assign["cluster_k2"]
        cl4 = assign["cluster_k4"]
        f.write("Cross-tab k=2 vs genotype:\n")
        f.write(pd.crosstab(cl2, assign["genotype"]).to_string() + "\n\n")
        f.write("Cross-tab k=4 vs timepoint:\n")
        f.write(pd.crosstab(cl4, assign["timepoint"]).to_string() + "\n")
    logger.info("[step08] " + " | ".join(lines[1:]))
    return assign


def _feature_clustermap_topvar(matrix, bio, sample_ids, sample_cols, out, fig_mirror,
                               top_n=500):
    var = matrix[sample_cols].var(axis=1).values
    idx = np.argsort(-var)[:min(top_n, len(var))]
    sub = matrix.iloc[idx]
    mat = sub[sample_cols].values.astype(float)
    z = np.clip((mat - mat.mean(axis=1, keepdims=True)) /
                (mat.std(axis=1, keepdims=True, ddof=0) + 1e-9), -3, 3)
    col_colors = utils.heatmap_col_colors(bio, sample_ids)
    row_link = linkage(pdist(z, metric="correlation"), method="average")
    col_link = linkage(pdist(z.T, metric="euclidean"), method=LINK_METHOD)
    g = sns.clustermap(
        pd.DataFrame(z, columns=sample_ids),
        row_linkage=row_link, col_linkage=col_link,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        col_colors=col_colors, figsize=(15.5, 12),
        xticklabels=False, yticklabels=False,
        cbar_kws=dict(label="Row z-score"), dendrogram_ratio=(0.10, 0.12),
        colors_ratio=0.02, linewidths=0)
    g.fig.suptitle(f"Top-{len(idx)} variable proteins — hierarchical clustering",
                   y=1.02, fontsize=config.TITLE_SIZE)
    legs = utils.add_heatmap_annotation_legends(g, x=1.04)
    png = utils.save_clustermap(g, out, "feature_clustermap_topvar", extra_artists=legs)
    plt.close(g.fig)
    _mirror(png, fig_mirror)


def _union_dep_heatmap(matrix, meta, deps_by_comparison, out, fig_mirror, logger):
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
                                                        config.PLASMA_GROUP_ORDER)
    sub = matrix[matrix[config.ID_COL].isin(union[config.ID_COL])]
    mat = sub[ordered_cols].values.astype(float)
    z = np.clip((mat - mat.mean(axis=1, keepdims=True)) /
                (mat.std(axis=1, keepdims=True, ddof=0) + 1e-9), -3, 3)
    sample_ids = ordered_meta["sample_id"].tolist()
    col_colors = utils.heatmap_col_colors(bio, sample_ids)
    row_link = linkage(pdist(z, metric="correlation"), method="average")
    g = sns.clustermap(
        pd.DataFrame(z, columns=sample_ids),
        row_linkage=row_link, col_cluster=True,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        col_colors=col_colors, figsize=(15.5, 12),
        xticklabels=False, yticklabels=False,
        cbar_kws=dict(label="Row z-score"), dendrogram_ratio=(0.10, 0.12),
        colors_ratio=0.02, linewidths=0)
    g.fig.suptitle(f"Union DEPs across all contrasts (n={len(union)}) — "
                   f"hierarchical clustering", y=1.02, fontsize=config.TITLE_SIZE)
    legs = utils.add_heatmap_annotation_legends(g, x=1.04)
    png = utils.save_clustermap(g, out, "heatmap_union_deps", extra_artists=legs)
    plt.close(g.fig)
    _mirror(png, fig_mirror)
    logger.info(f"[step08] union DEP heatmap: {len(union)} proteins.")


def run(dataset: str, logger, state: dict) -> dict:
    out = utils.step_path(dataset, "08_clustering")
    fig_mirror = utils.figures_dir(dataset)
    utils.apply_style()

    matrix = state["data_df"]
    meta = state["meta"]
    bio = meta[~meta["is_ref"]].copy()
    bio = bio[bio["group"].isin(config.PLASMA_GROUP_ORDER)].reset_index(drop=True)
    sample_cols = bio["column_name"].tolist()
    sample_ids = bio["sample_id"].tolist()
    X = matrix[sample_cols].values.T.astype(float)   # samples × proteins

    logger.info(f"[step08] {dataset}: hierarchical clustering of {len(sample_ids)} "
                f"samples × {matrix.shape[0]} proteins (Ward).")

    link = _sample_correlation_clustermap(X, sample_ids, bio, out, fig_mirror, logger)
    _sample_dendrogram(link, sample_ids, bio, out, fig_mirror)
    _cluster_agreement(link, bio, sample_ids, out, logger)
    _feature_clustermap_topvar(matrix, bio, sample_ids, sample_cols, out, fig_mirror)
    _union_dep_heatmap(matrix, meta, state.get("deps_by_comparison", {}), out,
                       fig_mirror, logger)

    return state


if __name__ == "__main__":
    pass
