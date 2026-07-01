"""Hierarchical clustering heatmap (seaborn.clustermap).

Adapted from KBSI_FAIMS_SC_260513/viz/viz_heatmap.py for the mouse plasma
dataset: uses config.ID_COL and a precomputed ``label`` column, and a
group-colour column annotation across the 8 genotype×timepoint groups.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

import config
import utils


def _zscore_rows(arr):
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True, ddof=0)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def plot_dep_heatmap(matrix_df: pd.DataFrame, meta: pd.DataFrame, dep_df: pd.DataFrame,
                     *, output_dir: str, name: str, title: str,
                     group_order: list[str], top_n: int = 60,
                     col_cluster: bool = True):
    """Top-N significant DEP heatmap over the given groups (samples grouped)."""
    sig = dep_df[dep_df["Significant"]].copy()
    if sig.empty:
        return None
    sig = sig.sort_values(config.SIG_PVALUE_COL).head(top_n)
    ids = set(sig[config.ID_COL].astype(str))
    work = matrix_df[matrix_df[config.ID_COL].astype(str).isin(ids)].copy()

    ordered_cols, ordered_meta = utils.ordered_bio_cols(meta, matrix_df.columns.tolist(),
                                                        group_order)
    sample_order = [c for c in ordered_cols
                    if ordered_meta.set_index("column_name").loc[c, "group"] in group_order]
    sample_ids = [ordered_meta.set_index("column_name").loc[c, "sample_id"]
                  for c in sample_order]
    groups = [ordered_meta.set_index("column_name").loc[c, "group"] for c in sample_order]

    mat = work[sample_order].values.astype(float)
    id_to_label = dict(zip(sig[config.ID_COL].astype(str), sig["label"]))
    row_labels = [id_to_label.get(str(pg), str(pg)) for pg in work[config.ID_COL]]

    z = np.clip(_zscore_rows(mat), -3, 3)
    n_rows, n_cols = z.shape
    height = min(max(7.0, 0.10 * n_rows + 4.0), 18.0)
    width = min(max(8.0, 0.12 * n_cols + 4.0), 22.0)

    bio = ordered_meta[ordered_meta["sample_id"].isin(sample_ids)]
    col_colors = utils.heatmap_col_colors(bio, sample_ids)
    try:
        row_link = linkage(pdist(z, metric="correlation"), method="average")
    except Exception:
        row_link = linkage(pdist(z, metric="euclidean"), method="average")
    col_link = linkage(pdist(z.T, metric="euclidean"), method="ward") if col_cluster else None

    show_row_labels = n_rows <= 70
    # Widen the figure to reserve right-margin room for the stacked legends.
    g = sns.clustermap(
        pd.DataFrame(z, index=row_labels, columns=sample_ids),
        row_linkage=row_link, col_linkage=col_link, col_cluster=col_cluster,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        col_colors=col_colors, yticklabels=show_row_labels, xticklabels=False,
        figsize=(width + 2.5, height), cbar_kws=dict(label="Row z-score"),
        dendrogram_ratio=(0.12, 0.08 if col_cluster else 0.0), linewidths=0)
    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
    g.fig.suptitle(title, y=1.02, fontsize=config.TITLE_SIZE)

    # Stacked Genotype / Timepoint / Group legends + repositioned colour bar.
    legs = utils.add_heatmap_annotation_legends(g, x=1.05)
    png = utils.save_clustermap(g, output_dir, name, extra_artists=legs)
    plt.close(g.fig)
    return png
