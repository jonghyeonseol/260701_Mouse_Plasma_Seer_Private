"""Hierarchical clustering heatmaps (seaborn.clustermap).

Shared drawing core (`_draw`) + three entry points used by the DEP step and the
pairwise module:
  - plot_dep_heatmap      : top-N significant DEPs (row labels shown)
  - plot_feature_heatmap  : top-variance N proteins, or ALL filtered features
Green-black-red z-score colormap; stacked annotation legends + repositioned
colorbar (via utils); titles kept clear of the plot via a tight bbox.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def _subset_samples(matrix_df, meta, group_order):
    """Ordered biological sample columns restricted to `group_order`."""
    ordered_cols, ordered_meta = utils.ordered_bio_cols(
        meta, matrix_df.columns.tolist(), group_order)
    gmap = ordered_meta.set_index("column_name")["group"]
    sample_order = [c for c in ordered_cols if gmap.get(c) in group_order]
    om = ordered_meta.set_index("column_name")
    sample_ids = [om.loc[c, "sample_id"] for c in sample_order]
    bio = ordered_meta[ordered_meta["column_name"].isin(sample_order)]
    return sample_order, sample_ids, bio


def _draw(z, row_labels, sample_ids, bio, *, output_dir, name, title,
          annotation_axes=("Genotype", "Timepoint", "Group"), group_subset=None,
          col_cluster=True, row_cluster=True, show_row_labels=None):
    n_rows, n_cols = z.shape
    if show_row_labels is None:
        show_row_labels = n_rows <= 70
    height = min(max(7.0, 0.10 * n_rows + 4.0), 18.0)
    width = min(max(8.0, 0.12 * n_cols + 4.0), 22.0)

    col_colors = utils.heatmap_col_colors(bio, sample_ids, which=annotation_axes)
    row_link = None
    if row_cluster:
        try:
            row_link = linkage(pdist(z, metric="correlation"), method="average")
        except Exception:
            row_link = linkage(pdist(z, metric="euclidean"), method="average")
    col_link = linkage(pdist(z.T, metric="euclidean"), method="ward") if col_cluster else None

    g = sns.clustermap(
        pd.DataFrame(z, index=row_labels, columns=sample_ids),
        row_linkage=row_link, col_linkage=col_link,
        row_cluster=row_cluster, col_cluster=col_cluster,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        col_colors=col_colors, yticklabels=show_row_labels, xticklabels=False,
        figsize=(width + 2.5, height), cbar_kws=dict(label=""),
        dendrogram_ratio=(0.12 if row_cluster else 0.0, 0.08 if col_cluster else 0.0),
        linewidths=0)
    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
    title_artist = utils.set_clustermap_title(g, title)

    legend_x = 1.16 if show_row_labels else 1.05
    legs = utils.add_heatmap_annotation_legends(
        g, which=annotation_axes, x=legend_x, group_subset=group_subset)
    png = utils.save_clustermap(g, output_dir, name, extra_artists=legs + [title_artist])
    plt.close(g.fig)
    return png


def plot_dep_heatmap(matrix_df: pd.DataFrame, meta: pd.DataFrame, dep_df: pd.DataFrame,
                     *, output_dir: str, name: str, title: str,
                     group_order: list[str], top_n: int = 60, col_cluster: bool = True,
                     annotation_axes=("Genotype", "Timepoint", "Group"),
                     group_subset=None):
    """Top-N significant DEP heatmap over the given groups (protein names shown)."""
    sig = dep_df[dep_df["Significant"]].copy()
    if sig.empty:
        return None
    sig = sig.sort_values(config.SIG_PVALUE_COL).head(top_n)
    ids = set(sig[config.ID_COL].astype(str))
    work = matrix_df[matrix_df[config.ID_COL].astype(str).isin(ids)].copy()

    sample_order, sample_ids, bio = _subset_samples(matrix_df, meta, group_order)
    mat = work[sample_order].values.astype(float)
    id_to_label = dict(zip(sig[config.ID_COL].astype(str), sig["label"]))
    row_labels = [id_to_label.get(str(pg), str(pg)) for pg in work[config.ID_COL]]

    z = np.clip(_zscore_rows(mat), -3, 3)
    return _draw(z, row_labels, sample_ids, bio, output_dir=output_dir, name=name,
                 title=title, annotation_axes=annotation_axes, group_subset=group_subset,
                 col_cluster=col_cluster)


def plot_feature_heatmap(matrix_df: pd.DataFrame, meta: pd.DataFrame, *,
                         output_dir: str, name: str, title: str, group_order: list[str],
                         mode: str = "topvar", n_top: int = 500, col_cluster: bool = True,
                         annotation_axes=("Genotype", "Timepoint", "Group"),
                         group_subset=None):
    """Feature heatmap over the given groups.  mode='topvar' → top ``n_top`` by
    variance across the subset samples; mode='total' → all filtered proteins
    (no row labels, no row dendrogram for speed)."""
    sample_order, sample_ids, bio = _subset_samples(matrix_df, meta, group_order)
    mat_all = matrix_df[sample_order].values.astype(float)

    if mode == "topvar":
        var = np.nanvar(mat_all, axis=1)
        idx = np.argsort(-var)[:min(n_top, mat_all.shape[0])]
        z = np.clip(_zscore_rows(mat_all[idx]), -3, 3)
        row_cluster, show_row_labels = True, False
    elif mode == "total":
        # All features: pre-order rows by hierarchical clustering leaf order
        # (row_cluster=False avoids rendering a meaningless thousands-leaf dendrogram).
        z = np.clip(_zscore_rows(mat_all), -3, 3)
        from scipy.cluster.hierarchy import leaves_list
        try:
            link = linkage(pdist(z, metric="correlation"), method="average")
            z = z[leaves_list(link)]
        except Exception:
            pass
        row_cluster, show_row_labels = False, False
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    row_labels = [""] * z.shape[0]
    return _draw(z, row_labels, sample_ids, bio, output_dir=output_dir, name=name,
                 title=title, annotation_axes=annotation_axes, group_subset=group_subset,
                 col_cluster=col_cluster, row_cluster=row_cluster,
                 show_row_labels=show_row_labels)
