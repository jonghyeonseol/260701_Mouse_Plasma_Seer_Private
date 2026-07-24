"""Step 11 — mouse-brain-protein-filtered re-analysis (main cohort only).

Restricts the normalized+imputed matrix to proteins classified as
mouse-brain proteins by step01b (`mouse_brain_annotation.tsv`,
`is_mouse_brain_protein == True`), then re-runs the same dimensionality-
reduction and clustering analyses as step07/step08 on that subset — to test
whether narrowing the plasma proteome to brain-associated proteins changes
the clustering/separation behavior seen on the whole proteome (which showed
weak separation, silhouette ~ 0; see diagnose_separation.py findings).

The "regionally elevated" subset (Region enriched/enhanced/Group enriched)
is not analyzed as a second, separate subset — it is highlighted as an extra
row-color strip on the feature clustermap within this one brain-protein
analysis, per the confirmed plan.

Outputs (output_data/MousePlasma/11_brain_filtered/):
  - brain_protein_matrix.tsv                    the filtered subset matrix
  - pca_scores.tsv / tsne_embedding.tsv / umap_embedding.tsv
  - dimreduction_by_<axis>.{png,svg}             1x3: PCA | t-SNE | UMAP
  - dimreduction_grid.{png,svg}                  3 methods x 3 axes
  - sample_correlation_clustermap.{png,svg}
  - feature_clustermap.{png,svg}                 rows annotated by elevated flag
  - feature_clustermap_topvar.{png,svg}          top-500-variance subset of the above
  - brain_filtered_separation_summary.tsv        silhouette vs genotype/timepoint/group
"""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import seaborn as sns

import config
import utils
from step07_dim_reduction import _color_spec, _scatter


def _mirror(png, fig_mirror):
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


def _dim_reduction(matrix, bio, cfg, out, fig_mirror, dataset, logger, step_tag="step11",
                   label="Brain-protein-filtered"):
    sample_cols = bio["column_name"].tolist()
    X = matrix[sample_cols].values.T.astype(float)
    emb = utils.compute_embeddings(X)
    pcs, var, tsne_emb, umap_emb, perp = (emb["pcs"], emb["var"], emb["tsne"],
                                          emb["umap"], emb["perplexity"])

    base = bio[["sample_id", "group", "genotype", "timepoint"]].reset_index(drop=True)
    for name, e, cols in [
        ("pca_scores", pcs[:, :5], [f"PC{i+1}" for i in range(5)]),
        ("tsne_embedding", tsne_emb, ["tSNE1", "tSNE2"]),
        ("umap_embedding", umap_emb, ["UMAP1", "UMAP2"]),
    ]:
        edf = pd.concat([base, pd.DataFrame(e, columns=cols)], axis=1)
        utils.save_tsv(edf, out, name)

    embeddings = [("PCA", pcs[:, :2], f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)"),
                  ("t-SNE", tsne_emb, "tSNE1", "tSNE2"),
                  ("UMAP", umap_emb, "UMAP1", "UMAP2")]
    axes_list = cfg.get("dimred_axes", ["genotype", "timepoint", "group"])

    for axis in axes_list:
        values, cmap, order, title = _color_spec(axis, bio, cfg)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
        for ax, (mname, e, xl, yl) in zip(axs, embeddings):
            _scatter(ax, e, values, cmap, order, xl, yl, mname)
        handles = [Patch(color=cmap.get(v, "#888"),
                         label=config.GROUP_DISPLAY.get(v, str(v)) if isinstance(v, str)
                         else f"{v}W") for v in order]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(order), 8), frameon=False)
        fig.suptitle(f"{label} dim-reduction — coloured by {title} — {dataset}",
                     fontsize=config.TITLE_SIZE, y=1.0)
        fig.tight_layout(rect=[0, 0.07, 1, 0.97])
        _mirror(utils.save_fig(fig, out, f"dimreduction_by_{axis}", tight=False), fig_mirror)
        plt.close(fig)

    ncol = len(axes_list)
    fig, axes = plt.subplots(3, ncol, figsize=(5.3 * ncol + 0.5, 15), squeeze=False)
    for r, (mname, e, xl, yl) in enumerate(embeddings):
        for c, axis in enumerate(axes_list):
            values, cmap, order, title = _color_spec(axis, bio, cfg)
            ax = axes[r, c]
            _scatter(ax, e, values, cmap, order, xl, yl, f"{mname} · {title}" if r == 0 else mname)
            if r == 0:
                handles = [Patch(color=cmap.get(v, "#888"),
                                 label=config.GROUP_DISPLAY.get(v, str(v))
                                 if isinstance(v, str) else f"{v}W") for v in order]
                ax.legend(handles=handles, fontsize=6, loc="best", ncol=1)
    fig.suptitle(f"{label} dimensionality reduction — {dataset}",
                fontsize=config.TITLE_SIZE, y=1.0)
    _mirror(utils.save_fig(fig, out, "dimreduction_grid", tight=True), fig_mirror)
    plt.close(fig)

    logger.info(f"[{step_tag}] {dataset}: dim-reduction — PC1={var[0]:.1f}%, "
                f"PC2={var[1]:.1f}%, perplexity={perp}, n={len(sample_cols)} samples.")
    return pcs


def _sample_correlation_clustermap(matrix, bio, out, fig_mirror, annotation_axes,
                                   label="Brain-protein subset"):
    sample_cols = bio["column_name"].tolist()
    sample_ids = bio["sample_id"].tolist()
    X = matrix[sample_cols].values.T.astype(float)
    corr = np.corrcoef(X)
    col_colors = utils.heatmap_col_colors(bio, sample_ids, which=annotation_axes)
    link = linkage(pdist(X, metric="euclidean"), method="ward")
    g = sns.clustermap(
        pd.DataFrame(corr, index=sample_ids, columns=sample_ids),
        row_linkage=link, col_linkage=link,
        cmap="magma", figsize=(15.5, 13),
        row_colors=col_colors, col_colors=col_colors,
        xticklabels=False, yticklabels=(len(sample_ids) <= 90),
        cbar_kws=dict(label=""), dendrogram_ratio=(0.12, 0.12),
        colors_ratio=0.018, linewidths=0)
    t = utils.set_clustermap_title(g, f"{label} — sample correlation clustering")
    legs = utils.add_heatmap_annotation_legends(g, which=annotation_axes, x=1.16,
                                                cbar_title="Pearson r")
    png = utils.save_clustermap(g, out, "sample_correlation_clustermap", extra_artists=legs + [t])
    plt.close(g.fig)
    _mirror(png, fig_mirror)
    return link


def _feature_clustermap(matrix, annotation, bio, out, fig_mirror, annotation_axes):
    """Feature clustermap of the full brain-protein subset, with an extra
    row-color strip marking the regionally-elevated proteins."""
    sample_cols = bio["column_name"].tolist()
    sample_ids = bio["sample_id"].tolist()
    mat = matrix[sample_cols].values.astype(float)
    z = np.clip((mat - mat.mean(axis=1, keepdims=True)) /
                (mat.std(axis=1, keepdims=True, ddof=0) + 1e-9), -3, 3)

    elevated_map = dict(zip(annotation[config.ID_COL], annotation["is_regionally_elevated"]))
    row_colors = pd.DataFrame({
        "Regionally elevated": ["#E00000" if elevated_map.get(pid, False) else "#DDDDDD"
                                for pid in matrix[config.ID_COL]]
    }, index=range(len(matrix)))

    col_colors = utils.heatmap_col_colors(bio, sample_ids, which=annotation_axes)
    row_link = linkage(pdist(z, metric="correlation"), method="average")
    col_link = linkage(pdist(z.T, metric="euclidean"), method="ward")
    g = sns.clustermap(
        pd.DataFrame(z, columns=sample_ids),
        row_linkage=row_link, col_linkage=col_link,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        row_colors=row_colors, col_colors=col_colors, figsize=(15.5, 12),
        xticklabels=False, yticklabels=False,
        cbar_kws=dict(label=""), dendrogram_ratio=(0.10, 0.12),
        colors_ratio=0.02, linewidths=0)
    t = utils.set_clustermap_title(
        g, f"Mouse-brain-protein subset (n={len(matrix)}) — hierarchical clustering")
    legs = utils.add_heatmap_annotation_legends(g, which=annotation_axes, x=1.04)
    handles = [Patch(color="#E00000", label="Regionally elevated"),
               Patch(color="#DDDDDD", label="Other brain protein")]
    leg2 = g.ax_heatmap.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 0.30),
                               frameon=False, fontsize=7, title="Regional status",
                               title_fontsize=8)
    png = utils.save_clustermap(g, out, "feature_clustermap", extra_artists=legs + [t, leg2])
    plt.close(g.fig)
    _mirror(png, fig_mirror)


def _feature_clustermap_topvar(matrix, annotation, bio, out, fig_mirror, annotation_axes,
                               top_n=500):
    """Same as `_feature_clustermap` but restricted to the top-`top_n`-variance
    rows (mirrors step08's `feature_clustermap_topvar`) — a smaller, more
    readable companion to the full brain-protein clustermap."""
    sample_cols = bio["column_name"].tolist()
    sample_ids = bio["sample_id"].tolist()
    var = matrix[sample_cols].var(axis=1).values
    idx = np.argsort(-var)[:min(top_n, len(var))]
    sub = matrix.iloc[idx]
    mat = sub[sample_cols].values.astype(float)
    z = np.clip((mat - mat.mean(axis=1, keepdims=True)) /
                (mat.std(axis=1, keepdims=True, ddof=0) + 1e-9), -3, 3)

    elevated_map = dict(zip(annotation[config.ID_COL], annotation["is_regionally_elevated"]))
    row_colors = pd.DataFrame({
        "Regionally elevated": ["#E00000" if elevated_map.get(pid, False) else "#DDDDDD"
                                for pid in sub[config.ID_COL]]
    }, index=range(len(sub)))

    col_colors = utils.heatmap_col_colors(bio, sample_ids, which=annotation_axes)
    row_link = linkage(pdist(z, metric="correlation"), method="average")
    col_link = linkage(pdist(z.T, metric="euclidean"), method="ward")
    g = sns.clustermap(
        pd.DataFrame(z, columns=sample_ids),
        row_linkage=row_link, col_linkage=col_link,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        row_colors=row_colors, col_colors=col_colors, figsize=(15.5, 12),
        xticklabels=False, yticklabels=False,
        cbar_kws=dict(label=""), dendrogram_ratio=(0.10, 0.12),
        colors_ratio=0.02, linewidths=0)
    t = utils.set_clustermap_title(
        g, f"Mouse-brain-protein subset — top-{len(idx)} variable proteins — hierarchical clustering")
    legs = utils.add_heatmap_annotation_legends(g, which=annotation_axes, x=1.04)
    handles = [Patch(color="#E00000", label="Regionally elevated"),
               Patch(color="#DDDDDD", label="Other brain protein")]
    leg2 = g.ax_heatmap.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 0.30),
                               frameon=False, fontsize=7, title="Regional status",
                               title_fontsize=8)
    png = utils.save_clustermap(g, out, "feature_clustermap_topvar", extra_artists=legs + [t, leg2])
    plt.close(g.fig)
    _mirror(png, fig_mirror)


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "11_brain_filtered")
    fig_mirror = utils.figures_dir(dataset)
    utils.apply_style()

    annotation_path = os.path.join(utils.step_path(dataset, "01_raw_summary"),
                                   "mouse_brain_annotation.tsv")
    if not os.path.exists(annotation_path):
        logger.warning(f"[step11] {dataset}: mouse_brain_annotation.tsv not found "
                       f"(step01b not run) — skipping.")
        return state
    annotation = pd.read_csv(annotation_path, sep="\t")

    full_matrix = state["data_df"]
    meta = state["meta"]
    bio = meta[(~meta["is_ref"]) & (meta["group"].isin(cfg["group_order"]))].reset_index(drop=True)

    brain_ids = set(annotation.loc[annotation["is_mouse_brain_protein"], config.ID_COL])
    matrix = full_matrix[full_matrix[config.ID_COL].isin(brain_ids)].reset_index(drop=True)
    n_elevated = int(annotation.loc[annotation[config.ID_COL].isin(matrix[config.ID_COL]),
                                    "is_regionally_elevated"].sum())
    logger.info(f"[step11] {dataset}: {len(matrix)}/{len(full_matrix)} filtered proteins are "
                f"mouse-brain proteins ({n_elevated} regionally elevated).")

    # Explicit last-column flag (string "True"/"False"), per user request. Since
    # `matrix` is already subset to is_mouse_brain_protein==True, every row here
    # reads "True" — this file only exists to be the brain-protein set. Re-looked
    # up from the annotation table rather than hardcoded, so it stays correct if
    # this function is ever called on an unfiltered matrix.
    brain_flag_map = dict(zip(annotation[config.ID_COL], annotation["is_mouse_brain_protein"]))
    matrix["Mouse_Brain_Protein"] = matrix[config.ID_COL].map(brain_flag_map).map(
        {True: "True", False: "False"})
    utils.save_tsv(matrix, out, "brain_protein_matrix")

    if len(matrix) < 10:
        logger.warning(f"[step11] {dataset}: only {len(matrix)} brain proteins in the filtered "
                       f"set — too few for a meaningful re-analysis, skipping downstream steps.")
        return state

    annotation_axes = cfg.get("annotation_axes", ["Genotype", "Timepoint", "Group"])

    pcs = _dim_reduction(matrix, bio, cfg, out, fig_mirror, dataset, logger)
    _sample_correlation_clustermap(matrix, bio, out, fig_mirror, annotation_axes)
    _feature_clustermap(matrix, annotation, bio, out, fig_mirror, annotation_axes)
    _feature_clustermap_topvar(matrix, annotation, bio, out, fig_mirror, annotation_axes)

    # Silhouette summary: does the brain-protein subset separate design groups
    # any better than the whole-proteome result (silhouette ~ 0)?
    rows = []
    for axis_name, labels in [("genotype", bio["genotype"].values),
                              ("timepoint", bio["timepoint"].astype(int).values),
                              ("group", bio["group"].values)]:
        sil = silhouette_score(pcs, labels) if len(set(labels)) > 1 else float("nan")
        rows.append({"axis": axis_name, "n_proteins": len(matrix),
                    "n_samples": len(bio), "silhouette_pca": round(sil, 4)})
    summary = pd.DataFrame(rows)
    utils.save_tsv(summary, out, "brain_filtered_separation_summary")
    logger.info(f"[step11] {dataset}: brain-protein-subset silhouette — " +
                ", ".join(f"{r['axis']}={r['silhouette_pca']:.3f}" for r in rows))

    return state


if __name__ == "__main__":
    pass
