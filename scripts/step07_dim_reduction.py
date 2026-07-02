"""Step 07 — dimensionality reduction (PCA + t-SNE + UMAP).

Embeddings are computed once on the normalized, imputed matrix (samples ×
proteins, standardized) and visualized coloured by each design axis:
genotype (WT/KI), timepoint (weeks) and the combined 8-group factor.

Outputs:
  - pca_scores.tsv / tsne_embedding.tsv / umap_embedding.tsv
  - pca_scree.{png,svg}
  - dimreduction_by_<axis>.{png,svg}          (1×3: PCA | t-SNE | UMAP)
  - dimreduction_grid.{png,svg}               (3 methods × 3 colour axes)
"""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import config
import utils


def _mirror(png, fig_mirror):
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


def _color_spec(axis: str, meta_bio: pd.DataFrame, cfg: dict):
    """Return (value_series, color_map, legend_order, title)."""
    if axis == "genotype":
        return (meta_bio["genotype"], config.GENOTYPE_COLORS,
                config.GENOTYPES, "Genotype")
    if axis == "timepoint":
        return (meta_bio["timepoint"].astype(int), config.TIMEPOINT_COLORS,
                cfg["timepoints"], "Timepoint (weeks)")
    return (meta_bio["group"], config.GROUP_COLORS,
            cfg["group_order"], "Group")


def _scatter(ax, emb, values, cmap, legend_order, xlab, ylab, title):
    for v in legend_order:
        m = (values == v).values
        if not m.any():
            continue
        lab = config.GROUP_DISPLAY.get(v, str(v)) if isinstance(v, str) else f"{v}W"
        ax.scatter(emb[m, 0], emb[m, 1], color=cmap.get(v, "#888"),
                   edgecolor="black", linewidth=0.4, s=48, alpha=0.85, label=lab)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="semibold")


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "07_dim_reduction")
    fig_mirror = utils.figures_dir(dataset)
    utils.apply_style()

    matrix = state["data_df"]
    meta = state["meta"]
    bio = meta[~meta["is_ref"]].copy()
    bio = bio[bio["group"].isin(cfg["group_order"])].reset_index(drop=True)
    sample_cols = bio["column_name"].tolist()

    X = matrix[sample_cols].values.T.astype(float)
    emb = utils.compute_embeddings(X)
    pcs, var, tsne_emb, umap_emb, perp = (emb["pcs"], emb["var"], emb["tsne"],
                                          emb["umap"], emb["perplexity"])

    base = bio[["sample_id", "group", "genotype", "timepoint"]].reset_index(drop=True)
    for name, emb, cols in [
        ("pca_scores", pcs[:, :5], [f"PC{i+1}" for i in range(5)]),
        ("tsne_embedding", tsne_emb, ["tSNE1", "tSNE2"]),
        ("umap_embedding", umap_emb, ["UMAP1", "UMAP2"]),
    ]:
        edf = pd.concat([base, pd.DataFrame(emb, columns=cols)], axis=1)
        utils.save_tsv(edf, out, name)

    embeddings = [("PCA", pcs[:, :2], f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)"),
                  ("t-SNE", tsne_emb, "tSNE1", "tSNE2"),
                  ("UMAP", umap_emb, "UMAP1", "UMAP2")]

    # Scree
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    ns = min(10, len(var))
    ax.bar(range(1, ns + 1), var[:ns], color="#56B4E9", edgecolor="black", linewidth=0.5)
    ax2 = ax.twinx()
    ax2.plot(range(1, ns + 1), np.cumsum(var[:ns]), color="#D55E00", marker="o", lw=1.6)
    ax2.axhline(80, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("Principal component"); ax.set_ylabel("Variance explained (%)")
    ax2.set_ylabel("Cumulative (%)", color="#D55E00")
    ax.set_xticks(range(1, ns + 1)); ax.set_title(f"PCA scree — {dataset}")
    _mirror(utils.save_fig(fig, out, "pca_scree"), fig_mirror)
    plt.close(fig)

    # Per-axis 1×3 (method) figures
    axes_list = cfg.get("dimred_axes", ["genotype", "timepoint", "group"])
    for axis in axes_list:
        values, cmap, order, title = _color_spec(axis, bio, cfg)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
        for ax, (mname, emb, xl, yl) in zip(axs, embeddings):
            _scatter(ax, emb, values, cmap, order, xl, yl, mname)
        handles = [Patch(color=cmap.get(v, "#888"),
                         label=config.GROUP_DISPLAY.get(v, str(v)) if isinstance(v, str)
                         else f"{v}W") for v in order]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(order), 8), frameon=False)
        fig.suptitle(f"Dimensionality reduction coloured by {title} — {dataset}",
                     fontsize=config.TITLE_SIZE, y=1.0)
        fig.tight_layout(rect=[0, 0.07, 1, 0.97])
        _mirror(utils.save_fig(fig, out, f"dimreduction_by_{axis}", tight=False), fig_mirror)
        plt.close(fig)

    # Combined 3 methods × N axes grid (N = number of design axes for this dataset)
    ncol = len(axes_list)
    fig, axes = plt.subplots(3, ncol, figsize=(5.3 * ncol + 0.5, 15), squeeze=False)
    for r, (mname, emb, xl, yl) in enumerate(embeddings):
        for c, axis in enumerate(axes_list):
            values, cmap, order, title = _color_spec(axis, bio, cfg)
            ax = axes[r, c]
            _scatter(ax, emb, values, cmap, order, xl, yl,
                     f"{mname} · {title}" if r == 0 else mname)
            if r == 0:
                handles = [Patch(color=cmap.get(v, "#888"),
                                 label=config.GROUP_DISPLAY.get(v, str(v))
                                 if isinstance(v, str) else f"{v}W") for v in order]
                ax.legend(handles=handles, fontsize=6, loc="best", ncol=1)
    fig.suptitle(f"Dimensionality reduction — {dataset}", fontsize=config.TITLE_SIZE, y=1.0)
    _mirror(utils.save_fig(fig, out, "dimreduction_grid", tight=True), fig_mirror)
    plt.close(fig)

    logger.info(f"[step07] {dataset}: PCA (PC1={var[0]:.1f}%, PC2={var[1]:.1f}%), "
                f"t-SNE (perplexity={perp}), UMAP done for {len(sample_cols)} samples.")
    return state


if __name__ == "__main__":
    pass
