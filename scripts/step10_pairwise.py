"""Step 10 — pairwise 2-group analysis module (main cohort).

The integrated modules (step07 dim-reduction, step08 clustering) analyze all
groups at once, where proteome heterogeneity masks group structure.  This module
reduces the problem to a simple 2-class comparison: for every 2-group contrast in
config.COMPARISONS (genotype_at_time + time_course = 16 pairs) it subsets to just
those two groups and produces PCA / t-SNE / UMAP + a top-DEP heatmap, plus a
silhouette separation score quantifying how cleanly the two groups split.

Outputs (per pair, under 10_pairwise/<contrast_name>/):
  - dimreduction.{png,svg}          PCA | t-SNE | UMAP, coloured by the 2 groups
  - pca_scores.tsv / tsne.tsv / umap.tsv
  - heatmap_top_deps.{png,svg}      top significant DEPs (reused limma results)
Cross-pair:
  - pairwise_separation_summary.tsv  (n, PC1/PC2 var%, silhouette, n_DEP)
  - pairwise_separation_bar.{png,svg}
"""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import silhouette_score

import config
import utils
from viz.viz_heatmap import plot_dep_heatmap, plot_feature_heatmap


def _pairwise_contrasts(comparisons):
    """2-group contrasts whose case/ctrl are real groups (not composite labels)."""
    return [c for c in comparisons
            if c["family"] in ("genotype_at_time", "time_course")]


def _scatter(ax, emb, groups_vec, order, xlab, ylab, title):
    for g in order:
        m = (groups_vec == g)
        if not m.any():
            continue
        ax.scatter(emb[m, 0], emb[m, 1], color=config.GROUP_COLORS.get(g, "#888"),
                   edgecolor="black", linewidth=0.4, s=60, alpha=0.85,
                   label=config.GROUP_DISPLAY.get(g, g))
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="semibold")


def _dimreduction_figure(emb, groups_vec, order, pair_dir, fig_mirror, name, title):
    utils.apply_style()
    var = emb["var"]
    panels = [("PCA", emb["pcs"][:, :2], f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)"),
              ("t-SNE", emb["tsne"], "tSNE1", "tSNE2"),
              ("UMAP", emb["umap"], "UMAP1", "UMAP2")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (mname, e, xl, yl) in zip(axes, panels):
        _scatter(ax, e, groups_vec, order, xl, yl, mname)
    handles = [Patch(color=config.GROUP_COLORS.get(g, "#888"),
                     label=config.GROUP_DISPLAY.get(g, g)) for g in order]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=config.TITLE_SIZE, y=1.0)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    png = utils.save_fig(fig, pair_dir, "dimreduction", tight=False)
    plt.close(fig)
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, f"pairwise_{name}_dimreduction{ext}"))


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out_root = utils.step_path(dataset, "10_pairwise")
    fig_mirror = utils.figures_dir(dataset)
    utils.apply_style()

    matrix = state["data_df"]
    meta = state["meta"]
    deps = state.get("deps_by_comparison", {})
    pairs = _pairwise_contrasts(cfg["comparisons"])

    rows = []
    for comp in pairs:
        name, case_g, ctrl_g = comp["name"], comp["case_grp"], comp["ctrl_grp"]
        order = [ctrl_g, case_g]
        sub = meta[(~meta["is_ref"]) & (meta["group"].isin(order))]
        sample_cols = sub["column_name"].tolist()
        groups_vec = sub["group"].values
        if len(sample_cols) < 4:
            logger.warning(f"[step10] {name}: too few samples ({len(sample_cols)}) — skipped.")
            continue
        pair_dir = utils.ensure_dir(os.path.join(out_root, name))

        X = matrix[sample_cols].values.T.astype(float)
        emb = utils.compute_embeddings(X)

        # Save embeddings
        base = sub[["sample_id", "group"]].reset_index(drop=True)
        for nm, e, cols in [("pca_scores", emb["pcs"][:, :5],
                             [f"PC{i+1}" for i in range(min(5, emb["pcs"].shape[1]))]),
                            ("tsne", emb["tsne"], ["tSNE1", "tSNE2"]),
                            ("umap", emb["umap"], ["UMAP1", "UMAP2"])]:
            edf = pd.concat([base, pd.DataFrame(e, columns=cols)], axis=1)
            utils.save_tsv(edf, pair_dir, nm)

        _dimreduction_figure(emb, groups_vec, order, pair_dir, fig_mirror, name,
                             f"{case_g} vs {ctrl_g}  (pairwise, n={len(sample_cols)})")

        # Silhouette separation on PCA space (2 groups).
        labels = np.array([0 if g == ctrl_g else 1 for g in groups_vec])
        sil = float(silhouette_score(emb["pcs"], labels)) if len(set(labels)) == 2 else np.nan

        # Three heatmaps per pair, annotated by the 2 groups only.
        ann = ["Group"]
        # (1) Top-DEP heatmap (reuse limma results for this contrast)
        n_dep = 0
        if name in deps:
            dep_df = deps[name]
            n_dep = int(dep_df["Significant"].sum())
            if n_dep > 0:
                plot_dep_heatmap(matrix, meta, dep_df, output_dir=pair_dir,
                                 name="heatmap_deps", group_order=order,
                                 title=f"DEP heatmap — {case_g} vs {ctrl_g} (n={n_dep})",
                                 top_n=60, annotation_axes=ann, group_subset=order)
            else:
                logger.info(f"[step10] {name}: 0 DEPs — DEP heatmap skipped.")
        # (2) Top-500 variance heatmap
        plot_feature_heatmap(matrix, meta, output_dir=pair_dir,
                             name="heatmap_top500var", group_order=order, mode="topvar",
                             n_top=500, annotation_axes=ann, group_subset=order,
                             title=f"Top-500 variance — {case_g} vs {ctrl_g}")
        # (3) Total features heatmap
        plot_feature_heatmap(matrix, meta, output_dir=pair_dir,
                             name="heatmap_total_features", group_order=order, mode="total",
                             annotation_axes=ann, group_subset=order,
                             title=f"All features ({matrix.shape[0]}) — {case_g} vs {ctrl_g}")
        for base in ("heatmap_deps", "heatmap_top500var", "heatmap_total_features"):
            for ext in (".png", ".svg"):
                src = os.path.join(pair_dir, f"{base}{ext}")
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(fig_mirror, f"pairwise_{name}_{base}{ext}"))

        rows.append({"pair": name, "family": comp["family"], "case": case_g,
                     "ctrl": ctrl_g, "n_samples": len(sample_cols),
                     "PC1_var_pct": round(float(emb["var"][0]), 2),
                     "PC2_var_pct": round(float(emb["var"][1]), 2),
                     "silhouette": round(sil, 4), "n_DEP": n_dep})
        logger.info(f"[step10] {name}: n={len(sample_cols)}, silhouette={sil:.3f}, "
                    f"n_DEP={n_dep}.")

    summary = pd.DataFrame(rows)
    utils.save_tsv(summary, out_root, "pairwise_separation_summary")

    # Silhouette bar across pairs (separation overview).
    if not summary.empty:
        s = summary.sort_values("silhouette", ascending=True)
        fam_color = {"genotype_at_time": "#6A51A3", "time_course": "#2171B5"}
        fig, ax = plt.subplots(figsize=(9, max(4.5, 0.4 * len(s) + 1.5)))
        colors = [fam_color.get(f, "#888") for f in s["family"]]
        ax.barh(range(len(s)), s["silhouette"], color=colors, edgecolor="black",
                linewidth=0.3)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s["pair"], fontsize=8)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel("Silhouette (2-group separation on PCA)")
        ax.set_title("Pairwise separation overview\n(higher = groups separate better)",
                     fontsize=12)
        ax.legend(handles=[Patch(color=c, label=f) for f, c in fam_color.items()],
                  loc="lower right", fontsize=8)
        png = utils.save_fig(fig, out_root, "pairwise_separation_bar")
        plt.close(fig)
        for ext in (".png", ".svg"):
            src = png.replace(".png", ext)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))

    logger.info(f"[step10] {dataset}: pairwise analysis complete for {len(rows)} pairs; "
                f"mean silhouette={summary['silhouette'].mean():.3f}" if rows else
                f"[step10] {dataset}: no pairs.")
    state = dict(state)
    state["pairwise_summary"] = summary
    return state


if __name__ == "__main__":
    pass
