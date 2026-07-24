"""Step 12 — regionally-elevated-only re-analysis (main cohort only).

A narrower companion to step11: instead of the full mouse-brain-protein subset
(~6,881 proteins), this restricts to just the "regionally elevated" genes from
the HPA DB (config.BRAIN_ELEVATED_CATEGORIES = Region enriched/enhanced, Group
enriched — 763 genes total in the DB) that are actually detected and survive
QC in this experiment. Re-runs the same dim-reduction + clustering pattern as
step11 on this much smaller, more specific set.

Coverage (DB -> real data), logged and saved for the record:
  - 763 genes total in the DB (config.BRAIN_ELEVATED_CATEGORIES union; verified
    exactly 333+266+164=763, no duplicates).
  - Of those, how many map to >=1 protein in the raw 9,769 (i.e. actually
    detected by SEER in this species/tissue at all) and how many survive
    step02's QC filter (7,899) — reported in
    `regionally_elevated_coverage_report.tsv`.

Outputs (output_data/MousePlasma/12_regionally_elevated/):
  - regionally_elevated_matrix.tsv      the filtered subset matrix (+ Mouse_Brain_Protein,
                                         Regional_Category last columns)
  - regionally_elevated_coverage_report.tsv   DB-vs-data coverage numbers
  - pca_scores.tsv / tsne_embedding.tsv / umap_embedding.tsv
  - dimreduction_by_<axis>.{png,svg} / dimreduction_grid.{png,svg}
  - sample_correlation_clustermap.{png,svg}
  - feature_clustermap.{png,svg}        rows coloured by elevated sub-category
  - regionally_elevated_separation_summary.tsv
"""

from __future__ import annotations

import os

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
from step11_brain_filtered import _dim_reduction, _sample_correlation_clustermap, _mirror

CATEGORY_COLORS = {
    "Region enriched": "#C0392B",
    "Region enhanced": "#E67E22",
    "Group enriched":  "#8E44AD",
}


def _coverage_report(db_elevated_genes, ann, filt_ids, out, logger):
    """DB (763 genes) -> raw (9,769) -> QC-filtered (7,899) coverage."""
    elevated_rows = ann[ann["is_regionally_elevated"]]
    raw_genes = set(elevated_rows["matched_brain_gene"])
    filt_rows = elevated_rows[elevated_rows[config.ID_COL].isin(filt_ids)]
    filt_genes = set(filt_rows["matched_brain_gene"])

    report = pd.DataFrame([
        {"stage": "DB (config.BRAIN_ELEVATED_CATEGORIES, unique genes)",
         "n_genes": len(db_elevated_genes), "n_proteins": len(db_elevated_genes),
         "pct_of_db": 100.0},
        {"stage": "Raw data (9,769 proteins) — any match",
         "n_genes": len(raw_genes), "n_proteins": len(elevated_rows),
         "pct_of_db": round(len(raw_genes) / len(db_elevated_genes) * 100, 1)},
        {"stage": "QC-filtered (7,899 proteins) — survives step02",
         "n_genes": len(filt_genes), "n_proteins": len(filt_rows),
         "pct_of_db": round(len(filt_genes) / len(db_elevated_genes) * 100, 1)},
    ])
    utils.save_tsv(report, out, "regionally_elevated_coverage_report")
    logger.info(f"[step12] coverage — DB: {len(db_elevated_genes)} genes | "
                f"raw data match: {len(raw_genes)} genes ({len(elevated_rows)} proteins, "
                f"{len(raw_genes)/len(db_elevated_genes)*100:.1f}% of DB) | "
                f"QC-filtered: {len(filt_genes)} genes ({len(filt_rows)} proteins, "
                f"{len(filt_genes)/len(db_elevated_genes)*100:.1f}% of DB).")
    return report, filt_rows


def _feature_clustermap_by_category(matrix, category_map, bio, out, fig_mirror, annotation_axes):
    sample_cols = bio["column_name"].tolist()
    sample_ids = bio["sample_id"].tolist()
    mat = matrix[sample_cols].values.astype(float)
    z = np.clip((mat - mat.mean(axis=1, keepdims=True)) /
                (mat.std(axis=1, keepdims=True, ddof=0) + 1e-9), -3, 3)

    row_colors = pd.DataFrame({
        "Regional category": [CATEGORY_COLORS.get(category_map.get(pid, ""), "#DDDDDD")
                              for pid in matrix[config.ID_COL]]
    }, index=range(len(matrix)))
    labels = utils.build_display_labels(matrix)

    col_colors = utils.heatmap_col_colors(bio, sample_ids, which=annotation_axes)
    row_link = linkage(pdist(z, metric="correlation"), method="average")
    col_link = linkage(pdist(z.T, metric="euclidean"), method="ward")
    show_labels = len(matrix) <= 200
    g = sns.clustermap(
        pd.DataFrame(z, index=labels if show_labels else None, columns=sample_ids),
        row_linkage=row_link, col_linkage=col_link,
        cmap=utils.zscore_cmap(), vmin=-3, vmax=3, center=0,
        row_colors=row_colors, col_colors=col_colors, figsize=(15.5, max(8.0, 0.09 * len(matrix) + 4)),
        xticklabels=False, yticklabels=show_labels,
        cbar_kws=dict(label=""), dendrogram_ratio=(0.14, 0.10),
        colors_ratio=0.02, linewidths=0)
    if show_labels:
        g.ax_heatmap.tick_params(axis="y", labelsize=5.5)
    t = utils.set_clustermap_title(
        g, f"Regionally-elevated mouse-brain proteins (n={len(matrix)}) — hierarchical clustering")
    legs = utils.add_heatmap_annotation_legends(g, which=annotation_axes,
                                                x=1.16 if show_labels else 1.04)
    handles = [Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    leg2 = g.ax_heatmap.legend(handles=handles, loc="upper left",
                               bbox_to_anchor=(1.16 if show_labels else 1.04, 0.30),
                               frameon=False, fontsize=7, title="Regional category",
                               title_fontsize=8)
    png = utils.save_clustermap(g, out, "feature_clustermap", extra_artists=legs + [t, leg2])
    plt.close(g.fig)
    _mirror(png, fig_mirror)


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "12_regionally_elevated")
    fig_mirror = utils.figures_dir(dataset)
    utils.apply_style()

    annotation_path = os.path.join(utils.step_path(dataset, "01_raw_summary"),
                                   "mouse_brain_annotation.tsv")
    if not os.path.exists(annotation_path):
        logger.warning(f"[step12] {dataset}: mouse_brain_annotation.tsv not found "
                       f"(step01b not run) — skipping.")
        return state
    ann = pd.read_csv(annotation_path, sep="\t")

    db = pd.read_csv(config.MOUSE_BRAIN_DB, sep="\t", low_memory=False)
    db_elevated_mask = db[config.BRAIN_DB_REGIONAL_COL].isin(config.BRAIN_ELEVATED_CATEGORIES)
    db_elevated_genes = set(db.loc[db_elevated_mask, config.BRAIN_DB_GENE_COL].str.upper())
    logger.info(f"[step12] {dataset}: DB has {len(db_elevated_genes)} unique regionally-elevated "
                f"genes ({dict(db.loc[db_elevated_mask, config.BRAIN_DB_REGIONAL_COL].value_counts())}).")

    full_matrix = state["data_df"]
    meta = state["meta"]
    bio = meta[(~meta["is_ref"]) & (meta["group"].isin(cfg["group_order"]))].reset_index(drop=True)

    filt_ids = set(full_matrix[config.ID_COL])
    report, filt_rows = _coverage_report(db_elevated_genes, ann, filt_ids, out, logger)

    elevated_ids = set(filt_rows[config.ID_COL])
    matrix = full_matrix[full_matrix[config.ID_COL].isin(elevated_ids)].reset_index(drop=True)
    category_map = dict(zip(filt_rows[config.ID_COL], filt_rows["mouse_brain_regional_category"]))
    matrix["Mouse_Brain_Protein"] = "True"
    matrix["Regional_Category"] = matrix[config.ID_COL].map(category_map)
    utils.save_tsv(matrix, out, "regionally_elevated_matrix")

    if len(matrix) < 10:
        logger.warning(f"[step12] {dataset}: only {len(matrix)} regionally-elevated proteins "
                       f"survive QC — too few for a meaningful re-analysis, skipping downstream.")
        return state

    annotation_axes = cfg.get("annotation_axes", ["Genotype", "Timepoint", "Group"])

    pcs = _dim_reduction(matrix, bio, cfg, out, fig_mirror, dataset, logger, step_tag="step12",
                        label="Regionally-elevated")
    _sample_correlation_clustermap(matrix, bio, out, fig_mirror, annotation_axes,
                                   label="Regionally-elevated subset")
    _feature_clustermap_by_category(matrix, category_map, bio, out, fig_mirror, annotation_axes)

    rows = []
    for axis_name, labels in [("genotype", bio["genotype"].values),
                              ("timepoint", bio["timepoint"].astype(int).values),
                              ("group", bio["group"].values)]:
        sil = silhouette_score(pcs, labels) if len(set(labels)) > 1 else float("nan")
        rows.append({"axis": axis_name, "n_proteins": len(matrix),
                    "n_samples": len(bio), "silhouette_pca": round(sil, 4)})
    summary = pd.DataFrame(rows)
    utils.save_tsv(summary, out, "regionally_elevated_separation_summary")
    logger.info(f"[step12] {dataset}: regionally-elevated-subset silhouette — " +
                ", ".join(f"{r['axis']}={r['silhouette_pca']:.3f}" for r in rows))

    return state


if __name__ == "__main__":
    pass
