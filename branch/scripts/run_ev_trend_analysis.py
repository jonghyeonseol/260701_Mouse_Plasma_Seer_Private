"""run_ev_trend_analysis.py

Mousebrain-elevated EV-protein trend analysis (per genotype branch).

This branch pipeline has its own step sequence, and every numbered output
folder below is that step's real position in THIS pipeline — not copied from
the reference project's unrelated step numbering (the reference project's own
"06_Trend_Analysis" reflects its 6th REPORT SECTION out of 10, covering a
completely different set of upstream analyses).

branch/output/tsv/ gets ALL 5 preprocessing stages (01-05), one audit TSV per
stage per genotype:
  01_EV_Candidate_Matching — load database/260812_mousebrain_elevated_
     exosome_database.tsv, keep genes with a non-empty Exosome_Classification
     (LC-EC / HC-EC / LC-VP / HC-VP), map each to real PG.ProteinGroups
     accessions via output_data/MousePlasma/01_raw_summary/
     mouse_brain_annotation.tsv (matched_brain_gene; built by step01b's
     UniProt gene-name resolution — same matching logic step11/step12 of the
     main pipeline already use).
  02_Filtered   — scripts/step02_filter (per-group valid-fraction filter),
                  reused as-is on this genotype's 4 timepoint groups.
  03_Log2_Transformed — scripts/step03_log2_transform, reused as-is.
  04_Normalized — scripts/step04_normalize (reference-ratio + median),
                  reused as-is.
  05_Imputed    — scripts/step05_impute (MinProb), reused as-is. Produces the
                  final per-genotype EV-subset matrix used for trend testing.

branch/output/figures/ only ever gets TWO stage folders, because steps 02-05
write tsv audit trails only (no plots) — so ITS OWN local sequence is:
  01_EV_Candidate_Matching — candidate_coverage_funnel.png
  02_Trend_Analysis — trend test per protein across the ordered timepoint
     axis (mirrors the reference project's branch/scripts/
     step08_trend_analysis.R): Jonckheere-Terpstra (normal approximation,
     same implementation as this project's own
     step09_trend_timeseries._jonckheere_terpstra) + Spearman rho vs
     timepoint + BH FDR. Figures (palette copied from the reference
     project's trends_common.R): trend_volcano.png, trend_heatmap.png,
     trend_boxplots_significant.png, trend_boxplots_top.png,
     trend_mean_se_lines.png. Run 3x, on 3 different data views, each in
     its own subfolder:
       WT/         — WT-only replicate values.
       KI/         — KI-only replicate values.
       KI_over_WT/ — tests whether the KI-vs-WT DIVERGENCE ITSELF trends
         monotonically over time (genotype x time interaction), not just
         each genotype's own time course. Restricted to the proteins
         detected + QC-surviving in BOTH branches. TWO complementary pieces:
           - Significance (run_genotype_interaction_test): a single limma
             linear model per protein, value ~ genotype*week (+peptide_ug),
             fit on the REAL un-expanded sample-level data (e.g. 10 WT + 10
             KI = 20 real mice at 8W). The genotype:week coefficient is the
             per-week rate of change of log2(KI/WT); its eBayes-moderated
             p-value/FDR is what's actually reported as significant. This
             is the methodologically correct approach (same limma pattern
             already used by step06_dep_limma/step09_trend_timeseries) —
             an EARLIER version of this analysis instead ran a
             Jonckheere-Terpstra test on an all-pairwise-expanded WTxKI
             ratio matrix (every WT_i(t)/KI_j(t) combination materialized
             as a pseudo-replicate), which double-counts JT's own internal
             pairwise-comparison structure and inflates significance —
             confirmed empirically (27/75 "significant" vs 0/75 once
             refit with real n). That approach is no longer used for
             p-values; kept only as a descriptive visualization layer, see
             below.
           - Visualization (build_ki_wt_ratio_matrix): every (WT_i(t),
             KI_j(t)) replicate pair -> KI_j(t) - WT_i(t) (log2 scale, so
             = log2(KI_j / WT_i)) is still materialized for the box/mean-SE/
             heatmap PLOTS, since WT and KI are unpaired mice with no
             natural 1:1 pairing and this Hodges-Lehmann-style all-pairs
             set is a legitimate way to DISPLAY the ratio's spread (its
             mean exactly equals mean(KI)-mean(WT) at each timepoint by
             linearity). It is not used for any p-value.

Each tsv/ stage 01-05 shows exactly which candidate proteins were carried
into / dropped at that step, so the final trend-analysis numbers are
traceable back to the raw 9,769-protein report.

Outputs (this project's branch/ tree):
  branch/output/tsv/01_EV_Candidate_Matching/ev_candidate_genes.tsv
  branch/output/tsv/02_Filtered/{WT,KI}_ev_subset_status.tsv
  branch/output/tsv/03_Log2_Transformed/{WT,KI}_ev_subset_log2.tsv
  branch/output/tsv/04_Normalized/{WT,KI}_ev_subset_normalized.tsv
  branch/output/tsv/05_Imputed/{WT,KI}_ev_subset_imputed.tsv
  branch/output/tsv/05_Imputed/KI_over_WT_ev_subset_ratio.tsv
  branch/output/figures/01_EV_Candidate_Matching/candidate_coverage_funnel.png
  branch/output/figures/02_Trend_Analysis/{WT,KI,KI_over_WT}/*.png
  branch/output/results/Mousebrain_elevated_EV_protein_list.tsv
  branch/output/results/trend_test_results_{WT,KI,KI_over_WT}.tsv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from adjustText import adjust_text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trends_common as tc

sys.path.insert(0, tc.SCRIPTS_DIR)
import config
import utils
import step02_filter as s2
import step03_log2_transform as s3
import step04_normalize as s4
import step05_impute as s5
from step09_trend_timeseries import _jonckheere_terpstra
from step06_dep_limma import _init_rpy2

EV_CLASS_ORDER = ["HC-EC", "LC-EC", "HC-VP", "LC-VP"]


# ══════════════════════════════════════════════════════════════════════════
# 1-2. Build the Mousebrain_elevated_EV_protein candidate -> accession list
# ══════════════════════════════════════════════════════════════════════════

def build_candidate_list(logger) -> pd.DataFrame:
    ev_db = pd.read_csv(tc.EV_DB_PATH, sep="\t", low_memory=False)
    ev_db = ev_db[ev_db["Exosome_Classification"].notna() &
                  (ev_db["Exosome_Classification"].astype(str).str.strip() != "")]
    ev_genes = dict(zip(ev_db["Gene"].astype(str).str.upper(), ev_db["Exosome_Classification"]))
    logger.info(f"[ev_trend] EV database: {len(ev_genes)} regionally-elevated genes "
                f"with a non-empty Exosome_Classification.")

    ann_path = os.path.join(config.DATASETS["MousePlasma"]["output_root"],
                            config.STEP_DIRS["01_raw_summary"], "mouse_brain_annotation.tsv")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"{ann_path} not found — run step01+step01b for MousePlasma first.")
    ann = pd.read_csv(ann_path, sep="\t")
    ann["matched_brain_gene"] = ann["matched_brain_gene"].fillna("").astype(str)

    rows = []
    for gene, classification in ev_genes.items():
        hits = ann[ann["matched_brain_gene"] == gene]
        if hits.empty:
            rows.append({"Gene": gene, "Exosome_Classification": classification,
                        config.ID_COL: None, "detected_raw": False})
        else:
            for _, h in hits.iterrows():
                rows.append({"Gene": gene, "Exosome_Classification": classification,
                            config.ID_COL: h[config.ID_COL], "detected_raw": True})

    cand = pd.DataFrame(rows)
    n_genes_detected = cand.loc[cand["detected_raw"], "Gene"].nunique()
    logger.info(f"[ev_trend] {n_genes_detected}/{len(ev_genes)} candidate genes detected in "
                f"the raw 9,769-protein report ({cand['detected_raw'].sum()} protein-group rows).")

    stage1_dir = tc.tsv_stage_dir(tc.STAGE_01_CANDIDATE)
    utils.save_tsv(cand, stage1_dir, "ev_candidate_genes")
    logger.info(f"[ev_trend] stage 1 audit saved: "
                f"{stage1_dir}/ev_candidate_genes.tsv ({len(cand)} rows).")
    return cand


# ══════════════════════════════════════════════════════════════════════════
# 3. Per-genotype branch preprocessing (reuses scripts/step02-05 as-is)
# ══════════════════════════════════════════════════════════════════════════

def _branch_initial_state(dataset: str, logger) -> dict:
    cfg = config.DATASETS[dataset]
    genotype = cfg["genotype"]
    df, meta = utils.load_raw(dataset)
    all_cols = meta["column_name"].tolist()
    df = utils.to_numeric_intensity(df, all_cols)
    sub_df, sub_meta, sample_cols, ref_col = utils.subset_by_genotype(df, meta, genotype)
    logger.info(f"[ev_trend] branch {dataset}: subset genotype={genotype}: "
                f"{sub_df.shape[0]} proteins x {len(sample_cols)} samples.")
    return {"data_df": sub_df, "meta": sub_meta, "sample_cols": sample_cols,
            "ref_col": ref_col, "all_sample_cols": sample_cols + ([ref_col] if ref_col else [])}


def preprocess_branch(genotype: str, logger) -> tuple[dict, dict]:
    """Run step02-05 and return (final_state, snapshots), where snapshots maps
    each stage name to the full data_df immediately after that step — the
    basis for the per-stage EV-subset audit TSVs written by
    save_stage_audit()."""
    dataset = f"MousePlasma_{genotype}"
    state = _branch_initial_state(dataset, logger)
    snapshots = {}
    stage_by_step = {
        s2: tc.STAGE_02_FILTERED,
        s3: tc.STAGE_03_LOG2,
        s4: tc.STAGE_04_NORMALIZED,
        s5: tc.STAGE_05_IMPUTED,
    }
    for step in (s2, s3, s4, s5):
        state = step.run(dataset, logger, state)
        snapshots[stage_by_step[step]] = state["data_df"].copy()
    return state, snapshots


def save_stage_audit(snapshots: dict, cand_detected: pd.DataFrame, genotype: str, logger) -> None:
    """Write the EV-candidate-subset audit TSV for each of stages 02-05: every
    raw-detected candidate accession, whether it is present in that stage's
    data_df (survived filtering) or not, plus that stage's sample values when
    present."""
    cand_ids = cand_detected[config.ID_COL].drop_duplicates()
    gene_map = cand_detected.drop_duplicates(subset=[config.ID_COL]).set_index(
        config.ID_COL)[["Gene", "Exosome_Classification"]]

    for stage, snap in snapshots.items():
        present = snap[snap[config.ID_COL].isin(cand_ids)].copy()
        present = present.merge(gene_map, on=config.ID_COL, how="left")
        missing_ids = sorted(set(cand_ids) - set(present[config.ID_COL]))
        if missing_ids:
            missing = pd.DataFrame({config.ID_COL: missing_ids})
            missing = missing.merge(gene_map, on=config.ID_COL, how="left")
            missing["dropped_at_or_before_this_stage"] = True
            present["dropped_at_or_before_this_stage"] = False
            audit = pd.concat([present, missing], ignore_index=True, sort=False)
        else:
            present["dropped_at_or_before_this_stage"] = False
            audit = present
        stage_dir = tc.tsv_stage_dir(stage)
        utils.save_tsv(audit, stage_dir, f"{genotype}_ev_subset")
        n_present = len(present)
        logger.info(f"[ev_trend] {genotype} stage audit [{stage}]: "
                    f"{n_present}/{len(cand_ids)} EV candidates present "
                    f"-> {stage_dir}/{genotype}_ev_subset.tsv")


def plot_candidate_coverage_funnel(n_total, n_raw, n_wt, n_ki, logger):
    """Stage 01 audit figure: how the 148 EV-database candidate genes narrow
    down to raw-report detection and per-branch QC survival."""
    labels = ["EV database\n(148 genes)", "Detected in\nraw report",
              "Survive WT\nQC filter", "Survive KI\nQC filter"]
    values = [n_total, n_raw, n_wt, n_ki]
    colors = ["#7F8C8D", "#2ecc71", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.4, width=0.6)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v}", ha="center", va="bottom",
               fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of genes")
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_title("Mousebrain-elevated EV candidate coverage funnel", fontsize=13,
                fontweight="bold", pad=12)
    fig.tight_layout()
    stage_dir = tc.figures_stage_dir(tc.STAGE_01_CANDIDATE)
    p = os.path.join(stage_dir, "candidate_coverage_funnel.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[ev_trend] saved stage 1 figure: {p}")


# ══════════════════════════════════════════════════════════════════════════
# 4. Trend test (Jonckheere-Terpstra + Spearman) across ordered timepoints
# ══════════════════════════════════════════════════════════════════════════

def _stage_group_means(matrix: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """Per-protein group mean at each ordered timepoint + a simple monotonic-
    ity flag. Used both by run_trend_test (WT/KI branches) and by the
    genotype-interaction analysis (descriptive-only there — see its own
    significance test)."""
    sample_cols = bio["column_name"].tolist()
    stage_vec = bio.set_index("column_name").loc[sample_cols, "timepoint"].astype(int).values
    vals = matrix[sample_cols].values.astype(float)
    gmeans = {t: vals[:, stage_vec == t].mean(axis=1) for t in tc.STAGE_ORDER}
    gmean_mat = np.column_stack([gmeans[t] for t in tc.STAGE_ORDER])
    diffs = np.diff(gmean_mat, axis=1)
    monotonic = np.all(diffs >= 0, axis=1) | np.all(diffs <= 0, axis=1)
    out = pd.DataFrame({f"Mean_{tc.STAGE_LABELS[t]}": gmeans[t] for t in tc.STAGE_ORDER})
    out["Monotonic"] = monotonic
    return out


def run_trend_test(matrix: pd.DataFrame, bio: pd.DataFrame, logger) -> pd.DataFrame:
    """Jonckheere-Terpstra + Spearman trend test. Valid when `matrix`'s
    sample columns are REAL, independent biological replicates (as they are
    for the WT-only and KI-only branches) — do NOT feed this a pairwise-
    expanded ratio matrix (see run_genotype_interaction_test for that case;
    JT/Spearman's variance formulas assume independent observations, which
    pairwise-expanded pseudo-replicates violate)."""
    sample_cols = bio["column_name"].tolist()
    stage_vec = bio.set_index("column_name").loc[sample_cols, "timepoint"].astype(int).values
    vals = matrix[sample_cols].values.astype(float)

    group_idx = [np.where(stage_vec == t)[0] for t in tc.STAGE_ORDER]
    jt_z, jt_p = _jonckheere_terpstra(vals, group_idx)

    sp_rho = np.full(vals.shape[0], np.nan)
    sp_p = np.full(vals.shape[0], np.nan)
    for i in range(vals.shape[0]):
        r = spearmanr(vals[i], stage_vec)
        sp_rho[i], sp_p[i] = r.statistic, r.pvalue

    out = matrix[[config.ID_COL, "Gene", "Exosome_Classification", "label"]].copy()
    out["JT_z"] = jt_z
    out["JT_p_value"] = jt_p
    out["Spearman_rho"] = sp_rho
    out["Spearman_p"] = sp_p
    out = pd.concat([out, _stage_group_means(matrix, bio)], axis=1)
    out["JT_FDR"] = multipletests(out["JT_p_value"].values, method="fdr_bh")[1]
    out["Spearman_FDR"] = multipletests(out["Spearman_p"].values, method="fdr_bh")[1]
    out["Direction"] = np.where(out["Spearman_rho"] > 0, "Increasing",
                                np.where(out["Spearman_rho"] < 0, "Decreasing", "Flat"))
    out["JT_sig"] = out["JT_FDR"].map(tc.sig_star)
    out["Spearman_sig"] = out["Spearman_FDR"].map(tc.sig_star)
    out = out.sort_values("JT_p_value").reset_index(drop=True)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 5. Figures (palette / layout mirrored from the reference project)
# ══════════════════════════════════════════════════════════════════════════

def plot_volcano(trend_df: pd.DataFrame, genotype: str, out_dir: str, logger, *,
                 effect_col: str = "Spearman_rho", fdr_col: str = "JT_FDR",
                 x_label: str = "Spearman rho (timepoint correlation)",
                 method_label: str = "Jonckheere-Terpstra test across 8W → 10W → 12W → 14W"):
    df = trend_df.copy()
    df["neg_log10_fdr"] = -np.log10(np.clip(df[fdr_col], 1e-300, None))
    df["sig"] = df[fdr_col] < tc.FDR_THRESHOLD

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(df["sig"], "#e74c3c", "#a6a6a6")
    ax.scatter(df[effect_col], df["neg_log10_fdr"], c=colors, s=45, alpha=0.85,
              edgecolor="none")
    ax.axhline(-np.log10(tc.FDR_THRESHOLD), ls="--", color="#808080", lw=0.8)
    ax.axvline(0, color="#b3b3b3", lw=0.6)

    if df["sig"].sum() > 0:
        label_df = df[df["sig"]]
    else:
        label_df = df.head(10)
    texts = [ax.text(r[effect_col], r["neg_log10_fdr"], r["label"], fontsize=8)
            for _, r in label_df.iterrows()]
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=7,
                      label=l) for c, l in [("#e74c3c", "FDR < 0.05"), ("#a6a6a6", "FDR >= 0.05")]]
    ax.legend(handles=handles, title="Significance", loc="lower left", frameon=False, fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$-\log_{10}$(FDR)")
    ax.set_title(f"Monotonic Trend Analysis — Mousebrain-elevated EV proteins ({genotype})\n"
                f"{method_label}",
                fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "trend_volcano.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[ev_trend] {genotype}: saved trend_volcano.png")


def _heatmap_col_fun_r(ro):
    ro.r.assign("hm_stops", ro.FloatVector(tc.HEATMAP_STOPS))
    ro.r.assign("hm_colors", ro.StrVector(tc.HEATMAP_COLORS))
    ro.r("hm_col_fun <- circlize::colorRamp2(hm_stops, hm_colors)")


def plot_heatmap(trend_df: pd.DataFrame, genotype: str, out_dir: str, logger, top_n=30, *,
                 effect_col: str = "Spearman_rho", fdr_col: str = "JT_FDR",
                 sort_label: str = "JT p-value"):
    sel = trend_df.head(min(top_n, len(trend_df))).copy()
    mean_cols = [f"Mean_{tc.STAGE_LABELS[t]}" for t in tc.STAGE_ORDER]
    gmean_mat = sel[mean_cols].values.astype(float)
    z = (gmean_mat - gmean_mat.mean(axis=1, keepdims=True)) / \
        (gmean_mat.std(axis=1, keepdims=True, ddof=0) + 1e-9)

    sel = sel.assign(_z_order=sel[effect_col])
    order = np.argsort(sel["_z_order"].values)
    z = z[order]
    sel = sel.iloc[order]

    row_labels = sel["label"].tolist()
    direction = sel["Direction"].tolist()
    fdr_star = sel[fdr_col].map(tc.sig_star).tolist()
    col_labels = [tc.STAGE_LABELS[t] for t in tc.STAGE_ORDER]

    n = len(sel)
    if n < 2:
        logger.warning(f"[ev_trend] {genotype}: too few proteins ({n}) for a heatmap; skipping.")
        return
    cell_h = max(8.0, min(18.0, 400.0 / n))
    row_fs = max(5.0, min(10.0, 300.0 / n))
    # +4.2in fixed margin (not +3.0) — the continuous z-score color bar in the
    # merged bottom legend row is taller than the discrete Trend-FDR/Direction
    # legends next to it, and the previous margin clipped it (its "-2" tick
    # and the bottom of the gradient fell outside the PNG canvas).
    dev_w, dev_h = 8.5, max(7.2, n * cell_h / 72.0 + 4.2)

    png_path = os.path.join(out_dir, "trend_heatmap.png")
    ro, _ = _init_rpy2()
    _heatmap_col_fun_r(ro)
    ro.r.assign("z_vec", ro.FloatVector(z.flatten("F")))
    ro.r.assign("n_row", n)
    ro.r.assign("n_col", z.shape[1])
    ro.r.assign("row_labels", ro.StrVector(row_labels))
    ro.r.assign("col_labels", ro.StrVector(col_labels))
    ro.r.assign("direction", ro.StrVector(direction))
    ro.r.assign("fdr_star", ro.StrVector(fdr_star))
    ro.r.assign("dir_cols", ro.StrVector(list(tc.DIRECTION_COLORS.values())))
    ro.r.assign("dir_names", ro.StrVector(list(tc.DIRECTION_COLORS.keys())))
    ro.r.assign("fdr_cols", ro.StrVector(list(tc.FDR_STAR_COLORS.values())))
    ro.r.assign("fdr_names", ro.StrVector(list(tc.FDR_STAR_COLORS.keys())))
    ro.r.assign("cell_h", float(cell_h))
    ro.r.assign("row_fs", float(row_fs))
    ro.r.assign("dev_w", float(dev_w))
    ro.r.assign("dev_h", float(dev_h))
    ro.r.assign("png_path", png_path)
    ro.r.assign("title_txt", f"Mousebrain-elevated EV proteins ({genotype}) — top {n} by {sort_label}")
    ro.r('''
    suppressMessages({library(ComplexHeatmap); library(circlize); library(grid)})
    m <- matrix(z_vec, nrow = n_row, ncol = n_col)
    names(dir_cols) <- dir_names
    names(fdr_cols) <- fdr_names
    # Factor with ALL defined FDR-star levels (not just the ones present among
    # this heatmap's rows), so the legend always shows what color each of
    # '*', '**', '***' (and 'ns') maps to — even when a level has zero rows.
    fdr_star_f <- factor(fdr_star, levels = fdr_names)
    ha <- rowAnnotation(
      `Trend FDR` = fdr_star_f,
      Direction   = direction,
      col = list(`Trend FDR` = fdr_cols, Direction = dir_cols),
      annotation_width = unit(c(5, 5), "mm"),
      annotation_name_gp = gpar(fontsize = 9),
      annotation_name_side = "top",
      gap = unit(1, "mm")
    )
    ht <- Heatmap(
      m, name = "z-score", col = hm_col_fun,
      cluster_rows = FALSE, cluster_columns = FALSE,
      row_labels = row_labels, column_labels = col_labels,
      row_names_side = "right", row_names_gp = gpar(fontsize = row_fs),
      column_names_gp = gpar(fontsize = 11), column_names_rot = 45,
      left_annotation = ha,
      border = "grey90", rect_gp = gpar(col = "grey90"),
      width  = unit(15 * n_col, "mm"),
      height = unit(cell_h * n_row, "pt"),
      heatmap_legend_param = list(title = "z-score", at = c(-2, -1, 0, 1, 2),
                                  border = "grey50", legend_height = unit(3, "cm"))
    )
    png(png_path, width = dev_w, height = dev_h, units = "in", res = 300)
    draw(ht, heatmap_legend_side = "bottom", annotation_legend_side = "bottom",
         merge_legend = TRUE, padding = unit(c(5, 10, 40, 5), "mm"))
    grid.text(title_txt, x = unit(5, "mm"), y = unit(1, "npc") - unit(6, "mm"),
              just = "left", gp = gpar(fontsize = 13, fontface = "bold"))
    dev.off()
    ''')
    logger.info(f"[ev_trend] {genotype}: saved trend_heatmap.png (n={n})")


def _finalize_grid_header(fig, fig_h_in: float, title_line: str, subtitle_line: str):
    """Reserve a FIXED absolute header height (in inches) for a 2-line panel
    title, independent of the grid's row count. A fixed *fraction* (e.g.
    rect top=0.93) reserves proportionally less absolute space for short
    (few-row) figures than for tall ones, which is what caused the title to
    collide with the first row of subplots when only 1-6 panels were drawn
    (small figure height) — see trend_boxplots_significant.png overlap."""
    header_in = 0.95
    top_frac = max(0.5, 1.0 - header_in / fig_h_in)
    fig.tight_layout(rect=[0, 0, 1, top_frac])
    fig.text(0.5, 1.0 - 0.32 / fig_h_in, title_line, ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 1.0 - 0.62 / fig_h_in, subtitle_line, ha="center", va="top",
             fontsize=9.5, color="#666666")


def _select_box_ids(trend_df: pd.DataFrame, logger, genotype: str, *,
                    fdr_col: str = "JT_FDR", sort_label: str = "JT p-value"):
    """trend_df must already be sorted ascending by the p-value backing
    `sort_label` (run_trend_test / run_genotype_interaction_test both do)."""
    sig = trend_df[trend_df[fdr_col] < tc.FDR_THRESHOLD]
    if len(sig) > 0:
        return sig[config.ID_COL].tolist(), False
    logger.info(f"[ev_trend] {genotype}: no protein with {fdr_col}<{tc.FDR_THRESHOLD}; "
               f"plotting top 6 by {sort_label} instead.")
    return trend_df.head(6)[config.ID_COL].tolist(), True


def _make_box_jitter_ax(ax, values_by_stage: dict, gene_lab: str, subtitle: str,
                        value_label: str = "Normalized log2 ratio"):
    positions = list(range(1, len(tc.STAGE_ORDER) + 1))
    data = [values_by_stage[t] for t in tc.STAGE_ORDER]
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False)
    for patch, t in zip(bp["boxes"], tc.STAGE_ORDER):
        patch.set_facecolor(tc.STAGE_COLORS[t]); patch.set_alpha(0.6)
        patch.set_edgecolor("#666666"); patch.set_linewidth(0.8)
    for med in bp["medians"]:
        med.set_color("#4D4D4D"); med.set_linewidth(1.0)

    rng = np.random.default_rng(42)
    means = []
    for pos, t in zip(positions, tc.STAGE_ORDER):
        vals = values_by_stage[t]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos + jitter, vals, color=tc.STAGE_COLORS[t], s=18, alpha=0.7,
                  edgecolor="none", zorder=3)
        means.append(np.mean(vals))
    ax.plot(positions, means, color="black", lw=1.2, ls="--", zorder=4)
    ax.scatter(positions, means, color="black", marker="D", s=45, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([tc.STAGE_LABELS[t] for t in tc.STAGE_ORDER], fontsize=9)
    ax.set_ylabel(value_label, fontsize=9)
    # `pad` (points) reserves headroom above the axes for the title; the
    # subtitle is offset from the axes edge in points too (not axes-fraction),
    # so the two never collide regardless of the figure's dpi or panel count —
    # axes-fraction y=1.0 for both, as before, put them directly on top of
    # each other.
    ax.set_title(gene_lab, fontsize=11, fontweight="bold", pad=20)
    ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
               xytext=(0, 3), textcoords="offset points",
               ha="center", va="bottom", fontsize=7.5, color="#666666")


def plot_boxplots(matrix, bio, trend_df, genotype, out_dir, logger, ids, name, title,
                  value_label: str = "Normalized log2 ratio", *,
                  fdr_col: str = "JT_FDR", sig_col: str = "JT_sig",
                  fdr_label: str = "JT FDR", effect_col: str = "Spearman_rho",
                  effect_label: str = "rho"):
    if not ids:
        return
    sample_cols = bio["column_name"].tolist()
    stage_vec = bio.set_index("column_name").loc[sample_cols, "timepoint"].astype(int)

    n = len(ids)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig_h = 4.2 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, fig_h), squeeze=False)

    row_by_id = trend_df.set_index(config.ID_COL)
    for i, pid in enumerate(ids):
        ax = axes[i // ncol][i % ncol]
        row = matrix.loc[matrix[config.ID_COL] == pid].iloc[0]
        r = row_by_id.loc[pid]
        values_by_stage = {t: row[sample_cols][stage_vec == t].values.astype(float)
                           for t in tc.STAGE_ORDER}
        subtitle = (f"{fdr_label}={r[fdr_col]:.4f}{r[sig_col]} | {effect_label}={r[effect_col]:.3f} | "
                   f"{r['Direction']}")
        _make_box_jitter_ax(ax, values_by_stage, r["label"], subtitle, value_label=value_label)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    _finalize_grid_header(
        fig, fig_h, title,
        "8W → 10W → 12W → 14W  |  Diamond = group mean  |  Dashed = mean trend line")
    p = os.path.join(out_dir, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[ev_trend] {genotype}: saved {name}.png ({n} proteins)")


def plot_mean_se_lines(matrix, bio, trend_df, genotype, out_dir, logger, ids,
                       value_label: str = "Mean normalized log2 ratio", *,
                       effect_col: str = "Spearman_rho", effect_label: str = "rho",
                       subtitle_line: str | None = None):
    if not ids:
        return
    sample_cols = bio["column_name"].tolist()
    stage_vec = bio.set_index("column_name").loc[sample_cols, "timepoint"].astype(int)

    n = len(ids)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig_h = 4.2 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, fig_h), squeeze=False)

    row_by_id = trend_df.set_index(config.ID_COL)
    x = list(range(1, len(tc.STAGE_ORDER) + 1))
    for i, pid in enumerate(ids):
        ax = axes[i // ncol][i % ncol]
        row = matrix.loc[matrix[config.ID_COL] == pid].iloc[0]
        r = row_by_id.loc[pid]
        means, ses = [], []
        for t in tc.STAGE_ORDER:
            vals = row[sample_cols][stage_vec == t].values.astype(float)
            means.append(vals.mean())
            ses.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        means, ses = np.array(means), np.array(ses)

        ax.fill_between(x, means - ses, means + ses, color="#CCCCCC", alpha=0.5)
        ax.plot(x, means, color="#2c3e50", lw=1.4)
        ax.errorbar(x, means, yerr=ses, fmt="none", ecolor="#666666", elinewidth=0.8, capsize=3)
        for xi, t in zip(x, tc.STAGE_ORDER):
            ax.scatter(xi, means[x.index(xi)], color=tc.STAGE_COLORS[t], s=60, zorder=5,
                      edgecolor="black", linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels([tc.STAGE_LABELS[t] for t in tc.STAGE_ORDER], fontsize=9)
        ax.set_xlabel("Timepoint", fontsize=9)
        ax.set_ylabel(value_label, fontsize=9)
        ax.set_title(r["label"], fontsize=11, fontweight="bold", pad=20)
        ax.annotate(f"{effect_label}={r[effect_col]:.3f} | {r['Direction']}",
                   xy=(0.5, 1.0), xycoords="axes fraction",
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=7.5, color="#666666")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    _finalize_grid_header(
        fig, fig_h, f"Mean ± SE Expression Across Timepoints ({genotype})",
        subtitle_line if subtitle_line is not None else f"Trend EV proteins (JT FDR < {tc.FDR_THRESHOLD})")
    p = os.path.join(out_dir, "trend_mean_se_lines.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[ev_trend] {genotype}: saved trend_mean_se_lines.png ({n} proteins)")


# ══════════════════════════════════════════════════════════════════════════
# 7. KI/WT ratio trend analysis (genotype-contrast, same JT+Spearman method)
# ══════════════════════════════════════════════════════════════════════════

def build_ki_wt_ratio_matrix(wt_matrix, wt_bio, ki_matrix, ki_bio, logger):
    """Convert signal intensity into a KI/WT ratio via ALL PAIRWISE replicate
    combinations within each timepoint: for every protein and timepoint t,
    every (WT_i(t), KI_j(t)) replicate pair yields one ratio value
    KI_j(t) - WT_i(t) (log2 scale, so = log2(KI_j / WT_i)) — e.g. 8W WT (10
    replicates) x 8W KI (10 replicates) = 100 pairwise ratio values at 8W.
    This is the Hodges-Lehmann-style "all pairwise differences" construction
    (same pairing the Mann-Whitney U / Jonckheere-Terpstra test itself
    computes internally), and it's used here because WT and KI are
    unpaired/independent mice — there is no natural 1:1 replicate pairing —
    so every WT/KI combination at a timepoint is an equally valid ratio.

    Caveat (documented, not hidden): pairwise combinations are NOT
    independent observations (each real replicate contributes to many
    pairs), so the resulting sample size per timepoint overstates the true
    n. This inflates apparent test power somewhat; the reported n. per
    timepoint should always be read as "instrument for computing an
    all-pairs JT/Spearman location statistic," not as instrument
    counts of independent biological replicates.

    Only proteins detected + QC-surviving in BOTH branches can form a ratio.
    Returns (ratio_matrix, ratio_bio) — ratio_bio describes the synthetic
    pairwise "sample" columns (column_name, timepoint) the same way a real
    bio table would, so run_trend_test/plotting need no special-casing.

    Audit trail (validity/traceability of this construction step) is written
    to branch/output/tsv/05_Imputed/KI_over_WT/:
      candidate_status.tsv           — every WT-QC or KI-QC candidate protein,
        whether it made it into the ratio (needs BOTH), and why not if not.
      pairwise_design.tsv            — the literal (WT sample, KI sample) ->
        synthetic ratio-column recipe for all 404 pairs, so any single ratio
        value can be traced back to the exact two real replicates that made it.
      replication_validity_summary.tsv — per timepoint: real WT/KI replicate
        counts vs. the nominal pairwise count actually fed to the JT/Spearman
        test, and the resulting pseudo-replication inflation factor — the
        number a reviewer needs to judge how much this method's nominal n
        overstates true biological replication (documented, not hidden, in
        the function docstring above; this file makes it a checkable number).
    """
    wt_ids, ki_ids = set(wt_matrix[config.ID_COL]), set(ki_matrix[config.ID_COL])
    common_ids = wt_ids & ki_ids
    ki_sub = ki_matrix[ki_matrix[config.ID_COL].isin(common_ids)].reset_index(drop=True)
    wt_indexed = wt_matrix.set_index(config.ID_COL)
    ki_indexed = ki_sub.set_index(config.ID_COL)
    ids_order = ki_sub[config.ID_COL].tolist()

    audit_dir = tc.tsv_stage_dir(os.path.join(tc.STAGE_05_IMPUTED, "KI_over_WT"))

    # ── Candidate status / validity table ────────────────────────────────
    meta_cols = [config.ID_COL, "Gene", "Exosome_Classification", "label"]
    combined_meta = pd.concat([wt_matrix[meta_cols], ki_matrix[meta_cols]]).drop_duplicates(
        subset=[config.ID_COL]).reset_index(drop=True)
    combined_meta["detected_WT_QC"] = combined_meta[config.ID_COL].isin(wt_ids)
    combined_meta["detected_KI_QC"] = combined_meta[config.ID_COL].isin(ki_ids)
    combined_meta["included_in_ratio"] = (combined_meta["detected_WT_QC"] &
                                          combined_meta["detected_KI_QC"])
    combined_meta["exclusion_reason"] = np.select(
        [combined_meta["included_in_ratio"],
         combined_meta["detected_WT_QC"] & ~combined_meta["detected_KI_QC"],
         ~combined_meta["detected_WT_QC"] & combined_meta["detected_KI_QC"]],
        ["", "WT_QC_only (no matching KI_QC measurement)", "KI_QC_only (no matching WT_QC measurement)"],
        default="")
    utils.save_tsv(combined_meta, audit_dir, "candidate_status")

    # ── Pairwise design + per-timepoint replication validity ────────────────
    id_cols = ki_sub[meta_cols].reset_index(drop=True)
    bio_rows, design_rows, validity_rows = [], [], []
    ratio_cols = {}
    for t in tc.STAGE_ORDER:
        wt_cols_t = wt_bio.loc[wt_bio["timepoint"].astype(int) == t, "column_name"].tolist()
        ki_cols_t = ki_bio.loc[ki_bio["timepoint"].astype(int) == t, "column_name"].tolist()
        wt_vals_t = wt_indexed.loc[ids_order, wt_cols_t].values   # n_proteins x n_wt_t
        ki_vals_t = ki_indexed.loc[ids_order, ki_cols_t].values   # n_proteins x n_ki_t
        for wi, wt_c in enumerate(wt_cols_t):
            for kj, ki_c in enumerate(ki_cols_t):
                col_name = f"ratio_{t}W_WT{wi + 1}_KI{kj + 1}"
                ratio_cols[col_name] = ki_vals_t[:, kj] - wt_vals_t[:, wi]
                bio_rows.append({"column_name": col_name, "timepoint": t})
                design_rows.append({"ratio_column": col_name, "timepoint": tc.STAGE_LABELS[t],
                                   "wt_sample_column": wt_c, "ki_sample_column": ki_c})
        n_wt, n_ki = len(wt_cols_t), len(ki_cols_t)
        n_pairwise = n_wt * n_ki
        n_independent = n_wt + n_ki
        validity_rows.append({
            "timepoint": tc.STAGE_LABELS[t], "n_WT_replicates": n_wt, "n_KI_replicates": n_ki,
            "n_pairwise_ratio_values": n_pairwise, "n_independent_replicates": n_independent,
            "pseudoreplication_inflation_factor": round(n_pairwise / n_independent, 2),
        })

    utils.save_tsv(pd.DataFrame(design_rows), audit_dir, "pairwise_design")
    utils.save_tsv(pd.DataFrame(validity_rows), audit_dir, "replication_validity_summary")

    ratio_df = pd.concat([id_cols, pd.DataFrame(ratio_cols)], axis=1)
    ratio_bio = pd.DataFrame(bio_rows)
    n_pairs_by_stage = {r["timepoint"]: r["n_pairwise_ratio_values"] for r in validity_rows}
    logger.info(f"[ev_trend] KI/WT ratio: {len(common_ids)}/{len(wt_ids | ki_ids)} candidate "
               f"proteins detected + QC-surviving in BOTH branches "
               f"({len(wt_ids - ki_ids)} WT-only + {len(ki_ids - wt_ids)} KI-only excluded); "
               f"all-pairwise WTxKI ratio per timepoint: "
               + ", ".join(f"{lbl}={n}" for lbl, n in n_pairs_by_stage.items())
               + f" ({len(ratio_bio)} total pairwise ratio values per protein).")
    logger.info(f"[ev_trend] KI/WT ratio audit trail -> {audit_dir}/ "
               f"(candidate_status.tsv, pairwise_design.tsv, replication_validity_summary.tsv)")
    return ratio_df, ratio_bio


def run_genotype_interaction_test(wt_matrix, wt_bio, ki_matrix, ki_bio, common_ids, logger):
    """The methodologically correct way to test "does the KI-vs-WT
    difference itself change over time": fit ONE limma linear model per
    protein on the REAL, un-expanded sample-level data —
        value ~ genotype * week (+ peptide_ug)
    — and test the genotype:week INTERACTION coefficient. This uses the
    true n (e.g. 10 WT + 10 KI = 20 independent mice at 8W, not the 100
    pairwise pseudo-replicates build_ki_wt_ratio_matrix constructs for
    visualization), so eBayes/topTable's p-values are not anti-conservative
    the way running JT/Spearman on the pairwise-expanded ratio matrix would
    be (JT's own variance formula already assumes independent pairs; feeding
    it pre-expanded pairwise differences double-counts that structure and
    inflates apparent significance — confirmed empirically: the JT-on-
    pairwise-ratio version flagged 27/75 proteins vs ~1/78 and 1/76 in the
    WT-only/KI-only branches, an implausible jump).

    Same limma machinery as step06_dep_limma / step09_trend_timeseries
    (this project's own established DEP/trend-testing pattern), just with a
    genotype*week design instead of ~0+group or ~week.
    """
    meta_cols = [config.ID_COL, "Gene", "Exosome_Classification", "label"]
    id_meta = wt_matrix[wt_matrix[config.ID_COL].isin(common_ids)][meta_cols].drop_duplicates(
        subset=[config.ID_COL]).reset_index(drop=True)
    ids_order = id_meta[config.ID_COL].tolist()

    wt_sample_cols = wt_bio["column_name"].tolist()
    ki_sample_cols = ki_bio["column_name"].tolist()
    wt_vals = wt_matrix.set_index(config.ID_COL).loc[ids_order, wt_sample_cols].values.astype(float)
    ki_vals = ki_matrix.set_index(config.ID_COL).loc[ids_order, ki_sample_cols].values.astype(float)
    expr = np.hstack([wt_vals, ki_vals])   # n_proteins x (n_wt_total + n_ki_total), REAL replicates only

    genotype_vec = ["WT"] * len(wt_sample_cols) + ["KI"] * len(ki_sample_cols)
    week_vec = (wt_bio.set_index("column_name").loc[wt_sample_cols, "timepoint"].astype(float).tolist() +
               ki_bio.set_index("column_name").loc[ki_sample_cols, "timepoint"].astype(float).tolist())
    pep_wt = pd.to_numeric(wt_bio.set_index("column_name").loc[wt_sample_cols, config.PEPTIDE_COL],
                          errors="coerce")
    pep_ki = pd.to_numeric(ki_bio.set_index("column_name").loc[ki_sample_cols, config.PEPTIDE_COL],
                          errors="coerce")
    pep_vec = pd.concat([pep_wt, pep_ki]).values.astype(float)

    ro, _ = _init_rpy2()
    ro.r.assign("expr_mat", ro.r["matrix"](ro.FloatVector(expr.flatten("F")), nrow=expr.shape[0]))
    ro.r.assign("genotype", ro.FactorVector(genotype_vec, levels=ro.StrVector(["WT", "KI"])))
    ro.r.assign("week", ro.FloatVector(week_vec))
    use_pep = config.USE_PEPTIDE_COVARIATE and np.isfinite(pep_vec).all()
    if use_pep:
        ro.r.assign("pep", ro.FloatVector(pep_vec))
        ro.r("design <- model.matrix(~ genotype * week + pep)")
    else:
        ro.r("design <- model.matrix(~ genotype * week)")
    ro.r("""
    fit <- eBayes(lmFit(expr_mat, design), trend=TRUE, robust=TRUE)
    tt  <- topTable(fit, coef="genotypeKI:week", number=Inf, sort.by="none", adjust.method="BH")
    """)
    tt = ro.r["tt"]
    if isinstance(tt, pd.DataFrame):
        stats = tt[["logFC", "t", "P.Value", "adj.P.Val"]].reset_index(drop=True)
    else:
        stats = pd.DataFrame({k: np.asarray(tt.rx2(k), dtype=float) for k in
                              ["logFC", "t", "P.Value", "adj.P.Val"]})
    stats.columns = ["Interaction_slope", "Interaction_t", "Interaction_P", "Interaction_FDR"]

    result = pd.concat([id_meta, stats], axis=1)
    result["Interaction_sig"] = result["Interaction_FDR"].map(tc.sig_star)
    result["Direction"] = np.where(result["Interaction_slope"] > 0, "Increasing",
                                   np.where(result["Interaction_slope"] < 0, "Decreasing", "Flat"))
    result = result.sort_values("Interaction_P").reset_index(drop=True)

    n_sig = int((result["Interaction_FDR"] < tc.FDR_THRESHOLD).sum())
    logger.info(f"[ev_trend] genotype x week interaction (limma; peptide covariate={use_pep}; "
               f"real n={expr.shape[1]} samples [{len(wt_sample_cols)} WT + {len(ki_sample_cols)} KI], "
               f"NO pairwise expansion): {len(result)} proteins tested; "
               f"{n_sig} significant (interaction FDR<{tc.FDR_THRESHOLD}).")
    return result


def run_ki_wt_ratio_analysis(wt_matrix, wt_bio, ki_matrix, ki_bio, logger):
    label = "KI_over_WT"
    display = "KI/WT ratio"
    ratio_matrix, ratio_bio = build_ki_wt_ratio_matrix(wt_matrix, wt_bio, ki_matrix, ki_bio, logger)

    stage5_dir = tc.tsv_stage_dir(tc.STAGE_05_IMPUTED)
    utils.save_tsv(ratio_matrix, stage5_dir, f"{label}_ev_subset_ratio")

    if len(ratio_matrix) < 4:
        logger.warning(f"[ev_trend] {label}: only {len(ratio_matrix)} proteins have both WT and "
                      f"KI QC-surviving data — too few for trend testing; skipping.")
        return pd.DataFrame()

    common_ids = ratio_matrix[config.ID_COL].tolist()
    interaction_df = run_genotype_interaction_test(wt_matrix, wt_bio, ki_matrix, ki_bio,
                                                    common_ids, logger)

    # Descriptive per-timepoint group means of the pairwise ratio distribution
    # (visualization only — NOT the significance test; see run_trend_test's
    # docstring on why JT/Spearman must not run on this pairwise-expanded
    # matrix). By linearity these means equal mean(KI)-mean(WT) per timepoint
    # regardless of the pairwise expansion, so they remain a faithful summary.
    stage_means = _stage_group_means(ratio_matrix, ratio_bio)
    trend_df = pd.concat([interaction_df, stage_means], axis=1)
    utils.save_tsv(trend_df, tc.RESULTS_DIR, f"trend_test_results_{label}")

    fig_dir = tc.figures_stage_dir(tc.STAGE_02_TREND, label)
    value_label = "log2(KI/WT) ratio"
    stat_kwargs = dict(effect_col="Interaction_slope", fdr_col="Interaction_FDR")
    plot_volcano(trend_df, display, fig_dir, logger, **stat_kwargs,
                x_label="Interaction slope (Δlog2(KI/WT) per week, limma)",
                method_label="genotype × week interaction (limma, real n — no pairwise expansion)")
    plot_heatmap(trend_df, display, fig_dir, logger, **stat_kwargs,
                sort_label="interaction P-value")

    sig_ids, is_fallback = _select_box_ids(trend_df, logger, display, fdr_col="Interaction_FDR",
                                           sort_label="interaction P-value")
    box_title = (f"Timepoint Trend — {'Significant' if not is_fallback else 'Top'} "
               f"EV Proteins ({display})")
    box_kwargs = dict(fdr_col="Interaction_FDR", sig_col="Interaction_sig", fdr_label="Interaction FDR",
                      effect_col="Interaction_slope", effect_label="slope/week")
    plot_boxplots(ratio_matrix, ratio_bio, trend_df, display, fig_dir, logger, sig_ids,
                 "trend_boxplots_significant", box_title, value_label=value_label, **box_kwargs)

    top_ids = trend_df.head(20)[config.ID_COL].tolist()
    plot_boxplots(ratio_matrix, ratio_bio, trend_df, display, fig_dir, logger, top_ids,
                 "trend_boxplots_top",
                 f"Timepoint Trend — Top {len(top_ids)} EV Proteins by Interaction P-value ({display})",
                 value_label=value_label, **box_kwargs)

    plot_mean_se_lines(ratio_matrix, ratio_bio, trend_df, display, fig_dir, logger, sig_ids,
                       value_label=f"Mean {value_label}", effect_col="Interaction_slope",
                       effect_label="slope/week",
                       subtitle_line=f"Trend EV proteins (genotype × week interaction FDR < {tc.FDR_THRESHOLD})")

    trend_df.insert(0, "Genotype", label)
    return trend_df


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def run_genotype(genotype: str, candidates: pd.DataFrame, logger):
    logger.info(f"\n{'=' * 70}\n[ev_trend] Branch: {genotype}\n{'=' * 70}")
    utils.apply_style()

    state, snapshots = preprocess_branch(genotype, logger)
    dataset = f"MousePlasma_{genotype}"
    cfg = config.DATASETS[dataset]
    full_matrix = state["data_df"]
    meta = state["meta"]
    bio = meta[(~meta["is_ref"]) & (meta["group"].isin(cfg["group_order"]))].reset_index(drop=True)

    cand_detected = candidates[candidates["detected_raw"]].copy()
    save_stage_audit(snapshots, cand_detected, genotype, logger)

    qc_ids = set(full_matrix[config.ID_COL])
    cand_qc = cand_detected[cand_detected[config.ID_COL].isin(qc_ids)].drop_duplicates(
        subset=[config.ID_COL])
    logger.info(f"[ev_trend] {genotype}: {cand_qc['Gene'].nunique()}/"
               f"{cand_detected['Gene'].nunique()} raw-detected EV genes survive QC "
               f"({len(cand_qc)} protein-group rows).")

    matrix = full_matrix[full_matrix[config.ID_COL].isin(cand_qc[config.ID_COL])].merge(
        cand_qc[[config.ID_COL, "Gene", "Exosome_Classification"]], on=config.ID_COL, how="left")
    dup_gene = matrix["Gene"].duplicated(keep=False)
    short_acc = matrix[config.ID_COL].astype(str).str.split(";").str[0]
    matrix["label"] = np.where(dup_gene, matrix["Gene"] + " (" + short_acc + ")", matrix["Gene"])
    stage5_dir = tc.tsv_stage_dir(tc.STAGE_05_IMPUTED)
    utils.save_tsv(matrix, stage5_dir, f"{genotype}_ev_subset_imputed")

    if len(matrix) < 4:
        logger.warning(f"[ev_trend] {genotype}: only {len(matrix)} EV proteins survive QC — "
                      f"too few for trend testing; skipping.")
        return pd.DataFrame(), matrix, bio

    trend_df = run_trend_test(matrix, bio, logger)
    utils.save_tsv(trend_df, tc.RESULTS_DIR, f"trend_test_results_{genotype}")

    n_sig = int((trend_df["JT_FDR"] < tc.FDR_THRESHOLD).sum())
    n_inc = int(((trend_df["JT_FDR"] < tc.FDR_THRESHOLD) & (trend_df["Direction"] == "Increasing")).sum())
    n_dec = int(((trend_df["JT_FDR"] < tc.FDR_THRESHOLD) & (trend_df["Direction"] == "Decreasing")).sum())
    logger.info(f"[ev_trend] {genotype}: {len(trend_df)} EV proteins tested; "
               f"{n_sig} significant (JT FDR<{tc.FDR_THRESHOLD}: {n_inc} increasing, "
               f"{n_dec} decreasing).")

    fig_dir = os.path.join(tc.FIGURES_ROOT, genotype)
    plot_volcano(trend_df, genotype, fig_dir, logger)
    plot_heatmap(trend_df, genotype, fig_dir, logger)

    sig_ids, is_fallback = _select_box_ids(trend_df, logger, genotype)
    box_title = (f"Timepoint Trend — {'Significant' if not is_fallback else 'Top'} "
               f"EV Proteins ({genotype})")
    plot_boxplots(matrix, bio, trend_df, genotype, fig_dir, logger, sig_ids,
                 "trend_boxplots_significant", box_title)

    top_ids = trend_df.head(20)[config.ID_COL].tolist()
    plot_boxplots(matrix, bio, trend_df, genotype, fig_dir, logger, top_ids,
                 "trend_boxplots_top",
                 f"Timepoint Trend — Top {len(top_ids)} EV Proteins by JT p-value ({genotype})")

    plot_mean_se_lines(matrix, bio, trend_df, genotype, fig_dir, logger, sig_ids)

    trend_df.insert(0, "Genotype", genotype)
    return trend_df, matrix, bio


def main():
    tc.ensure_dirs()
    logger = utils.setup_logger("EV_Trend_Analysis")
    logger.info("Mousebrain-elevated EV protein trend analysis (branch)")

    candidates = build_candidate_list(logger)

    all_trend = []
    matrix_by_genotype = {}
    for genotype in tc.GENOTYPES:
        trend_df, matrix, bio = run_genotype(genotype, candidates, logger)
        matrix_by_genotype[genotype] = (matrix, bio)
        if not trend_df.empty:
            all_trend.append(trend_df)

    # ── KI/WT ratio trend analysis (Task: signal intensity -> KI/WT ratio) ──
    logger.info(f"\n{'=' * 70}\n[ev_trend] Branch: KI/WT ratio\n{'=' * 70}")
    wt_matrix, wt_bio = matrix_by_genotype["WT"]
    ki_matrix, ki_bio = matrix_by_genotype["KI"]
    ratio_trend_df = run_ki_wt_ratio_analysis(wt_matrix, wt_bio, ki_matrix, ki_bio, logger)
    if not ratio_trend_df.empty:
        all_trend.append(ratio_trend_df)

    if all_trend:
        combined = pd.concat(all_trend, ignore_index=True)
        utils.save_tsv(combined, tc.RESULTS_DIR, "trend_test_results_combined")

    # ── Consolidated candidate coverage table ───────────────────────────────
    stage5_dir = tc.tsv_stage_dir(tc.STAGE_05_IMPUTED)
    detected_ids_by_genotype = {}
    for genotype in tc.GENOTYPES:
        p = os.path.join(stage5_dir, f"{genotype}_ev_subset_imputed.tsv")
        if os.path.exists(p):
            detected_ids_by_genotype[genotype] = set(
                pd.read_csv(p, sep="\t")[config.ID_COL].astype(str))
        else:
            detected_ids_by_genotype[genotype] = set()

    coverage = candidates.copy()
    for genotype in tc.GENOTYPES:
        coverage[f"detected_{genotype}_QC"] = coverage[config.ID_COL].astype(str).isin(
            detected_ids_by_genotype[genotype])
    utils.save_tsv(coverage, tc.RESULTS_DIR, "Mousebrain_elevated_EV_protein_list")

    n_genes_total = coverage["Gene"].nunique()
    n_genes_raw = coverage.loc[coverage["detected_raw"], "Gene"].nunique()
    n_genes_wt = coverage.loc[coverage["detected_WT_QC"], "Gene"].nunique()
    n_genes_ki = coverage.loc[coverage["detected_KI_QC"], "Gene"].nunique()
    n_genes_either = coverage.loc[coverage["detected_WT_QC"] | coverage["detected_KI_QC"],
                                  "Gene"].nunique()
    logger.info(f"\n{'=' * 70}\n[ev_trend] SUMMARY\n{'=' * 70}")
    logger.info(f"EV candidate genes (DB): {n_genes_total}")
    logger.info(f"  detected in raw report:        {n_genes_raw}")
    logger.info(f"  detected + survive WT QC:       {n_genes_wt}")
    logger.info(f"  detected + survive KI QC:       {n_genes_ki}")
    logger.info(f"  detected + survive QC (either): {n_genes_either}")

    plot_candidate_coverage_funnel(n_genes_total, n_genes_raw, n_genes_wt, n_genes_ki, logger)
    logger.info(f"Outputs written under: {tc.BRANCH_DIR}/output/")


if __name__ == "__main__":
    main()
