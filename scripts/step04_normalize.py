"""Step 04 — reference-ratio + median normalization (log2 scale).

Two stages (config.NORMALIZE_METHOD == 'ref_ratio_median'):
  1. Reference ratio  : subtract the pooled-ref log2 value per protein
                        ⇒ each value becomes log2(sample / pooled_ref).
                        Proteins missing in the ref fall back to the row mean
                        of observed biological samples (row-centered).
  2. Median centering : subtract each sample's median so loading/depth
                        differences between samples are removed.

The pooled-ref column is consumed here and dropped from the downstream matrix.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import config
import utils
from step03_log2_transform import boxplot_by_sample, density_by_group


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "04_normalized")
    utils.apply_style()

    if cfg["normalize_method"] != "ref_ratio_median":
        raise ValueError(f"Unsupported normalize_method={cfg['normalize_method']!r}")

    df = state["data_df"].copy()
    sample_cols = list(state["sample_cols"])
    ref_col = state.get("ref_col")
    meta = state["meta"]

    if not ref_col or ref_col not in df.columns:
        raise ValueError("Reference-ratio normalization requires the pooled-ref column.")

    # Stage 1 — reference ratio (per-protein) on the biological samples.
    ratio_df, ref_tbl = utils.reference_ratio_log2(df, sample_cols, ref_col)
    n_fallback = int((ref_tbl["ref_source"] == "row_mean_fallback").sum())
    logger.info(f"[step04] {dataset}: reference-ratio applied; "
                f"{len(ref_tbl) - n_fallback} proteins used pooled ref, "
                f"{n_fallback} used row-mean fallback (ref missing).")

    # Stage 2 — sample-wise median centering.
    norm_df, factors = utils.median_normalize_log2(ratio_df, sample_cols)

    # Drop the pooled-ref column from the downstream matrix (its job is done).
    norm_df = norm_df.drop(columns=[ref_col])

    utils.save_tsv(norm_df, out, "normalized_matrix")
    utils.save_tsv(factors, out, "normalization_factors")
    utils.save_tsv(ref_tbl, out, "reference_ratio_table")

    boxplot_by_sample(norm_df, sample_cols, meta, cfg["group_order"], out,
                      "boxplot_after_normalization",
                      f"Sample log2 distributions — after ref-ratio + median ({dataset})",
                      ylabel="log2(sample / pooled ref), median-centered")
    density_by_group(norm_df, sample_cols, meta, cfg["group_order"], out,
                     "distribution_after",
                     f"Distribution — after normalization ({dataset})",
                     "Normalized log2 ratio")

    # Shift factors bar
    cols_ordered, ordered_meta = utils.ordered_bio_cols(meta, sample_cols, cfg["group_order"])
    f_map = dict(zip(factors["column_name"], factors["shift_subtracted"]))
    shifts = [f_map[c] for c in cols_ordered]
    palette = [config.GROUP_COLORS.get(g, "#888") for g in ordered_meta["group"]]
    n = len(cols_ordered)
    fig, ax = plt.subplots(figsize=(min(28.0, max(8.0, 0.10 * n + 4.0)), 4.5))
    ax.bar(range(n), shifts, color=palette, edgecolor="black", linewidth=0.3, width=0.85)
    stride = max(1, int(np.ceil(n / 24)))
    ax.set_xticks(range(0, n, stride))
    ax.set_xticklabels(ordered_meta["sample_id"].iloc[::stride].tolist(),
                       rotation=70, ha="right", fontsize=7)
    ax.axhline(0, color="#666", lw=0.6)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Median shift subtracted")
    ax.set_title(f"Sample median-normalization shifts — {dataset}")
    utils.save_fig(fig, out, "normalization_shifts")
    plt.close(fig)

    with open(os.path.join(out, "normalization_summary.txt"), "w") as f:
        f.write(f"method: {cfg['normalize_method']}\n")
        f.write(f"n_proteins: {len(norm_df)}\n")
        f.write(f"n_bio_samples: {len(sample_cols)}\n")
        f.write(f"ref_used_pooled: {len(ref_tbl) - n_fallback}\n")
        f.write(f"ref_used_rowmean_fallback: {n_fallback}\n")

    logger.info(f"[step04] {dataset}: normalized (ref-ratio + median); "
                f"pooled-ref column dropped.")
    state = dict(state)
    state["data_df"] = norm_df
    state["sample_cols"] = sample_cols
    state["ref_col"] = None
    return state


if __name__ == "__main__":
    pass
