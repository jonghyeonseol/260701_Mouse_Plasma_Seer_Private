"""Step 01 — raw summary diagnostics + plasma-specific QC.

Diagnostics (adapted from the reference NK-cell step01):
  - raw intensity matrix + joined sample metadata
  - protein IDs per sample (bar) and per group (box)
  - missing-value heatmap, per-protein CV distribution

Plasma-specific additions (this project):
  - Peptide-recovery QC: per-sample / per-group µg recovered, and its
    correlation with proteome depth (loading effect check).
  - Plate/well batch check: proteome depth and median intensity across plate
    row/column to surface positional (edge/row) effects.
"""

from __future__ import annotations

import os
import warnings as _w

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as _sps

import config
import utils


def _tick_stride(n: int) -> int:
    return max(1, int(np.ceil(n / 24)))


def _peptide_qc(meta_bio, id_df, out):
    """Peptide recovery per sample/group and vs proteome depth."""
    order = config.PLASMA_GROUP_ORDER
    m = meta_bio.merge(id_df[["column_name", "n_proteins"]], on="column_name")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # (a) per-group boxplot of peptide µg
    data = [m.loc[m["group"] == g, "peptide_ug"].dropna().values for g in order]
    bp = axes[0].boxplot(data, labels=[config.GROUP_DISPLAY[g] for g in order],
                         patch_artist=True, widths=0.6, showmeans=True,
                         meanprops={"marker": "D", "markerfacecolor": "white",
                                    "markeredgecolor": "black", "markersize": 5})
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(config.GROUP_COLORS[g]); patch.set_alpha(0.75)
    axes[0].set_ylabel("Peptide recovered (µg)")
    axes[0].set_title("Peptide recovery per group")
    axes[0].tick_params(axis="x", rotation=60)

    # (b) peptide µg vs proteome depth
    x = m["peptide_ug"].values
    y = m["n_proteins"].values
    colors = [config.GROUP_COLORS[g] for g in m["group"]]
    axes[1].scatter(x, y, c=colors, edgecolor="black", linewidth=0.3, s=40, alpha=0.85)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 2:
        r, p = _sps.pearsonr(x[ok], y[ok])
        b, a = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        axes[1].plot(xs, a + b * xs, color="#333", lw=1.0, ls="--")
        axes[1].set_title(f"Peptide µg vs proteome depth\nPearson r={r:.2f}, p={p:.2g}")
    axes[1].set_xlabel("Peptide recovered (µg)")
    axes[1].set_ylabel("Proteins identified")
    utils.save_fig(fig, out, "qc_peptide_recovery")
    plt.close(fig)
    return (r, p) if ok.sum() > 2 else (np.nan, np.nan)


def _plate_qc(meta_bio, id_df, df, sample_cols, out):
    """Plate row/column positional (batch) effect on depth and intensity."""
    m = meta_bio.merge(id_df[["column_name", "n_proteins"]], on="column_name")
    med_int = {c: np.nanmedian(df[c].values) for c in sample_cols}
    m["median_intensity"] = m["column_name"].map(med_int)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    row_order = sorted([r for r in m["plate_row"].dropna().unique()])
    col_order = sorted([int(c) for c in m["plate_col"].dropna().unique()])

    def _box(ax, by, order, vals, ylabel, title):
        data = [m.loc[m[by] == o, vals].dropna().values for o in order]
        ax.boxplot(data, labels=[str(o) for o in order], patch_artist=True,
                   widths=0.6, boxprops=dict(facecolor="#B0C4DE", alpha=0.8))
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_xlabel(by)

    _box(axes[0, 0], "plate_row", row_order, "n_proteins",
         "Proteins identified", "Proteome depth by plate row")
    _box(axes[0, 1], "plate_col", col_order, "n_proteins",
         "Proteins identified", "Proteome depth by plate column")
    _box(axes[1, 0], "plate_row", row_order, "median_intensity",
         "Median raw intensity", "Median intensity by plate row")
    _box(axes[1, 1], "plate_col", col_order, "median_intensity",
         "Median raw intensity", "Median intensity by plate column")

    # ANOVA F-test for positional effect on depth (documented, not corrected).
    stats_txt = {}
    for by, order in [("plate_row", row_order), ("plate_col", col_order)]:
        groups = [m.loc[m[by] == o, "n_proteins"].dropna().values for o in order]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            F, p = _sps.f_oneway(*groups)
            stats_txt[by] = (float(F), float(p))
    fig.suptitle("Plate positional (batch) QC — proteome depth & intensity",
                 fontsize=config.TITLE_SIZE, y=1.0)
    utils.save_fig(fig, out, "qc_plate_batch", tight=False)
    plt.close(fig)
    return stats_txt


def run(dataset: str, logger, state: dict | None = None) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "01_raw_summary")
    utils.apply_style()

    df, meta = utils.load_raw(dataset)
    meta_cols = cfg["meta_cols"]
    all_sample_cols = meta["column_name"].tolist()
    df = utils.to_numeric_intensity(df, all_sample_cols)

    bio_cols, ref_col = utils.split_sample_columns(meta)
    meta_bio = meta[~meta["is_ref"]].copy()

    keep_meta = [c for c in meta_cols if c in df.columns]
    utils.save_tsv(df[keep_meta + all_sample_cols], out, "raw_intensity_matrix")
    utils.save_tsv(meta, out, "sample_metadata")
    logger.info(f"[step01] {dataset}: {df.shape[0]} proteins × {len(bio_cols)} "
                f"biological samples (+1 pooled ref).")

    # ── Protein identifications per sample ───────────────────────────────────
    id_counts = df[all_sample_cols].notna().sum(axis=0)
    id_df = pd.DataFrame({"column_name": all_sample_cols,
                          "n_proteins": id_counts.values})
    id_df = id_df.merge(meta[["column_name", "sample_id", "group",
                              "sample_number", "is_ref"]], on="column_name")
    rank = {g: i for i, g in enumerate(cfg["group_order"] + [config.REF_GROUP])}
    id_df["_rank"] = id_df["group"].map(rank).fillna(99).astype(int)
    id_df = id_df.sort_values(["_rank", "sample_number"]).reset_index(drop=True)
    id_df = id_df.drop(columns=["_rank"])
    utils.save_tsv(id_df, out, "protein_id_per_sample")

    n = len(id_df)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.10 * n + 4.0), 5.5))
    palette = [config.GROUP_COLORS.get(g, "#888888") for g in id_df["group"]]
    ax.bar(range(n), id_df["n_proteins"], color=palette, edgecolor="black",
           linewidth=0.3, width=0.85)
    stride = _tick_stride(n)
    ax.set_xticks(range(0, n, stride))
    ax.set_xticklabels([id_df["sample_id"].iloc[i] for i in range(0, n, stride)],
                       rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Proteins identified")
    ax.set_xlim(-0.6, n - 0.4)
    median_n = float(id_df.loc[~id_df["is_ref"], "n_proteins"].median())
    ax.axhline(median_n, color="#888", lw=0.8, ls="--")
    ax.text(n - 0.3, median_n, f" median = {int(median_n):,}",
            va="center", ha="left", fontsize=9, color="#555")
    prev, start = None, 0
    for i, g in enumerate(id_df["group"]):
        if prev is not None and g != prev:
            ax.axvline(i - 0.5, color="#bbbbbb", lw=0.8)
            ax.text((start + i - 1) / 2, ax.get_ylim()[1] * 0.98,
                    config.GROUP_DISPLAY.get(prev, prev), ha="center", va="top",
                    fontsize=8, color="#333", weight="bold")
            start = i
        prev = g
    ax.text((start + n - 1) / 2, ax.get_ylim()[1] * 0.98,
            config.GROUP_DISPLAY.get(prev, prev), ha="center", va="top",
            fontsize=8, color="#333", weight="bold")
    ax.set_title(f"Protein IDs per sample — {dataset} (n={n})")
    utils.save_fig(fig, out, "protein_id_per_sample")
    plt.close(fig)

    # Per-group box (biological groups only)
    fig, ax = plt.subplots(figsize=(9, 6))
    order = cfg["group_order"]
    data = [id_df.loc[id_df["group"] == g, "n_proteins"].values for g in order]
    bp = ax.boxplot(data, labels=[config.GROUP_DISPLAY[g] for g in order],
                    patch_artist=True, widths=0.6, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 5})
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(config.GROUP_COLORS[g]); patch.set_alpha(0.75)
    ax.set_ylabel("Proteins identified per sample")
    ax.set_title(f"Per-group identification — {dataset}")
    ax.tick_params(axis="x", rotation=60)
    utils.save_fig(fig, out, "protein_id_per_group_box")
    plt.close(fig)

    # ── Missing-value heatmap (biological samples, grouped) ──────────────────
    ordered_cols, ordered_meta = utils.ordered_bio_cols(meta, bio_cols, order)
    miss_mat = df[ordered_cols].isna().astype(int).values
    if miss_mat.shape[0] > 4000:
        row_miss = miss_mat.sum(axis=1)
        oi = np.argsort(-row_miss)
        sel = np.concatenate([oi[:2000], np.random.default_rng(42).choice(
            oi[2000:], size=2000, replace=False)])
        miss_mat = miss_mat[sel]
    fig, ax = plt.subplots(figsize=(max(10, 0.10 * len(ordered_cols) + 3), 8))
    sns.heatmap(miss_mat, ax=ax, cmap="Greys", cbar=False,
                xticklabels=False, yticklabels=False)
    grp_seq = list(ordered_meta["group"])
    for i in range(1, len(grp_seq)):
        if grp_seq[i] != grp_seq[i - 1]:
            ax.axvline(i, color="red", lw=1.0)
    ax.set_title(f"Missing value heatmap — {dataset} (black = missing)")
    ax.set_xlabel("Samples (ordered by group)")
    ax.set_ylabel("Proteins (subsampled to 4 000 for display)")
    utils.save_fig(fig, out, "missing_value_heatmap", also_svg=False)
    plt.close(fig)

    # ── CV distribution (per-protein CV across biological samples) ────────────
    arr = df[bio_cols].values
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(arr, axis=1)
        sd = np.nanstd(arr, axis=1, ddof=1)
    cv = np.where(mu > 0, sd / mu, np.nan) * 100
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    ax.hist(cv[~np.isnan(cv)], bins=80, color="#56B4E9", edgecolor="black", linewidth=0.4)
    ax.axvline(float(np.nanmedian(cv)), color="#D55E00", ls="--", lw=1.0)
    ax.set_xlabel("Per-protein CV (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Protein-level CV distribution — {dataset}")
    utils.save_fig(fig, out, "cv_distribution")
    plt.close(fig)

    # ── Plasma-specific QC ───────────────────────────────────────────────────
    pep_r, pep_p = _peptide_qc(meta_bio, id_df, out)
    plate_stats = _plate_qc(meta_bio, id_df, df, bio_cols, out)
    logger.info(f"[step01] {dataset}: peptide µg vs depth Pearson r={pep_r:.3f} "
                f"(p={pep_p:.2g}); plate positional ANOVA {plate_stats}")

    summary = {
        "dataset": dataset,
        "n_proteins_raw": int(df.shape[0]),
        "n_biological_samples": len(bio_cols),
        "has_pooled_ref": ref_col is not None,
        "groups": meta_bio["group"].value_counts().to_dict(),
        "missing_overall_pct": float(np.isnan(arr).mean() * 100),
        "median_proteins_per_sample": float(id_df.loc[~id_df["is_ref"], "n_proteins"].median()),
        "min_proteins_per_sample": int(id_df.loc[~id_df["is_ref"], "n_proteins"].min()),
        "max_proteins_per_sample": int(id_df.loc[~id_df["is_ref"], "n_proteins"].max()),
        "peptide_vs_depth_pearson_r": float(pep_r),
        "peptide_vs_depth_p": float(pep_p),
        "plate_positional_anova": plate_stats,
    }
    with open(os.path.join(out, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"[step01] summary: {summary}")

    return {"data_df": df, "meta": meta, "sample_cols": bio_cols,
            "ref_col": ref_col, "all_sample_cols": all_sample_cols,
            "summary": summary}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    log = utils.setup_logger("MousePlasma")
    run("MousePlasma", log)
