"""Step 05 — MinProb imputation (per-feature, left-censored Gaussian downshift).

For each protein row with ≥2 observed values:
    mu_imp = mean(observed) − downshift × sd(observed)
    sd_imp = width × sd(observed)
Rows with <2 observations fall back to the global observed pool.  Operates on
the normalized log2-ratio matrix (biological samples only).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
import utils


def minprob_impute(matrix: np.ndarray, *, downshift: float, width: float, seed: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = matrix.copy()
    mask = np.isnan(out)
    obs = matrix[~np.isnan(matrix)]
    global_med = float(np.median(obs))
    global_sd = float(np.std(obs, ddof=1)) if obs.size > 1 else 1.0
    for i in range(out.shape[0]):
        row = out[i]
        observed = row[~np.isnan(row)]
        if observed.size >= 2:
            mu = observed.mean()
            sd = observed.std(ddof=1)
            if not np.isfinite(sd) or sd == 0:
                sd = global_sd
        else:
            mu, sd = global_med, global_sd
        mu_imp = mu - downshift * sd
        sd_imp = max(width * sd, 1e-6)
        nan_idx = np.where(np.isnan(row))[0]
        if nan_idx.size:
            out[i, nan_idx] = rng.normal(mu_imp, sd_imp, size=nan_idx.size)
    return out, mask


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "05_imputed")
    utils.apply_style()

    df = state["data_df"].copy()
    sample_cols = state["sample_cols"]
    meta = state["meta"]

    mat = df[sample_cols].values.astype(float)
    n_missing = int(np.isnan(mat).sum())
    total = mat.size
    logger.info(f"[step05] {dataset}: missing before = {n_missing}/{total} "
                f"({n_missing/total*100:.1f}%)")

    imputed, miss_mask = minprob_impute(
        mat, downshift=config.IMPUTE_DOWNSHIFT, width=config.IMPUTE_WIDTH,
        seed=config.IMPUTE_SEED)

    df[sample_cols] = imputed
    utils.save_tsv(df, out, "imputed_matrix")
    pd.DataFrame(miss_mask.astype(int), columns=sample_cols).to_csv(
        os.path.join(out, "imputed_value_mask.tsv"), sep="\t", index=False)

    # Observed vs imputed distribution
    obs_vals = mat[~miss_mask]
    imp_vals = imputed[miss_mask]
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    bins = np.linspace(np.nanmin(obs_vals), np.nanmax(obs_vals), 80)
    ax.hist(obs_vals, bins=bins, color="#56B4E9", alpha=0.6,
            label=f"Observed (n={obs_vals.size:,})", edgecolor="white", linewidth=0.3)
    ax.hist(imp_vals, bins=bins, histtype="step", color="#D55E00", linewidth=1.4,
            label=f"Imputed (n={imp_vals.size:,})")
    ax.set_xlabel("Normalized log2 ratio")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Observed vs imputed distribution — {dataset}")
    ax.legend(frameon=False, loc="upper left")
    utils.save_fig(fig, out, "imputation_distribution")
    plt.close(fig)

    # Per-group missing rate (pre-imputation)
    groups = cfg["group_order"]
    rates = {}
    for g in groups:
        cols = meta[meta["group"] == g]["column_name"].tolist()
        idx = [sample_cols.index(c) for c in cols if c in sample_cols]
        if idx:
            rates[g] = float(np.isnan(mat[:, idx]).mean() * 100)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar([config.GROUP_DISPLAY[g] for g in rates],
           list(rates.values()),
           color=[config.GROUP_COLORS[g] for g in rates],
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Missing rate (%)")
    ax.set_title(f"Per-group missing rate (pre-imputation) — {dataset}")
    ax.tick_params(axis="x", rotation=60)
    for i, v in enumerate(rates.values()):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    utils.save_fig(fig, out, "imputation_missing_rate_by_group")
    plt.close(fig)

    # Before/after group violins
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, source, title in [(axes[0], mat, "Before (observed only)"),
                              (axes[1], imputed, "After imputation")]:
        rows = []
        for g in groups:
            cols = meta[meta["group"] == g]["column_name"].tolist()
            idx = [sample_cols.index(c) for c in cols if c in sample_cols]
            if idx:
                rows.append(pd.DataFrame({"group": g, "value": source[:, idx].ravel()}))
        long_df = pd.concat(rows, ignore_index=True).dropna()
        long_df["group"] = pd.Categorical(long_df["group"], categories=groups, ordered=True)
        sns.violinplot(data=long_df, x="group", y="value", hue="group",
                       palette=config.GROUP_COLORS, ax=ax, inner="box", legend=False)
        ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("log2 ratio (normalized)")
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle(f"Group distributions — {dataset}", y=1.02)
    utils.save_fig(fig, out, "imputation_group_distributions", tight=False)
    plt.close(fig)

    stats = {
        "dataset": dataset, "n_missing_before": n_missing, "n_total": int(total),
        "missing_pct_before": float(n_missing / total * 100),
        "n_imputed": int(miss_mask.sum()),
        "downshift": config.IMPUTE_DOWNSHIFT, "width": config.IMPUTE_WIDTH,
        "seed": config.IMPUTE_SEED,
    }
    with open(os.path.join(out, "imputation_summary.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"[step05] {dataset}: imputed {stats['n_imputed']} values "
                f"(MinProb downshift={stats['downshift']}, width={stats['width']}).")

    state = dict(state)
    state["data_df"] = df
    state["imputed_mask"] = miss_mask
    return state


if __name__ == "__main__":
    pass
