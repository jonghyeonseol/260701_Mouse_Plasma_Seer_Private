"""Step 09 — trend & time-series analysis (branch projects only).

Operates on a per-genotype branch's normalized+imputed matrix (proteins × that
genotype's samples across 4 timepoints).  Two complementary analyses:

1. Temporal-profile soft-clustering (Mfuzz-style):
   - per-protein profile = mean over replicates at each timepoint, z-scored
     across the 4 points;
   - fuzzy c-means (numpy implementation, no external dependency) →
     soft membership + hard label;
   - outputs cluster trajectory panels and a cluster-ordered profile heatmap.

2. Monotonic trend testing:
   - limma linear-week fit  ~ week (+ peptide_ug)  → per-week slope, moderated t, BH;
   - Jonckheere–Terpstra (normal approximation) across ordered timepoints;
   - Spearman-vs-week cross-check;
   - proteins classified Increasing / Decreasing / NS by the week slope.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import config
import utils
from step06_dep_limma import _init_rpy2


# ── Fuzzy c-means (Mfuzz-style soft clustering) ──────────────────────────────

def fuzzy_cmeans(X: np.ndarray, k: int, m: float, seed: int,
                 max_iter: int = 300, tol: float = 1e-5):
    """Return (membership U [n×k], centers [k×d]).  Standard FCM with Euclidean
    distance on z-scored temporal profiles."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n, k))
    U /= U.sum(axis=1, keepdims=True)
    centers = np.zeros((k, X.shape[1]))
    for _ in range(max_iter):
        Um = U ** m
        centers = (Um.T @ X) / np.fmax(Um.sum(axis=0)[:, None], 1e-12)
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # n×k
        d = np.fmax(d, 1e-10)
        ratio = d[:, :, None] / d[:, None, :]           # n×k×k
        U_new = 1.0 / np.sum(ratio ** (2.0 / (m - 1)), axis=2)
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
    return U, centers


def _temporal_profiles(matrix, bio, sample_cols, timepoints):
    """Return (profiles [n×T] mean per timepoint, z_profiles, ids, labels)."""
    tp_cols = {t: bio.loc[bio["timepoint"] == t, "column_name"].tolist()
               for t in timepoints}
    prof = np.column_stack([matrix[tp_cols[t]].mean(axis=1).values for t in timepoints])
    z = (prof - prof.mean(axis=1, keepdims=True)) / \
        (prof.std(axis=1, keepdims=True, ddof=0) + 1e-9)
    ids = matrix[config.ID_COL].astype(str).values
    labels = utils.build_display_labels(matrix)
    return prof, z, ids, labels


def _plot_cluster_trajectories(z, hard, U, timepoints, k, out, fig_mirror, dataset):
    utils.apply_style()
    ncol = 3
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    x = list(timepoints)
    for c in range(k):
        ax = axes[c // ncol][c % ncol]
        members = np.where(hard == c)[0]
        # colour lines by membership strength (Mfuzz-style)
        for i in members:
            ax.plot(x, z[i], color=plt.cm.viridis(U[i, c]), alpha=0.25, lw=0.5)
        if len(members):
            ax.plot(x, z[members].mean(axis=0), color="#C0392B", lw=2.4,
                    marker="o", label="centroid")
        ax.set_title(f"Cluster {c+1}  (n={len(members)})", fontsize=11)
        ax.set_xlabel("Week"); ax.set_ylabel("z-score")
        ax.set_xticks(x)
        ax.axhline(0, color="#999", lw=0.6, ls="--")
    for j in range(k, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle(f"Temporal profile clusters (fuzzy c-means, k={k}) — {dataset}",
                 fontsize=config.TITLE_SIZE, y=1.0)
    png = utils.save_fig(fig, out, "cluster_trajectories", tight=True)
    plt.close(fig)
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            import shutil; shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


def _plot_profile_heatmap(z, hard, timepoints, k, out, fig_mirror, dataset):
    utils.apply_style()
    # order proteins by cluster, then by peak timepoint within cluster
    order = np.lexsort((z.argmax(axis=1), hard))
    zo = np.clip(z[order], -3, 3)
    clab = hard[order]
    fig, (axc, axh) = plt.subplots(
        1, 2, figsize=(6, 10), gridspec_kw={"width_ratios": [0.04, 1], "wspace": 0.02})
    # cluster colour strip
    cmap_clusters = plt.cm.tab10
    axc.imshow(clab[:, None], aspect="auto", cmap=cmap_clusters, vmin=0, vmax=9)
    axc.set_xticks([]); axc.set_yticks([]); axc.set_ylabel("Proteins (by cluster)")
    im = axh.imshow(zo, aspect="auto", cmap=utils.zscore_cmap(), vmin=-3, vmax=3)
    axh.set_xticks(range(len(timepoints)))
    axh.set_xticklabels([f"{t}W" for t in timepoints]); axh.set_yticks([])
    # cluster boundary lines
    bnd = np.where(np.diff(clab) != 0)[0]
    for b in bnd:
        axh.axhline(b + 0.5, color="black", lw=0.5)
    cbar = fig.colorbar(im, ax=axh, fraction=0.05, pad=0.02)
    cbar.set_label("z-score")
    fig.suptitle(f"Time-series profile heatmap (n={z.shape[0]}) — {dataset}",
                 fontsize=13, y=0.99)
    png = utils.save_fig(fig, out, "timeseries_profile_heatmap", tight=False)
    plt.close(fig)
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            import shutil; shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


# ── Monotonic trend ──────────────────────────────────────────────────────────

def _jonckheere_terpstra(values: np.ndarray, group_idx: list[np.ndarray]):
    """Vectorized JT statistic + normal-approx z across proteins.

    values: n_proteins × n_samples ; group_idx: list of column-index arrays,
    ordered by timepoint.  Returns (z [n], p [n]); positive z = increasing trend.
    """
    P = values.shape[0]
    U = np.zeros(P)
    ns = [len(g) for g in group_idx]
    N = sum(ns)
    for i in range(len(group_idx)):
        for j in range(i + 1, len(group_idx)):
            xi = values[:, group_idx[i]]        # P × n_i
            yj = values[:, group_idx[j]]        # P × n_j
            gt = (yj[:, None, :] > xi[:, :, None]).sum(axis=(1, 2))
            eq = (yj[:, None, :] == xi[:, :, None]).sum(axis=(1, 2))
            U += gt + 0.5 * eq
    sum_n2 = sum(n * n for n in ns)
    E = (N * N - sum_n2) / 4.0
    var = (N * N * (2 * N + 3) - sum(n * n * (2 * n + 3) for n in ns)) / 72.0
    z = (U - E) / np.sqrt(max(var, 1e-12))
    p = 2 * norm.sf(np.abs(z))
    return z, p


def _limma_week_trend(matrix, bio, sample_cols, logger):
    """limma moderated linear-week trend (slope per week)."""
    ro, limma = _init_rpy2()
    expr = matrix[sample_cols].values.astype(float)
    week = bio.set_index("column_name").loc[sample_cols, "timepoint"].astype(float).values
    pep = pd.to_numeric(bio.set_index("column_name").loc[sample_cols, config.PEPTIDE_COL],
                        errors="coerce").values.astype(float)
    ro.r.assign("expr_mat", ro.r["matrix"](ro.FloatVector(expr.flatten("F")),
                                            nrow=expr.shape[0]))
    ro.r.assign("week", ro.FloatVector(week))
    use_pep = config.USE_PEPTIDE_COVARIATE and np.isfinite(pep).all()
    if use_pep:
        ro.r.assign("pep", ro.FloatVector(pep))
        ro.r("design <- model.matrix(~ week + pep)")
    else:
        ro.r("design <- model.matrix(~ week)")
    ro.r("""
    fit <- eBayes(lmFit(expr_mat, design), trend=TRUE, robust=TRUE)
    tt  <- topTable(fit, coef="week", number=Inf, sort.by="none", adjust.method="BH")
    """)
    tt = ro.r["tt"]
    if isinstance(tt, pd.DataFrame):
        out = tt[["logFC", "t", "P.Value", "adj.P.Val"]].reset_index(drop=True)
    else:
        out = pd.DataFrame({k: np.asarray(tt.rx2(k), dtype=float) for k in
                            ["logFC", "t", "P.Value", "adj.P.Val"]})
    out.columns = ["week_slope", "week_t", "week_P", "week_adjP"]
    logger.info(f"[step09] limma linear-week trend fit (peptide covariate={use_pep}).")
    return out


def _trend_line_plots(prof, ids, labels, trend_df, timepoints, out, fig_mirror, dataset,
                      top_n=12):
    utils.apply_style()
    for direction, ascending in [("increasing", False), ("decreasing", True)]:
        sig = trend_df[trend_df["direction"] == direction.capitalize()]
        if sig.empty:
            continue
        sig = sig.reindex(sig["week_t"].sort_values(ascending=ascending).index).head(top_n)
        idx = [np.where(ids == pid)[0][0] for pid in sig[config.ID_COL].astype(str)]
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        x = list(timepoints)
        for i, lab in zip(idx, sig["label"]):
            p = prof[i]
            ax.plot(x, p, marker="o", lw=1.2, alpha=0.8, label=str(lab)[:16])
        ax.set_xlabel("Week"); ax.set_ylabel("Normalized log2 ratio (mean)")
        ax.set_xticks(x)
        ax.set_title(f"Top {len(idx)} {direction} proteins — {dataset}")
        ax.legend(fontsize=7, ncol=2, frameon=False)
        png = utils.save_fig(fig, out, f"trend_top_{direction}", tight=True)
        plt.close(fig)
        for ext in (".png", ".svg"):
            src = png.replace(".png", ext)
            if os.path.exists(src):
                import shutil; shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "09_trend_timeseries")
    fig_mirror = utils.figures_dir(dataset)
    utils.apply_style()

    matrix = state["data_df"].copy()
    meta = state["meta"]
    timepoints = cfg["timepoints"]
    bio = meta[(~meta["is_ref"]) & (meta["group"].isin(cfg["group_order"]))].copy()
    sample_cols = bio["column_name"].tolist()

    # ── Temporal profiles + fuzzy c-means ────────────────────────────────────
    prof, z, ids, labels = _temporal_profiles(matrix, bio, sample_cols, timepoints)
    k = config.TREND_N_CLUSTERS
    U, centers = fuzzy_cmeans(z, k, m=config.TREND_FUZZIFIER, seed=config.TREND_SEED)
    hard = U.argmax(axis=1)
    memb = U.max(axis=1)

    clu = pd.DataFrame({config.ID_COL: ids, "label": labels,
                        "cluster": hard + 1, "max_membership": memb,
                        "core": memb >= config.TREND_CORE_MEMBERSHIP})
    for c in range(k):
        clu[f"membership_c{c+1}"] = U[:, c]
    for ti, t in enumerate(timepoints):
        clu[f"mean_{t}W"] = prof[:, ti]
    utils.save_tsv(clu, out, "profile_cluster_assignments")

    _plot_cluster_trajectories(z, hard, U, timepoints, k, out, fig_mirror, dataset)
    _plot_profile_heatmap(z, hard, timepoints, k, out, fig_mirror, dataset)
    logger.info(f"[step09] {dataset}: {k} temporal clusters; sizes="
                f"{[int((hard == c).sum()) for c in range(k)]}, "
                f"core proteins={int(clu['core'].sum())}.")

    # ── Monotonic trend ──────────────────────────────────────────────────────
    week_df = _limma_week_trend(matrix, bio, sample_cols, logger)
    group_idx = [np.array([sample_cols.index(c) for c in
                           bio.loc[bio["timepoint"] == t, "column_name"]])
                 for t in timepoints]
    jt_z, jt_p = _jonckheere_terpstra(matrix[sample_cols].values.astype(float), group_idx)
    # Spearman vs week (per protein)
    week_vec = bio.set_index("column_name").loc[sample_cols, "timepoint"].astype(float).values
    vals = matrix[sample_cols].values.astype(float)
    sp_rho = np.array([spearmanr(vals[i], week_vec).statistic for i in range(vals.shape[0])])

    trend = pd.DataFrame({config.ID_COL: ids, "label": labels})
    trend = pd.concat([trend, week_df], axis=1)
    trend["JT_z"] = jt_z
    trend["JT_p"] = jt_p
    trend["spearman_rho"] = sp_rho
    sig = trend["week_P"] < config.PVALUE_THRESHOLD
    trend["direction"] = np.where(sig, np.where(trend["week_slope"] > 0,
                                                "Increasing", "Decreasing"), "NS")
    trend = trend.sort_values("week_P")
    utils.save_tsv(trend, out, "trend_results")

    n_inc = int((trend["direction"] == "Increasing").sum())
    n_dec = int((trend["direction"] == "Decreasing").sum())
    with open(os.path.join(out, "trend_summary.txt"), "w") as f:
        f.write(f"Monotonic trend over weeks {timepoints} — {dataset}\n")
        f.write(f"criterion: limma linear-week raw P<{config.PVALUE_THRESHOLD}\n")
        f.write(f"(BH adj.P and Jonckheere-Terpstra / Spearman also reported)\n\n")
        f.write(f"Increasing: {n_inc}\nDecreasing: {n_dec}\n"
                f"NS: {len(trend) - n_inc - n_dec}\n")
    _trend_line_plots(prof, ids, labels, trend, timepoints, out, fig_mirror, dataset)
    logger.info(f"[step09] {dataset}: monotonic trend — {n_inc} increasing, "
                f"{n_dec} decreasing (limma week raw p<{config.PVALUE_THRESHOLD}).")

    state = dict(state)
    state["trend"] = trend
    state["profile_clusters"] = clu
    return state


if __name__ == "__main__":
    pass
