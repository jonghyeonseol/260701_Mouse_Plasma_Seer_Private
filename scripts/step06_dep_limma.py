"""Step 06 — DEP analysis via R/limma (rpy2): 14 contrasts, 4 analysis goals.

A SINGLE means-model is fit over all 80 biological samples:

    design = model.matrix(~0 + group + peptide_ug)      (8 group means + covariate)

so eBayes moderates variance across the whole design.  All 14 contrasts
(config.COMPARISONS) are then extracted from one contrasts.fit via makeContrasts:

  Goal 1 genotype_at_time (4):  KI_tW − WT_tW           for t ∈ {8,10,12,14}
  Goal 2 time_course      (6):  g_tW − g_8W             for g ∈ {WT,KI}, t>8
  Goal 3 interaction      (3):  (KI_tW−WT_tW) − (KI_8W−WT_8W)
  Goal 4 global           (1):  mean(KI) − mean(WT)

Per-contrast outputs: dep_results_<name>.tsv, dep_significant_<name>.tsv,
volcano_<name>.{png,svg}.  Cross-contrast: master summary, DEP count bar,
per-family volcano grids, genotype-at-time UpSet, and a global-contrast heatmap.
"""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils
from viz.viz_volcano import plot_volcano
from viz.viz_heatmap import plot_dep_heatmap


def _init_rpy2():
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr
    try:
        conv = ro.default_converter + numpy2ri.converter + pandas2ri.converter
        ro.conversion.set_conversion(conv)
    except (AttributeError, TypeError):
        numpy2ri.activate(); pandas2ri.activate()
    return ro, importr("limma")


def _fit_all_contrasts(df, meta, comparisons, logger):
    """One limma fit over all groups; return {contrast_name: results_df}."""
    ro, limma = _init_rpy2()

    bio = meta[~meta["is_ref"]].copy()
    group_order = config.PLASMA_GROUP_ORDER
    bio = bio[bio["group"].isin(group_order)]
    sample_cols = bio["column_name"].tolist()

    expr = df[sample_cols].values.astype(float)
    group_vec = bio["group"].tolist()
    pep_vec = pd.to_numeric(bio[config.PEPTIDE_COL], errors="coerce").values.astype(float)

    ro.r.assign("expr_mat", ro.r["matrix"](
        ro.FloatVector(expr.flatten("F")), nrow=expr.shape[0]))
    ro.r.assign("grp", ro.FactorVector(group_vec, levels=ro.StrVector(group_order)))
    use_pep = config.USE_PEPTIDE_COVARIATE and np.isfinite(pep_vec).all()
    if use_pep:
        ro.r.assign("pep", ro.FloatVector(pep_vec))
        ro.r("design <- model.matrix(~0 + grp + pep); "
             "colnames(design) <- c(levels(grp), 'pep')")
    else:
        ro.r("design <- model.matrix(~0 + grp); colnames(design) <- levels(grp)")
    logger.info(f"[step06] design: ~0 + group"
                f"{' + peptide_ug' if use_pep else ''} "
                f"({len(group_order)} groups, n={len(sample_cols)} samples).")

    exprs = [c["expr"] for c in comparisons]
    ro.r.assign("contr_exprs", ro.StrVector(exprs))
    ro.r("""
    fit  <- lmFit(expr_mat, design)
    cont <- makeContrasts(contrasts=contr_exprs, levels=design)
    fit2 <- eBayes(contrasts.fit(fit, cont), trend=TRUE, robust=TRUE)
    """)

    id_vals = df[config.ID_COL].values
    labels = utils.build_display_labels(df)
    # Per-group means (log2 ratio) for annotation.
    group_means = {g: np.nanmean(df[bio[bio["group"] == g]["column_name"]].values, axis=1)
                   for g in group_order}

    results = {}
    for i, comp in enumerate(comparisons, start=1):
        ro.r.assign("ci", i)
        ro.r('tt <- topTable(fit2, coef=ci, number=Inf, sort.by="none", adjust.method="BH")')
        tt = ro.r["tt"]
        if isinstance(tt, pd.DataFrame):
            res = tt[["logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B"]].reset_index(drop=True)
        else:
            res = pd.DataFrame({k: np.asarray(tt.rx2(k), dtype=float) for k in
                                ["logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B"]})
        res.insert(0, config.ID_COL, id_vals)
        res.insert(1, "label", labels)
        res.insert(2, config.DESC_COL,
                   df.get(config.DESC_COL, pd.Series([""] * len(df))).values)
        sig = (res[config.SIG_PVALUE_COL] < config.PVALUE_THRESHOLD) & \
              (res["logFC"].abs() > config.LOG2FC_THRESHOLD)
        res["Direction"] = np.where(sig, np.where(res["logFC"] > 0, "Up", "Down"), "NS")
        res["Significant"] = sig
        res["contrast"] = comp["name"]
        res["family"] = comp["family"]
        for grp in (comp["case_grp"], comp["ctrl_grp"]):
            if grp in group_means:
                res[f"mean_{grp}"] = group_means[grp]
        results[comp["name"]] = res
    return results


def _master_summary(comparisons, results, out):
    rows = []
    for comp in comparisons:
        res = results[comp["name"]]
        n_up = int((res["Direction"] == "Up").sum())
        n_dn = int((res["Direction"] == "Down").sum())
        rows.append({"contrast": comp["name"], "family": comp["family"],
                     "case": comp["case_grp"], "ctrl": comp["ctrl_grp"],
                     "n_tested": len(res), "n_up": n_up, "n_down": n_dn,
                     "n_significant": n_up + n_dn})
    summary = pd.DataFrame(rows)
    utils.save_tsv(summary, out, "dep_master_summary")
    return summary


def _count_bar(summary, out, fig_mirror):
    utils.apply_style()
    fam_color = {"genotype_at_time": "#6A51A3", "time_course": "#2171B5",
                 "interaction": "#238B45", "global": "#CB181D"}
    s = summary.iloc[::-1].reset_index(drop=True)   # top-to-bottom
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.45 * len(s) + 1.5)))
    ax.barh(y, s["n_up"], color="#FF3F3F", edgecolor="black", linewidth=0.3, label="Up")
    ax.barh(y, -s["n_down"], color="#3232FF", edgecolor="black", linewidth=0.3, label="Down")
    for i in range(len(s)):
        if s["n_up"].iloc[i]:
            ax.text(s["n_up"].iloc[i], y[i], f" {s['n_up'].iloc[i]}", va="center",
                    ha="left", fontsize=8)
        if s["n_down"].iloc[i]:
            ax.text(-s["n_down"].iloc[i], y[i], f"{s['n_down'].iloc[i]} ", va="center",
                    ha="right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(s["contrast"], fontsize=8)
    for tick, fam in zip(ax.get_yticklabels(), s["family"]):
        tick.set_color(fam_color.get(fam, "#333"))
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("← Down          Number of DEPs          Up →")
    ax.set_title(f"DEP counts per contrast\n({config.sig_criterion_label()})", fontsize=11)
    from matplotlib.patches import Patch
    fam_handles = [Patch(color=c, label=f) for f, c in fam_color.items()]
    ax.legend(handles=fam_handles, loc="lower right", fontsize=8, title="family (label color)")
    png = utils.save_fig(fig, out, "dep_count_bar")
    plt.close(fig)
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


def _volcano_grid(comps, results, label, out, fig_mirror):
    """Volcano grid for an explicit list of contrasts (kept ≤6 wide for legibility)."""
    if not comps:
        return
    utils.apply_style()
    n = len(comps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    for ax, comp in zip(axes[0], comps):
        res = results[comp["name"]]
        x = res["logFC"].values
        y = -np.log10(res[config.SIG_PVALUE_COL].clip(lower=1e-300).values)
        sig = res["Significant"].values
        up = sig & (res["logFC"] > 0).values
        dn = sig & (res["logFC"] < 0).values
        ax.scatter(x[~sig], y[~sig], s=8, color="#95A5A6", alpha=0.4, linewidths=0)
        ax.scatter(x[dn], y[dn], s=16, color="#3232FF", alpha=0.75, linewidths=0)
        ax.scatter(x[up], y[up], s=16, color="#FF3F3F", alpha=0.75, linewidths=0)
        ax.axhline(-np.log10(config.PVALUE_THRESHOLD), color="gray", ls="--", lw=0.8)
        ax.axvline(config.LOG2FC_THRESHOLD, color="gray", ls="--", lw=0.8)
        ax.axvline(-config.LOG2FC_THRESHOLD, color="gray", ls="--", lw=0.8)
        ax.set_xlabel(f"log2FC ({comp['case_grp']}/{comp['ctrl_grp']})", fontsize=10)
        ax.set_ylabel(f"-log10({'raw p' if config.SIG_PVALUE_COL == 'P.Value' else 'adj.p'})")
        ax.set_title(f"{comp['name']}\n↑{int(up.sum())} / ↓{int(dn.sum())}", fontsize=10)
    fig.suptitle(f"Volcano grid — {label}", y=1.02, fontsize=13, fontweight="semibold")
    png = utils.save_fig(fig, out, f"volcano_grid_{label}", tight=False)
    plt.close(fig)
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


def _upset(comps, results, label, title, out, fig_mirror, strip_prefix=""):
    """UpSet of DEP-set intersections for an explicit list of contrasts."""
    try:
        from upsetplot import UpSet, from_contents
    except Exception:
        return
    contents = {}
    for comp in comps:
        res = results[comp["name"]]
        key = comp["name"].replace(strip_prefix, "") if strip_prefix else comp["name"]
        contents[key] = res.loc[res["Significant"], config.ID_COL].astype(str).tolist()
    if not any(contents.values()):
        return
    data = from_contents(contents)
    fig = plt.figure(figsize=(max(9, 1.1 * len(comps) + 4), 5.5))
    UpSet(data, subset_size="count", show_counts=True, sort_by="cardinality",
          min_subset_size=1).plot(fig=fig)
    fig.suptitle(title, y=1.02, fontsize=12, fontweight="semibold")
    png = utils.save_fig(fig, out, f"dep_upset_{label}", tight=False)
    plt.close(fig)
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))


def _dep_union_overview(comparisons, results, out, fig_mirror, logger):
    """Total-DEP overview across all contrasts: per-protein recurrence table +
    a bar of union DEP size per family and overall unique union."""
    families = ["genotype_at_time", "time_course", "interaction", "global"]
    # Per-protein: in how many contrasts / which families it is a DEP.
    rec = {}
    fam_sets = {f: set() for f in families}
    for comp in comparisons:
        res = results[comp["name"]]
        for pid, lab in zip(res.loc[res["Significant"], config.ID_COL].astype(str),
                            res.loc[res["Significant"], "label"]):
            r = rec.setdefault(pid, {"label": lab, "n_contrasts": 0,
                                     "families": set(), "contrasts": []})
            r["n_contrasts"] += 1
            r["families"].add(comp["family"])
            r["contrasts"].append(comp["name"])
            fam_sets[comp["family"]].add(pid)
    rows = [{config.ID_COL: pid, "label": r["label"], "n_contrasts": r["n_contrasts"],
             "n_families": len(r["families"]),
             "families": ";".join(sorted(r["families"])),
             "contrasts": ";".join(r["contrasts"])}
            for pid, r in rec.items()]
    overview = pd.DataFrame(rows).sort_values(
        ["n_contrasts", "n_families"], ascending=False) if rows else pd.DataFrame(
        columns=[config.ID_COL, "label", "n_contrasts", "n_families", "families", "contrasts"])
    utils.save_tsv(overview, out, "dep_union_overview")

    total_union = len(rec)
    utils.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # (a) union DEP size per family (+ overall unique union)
    labels = families + ["ALL (unique)"]
    vals = [len(fam_sets[f]) for f in families] + [total_union]
    colors = ["#6A51A3", "#2171B5", "#238B45", "#CB181D", "#333333"]
    axes[0].bar(labels, vals, color=colors, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Unique DEP proteins")
    axes[0].set_title("Union DEP size per family")
    axes[0].tick_params(axis="x", rotation=25)
    # (b) DEP recurrence: how many proteins are DEP in exactly k contrasts
    if not overview.empty:
        rc = overview["n_contrasts"].value_counts().sort_index()
        axes[1].bar(rc.index, rc.values, color="#0072B2", edgecolor="black", linewidth=0.4)
        axes[1].set_xlabel("# contrasts a protein is a DEP in")
        axes[1].set_ylabel("Protein count")
        axes[1].set_title("DEP recurrence across contrasts")
        axes[1].set_xticks(range(1, int(rc.index.max()) + 1))
    fig.suptitle(f"Total DEP overview — {total_union} unique DEP proteins "
                 f"across {len(comparisons)} contrasts",
                 y=1.03, fontsize=13, fontweight="semibold")
    png = utils.save_fig(fig, out, "dep_union_overview", tight=False)
    plt.close(fig)
    for ext in (".png", ".svg"):
        src = png.replace(".png", ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))
    logger.info(f"[step06] union overview: {total_union} unique DEP proteins; "
                f"per-family unique = "
                f"{ {f: len(fam_sets[f]) for f in families} }.")


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    comparisons = cfg["comparisons"]
    out = utils.step_path(dataset, "06_dep_limma")
    fig_mirror = utils.figures_dir(dataset)
    utils.apply_style()

    df = state["data_df"].copy()
    meta = state["meta"]

    results = _fit_all_contrasts(df, meta, comparisons, logger)

    for comp in comparisons:
        res = results[comp["name"]]
        n_up = int((res["Direction"] == "Up").sum())
        n_dn = int((res["Direction"] == "Down").sum())
        logger.info(f"[step06] {comp['name']} ({comp['family']}): "
                    f"↑{n_up} ↓{n_dn} (of {len(res)}).")
        utils.save_tsv(res, out, f"dep_results_{comp['name']}")
        utils.save_tsv(res[res["Significant"]].sort_values(config.SIG_PVALUE_COL),
                       out, f"dep_significant_{comp['name']}")
        plot_volcano(res, comparison_name=comp["name"], output_dir=out,
                     case_grp=comp["case_grp"], ctrl_grp=comp["ctrl_grp"],
                     figures_mirror=fig_mirror)

    summary = _master_summary(comparisons, results, out)
    _count_bar(summary, out, fig_mirror)

    # Volcano grids: genotype_at_time (4) and interaction (3) as single grids;
    # time_course (12) split per genotype so each grid stays ≤6 panels wide.
    gat = [c for c in comparisons if c["family"] == "genotype_at_time"]
    inter = [c for c in comparisons if c["family"] == "interaction"]
    tc_wt = [c for c in comparisons if c["family"] == "time_course"
             and c["case_grp"].startswith("WT")]
    tc_ki = [c for c in comparisons if c["family"] == "time_course"
             and c["case_grp"].startswith("KI")]
    _volcano_grid(gat, results, "genotype_at_time", out, fig_mirror)
    _volcano_grid(inter, results, "interaction", out, fig_mirror)
    _volcano_grid(tc_wt, results, "time_course_WT", out, fig_mirror)
    _volcano_grid(tc_ki, results, "time_course_KI", out, fig_mirror)

    # Per-family UpSets (total-DEP overview).
    _upset(gat, results, "genotype_at_time",
           "KI-vs-WT DEP overlap across timepoints", out, fig_mirror,
           strip_prefix="KI_vs_WT_")
    _upset(tc_wt, results, "time_course_WT",
           "WT time-course DEP overlap (pairwise)", out, fig_mirror,
           strip_prefix="WT_")
    _upset(tc_ki, results, "time_course_KI",
           "KI time-course DEP overlap (pairwise)", out, fig_mirror,
           strip_prefix="KI_")
    _dep_union_overview(comparisons, results, out, fig_mirror, logger)

    # Global KI-vs-WT DEP heatmap across all 8 groups.
    plot_dep_heatmap(df, meta, results["KI_vs_WT_global"], output_dir=out,
                     name="heatmap_top_deps_global", group_order=config.PLASMA_GROUP_ORDER,
                     title="Top DEPs — global KI vs WT (all groups)", top_n=60)
    for ext in (".png", ".svg"):
        src = os.path.join(out, f"heatmap_top_deps_global{ext}")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(fig_mirror, os.path.basename(src)))

    with open(os.path.join(out, "dep_count_summary.txt"), "w") as f:
        f.write(f"DEP criterion: {config.sig_criterion_label()}\n"
                f"(adj.P.Val [BH] also reported per protein but NOT used for calling)\n"
                f"model: ~0 + group"
                f"{' + peptide_ug' if config.USE_PEPTIDE_COVARIATE else ''}\n\n")
        f.write(summary.to_string(index=False) + "\n")

    logger.info(f"[step06] {dataset}: {len(comparisons)} contrasts complete; "
                f"total DEPs = {int(summary['n_significant'].sum())}.")
    state = dict(state)
    state["deps_by_comparison"] = results
    state["comparisons"] = comparisons
    state["dep_summary"] = summary
    return state


if __name__ == "__main__":
    pass
