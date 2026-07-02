"""Shared utilities: IO, sample parsing, metadata join, normalization, plotting.

Adapted from KBSI_FAIMS_SC_260513/scripts/utils.py for the bulk mouse plasma
SEER dataset.  The main additions are:
  - ``parse_plasma_columns``  : genotype × timepoint parsing (+ pooled ref).
  - ``load_sample_info``      : join the plasma_sample_information.xlsx metadata
                                (well location + peptide recovery µg).
  - ``reference_ratio_log2``  : per-protein ratio to the pooled reference sample.
  - ``build_display_labels``  : protein display label from PG.ProteinNames.
"""

from __future__ import annotations

import os
import re
import logging
import datetime as _dt
from typing import Iterable

import numpy as np
import pandas as pd

import config


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def step_path(dataset: str, step_key: str) -> str:
    root = config.DATASETS[dataset]["output_root"]
    return ensure_dir(os.path.join(root, config.STEP_DIRS[step_key]))


def figures_dir(dataset: str) -> str:
    return ensure_dir(os.path.join(config.DATASETS[dataset]["output_root"], "figures"))


def setup_logger(dataset: str) -> logging.Logger:
    ensure_dir(config.LOG_DIR)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.LOG_DIR, f"run_{ts}_{dataset}.log")
    logger = logging.getLogger(f"plasma.{dataset}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logger initialized for dataset='{dataset}' -> {log_path}")
    return logger


# ── Plot style ───────────────────────────────────────────────────────────────

def apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [config.FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size": config.TICK_SIZE,
        "axes.titlesize": config.TITLE_SIZE,
        "axes.titleweight": "semibold",
        "axes.labelsize": config.LABEL_SIZE,
        "xtick.labelsize": config.TICK_SIZE,
        "ytick.labelsize": config.TICK_SIZE,
        "legend.fontsize": config.LEGEND_SIZE,
        "legend.frameon": False,
        "legend.handletextpad": 0.4,
        "figure.dpi": 110,
        "savefig.dpi": config.DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def heatmap_col_colors(bio: pd.DataFrame, sample_ids: list[str],
                       which=("Genotype", "Timepoint", "Group")) -> pd.DataFrame:
    """Column-annotation colour DataFrame (Genotype/Timepoint/Group) for a
    clustermap, indexed by sample_id."""
    m = bio.set_index("sample_id")
    cols = {}
    if "Genotype" in which:
        cols["Genotype"] = [config.GENOTYPE_COLORS.get(m.loc[s, "genotype"], "#888")
                            for s in sample_ids]
    if "Timepoint" in which:
        cols["Timepoint"] = [config.TIMEPOINT_COLORS.get(int(m.loc[s, "timepoint"]), "#888")
                             for s in sample_ids]
    if "Group" in which:
        cols["Group"] = [config.GROUP_COLORS.get(m.loc[s, "group"], "#888")
                         for s in sample_ids]
    return pd.DataFrame(cols, index=sample_ids)


def zscore_cmap():
    """Diverging colormap for expression z-scores (config.ZSCORE_COLORS)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("z_gbr", config.ZSCORE_COLORS)


def compute_embeddings(X: np.ndarray, seed: int = config.RANDOM_SEED, n_pcs: int = 10):
    """Standardize (per feature) then compute PCA + t-SNE + UMAP embeddings.

    X is samples × features.  Sample-count-adaptive t-SNE perplexity and UMAP
    neighbours so the same helper serves the full cohort (step07) and small
    2-group pairwise subsets (step10).  Returns a dict of embeddings + PCA
    variance-explained (%) + the perplexity used.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    Xs = StandardScaler().fit_transform(X)
    n = Xs.shape[0]
    pca = PCA(n_components=min(n_pcs, n - 1), random_state=seed)
    pcs = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_ * 100
    perp = max(5, min(30, (n - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=seed, learning_rate="auto").fit_transform(Xs)
    umap_emb = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.3,
                         random_state=seed).fit_transform(Xs)
    return {"pcs": pcs, "var": var, "tsne": tsne, "umap": umap_emb, "perplexity": perp}


def add_heatmap_annotation_legends(clustergrid, which=("Genotype", "Timepoint", "Group"),
                                   x: float = 1.04, cbar_pos=(0.015, 0.905, 0.018, 0.065),
                                   cbar_title: str = "Row z-score", group_subset=None):
    """Place stacked, titled annotation legends to the RIGHT of a seaborn
    clustermap (one per annotation strip) and move the colorbar to a clean
    top-left slot, so nothing overlaps the dendrogram, labels or colour bar.

    ``x`` is in heatmap-axes fraction (increase it when row tick-labels are shown).
    """
    from matplotlib.patches import Patch
    from matplotlib.legend import Legend
    group_list = group_subset if group_subset else config.PLASMA_GROUP_ORDER
    specs = {
        "Genotype": [(gt, config.GENOTYPE_COLORS[gt]) for gt in config.GENOTYPES],
        "Timepoint": [(f"{t}W", config.TIMEPOINT_COLORS[t]) for t in config.TIMEPOINTS],
        "Group": [(config.GROUP_DISPLAY[g], config.GROUP_COLORS[g]) for g in group_list],
    }
    # Vertical anchor (axes-fraction top) per legend; sized so the 8-row Group
    # legend has room beneath the shorter Genotype/Timepoint legends.
    tops = {"Genotype": 0.99, "Timepoint": 0.83, "Group": 0.58}
    ax = clustergrid.ax_heatmap
    legends = []
    for title in which:
        handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in specs[title]]
        leg = Legend(ax, handles, [h.get_label() for h in handles], title=title,
                     loc="upper left", bbox_to_anchor=(x, tops[title]),
                     bbox_transform=ax.transAxes, frameon=False,
                     fontsize=7, title_fontsize=8.5, handlelength=1.1,
                     handleheight=1.1, labelspacing=0.3, borderaxespad=0.0)
        leg.get_title().set_fontweight("bold")
        ax.add_artist(leg)
        legends.append(leg)
    if getattr(clustergrid, "ax_cbar", None) is not None:
        clustergrid.ax_cbar.set_position(list(cbar_pos))
        clustergrid.ax_cbar.set_title(cbar_title, fontsize=8, pad=4)
    # Returned so callers can pass them to savefig(bbox_extra_artists=...) — a
    # tight bbox otherwise clips legends anchored outside the heatmap axes.
    return legends


def set_clustermap_title(clustergrid, title: str):
    """Place a clustermap title above the column dendrogram (reliably visible on
    tall clustermaps, where a figure suptitle can fall outside the tight bbox).
    Returns the title artist so it can be added to savefig's extra_artists."""
    if getattr(clustergrid, "ax_col_dendrogram", None) is not None:
        clustergrid.ax_col_dendrogram.set_title(
            title, fontsize=config.TITLE_SIZE, pad=12, fontweight="semibold")
        return clustergrid.ax_col_dendrogram.title
    return clustergrid.fig.suptitle(title, y=1.02, fontsize=config.TITLE_SIZE)


def save_clustermap(clustergrid, out_dir: str, name: str, extra_artists=None):
    """Save a seaborn clustermap as PNG+SVG, guaranteeing outside-axes legends
    are included in the tight bounding box."""
    ensure_dir(out_dir)
    png = os.path.join(out_dir, f"{name}.png")
    kw = dict(bbox_inches="tight")
    if extra_artists:
        kw["bbox_extra_artists"] = extra_artists
    clustergrid.savefig(png, dpi=config.DPI, **kw)
    clustergrid.savefig(png.replace(".png", ".svg"), **kw)
    return png


def save_fig(fig, out_dir: str, name: str, also_svg: bool = True, tight: bool = True):
    ensure_dir(out_dir)
    if tight:
        fig.tight_layout()
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(png, dpi=config.DPI, bbox_inches="tight")
    if also_svg:
        fig.savefig(os.path.join(out_dir, f"{name}.svg"), bbox_inches="tight")
    return png


def save_tsv(df: pd.DataFrame, out_dir: str, name: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{name}.tsv")
    df.to_csv(path, sep="\t", index=False)
    return path


# ── Sample / column parsing ──────────────────────────────────────────────────

def parse_plasma_columns(cols: Iterable[str]) -> pd.DataFrame:
    """Parse plasma sample columns into structured metadata.

    Biological columns → genotype (WT/KI) × timepoint (weeks).  The single
    "Total_pooled_ref" column is tagged with group=``config.REF_GROUP`` and
    ``is_ref=True`` (genotype/timepoint NaN).  Unmatched columns are skipped.
    """
    rows = []
    for c in cols:
        m = config.PLASMA_COL_PATTERN.match(c)
        if m:
            idx, date, weeks, geno, num = m.groups()
            tp = int(weeks)
            grp = f"{geno}_{tp}W"
            rows.append({
                "column_name":  c,
                "sample_id":    f"{geno}_{tp}W_{int(num)}",
                "index":        int(idx),
                "date":         date,
                "genotype":     geno,
                "timepoint":    tp,
                "sample_number": int(num),
                "group":        grp,
                "is_ref":       False,
            })
            continue
        r = config.PLASMA_REF_PATTERN.match(c)
        if r:
            idx, date = r.groups()
            rows.append({
                "column_name":  c,
                "sample_id":    "pooled_ref",
                "index":        int(idx),
                "date":         date,
                "genotype":     np.nan,
                "timepoint":    np.nan,
                "sample_number": 0,
                "group":        config.REF_GROUP,
                "is_ref":       True,
            })
    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    return df


_XLSX_SAMPLE_PATTERN = re.compile(r"(\d+)\s*week\s*(WT|KI)\s*(\d+)", re.IGNORECASE)


def load_sample_info(dataset: str) -> pd.DataFrame:
    """Parse plasma_sample_information.xlsx → per-sample well + peptide µg.

    Returns a DataFrame keyed by (genotype, timepoint, sample_number) with
    ``well`` and ``peptide_ug`` columns, plus a single pooled-ref row
    (is_ref=True).  Column headers in the workbook live on the 2nd physical row.
    """
    path = config.DATASETS[dataset]["sample_info"]
    raw = pd.read_excel(path, header=1)
    # Resolve columns by fuzzy header match (workbook labels vary slightly).
    def _find(sub):
        for c in raw.columns:
            if sub.lower() in str(c).lower():
                return c
        return None
    c_sample = _find("sample")
    c_well = _find("well")
    c_pep = _find("peptide")
    if c_sample is None or c_pep is None:
        raise ValueError(f"Could not locate Sample/peptide columns in {path}: "
                         f"{list(raw.columns)}")

    rows = []
    for _, r in raw.iterrows():
        s = str(r[c_sample]).strip()
        pep = pd.to_numeric(r[c_pep], errors="coerce")
        well = str(r[c_well]).strip() if c_well is not None else ""
        m = _XLSX_SAMPLE_PATTERN.search(s)
        if m:
            tp, geno, num = int(m.group(1)), m.group(2).upper(), int(m.group(3))
            rows.append({"genotype": geno, "timepoint": tp, "sample_number": num,
                         "well": well, "peptide_ug": pep, "is_ref": False})
        elif "pool" in s.lower():
            rows.append({"genotype": np.nan, "timepoint": np.nan,
                         "sample_number": 0, "well": well, "peptide_ug": pep,
                         "is_ref": True})
    return pd.DataFrame(rows)


def split_sample_columns(meta: pd.DataFrame) -> tuple[list[str], str | None]:
    """Return (biological_sample_columns, pooled_ref_column)."""
    bio = meta.loc[~meta["is_ref"], "column_name"].tolist()
    ref = meta.loc[meta["is_ref"], "column_name"]
    return bio, (ref.iloc[0] if len(ref) else None)


def ordered_bio_cols(meta: pd.DataFrame, sample_cols: list[str],
                     group_order: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Biological sample columns ordered by group_order then sample_number."""
    rank = {g: i for i, g in enumerate(group_order)}
    sub = meta[meta["column_name"].isin(sample_cols) & ~meta["is_ref"]].copy()
    sub["_rank"] = sub["group"].map(rank).fillna(99).astype(int)
    sub = sub.sort_values(["_rank", "sample_number"])
    return sub["column_name"].tolist(), sub


def get_sample_metadata(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    """Structured metadata for all sample columns, joined with xlsx info."""
    meta_cols = config.DATASETS[dataset]["meta_cols"]
    sample_cols = [c for c in df.columns if c not in meta_cols]
    meta = parse_plasma_columns(sample_cols)

    info = load_sample_info(dataset)
    bio_info = info[~info["is_ref"]][
        ["genotype", "timepoint", "sample_number", "well", "peptide_ug"]]
    meta = meta.merge(bio_info, on=["genotype", "timepoint", "sample_number"],
                      how="left")
    # Attach pooled-ref well/peptide to the ref row.
    ref_info = info[info["is_ref"]]
    if not ref_info.empty and (meta["is_ref"]).any():
        meta.loc[meta["is_ref"], "well"] = ref_info["well"].iloc[0]
        meta.loc[meta["is_ref"], "peptide_ug"] = ref_info["peptide_ug"].iloc[0]

    # Plate row/column decomposition from the well (e.g. "A5" → row A, col 5).
    def _row(w):
        w = str(w)
        return w[0] if w and w[0].isalpha() else np.nan
    def _col(w):
        m = re.search(r"(\d+)", str(w))
        return int(m.group(1)) if m else np.nan
    meta["plate_row"] = meta["well"].map(_row)
    meta["plate_col"] = meta["well"].map(_col)
    return meta


# ── Data IO ──────────────────────────────────────────────────────────────────

def load_raw(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (data_df, sample_meta_df).  data_df keeps meta_cols + sample cols."""
    cfg = config.DATASETS[dataset]
    df = pd.read_csv(cfg["input"], sep="\t")
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    meta = get_sample_metadata(dataset, df)
    return df, meta


def subset_by_genotype(df: pd.DataFrame, meta: pd.DataFrame, genotype: str
                       ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str | None]:
    """Subset the full matrix/metadata to one genotype's biological samples,
    keeping the pooled-ref column (needed for reference-ratio normalization).

    Returns (sub_df, sub_meta, sample_cols, ref_col) — a state analogous to
    step01's output, ready to feed step02 onward for a branch project.
    """
    bio = meta[(~meta["is_ref"]) & (meta["genotype"] == genotype)].copy()
    ref = meta[meta["is_ref"]].copy()
    sub_meta = pd.concat([bio, ref], ignore_index=True)
    sample_cols = bio["column_name"].tolist()
    ref_col = ref["column_name"].iloc[0] if len(ref) else None

    meta_cols = [c for c in config.PLASMA_META_COLS if c in df.columns]
    keep = meta_cols + sample_cols + ([ref_col] if ref_col else [])
    sub_df = df[keep].copy()
    return sub_df, sub_meta, sample_cols, ref_col


def to_numeric_intensity(df: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    """Coerce sample cols to float; treat sentinels and non-positive as missing."""
    df = df.copy()
    df[sample_cols] = df[sample_cols].replace(
        {"NaN": np.nan, "Filtered": np.nan, "": np.nan, "NA": np.nan})
    for c in sample_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[sample_cols] = df[sample_cols].mask(df[sample_cols] <= 0)
    return df


# ── Display labels ───────────────────────────────────────────────────────────

def build_display_labels(df: pd.DataFrame) -> list[str]:
    """Protein display label: UniProt entry name (PG.ProteinNames) with the
    trailing ``_MOUSE`` organism suffix stripped; falls back to the accession.
    Multi-protein groups keep only the first entry."""
    names = df.get(config.NAME_COL, pd.Series([""] * len(df))).astype(str)
    ids = df[config.ID_COL].astype(str)
    out = []
    for nm, acc in zip(names, ids):
        first = nm.split(";")[0].strip()
        if first and first.lower() != "nan":
            out.append(re.sub(r"_MOUSE$", "", first))
        else:
            out.append(acc.split(";")[0])
    return out


# ── Filtering ────────────────────────────────────────────────────────────────

def valid_per_group(values: np.ndarray, meta: pd.DataFrame, sample_cols: list[str],
                    groups: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for g in groups:
        cols = meta[meta["group"] == g]["column_name"].tolist()
        idx = [sample_cols.index(c) for c in cols]
        sub = values[:, idx]
        out[g] = np.sum(~np.isnan(sub), axis=1) / max(len(idx), 1)
    return out


def filter_valid_fraction(df: pd.DataFrame, meta: pd.DataFrame, sample_cols: list[str],
                          groups: list[str], min_frac: float = config.VALID_FRACTION,
                          mode: str = config.VALID_FRACTION_MODE
                          ) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Filter proteins by per-group valid fraction (≥min_frac in ALL/ANY group)."""
    values = df[sample_cols].values.astype(float)
    per_group = valid_per_group(values, meta, sample_cols, groups)
    if mode == "all":
        keep = np.ones(values.shape[0], dtype=bool)
        for frac in per_group.values():
            keep &= (frac >= min_frac)
    elif mode == "any":
        keep = np.zeros(values.shape[0], dtype=bool)
        for frac in per_group.values():
            keep |= (frac >= min_frac)
    else:
        raise ValueError(f"Unknown mode={mode!r}; expected 'all' or 'any'.")
    stats = {
        "n_before": int(values.shape[0]),
        "n_after": int(keep.sum()),
        "n_removed": int((~keep).sum()),
        "min_fraction": min_frac,
        "mode": mode,
    }
    return df.loc[keep].reset_index(drop=True), keep, stats


# ── Normalization ────────────────────────────────────────────────────────────

def reference_ratio_log2(df_log2: pd.DataFrame, sample_cols: list[str], ref_col: str
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-protein reference-ratio on log2 data: subtract the pooled-ref log2
    value from every biological sample (⇒ log2(sample / pooled_ref)).

    For proteins where the pooled ref is missing, subtract the per-protein mean
    of the observed biological samples instead, so the row is still centered
    (documented in the returned reference table via ``ref_source``).
    """
    mat = df_log2[sample_cols].values.astype(float)
    ref = df_log2[ref_col].values.astype(float)
    row_mean = np.nanmean(mat, axis=1)
    ref_source = np.where(np.isnan(ref), "row_mean_fallback", "pooled_ref")
    offset = np.where(np.isnan(ref), row_mean, ref)
    offset = np.where(np.isnan(offset), 0.0, offset)   # fully-missing row → no shift
    mat_ratio = mat - offset[:, None]
    out = df_log2.copy()
    out[sample_cols] = mat_ratio
    ref_tbl = pd.DataFrame({
        config.ID_COL: df_log2[config.ID_COL].values,
        "ref_log2": ref,
        "offset_subtracted": offset,
        "ref_source": ref_source,
    })
    return out, ref_tbl


def median_normalize_log2(df_log2: pd.DataFrame, sample_cols: list[str]
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Center each sample's log2 distribution to the overall median across samples."""
    mat = df_log2[sample_cols].values.astype(float)
    per_sample_med = np.nanmedian(mat, axis=0)
    global_med = np.nanmedian(per_sample_med)
    factors = per_sample_med - global_med
    mat_norm = mat - factors[None, :]
    out = df_log2.copy()
    out[sample_cols] = mat_norm
    fdf = pd.DataFrame({
        "column_name": sample_cols,
        "sample_median_log2": per_sample_med,
        "global_median_log2": global_med,
        "shift_subtracted": factors,
    })
    return out, fdf
