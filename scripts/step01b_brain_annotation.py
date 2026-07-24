"""Step 01b — mouse-brain protein annotation (runs after step01, before step02).

Classifies every raw protein against an HPA-derived reference
(`config.MOUSE_BRAIN_DB`): human gene records carrying mouse-brain RNA
regional-specificity data. Since every row in that file already has a
non-null regional-specificity value, **membership in the file = "mouse brain
protein"**; a separate "elevated" flag marks the subset with
Region-enriched/enhanced/Group-enriched status (config.BRAIN_ELEVATED_CATEGORIES).

ID mapping: our protein IDs are mouse UniProt accessions with an entry-name
label (e.g. "BL1S2") that does NOT reliably match the DB's official human gene
symbol (e.g. "BLOC1S2" for the same protein) — verified during planning. So
each accession's real gene name(s) are fetched from the UniProt REST API
(batched, cached locally) and matched case-insensitively against the DB.

This step is protein-identity-based, not detection-based, so it runs on the
FULL raw protein list (independent of any sample filter) and is a **disk-only
augmentation**: the 3 annotation columns are appended to the saved
`01_raw_summary/raw_intensity_matrix.tsv`, but `state["data_df"]` returned to
step02+ is unchanged — the existing numeric pipeline is unaffected.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

import config
import utils

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
BATCH_SIZE = 90
REQUEST_TIMEOUT = 30
HEADERS = {"User-Agent": "MousePlasmaSeer-pipeline/1.0 (research use)"}


def _load_gene_name_cache() -> dict[str, str]:
    if os.path.exists(config.UNIPROT_GENE_NAME_CACHE):
        df = pd.read_csv(config.UNIPROT_GENE_NAME_CACHE, sep="\t", dtype=str).fillna("")
        return dict(zip(df["accession"], df["gene_names"]))
    return {}


def _save_gene_name_cache(cache: dict[str, str]) -> None:
    utils.ensure_dir(config.SHARED_DIR)
    df = pd.DataFrame({"accession": list(cache.keys()), "gene_names": list(cache.values())})
    df.to_csv(config.UNIPROT_GENE_NAME_CACHE, sep="\t", index=False)


def _primary_accession(acc: str) -> str:
    """Multi-accession protein-group IDs (e.g. 'P86792;P86793') break the UniProt
    query syntax; use the first component as the representative accession to query,
    while the ORIGINAL full ID is still used as the cache/result key."""
    return acc.split(";")[0].strip()


def _query_uniprot_batch(id_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """id_pairs: list of (original_id, primary_accession_to_query). Raises on
    HTTP/network failure so the caller can retry with a smaller batch."""
    if not id_pairs:
        return {}
    query = " OR ".join(f"accession:{primary}" for _, primary in id_pairs)
    r = requests.get(UNIPROT_SEARCH_URL, params={
        "query": query, "fields": "accession,gene_names", "format": "tsv", "size": 500,
    }, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    by_primary = {}
    for line in lines[1:]:  # skip header
        parts = line.split("\t")
        if parts and parts[0]:
            by_primary[parts[0]] = parts[1] if len(parts) > 1 else ""
    return {orig: by_primary.get(primary, "") for orig, primary in id_pairs}


def _fetch_gene_names(accessions: list[str], logger) -> dict[str, str]:
    """Batched UniProt lookup: original accession -> space-separated gene-name
    synonyms. Batches that fail (e.g. one malformed ID breaking the query) are
    recursively halved and retried so a single bad ID costs at most itself, not
    the whole batch — this fixed a ~20% coverage loss seen on the first run."""
    id_pairs = [(a, _primary_accession(a)) for a in accessions]

    def _resolve(pairs: list[tuple[str, str]]) -> dict[str, str]:
        if not pairs:
            return {}
        try:
            return _query_uniprot_batch(pairs)
        except Exception as e:
            if len(pairs) == 1:
                logger.warning(f"[step01b] UniProt lookup failed for {pairs[0][0]}: {e}")
                return {pairs[0][0]: ""}
            mid = len(pairs) // 2
            return {**_resolve(pairs[:mid]), **_resolve(pairs[mid:])}

    out: dict[str, str] = {}
    chunks = [id_pairs[i:i + BATCH_SIZE] for i in range(0, len(id_pairs), BATCH_SIZE)]
    for i, chunk in enumerate(chunks, start=1):
        out.update(_resolve(chunk))
        time.sleep(0.2)  # be polite to the API
        if i % 10 == 0 or i == len(chunks):
            logger.info(f"[step01b] UniProt gene-name lookup: batch {i}/{len(chunks)} done "
                        f"({len(out)} accessions resolved so far).")
    return out


def run(dataset: str, logger, state: dict) -> dict:
    out = utils.step_path(dataset, "01_raw_summary")
    df = state["data_df"]
    accessions = df[config.ID_COL].astype(str).tolist()

    # ── 1. Resolve real gene names per accession (cached) ──────────────────
    cache = _load_gene_name_cache()
    missing = [a for a in accessions if a not in cache]
    logger.info(f"[step01b] {len(accessions)} proteins; {len(missing)} not yet in "
                f"gene-name cache ({config.UNIPROT_GENE_NAME_CACHE}).")
    if missing:
        fetched = _fetch_gene_names(missing, logger)
        cache.update(fetched)
        # Accessions UniProt didn't return (e.g. obsolete/merged) get an empty entry
        # so they aren't re-queried every run.
        for a in missing:
            cache.setdefault(a, "")
        _save_gene_name_cache(cache)
    gene_names_by_acc = {a: cache.get(a, "") for a in accessions}

    # ── 2. Load DB, build case-insensitive gene-symbol lookup ───────────────
    brain_db = pd.read_csv(config.MOUSE_BRAIN_DB, sep="\t", low_memory=False)
    gene_to_category = {}
    for _, row in brain_db[[config.BRAIN_DB_GENE_COL, config.BRAIN_DB_REGIONAL_COL]].dropna(
            subset=[config.BRAIN_DB_GENE_COL]).iterrows():
        gene_to_category[str(row[config.BRAIN_DB_GENE_COL]).upper()] = row[config.BRAIN_DB_REGIONAL_COL]

    # ── 3. Classify each protein: any gene-name synonym matches the DB? ─────
    is_brain, category, matched_gene = [], [], []
    for acc in accessions:
        synonyms = [s.upper() for s in gene_names_by_acc[acc].split() if s]
        hit = next((s for s in synonyms if s in gene_to_category), None)
        if hit:
            is_brain.append(True)
            category.append(gene_to_category[hit])
            matched_gene.append(hit)
        else:
            is_brain.append(False)
            category.append("")
            matched_gene.append("")
    is_elevated = [c in config.BRAIN_ELEVATED_CATEGORIES for c in category]

    annotation = pd.DataFrame({
        config.ID_COL: accessions,
        "uniprot_gene_names": [gene_names_by_acc[a] for a in accessions],
        "matched_brain_gene": matched_gene,
        "is_mouse_brain_protein": is_brain,
        "mouse_brain_regional_category": category,
        "is_regionally_elevated": is_elevated,
    })
    utils.save_tsv(annotation, out, "mouse_brain_annotation")

    n_brain = sum(is_brain)
    n_elevated = sum(is_elevated)
    logger.info(f"[step01b] {dataset}: {n_brain}/{len(accessions)} raw proteins classified "
                f"as mouse-brain proteins ({n_elevated} regionally elevated).")

    # ── 4. Disk-only augmentation of the already-saved raw_intensity_matrix ─
    raw_path = os.path.join(out, "raw_intensity_matrix.tsv")
    raw_df = pd.read_csv(raw_path, sep="\t")
    ann_cols = ["is_mouse_brain_protein", "mouse_brain_regional_category", "is_regionally_elevated"]
    raw_df = raw_df.merge(annotation[[config.ID_COL] + ann_cols], on=config.ID_COL, how="left")
    utils.save_tsv(raw_df, out, "raw_intensity_matrix")
    logger.info(f"[step01b] appended {ann_cols} to raw_intensity_matrix.tsv "
                f"(disk-only; state['data_df'] passed to step02+ is unchanged).")

    # state is returned unmodified — step02 onward sees the same numeric pipeline.
    return state


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import step01_raw_summary as s1
    log = utils.setup_logger("MousePlasma")
    st = s1.run("MousePlasma", log)
    run("MousePlasma", log, st)
