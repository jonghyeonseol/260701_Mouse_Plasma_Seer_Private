# 260701_Mouse_Plasma_Seer_Private

Mouse plasma proteomics analysis pipeline for a SEER Proteograph dataset — a
2-factor longitudinal design (genotype × timepoint) built on top of the
KBSI FAIMS reference pipeline (`KBSI_FAIMS_SC_260513`).

## 인수인계 요약 (Handover summary)

- **목적**: 마우스 혈장 SEER Proteograph 데이터(genotype × timepoint, 8군)에
  대해 전처리 → DEP(limma) → 차원축소/클러스터링 → 시계열 트렌드까지 이어지는
  재현 가능한 분석 파이프라인을 스크립트(`scripts/step0*_*.py`)로 관리한다.
  각 단계 산출물은 `output_data/<Project>/`에, 결과를 독립적으로 재검증하는
  스크립트는 `tests/`에 있다.
- **현재 상태 (2026-07-23 기준)**: 파이프라인 01–12 전 단계 구현 완료. 뇌 단백질
  주석(01b) 및 그에 기반한 재분석(11, 12)이 로컬 작업트리에 추가되어 있으나
  **아직 커밋되지 않은 상태**(`git status`로 확인 가능) — 커밋은 프로젝트 담당자가
  직접 진행할 예정이다. 데이터 무결성 검증 5종은 전부 통과하는 것을 전제로 하며,
  미해결 항목(샘플 페어 이상치 1건, `claudedocs/DATA_INTEGRITY_PM_PLAN.md` Phase A)이
  하나 남아 있다.
- **다음 담당자가 먼저 볼 것**: 아래 `## Changelog`, 그리고
  `claudedocs/DATA_INTEGRITY_PM_PLAN.md`(진행 중인 거버넌스 계획),
  `claudedocs/DATA_INTEGRITY_REPORT.md`(무결성 검증 상세 결과).

## Data

- **Input**: `input_data/20260630_142950_20260529_Report.tsv` — Spectronaut PG-level
  export, 9,769 protein groups × 81 quantity columns (80 biological samples + 1
  pooled reference), plus `input_data/plasma_sample_information.xlsx` (well
  location, peptide recovery µg).
- **Design**: genotype `{WT, KI}` × timepoint `{8, 10, 12, 14} weeks` → 8 groups
  (`WT_8W` … `KI_14W`), unbalanced replicate counts (8–12 per group).

## Pipeline structure

Three project scopes share the same step scripts (`scripts/step0*_*.py`,
`utils.py`, `config.py`):

| Project | Scope | Orchestrator | Output root |
|---|---|---|---|
| **Main** | all 8 groups | `python3 scripts/run_pipeline.py` | `output_data/MousePlasma/` |
| **WT branch** | WT only, 4 timepoints, re-filtered | `python3 scripts/run_branch.py --dataset MousePlasma_WT` | `output_data/MousePlasma_WT/` |
| **KI branch** | KI only, 4 timepoints, re-filtered | `python3 scripts/run_branch.py --dataset MousePlasma_KI` | `output_data/MousePlasma_KI/` |

`python3 scripts/run_branch.py` (no `--dataset`) runs both branches.

### Steps

| Step | Script | Main | Branch | Description |
|---|---|:---:|:---:|---|
| 01 | `step01_raw_summary.py` | ✅ | — | Raw QC: protein IDs/sample, CV distribution, missing-value heatmap, peptide-recovery vs depth, plate positional (batch) check |
| 01b | `step01b_brain_annotation.py` | ✅ | — | Mouse-brain protein classification vs an HPA reference DB (`database/`); disk-only — appends 3 annotation columns to `raw_intensity_matrix.tsv` without altering the numeric pipeline |
| 02 | `step02_filter.py` | ✅ | ✅ | Per-group valid-fraction filter (≥50% detected in **every** group) |
| 03 | `step03_log2_transform.py` | ✅ | ✅ | log2 transform |
| 04 | `step04_normalize.py` | ✅ | ✅ | Reference-ratio (÷ pooled ref) + sample-wise median normalization |
| 05 | `step05_impute.py` | ✅ | ✅ | MinProb imputation (per-protein left-censored Gaussian downshift) |
| 06 | `step06_dep_limma.py` | ✅ | — | limma DEP: single means-model fit (`~0 + group + peptide_ug`), **20 contrasts** (see below), volcano plots, UpSet overlaps, union-DEP overview |
| 07 | `step07_dim_reduction.py` | ✅ | ✅ | PCA / t-SNE / UMAP, colored by design axes (branch: timepoint only) |
| 08 | `step08_clustering.py` | ✅ | ✅ | Sample correlation clustermap + dendrogram, ARI/purity vs design labels, top-variance & union-DEP feature heatmaps |
| 09 | `step09_trend_timeseries.py` | — | ✅ | Fuzzy c-means temporal-profile clustering + monotonic trend test (limma linear-week + Jonckheere–Terpstra + Spearman), R/ComplexHeatmap `trend_heatmap` |
| 10 | `step10_pairwise.py` | ✅ | — | Per-pair (2-group) PCA/t-SNE/UMAP + DEP/top-500-variance/total-feature heatmaps, silhouette separation summary |
| 11 | `step11_brain_filtered.py` | ✅ | — | Re-runs dim-reduction + clustering on the mouse-brain-protein subset (step01b) only, to test whether restricting to brain-associated plasma proteins changes clustering/separation vs. the whole proteome |
| 12 | `step12_regionally_elevated.py` | ✅ | — | Narrower companion to step11: re-runs dim-reduction + clustering on just the "regionally elevated" genes (`config.BRAIN_ELEVATED_CATEGORIES`, 763 in the DB) that are actually detected and QC-filtered in this data; reports DB→real-data coverage |

### Mouse-brain protein classification (step01b/step11/step12)

`database/260525_v25.1_mouse_brain_category_rna_Any_Region.tsv` is an HPA export of
**human** gene records (`Gene`/`Ensembl`/`Uniprot` columns) carrying RNA regional-
specificity data measured in **mouse brain** tissue. Since our protein IDs are mouse
UniProt accessions, step01b fetches each accession's real gene name(s) via the UniProt
REST API (batched, cached to `output_data/_shared/uniprot_gene_names_cache.tsv`) and
matches case-insensitively against the DB — matching on our display label (a UniProt
entry-name mnemonic, e.g. `BL1S2`) instead was verified during development to silently
miss real matches (e.g. that same protein's official gene name is `Bloc1s2` →
`BLOC1S2`, which **is** in the DB). Presence in the DB = "mouse brain protein";
`config.BRAIN_ELEVATED_CATEGORIES` (`Region enriched`/`Region enhanced`/`Group
enriched`) = "regionally elevated".

**DB → real-data coverage** (logged by step12, saved to
`12_regionally_elevated/regionally_elevated_coverage_report.tsv`): the DB has
**763** unique regionally-elevated genes (333 + 266 + 164, no duplicates, confirmed
2026-07-13). Of those, **259 genes (260 proteins, 33.9% of the DB)** match at least
one protein in the raw 9,769-protein report, and **180 genes (181 proteins, 23.6% of
the DB)** survive step02's QC filter and are what step12 actually analyzes. The gap
reflects real biology/coverage (many brain-enriched genes are simply not abundant
enough to be detected in bulk plasma) rather than a mapping failure — see step01b's
UniProt-based matching methodology above.

### DEP contrast matrix (main, 20 contrasts)

| Family | Count | Contrasts |
|---|:---:|---|
| `genotype_at_time` | 4 | KI vs WT @ 8W, 10W, 12W, 14W |
| `time_course` | 12 | All pairwise timepoints within each genotype (6 per genotype) |
| `interaction` | 3 | (KI−WT)@t − (KI−WT)@8W, for t = 10/12/14W |
| `global` | 1 | KI(all) vs WT(all), time averaged out |

Significance: `raw p<0.05, |log2FC|>1.0` (config: `SIG_PVALUE_COL`/`PVALUE_THRESHOLD`/
`LOG2FC_THRESHOLD` in `scripts/config.py`); BH-adjusted p is always reported alongside.

## Key design choices

- **Normalization**: pooled-reference ratio (per-protein, log2 scale) followed by
  sample-median centering — anchors proteins to a common reference while removing
  per-sample loading differences.
- **Peptide covariate**: total µg peptide recovered is included as a limma design
  covariate (`peptide_ug`), since it correlates with proteome depth.
- **Branch re-filtering**: WT/KI branches re-apply the 50%-valid-fraction filter
  within their own 4-group subset (not inherited from the main 8-group filter),
  since the constraining group set changes.
- **z-score colormap**: green–black–red across all clustermaps/heatmaps
  (`config.ZSCORE_COLORS`).

## Setup

```bash
pip install -r requirements.txt
```

Requires R with the **limma** (Bioconductor), **ComplexHeatmap**, and **circlize**
packages installed, accessed via `rpy2` (used for DEP fitting and the trend
heatmap).

## Running

```bash
python3 scripts/run_pipeline.py           # main: preprocessing -> DEP -> dim-reduction -> clustering -> pairwise
python3 scripts/run_branch.py             # both branches: preprocessing -> dim-reduction -> clustering -> trend/time-series
```

## Verification

`tests/` holds a persisted data-integrity verification suite — each script
independently re-derives its target transform from the raw inputs and compares
to the pipeline's saved outputs (exits 0 on success):

```bash
bash tests/run_all.sh                     # runs all 5 scripts below, fail-fast
```

```bash
python3 tests/verify_preprocessing.py     # steps 01-05 (main), independent recompute
python3 tests/verify_integrity.py         # cross-step ID/order + metadata-join + sanity
python3 tests/verify_branch.py            # WT/KI branch subset + steps 02-05
python3 tests/verify_analysis.py          # DEP/pairwise/trend layer + completeness
python3 tests/verify_practical_audit.py   # implementation-independent audit (file
                                           # checksums, raw regex parsing, sample-
                                           # identity/duplication screening, BH math)
```

**`bash tests/run_all.sh` must be run (and must pass) before merging any change
to `scripts/`** — see `claudedocs/DATA_INTEGRITY_PM_PLAN.md` Phase B/C.

See `claudedocs/DATA_INTEGRITY_REPORT.md` for the full results and one open item
flagged for manual review (a high-correlation sample pair — not a proven defect,
see the report for details).

## Output layout

Each project root (`output_data/<Project>/`) contains one directory per step
(`01_raw_summary/` … `12_regionally_elevated/`) plus a `figures/` directory
mirroring every plot in one place. See `claudedocs/DATA_INTEGRITY_REPORT.md`
and the step scripts for per-file descriptions.

## Changelog (변경 이력)

인수인계 시점 기준, 최근 변경 순으로 정리. 커밋 해시가 없는 항목은 아직
커밋되지 않고 로컬 작업트리에만 존재하는 변경사항이다(커밋은 담당자가 직접 진행).

| 날짜 | 무엇을 | 왜 | 커밋 |
|---|---|---|---|
| 2026-07-13 ~ 07-14 | 뇌 단백질 주석 스텝(`step01b_brain_annotation.py`)과 이를 활용한 재분석 두 스텝(`step11_brain_filtered.py`: 뇌-연관 단백질 subset만으로 재클러스터링, `step12_regionally_elevated.py`: HPA "regionally elevated" 763유전자 subset만으로 재클러스터링) 추가. `database/`에 HPA 참조 DB(`260525_v25.1_mouse_brain_category_rna_Any_Region.tsv`) 추가, `scripts/config.py`·`scripts/run_pipeline.py`·`tests/verify_analysis.py`·`README.md`를 함께 갱신. | 혈장에서 검출된 단백질 중 뇌 유래/뇌-특이적 단백질만 따로 떼어 봤을 때 전체 프로테옴 대비 genotype/timepoint 분리(clustering separation)가 달라지는지 확인하기 위함 — 즉 "뇌 관련 신호가 혈장 전체 신호에 묻히는지"를 검증하려는 목적. | *(미커밋, 작업트리)* |
| 2026-07-04 | `tests/run_all.sh` 추가(5개 검증 스크립트를 순차 실행, 실패 시 즉시 중단), `README.md`·`run_pipeline.py`/`run_branch.py` docstring에 필수 단계로 명시. `tests/verify_practical_audit.py`에 raw 입력 TSV의 SHA-256 체크(`EXPECTED_RAW_SHA256`) 추가로 원본 데이터 변조/교체를 자동 탐지하도록 함. | 매번 검증 스크립트 5개를 수동으로 각각 돌리던 것을 한 번의 명령으로 통합해 재검증 프로세스를 자동화하고, "원본 파일이 사람도 모르게 바뀌는" 리스크를 코드로 탐지하기 위함(`claudedocs/DATA_INTEGRITY_PM_PLAN.md` Phase B/D). | `6002325`, `16179e4` |
| 2026-07-04 | `claudedocs/DATA_INTEGRITY_PM_PLAN.md`, `claudedocs/DATA_INTEGRITY_REPORT.md`, 최초 `README.md` 추가. | 파이프라인 무결성 검증 결과(125/125 통과, 미해결 1건)를 문서화하고, 남은 이슈(샘플 페어 `10W_KI8` vs `14W_WT10`, r=0.997 이상 고상관)를 추적 가능한 액션 아이템으로 관리하기 위함. | `73041ab` |
| 2026-07-04 | `.gitignore` 추가(`__pycache__/`, `*.pyc`, `*.pyo`), 이미 추적 중이던 `__pycache__`를 저장소에서 제외. | 파이썬 바이트코드 캐시가 버전관리에 잡히지 않도록 워크스페이스 위생 확보. | `9c596d7` |
| 2026-07-04 | `trend_heatmap`을 Python 구현에서 R `ComplexHeatmap` 기반으로 재작성, 컬러맵을 green–black–red로 통일하고 일부 레이아웃 수정 후 전체 파이프라인 산출물 재생성. | 참조 파이프라인(KBSI FAIMS)의 레이아웃과 시각적으로 일치시키기 위함. | `a24e4f9`, 및 그 앞 커밋 |
| 2026-07-04 | `tests/verify_preprocessing.py`, `verify_integrity.py`, `verify_branch.py`, `verify_analysis.py`, `verify_practical_audit.py` 5종 데이터 무결성 검증 스위트 최초 추가. | 파이프라인이 산출한 결과를 원본 입력으로부터 독립적으로 재계산해 대조함으로써, 구현 버그로 인한 조용한 데이터 오류를 방지하기 위함. | `6f3abb5` |
| 2026-07-01 ~ 07-02 | 초기 파이프라인 커밋 및 결과 최초 export. | 프로젝트 시작. | `f4e83de`, `a570de8`, `90510f9` |
