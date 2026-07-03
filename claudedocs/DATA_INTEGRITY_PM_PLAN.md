# Data Integrity — PM Action Plan

**Owner:** Project maintainer (jonghyeonseol)
**Prepared by:** PM Agent meta-layer review
**Date:** 2026-07-04 (Phase A status confirmed 2026-07-04)
**Related:** `claudedocs/DATA_INTEGRITY_REPORT.md` (technical findings this plan acts on)

## Status log

| Date | Update |
|---|---|
| 2026-07-04 | Plan created. Phases B/C/D not yet started. |
| 2026-07-04 | **Phase A confirmed PENDING** — maintainer has **not yet** contacted the core facility (KBSI) for the LC-MS run log. Request draft added below (§4, Phase A) so this can move to "sent" as soon as it goes out. |
| 2026-07-04 | **Phase B DONE.** Added `tests/run_all.sh` (runs all 5 `verify_*.py`, fail-fast); documented as the required pre-merge command in `README.md`; referenced in `run_pipeline.py`/`run_branch.py` docstrings. Full suite re-verified passing (18/18 + 1 flag) after the change. |
| 2026-07-04 | **Phase D bullet 2 DONE.** `tests/verify_practical_audit.py` now compares the raw TSV's SHA-256 against a recorded `EXPECTED_RAW_SHA256` constant and **fails** on mismatch, instead of merely printing the hash — input drift is now actually detected, not just loggable by a human who remembers to look. Remaining Phase D bullets (recurring cadence, plan housekeeping) still require human process, not code. |
| 2026-07-04 | **Phase A still PENDING** — not advanced; requires the maintainer to actually send the request in §4a. Not fabricated as done. |

## 1. Objective

Move data integrity from **"125/125 automated checks passed, 1 item open"** to
**"fully closed, with a governance process that keeps it closed as the pipeline
evolves."** Automated verification proved the *pipeline* is mechanically sound;
this plan covers what verification alone cannot close (external evidence,
recurring cadence, ownership) and turns the one-off audit into a durable process.

## 2. Current state (inherited from the technical report)

| Layer | Status |
|---|---|
| Preprocessing (01–05), all 3 projects | ✅ Closed — 84/84 checks, independent recompute |
| Cross-step integrity + metadata join | ✅ Closed — 17/17 |
| Analytical layer (DEP/pairwise/trend) | ✅ Closed — 24/24 |
| Implementation-independent practical audit | ✅ Closed — 18/18 (added raw-input SHA-256 drift check) |
| **Sample-pair anomaly (`10W_KI8` vs `14W_WT10`, r=0.9966)** | 🟡 **Open — Phase A PENDING (not yet requested)**. Plate/report-order checked from existing files, no adjacency found; true LC-MS run-order record does not exist in this repo and has not yet been requested from the core facility. |
| Governance / recurring re-verification | 🟡 **Partial** — Phase B done (single fail-fast command, documented); Phase D bullet 2 (input-drift check) done; Phase C (keep-in-lockstep discipline) and remaining Phase D cadence still process, not code |

## 3. Gap analysis — what "achieved" actually requires

1. **Evidentiary gap**: the flagged sample pair cannot be resolved from data in
   this repository alone. Closing it requires an artifact this project does not
   control (the acquisition core facility's instrument queue/run log).
2. **Process gap**: nothing currently prevents a future pipeline change from
   silently breaking an invariant the test suite checks today — there is no
   trigger that re-runs `tests/verify_*.py` on change.
3. **Ownership gap**: the sample-pair follow-up has no assigned owner or
   deadline; "recommend contacting the core facility" is not the same as a
   tracked action.
4. **Scope-drift gap**: `tests/verify_analysis.py` hardcodes today's contrast
   counts (20 DEP contrasts, 16 pairwise, 6 clusters). If `config.py` changes
   those numbers, the suite will fail loudly — correct behavior, but only if
   someone is running it.

## 4. Action plan

### Phase A — Close the evidentiary gap (owner: maintainer, target: next data delivery cycle)
**Status: 🔴 PENDING — not yet initiated (confirmed with maintainer 2026-07-04).**
No request has been sent to the core facility. Nothing in this phase can be
marked done until that happens — the checklist below is unchanged in intent,
only its status is now accurately tracked instead of implied.

- [ ] **Request the LC-MS acquisition log / autosampler injection queue** for
      the `20260529` batch from the core facility (KBSI), specifically
      covering wells **F5** (`10W_KI8`) and **F9** (`14W_WT10`). A ready-to-send
      draft is in **§4a** below — copy it in, attach/send, then flip this item
      to "sent (date)".
- [ ] If the log confirms non-adjacent, non-sequential injection → **downgrade
      the flag to "resolved, coincidental similarity"** in
      `DATA_INTEGRITY_REPORT.md`.
- [ ] If the log shows adjacent/sequential injection or any carryover flag →
      re-run all `KI_10W`/`WT_14W`-involving DEP contrasts with that sample
      pair excluded as a sensitivity check, and note any result that depends on
      it.
- [ ] Either outcome: append a dated resolution note to §5 of the integrity
      report (do not delete the original finding — audit trail matters more
      than a clean-looking report).

#### §4a. Ready-to-send request draft

> **Subject:** Run-order/plate-map confirmation request — batch 20260529 (wells F5, F9)
>
> Hi [core facility contact],
>
> While QC-reviewing the SEER Proteograph results from the `20260529` mouse
> plasma batch, we found that two samples — **well F5** (`10week KI8`) and
> **well F9** (`14week WT10`) — have the highest pairwise proteome correlation
> in the entire batch (Pearson r = 0.997, out of 3,240 sample pairs), despite
> being different genotype and different timepoint groups.
>
> Before we finalize the analysis, could you confirm from the instrument
> records:
> 1. The LC-MS injection order for this batch (were F5 and F9 injected
>    adjacently or with any temporal proximity?).
> 2. Whether any carryover, mis-injection, or autosampler flag was logged
>    around either injection.
> 3. Whether the physical plate loading matches our records (F5 = `10week
>    KI8`, F9 = `14week WT10`) — i.e. no relabeling occurred between plate
>    preparation and injection.
>
> This is a routine QC check, not an indication of a known problem — we just
> want to rule out a run-order or labeling artifact before treating these two
> samples as fully independent in downstream comparisons.
>
> Thanks,
> [name]

### Phase B — Make re-verification automatic, not manual (owner: maintainer, target: 1 week)
**Status: ✅ DONE (2026-07-04).**
- [x] Add a `tests/run_all.sh` (or Makefile target) that runs all 5
      `verify_*.py` scripts in sequence and fails fast on the first non-zero
      exit — one command instead of five.
- [x] Document in `README.md` that it must be run before merging any change to
      `scripts/`. (Pre-commit/pre-push git hook not wired up — plain
      documentation was judged sufficient for a single-maintainer repo; revisit
      if the contributor count grows.)
- [x] Add it to the `run_pipeline.py` / `run_branch.py` docstring as the
      required last step after any regeneration.

### Phase C — Keep the suite honest as the pipeline evolves (owner: maintainer, ongoing)
- [ ] Whenever `config.py` changes a structural constant checked by
      `verify_analysis.py` (contrast count, cluster count `k`, thresholds),
      update the corresponding assertion in the same commit — treat test
      updates as part of the change, not a follow-up.
- [ ] Re-run the **full** suite (not just the changed step) after any edit to
      `utils.py` or `config.py`, since both are imported by nearly every step.

### Phase D — Recurring integrity health-check (owner: maintainer, cadence: monthly or per data delivery)
- [ ] Re-run `tests/verify_practical_audit.py` whenever `input_data/` changes
      (new report export, corrected xlsx) — it is the only script that checks
      the raw files themselves rather than pipeline-derived state. *(process
      discipline — still requires a human to trigger it; not automated.)*
- [x] **Automated (2026-07-04):** the SHA-256 of
      `input_data/20260630_142950_20260529_Report.tsv` is now checked against a
      recorded `EXPECTED_RAW_SHA256` in `tests/verify_practical_audit.py` and
      **fails the suite** on mismatch — this no longer depends on a human
      remembering to "spot-check" it.
- [ ] Revisit this plan itself after Phase A closes — retire completed items,
      don't let the document accumulate stale checkboxes.

## 5. Success criteria (definition of "data integrity achieved")

Integrity is considered **fully achieved** when all of the following hold
simultaneously:
1. All 5 `tests/verify_*.py` scripts exit 0 on the current `output_data/`.
2. The sample-pair flag in §5 of the integrity report carries a **dated
   resolution** (not just a mitigation argument).
3. Re-verification is a single documented command, not five manual ones.
4. No structural constant in `config.py` has drifted from what
   `verify_analysis.py` asserts (i.e., the suite was updated in lockstep with
   the last pipeline change, not after the fact).

## 6. Non-goals (explicitly out of scope for this plan)

- Improving statistical power / group separation (silhouette ≈ 0) — that is a
  **scientific methodology** question, not a data-integrity defect, and is
  already documented separately (`diagnose_separation.py` findings).
- Switching the significance criterion from raw-p to adjusted-p — a deliberate,
  already-documented user decision, not an integrity gap.
