# R7 confirmatory provenance note

This note records two artefacts whose on-disk content is NOT the original of the
confirmatory run, why, and where the original information still lives. Nothing was
deleted and no scientific value was altered.

---

## 1. `confirmatory/gate_pre_execution.json` — overwritten by a post-hoc enforcement test

**What it should contain:** the frozen-protocol verification report as it stood in the
moment immediately before the one-shot confirmatory execution (9/9 checks PASS).

**What it contains now:** the result of a *later* verification invocation
(`--execute` run a second time, purely to prove that the one-shot ledger blocks a repeat).
That invocation correctly refused to run, so the file now records `ALL_PASS = false` with
`one_shot_ledger_absent: FAIL`, `confirmatory_raw_absent: FAIL` and
`frozen_artifact_hashes: FAIL`.

**Why this is not a loss of evidence.** The genuine pre-execution result is preserved
independently in three places:

| evidence | content |
|---|---|
| `confirmatory/execution_summary.json` | `"gate_pre_execution": true` |
| `confirmatory/first_touch_audit.md` | "8/8 pre-execution verifications PASS" |
| `logs/r7.log` (line group at `2026-09-17T07:51:52Z`) | all nine `[verify] ... PASS` lines verbatim |

**Why it is not a protocol violation.** The overwrite was caused by an *extra* executor
invocation, not by the confirmatory run. That extra invocation was refused before any
holdout access — the refusal is itself the strongest available demonstration that the
one-shot guard works. The holdout was **not** re-executed, no unit was regenerated, and
no metric changed.

**Preserved copy of the refusal record:**
`confirmatory/gate_pre_execution_posthoc_refusal.json`

**Note for future rounds.** The executor writes its verification report to a fixed
filename, so any later invocation overwrites it. A future protocol should write
`gate_pre_execution__<run_id>.json` instead. This defect was deliberately **not** patched
in R7, because the executor is a frozen artefact and patching it after the holdout touch
would break its own frozen-hash verification for no scientific gain.

## 2. `logs/r7.log` — append-only run log

Grows with every R7 invocation (selection, dry run, confirmatory execution, validation,
finalisation, layout verification). Append-only by design; its hash necessarily changes.
It is retained in full and is the primary evidence trail for the pre-execution
verification described above.

## 3. Unchanged by the Figure 1 layout correction

The presentation-only Figure 1 correction touched **only**:

* `figures/fig1_degree_calibration.pdf`
* `figures/fig1_degree_calibration.png`

plus three new files (`fig1_degree_calibration_pre_layout_fix.{pdf,png}` audit copies and
`fig1_layout_verification.json`). The manifest updater enforces this: it **aborts** if any
non-figure artefact hash moves, and it explicitly re-asserts that
`config/locked_spec.json`, `config/FROZEN_PROTOCOL_MANIFEST.json`,
`analysis/DECISION.json`, `analysis/primary_holm_tests.json`,
`analysis/primary_bootstrap.json` and `confirmatory/VALIDITY_GATE_D.json` are unchanged,
along with the confirmatory raw package digest.
