# UOT_KR — Transport Representation and Conditional Decoding for Cross-Chain Forensic Fund-Flow Correspondence

Artifact release accompanying the manuscript

> **Non-One-to-One Cross-Chain Forensic Fund Flow Correspondence: Transport Representation and Conditional Decoding**
> (Chinese title: 跨链资金流对应的运输表示：排序失真诊断与半合成边界评估)

**Submission tag:** `v1.0-vldb-submission`

This repository contains the **code, frozen data artifacts and implementation artifacts** that produce
every number, table and figure in the paper. It is a curated, self-contained release: it does **not**
carry the development history of the private working repository, and it contains no raw on-chain
corpus and no credentials.

| | |
|---|---|
| Paper package (figures, tables, provenance audits) | `paper/ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/` — 409 files |
| Submission-ready manuscript | `paper/ZN_TIFS_CN_R11_SUBMISSION_READY.docx` (+ extracted text, reference list) |
| Headline confirmatory result | macro edge F1 **0.429210** (UOT-KR) vs **0.403679** (Hungarian 1-1), H1 effect **+0.025532**, 95 % CI **[+0.013405, +0.037100]**, Holm p = 5.49973e-04 |
| Confirmatory protocol | hash-locked, one-shot; `locked_spec.json` sha256 `9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d` |
| Independent validation | frozen validator recomputes all metrics from primitives, max abs diff **2.220e-16**; **Gate E = PASS** |
| Repository size | ~153 MB, 2 074 files, no file larger than 18 MB |

**First thing to run (about a minute, no external data):**

```bash
python release_check.py
```

It verifies the layout, the manuscript hash, the locked protocol hash, the presence of the 30
confirmatory units, **all 510 entries of the frozen R7 manifest** (116 checked directly, 301 inside
the packaged evidence archives, 93 `.npz` excluded by design — 0 mismatched, 0 absent), then runs the
frozen independent validator and confirms that running it left every manifest-listed file byte-identical.
Expected: `11/11 offline checks passed`.

---

## 1. Environment dependencies

Verified run environment (recorded in the R7 one-shot ledger and in every experiment report):

| component | version |
|---|---|
| Python | **3.11.11** (CPython, `MSC v.1943 64 bit (AMD64)`) |
| OS used for the released runs | Windows 10 (`10.0.26200`); all commands below are cross-platform |
| CPU execution | no GPU required anywhere in this release |

Exact pinned dependencies used to produce the released results
(`requirements.txt`; recorded in `out/r7_confirmatory_kernel_ranking_20260917/FINAL_EXPERIMENT_REPORT.md` §18):

| package | version | role |
|---|---|---|
| **`POT`** (`import ot`) | **0.9.6.post1** | entropic unbalanced OT solver (`ot.unbalanced.sinkhorn_unbalanced`) |
| `numpy` | 1.26.4 | arrays |
| `scipy` | 1.17.1 | linear assignment / stats |
| `pandas` | 2.3.3 | tables |
| `scikit-learn` | 1.8.0 | metrics |
| `matplotlib` | 3.10.9 | figures |
| `torch` | 2.11.0+cpu | declared project dependency (CPU build) |
| `pytest` | 9.0.3 | test suite |
| `jsonschema` | 4.26.0 | config/schema validation |

> **Solver-version caveat (important for reproducing the OT numbers).**
> `POT >= 0.9.5` changed the default `reg_type` to `'kl'`, so the effective unbalanced kernel is
> `exp(-C / reg) * outer(a, b)`. Every optimality claim in this release was evaluated against that
> kernel (maximum KKT marginal residual across all 30 confirmatory cells: **4.669e-13**, tolerance
> 1e-7). Pinning `POT==0.9.6.post1` is therefore part of the protocol, not a convenience.

Install:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # optional: installs the `cross` package from src/
```

`pyproject.toml` declares `requires-python = ">=3.10"`; `requirements.txt` pins the exact versions.
There is no conda `environment.yml` in this release — `requirements.txt` is authoritative.

---

## 2. Directory structure

| path | what it is |
|---|---|
| **`src/`** | The installable `cross` package: the method itself. `src/cross/domain/uot/` holds the cost matrix, delay policy, transport solver and decoders (`rc_uot_v2.py`, `decode_transport.py`, `cost_matrix.py`); `src/cross/domain/evaluation/` holds the semi-synthetic flow generator that injects non-one-to-one topology onto real flow features; `src/cross/baseline_compare/` holds the baseline adapters; `src/cross/application/` and `src/cross/interfaces/` hold the experiment entry points and CLI. |
| **`scripts/`** | The runnable experiment drivers. Top level holds the three released experiment lines (`run_r5_posthoc_hparam_sensitivity.py`, `run_r6_posthoc_kernel_k_control.py`, `run_r7_confirmatory_kernel_ranking.py`, `run_r7_preflight.py`, `run_r7_selection.py`, `validate_r7_confirmatory_results.py`) plus the earlier phase drivers. `scripts/multi_bridge/` holds the frozen development machinery imported by those drivers (`diag/ctd_common.py`, `dev_candidate/af_common.py`, `dev_candidate2/cp_common.py`, `decoder_audit/da_common.py`, `holdout/`) — these are the *same* modules the frozen pipeline used, never re-implementations. |
| **`tests/`** | `pytest` suite (34 test modules) covering cost matrices, transport decoding, the RC-UOT v2 protocol, window sensitivity, anchor masking, and the semi-synthetic generator. `tests/fixtures/rc_uot_v2_2_synthetic/` holds the small frozen fixture used by the protocol tests. |
| **`config/`** | Frozen run configuration: `defaults.json` (pipeline defaults and the cost-weight block), the Celer cBridge ABI registry, routing rules and token routes. **No secrets** — provider keys and local overrides live in a git-ignored `config/local.json`. `config/local.example.json` is the shipped template to copy from. The credential-bearing `config/local.json`, `config/local.runtime.json` and `config/api-key-cross.json` of the private workspace are **not** part of this release; see `SECURITY.md`. |
| **`schemas/`** | JSON Schemas for the run configuration and the artifact contracts; used by the validators. |
| **`docs/`** | Design and results notes: `baseline_comparison_note.md` (how Connector/ABCTracer are mapped into the flow-level label space), `m1_ablation_runbook.md`, the per-table notes, the claim bank, the paper-readiness checklist, and the three-bridge split/merge handoff note. |
| **`paper/`** | The paper authoring package (409 files): figure/table source indices, per-figure generation scripts and value audits, the frozen `core_results/` artifact set, the 70-item risk register, and the submission-ready DOCX + its `SHA256SUMS.txt` (1 008 entries). |
| **`out/`** | The three released experiment lines with their reports, tables, figures and frozen raw results (see §3). |
| **`audit/risk_field/`** | The 2026-09-21 risk-field empirical audit (negative result: the released risk score column is constant-zero on the destination side and `aml_risk_level` is all-NaN repo-wide). Released as-is. |
| **`release_check.py`** | Offline release self-verification: layout, manuscript hash, locked protocol hash, the 30 confirmatory units, all 510 frozen R7 manifest entries, the frozen independent validator, and a proof that verifying did not modify the tree (run this first). |
| **`verify_release.py`** | Authoritative per-file integrity check against `RELEASE_SHA256SUMS.txt` (handles non-ASCII paths). |
| **`unpack_release.py`** | Re-expands the three packaged evidence trails and verifies every extracted file against `RELEASE_MANIFEST.json`. |
| **`generate_sums.py`** / **`generate_manifest.py`** | Regenerate `RELEASE_SHA256SUMS.txt` / the canonical `RELEASE_MANIFEST.json`. |
| **`reproducibility/build/`** | The curation pipeline and `BUILD.md`, documenting exactly what was kept, what was dropped, and why the manifest generation order matters. Not needed to use the artifact. |

---

## 3. Reproducing the experiments

Three experiment lines are released. Each has its own output root under `out/`, holding
`FINAL_EXPERIMENT_REPORT.md`, `VALIDATION_CHECKLIST.md` / gate JSONs, `results/`, `figures/`, `paper/`
(manuscript patches) and `provenance/`.

> **Read this before running.** The R5 and R6 drivers solve UOT on the *development cells*, whose
> cost/transport arrays are ≈133 MB of `.npz`. Those arrays are **not** shipped (see
> `EXTERNAL_DATA_MANIFEST.md`); without them `--check`, `--smoke`, `--sweep`, `--ablation`, `--full`
> and `--verify-reuse` stop at `FileNotFoundError` on
> `out/multi_bridge_expansion/cost_transport_diagnosis/plans/dev/<BRIDGE>/seed_<N>/cost.npz`.
> All *released* inputs and *all* released outputs are present, so every reported number can be
> inspected and re-aggregated offline; the offline-verifiable path is `release_check.py` plus the R7
> validator, which needs no external data at all.

### 3.1 Line R5 — post-hoc hyper-parameter sensitivity and cost-component ablation

* **Question.** How sensitive are the two decoders to `k`, `epsilon`, `lambda`, and to removing a
  single cost component, on the development distribution?
* **Data.** Development seeds **201–205** × bridges **{Celer, Multichain, PolyNetwork}** × 48 templates
  = 210 cells; 14 unique hyper-parameter configurations; methods `RAW_UOT_PLAN_D4` and
  `CONDITIONAL_UOT_D4` on the *same* plan. Holdout seeds 301–305 are hard-refused at argument
  validation, before any data access.
* **Cost matrix.** The paper's primary amount-free renormalised cost (amount removed; time / route /
  risk / evidence / novelty renormalised to sum 1).
* **Command (from the repository root):**

  ```bash
  python scripts/run_r5_posthoc_hparam_sensitivity.py --sweep      # 210 cells
  python scripts/run_r5_posthoc_hparam_sensitivity.py --ablation   # leave-one-cost-component-out
  python scripts/run_r5_posthoc_hparam_sensitivity.py --aggregate  # -> results/ and ablation/
  ```

* **Released outputs and expected values.** `out/r5_posthoc_hparam_sensitivity_20260917/`
  — `FINAL_EXPERIMENT_REPORT.md`, `results/`, `ablation/`, `figures/` (8 files),
  `VALIDATION_CHECKLIST.md` (**16/16 checks PASS**).
  Reproduction anchors: `RAW_UOT_PLAN_D4` = **0.237428** (published 0.2374) and
  `CONDITIONAL_UOT_D4` = **0.312909** (published 0.3129); max deviation 2.82e-05 < 5e-04 tolerance.
  Headline sensitivity: at the default (`k=5, eps=0.05, lambda=0.5`) RAW = 0.2374 / CONDITIONAL =
  0.3129; the grid optimum for CONDITIONAL is `k=3` at **0.4797**.
* **Runtime / hardware.** One UOT solve per unique `(bridge, seed, k, epsilon, lambda)`; 210 cells.
  Single CPU core is sufficient; no GPU. Measure it on your machine with `--smoke` first.

### 3.2 Line R6 — post-hoc direct-kernel `k` control

* **Question.** On the development distribution, how does the ordering of `CONDITIONAL_UOT_D4` vs
  `AMOUNT_FREE_COST_D4` change with decoder width `k`?
* **Status.** **Post-hoc, not preregistered.** It cannot answer whether the frozen-holdout `k=5`
  difference survives at `k != 5`, because the holdout exists only at the default `k=5` and is never
  re-run (`HOLDOUT_REEXECUTED = NO`).
* **Command:**

  ```bash
  python scripts/run_r6_posthoc_kernel_k_control.py --lock-spec
  python scripts/run_r6_posthoc_kernel_k_control.py --preflight
  python scripts/run_r6_posthoc_kernel_k_control.py --full --resume
  python scripts/run_r6_posthoc_kernel_k_control.py --analyze
  python scripts/run_r6_posthoc_kernel_k_control.py --figures
  python scripts/run_r6_posthoc_kernel_k_control.py --report
  ```

* **Released outputs and expected values.** `out/r6_posthoc_kernel_k_control_20260917/`
  — `FINAL_EXPERIMENT_REPORT.md`, `results/kernel_k_summary.csv`, `kernel_k_per_seed.csv`,
  `kernel_k_long.csv`, `primary_contrasts.csv`, `classification.json`, `figures/`, `paper/`.
  Reproduction gate (must be inside 5e-04):
  `AMOUNT_FREE_COST_D4` = **0.3171931003584229** vs 0.3172 (diff 6.90e-06);
  `CONDITIONAL_UOT_D4` = **0.3129093352883675** vs 0.3129 (diff 9.34e-06);
  `RAW_UOT_PLAN_D4` = **0.23742822838252942** vs 0.2374 (diff 2.82e-05). All **PASS**.
  Locked decision rule sha256: `2733d2f5b24d685ca7eba667932a32a2eb96b43459c7d2e49b30c8af70b6f031`.
* **Runtime / hardware.** 15 development units (3 bridges × 5 seeds) plus two decoder widths per
  cell; single CPU core, no GPU.

### 3.3 Line R7 — degree-calibrated confirmatory kernel ranking (the headline result)

This is the preregistered line. The protocol is **hash-locked** and the holdout was executed
**exactly once**.

* **Design (in order).** (0A) recalibrate the generator's split/merge degree distributions from the
  frozen v4/v5 audit windows, removing the fixed-degree artefact that made the R6 `k=3` advantage
  inseparable from generator construction; (0B) expand to **24 distinct template families** per
  cell (48 instances); (1) run a **selection** stage on a fresh seed block to fix the decoding rule
  and every parameter; (2) **hash-lock** the protocol (42 hashed files); (3) execute **one single
  confirmatory run** on block **411–420**, which had never been generated or read; (4) verify with a
  frozen independent validator.
* **Frozen protocol.** `out/r7_confirmatory_kernel_ranking_20260917/config/locked_spec.json`,
  sha256 `9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d`.
  Cost: amount-free renormalised five-component cost, `epsilon = 0.05`, `lambda = 0.5`, support
  threshold `1e-09`. Statistics: bootstrap B = 4000 (RNG 20240101), permutation n_perm = 20000
  (RNG 20240102), Holm over H1/H2, alpha = 0.05.
  Selected rule: **`R-const@k3`** (selection score 0.432151).
* **Primary command — verify the released confirmatory result (no external data needed):**

  ```bash
  python scripts/validate_r7_confirmatory_results.py --out /tmp/gate_e_check.json
  ```

  > **Always pass `--out`.** Without it the validator writes to its default location,
  > `confirmatory/VALIDITY_GATE_E.json`, which is **hash-locked in `MANIFEST.json`**. The report is
  > stamped with `generated_at_utc` and `runtime_sec`, so re-writing it in place would silently
  > break the frozen record. `release_check.py` redirects it correctly and then re-hashes the whole
  > manifest to prove nothing moved.

  Expected output: all checks `PASS` and `GATE_E = PASS` (exit 0). The validator recomputes every
  metric from primitives; agreement with the executor is ≤ **2.220e-16** against a 1e-9 tolerance,
  over 30 units × 7 methods × 7 metric fields. Consistency check: the 30 units match
  `confirmatory/raw/INDEX.json`, the retired `401–410` partial artefacts are preserved under
  `confirmatory/retired_401_410/`, and the one-shot ledger
  `confirmatory/CONFIRMATORY_TOUCH_ONCE__411_420.json` is present.
* **Re-executing the holdout is refused by design.** This is a feature, not a defect:

  ```bash
  python scripts/run_r7_confirmatory_kernel_ranking.py --execute
  ```

  Expected: pre-execution gate prints `FAILED -> refusing to execute; holdout untouched`, exit 1.
  The gate reports `one_shot_ledger_absent` and `confirmatory_raw_absent` as `FAIL` **because the
  holdout has already been spent** — exactly the state a spent one-shot ledger should produce.
  (`--verify-only` additionally reports `frozen_artifact_hashes` as `FAIL`: two entries of the
  frozen manifest hash files from the R5 authoring tree that are deliberately outside this release's
  curated scope. This does not affect any metric, threshold or data; it is recorded in **Known
  issues**.)
* **Main results** (`analysis/confirmatory_bridge_summary.csv`, `analysis/confirmatory_cell_level.csv`):

  | method | macro edge F1 | precision | recall | edges/family |
  |---|---:|---:|---:|---:|
  | **UOT_KR** | **0.429210** | 0.3761 | 0.5089 | 17.45 |
  | SUPPORT_PLUS_K | 0.417081 | 0.3652 | 0.4951 | 17.08 |
  | CONDITIONAL_UOT | 0.414999 | 0.3636 | 0.4922 | 17.80 |
  | HUNGARIAN_1TO1 | 0.403679 | 0.4148 | 0.3962 | 12.04 |
  | DUAL_SOFTMAX | 0.247960 | 0.9681 | 0.1428 | 1.98 |
  | RAW_UOT_PLAN | 0.109907 | 0.0976 | 0.1286 | 16.24 |
  | THRESHOLD_MM | 0.062171 | 0.0372 | 0.1962 | 16.89 |

  H1 (UOT_KR − HUNGARIAN_1TO1): **+0.025532**, 95 % CI **[+0.013405, +0.037100]**, Holm p =
  5.49973e-04, **Gate A PASS**. H2 (UOT_KR − THRESHOLD_MM): **+0.367040**, 95 % CI
  **[+0.360342, +0.373424]**, Holm p = 9.9995e-05, **Gate B PASS**. Cross-bridge **Gate C PASS**;
  validity **Gates D and E PASS**.
* **Runtime / hardware.** The one-shot confirmatory execution recorded
  `wall_sec = 52.65` for **30 units** (3 bridges × 10 seeds) on the run host
  (Windows 10, Python 3.11.11, CPU only). Validation is a few seconds. This is an
  embarrassingly-parallel single-core workload; no GPU.

### 3.4 Cross-cutting: run the unit tests

```bash
# Windows (PowerShell)
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest tests -q

# Linux / macOS
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

`pyproject.toml` already sets `testpaths = ["tests"]`, `pythonpath = ["src"]` and `-p no:web3`.
The extra `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed **only** when a third-party `pytest` plugin in
your environment drags in the broken `parsimonious`/`eth_abi` chain (`ImportError: cannot import name
'getargspec' from 'inspect'`). This is a property of the environment, not of this repository.

Verified on a fresh copy of this release:

```text
python -m pytest tests/test_cost_matrix.py tests/test_decode_transport.py tests/test_flow_metrics.py -q
10 passed
```

`tests/test_p0_validator_self.py` is a standalone validator script with a module-level `sys.exit(0)`,
so it must be run directly rather than through `pytest` (under `pytest` it raises
`INTERNALERROR> SystemExit` at collection — that is the script's own design, not a broken test).
**It is only partially reproducible from this release:** 6 of its 16 checks pass here
(`test_valid_gate`, `test_trivial`, `test_prematched`, `test_zero_fraction`, `test_no_solver_import`,
`test_no_labels_access`). The other 10 need artifacts outside the approved scope:

```bash
python tests/test_p0_validator_self.py
# 6/16 passed
```

| failing group | cause |
|---|---|
| `test_full_sha256` | reads `out/rc_uot_v2_2/provider_contract/candidate_generation_policy.json` — the whole `out/rc_uot_v2_2/` tree is **not present in the source workspace any more** and is outside this release's scope. Not recoverable. |
| 9 × `tempfile.TemporaryDirectory` checks | `[WinError 5]` on temp-dir creation. This is an environment restriction of the sandbox this release was assembled in, not a logic failure; the checks read and write only throwaway temp files and are expected to pass in a normal environment. Re-run with a writable `TEMP`/`TMP` to confirm. |

### 3.5 Which of the three lines can you actually run here?

Stated up front, because it is the first question an artifact reviewer asks. All three lines
ship every released **input to the analysis** and every released **output**, so every reported
number can be inspected and re-aggregated offline. What differs is whether the *solver* can be
re-run, which for R5/R6 needs development-cell arrays that are outside this release's scope.

| line | runnable from this release alone? | entry point | what it produces |
|---|---|---|---|
| **R7** — confirmatory | **Yes, fully.** No external data. | `python release_check.py`<br>`python scripts/validate_r7_confirmatory_results.py --out <scratch>` | Recomputes all 7 methods × 7 metrics over the 30 frozen confirmatory units from primitives and prints `GATE_E = PASS`; verifies the locked protocol hash and all 510 manifest entries. |
| **R6** — post-hoc k control | **Partially.** Best-effort path: `--analyze`, `--figures`, `--report`, `--validate` re-aggregate and re-verify from the shipped `runs/`; `--full` and `--verify-reuse` need the external cells. | `python scripts/run_r6_posthoc_kernel_k_control.py --analyze` | Rebuilds `results/kernel_k_summary.csv` and the primary contrasts from the 15 shipped unit files. |
| **R5** — sensitivity/ablation | **Inspection only.** The 210 swept cells are shipped as results; re-solving needs the external cells. | see §3.1 | The released `results/`, `ablation/` and `figures/` are complete and can be re-aggregated with `--aggregate` once cells exist. |

The external dependency in both partial cases is the same ≈133 MB of development-cell
`.npz` arrays at `out/multi_bridge_expansion/cost_transport_diagnosis/plans/dev/`. They are
excluded as regenerable intermediates and, more fundamentally, because regenerating them needs
the restricted raw on-chain corpora. Without them the R5/R6 drivers stop at a clear
`FileNotFoundError` on that path — they do not fail silently or produce wrong numbers. See
`EXTERNAL_DATA_MANIFEST.md` §1–2.

**Hardware:** everything above runs on a single CPU core with no GPU. R7's one-shot
confirmatory execution recorded 52.65 s for 30 units on the run host; validation takes seconds.

---

## 4. Data availability

### 4.1 Shipped inside this repository

Everything needed to inspect and independently validate the **confirmatory** result, plus the frozen
downstream artifacts behind every released figure and table:

| content | location | size |
|---|---|---|
| Code (method, drivers, tests, config, schemas) | `src/`, `scripts/`, `tests/`, `config/`, `schemas/`, `docs/` | ≈11 MB |
| Manuscript, supplements, figure/table sources, provenance audits | `paper/` | ≈38 MB |
| R7 frozen confirmatory package (30 units + raw index + gates + ledger) | `out/r7_confirmatory_kernel_ranking_20260917/confirmatory/` | ≈38 MB |
| R5 / R6 / R7 reports, tables, figures, provenance | `out/r5_*`, `out/r6_*`, `out/r7_*` | ≈26 MB |
| R11 forensic evidence package + `SHA256SUMS.txt` (1 008 entries) | `out/handoff_r11_remaining6/` | ≈21 MB |
| Risk-field audit (negative result) | `audit/risk_field/` | ≈1 MB |

Per-file integrity: **`RELEASE_SHA256SUMS.txt`** lists the SHA256 of every loose file in this release
(archives included, as they ship). Verify with the authoritative checker (Python, so non-ASCII paths
are handled correctly):

```bash
python verify_release.py
```

Expected: `listed files 2072 ... RESULT: PASS` — every listed file present and matching, and no
unlisted file present.

**`RELEASE_MANIFEST.json`** is the canonical manifest: it holds the SHA256 of **every logical file**,
i.e. loose files as stored *plus* each archive member as it appears after extraction (3 621 entries
over 3 archives). It is what makes "expand the archives, then re-verify" work — see §4.2.

### 4.2 Packaged evidence trails (inside the repo, re-expandable)

Three bulky *evidence trails* ship as ZIP archives so the repository stays small. Nothing is lost —
`python unpack_release.py` expands them **and** verifies every extracted file against
`RELEASE_MANIFEST.json`, then (by default) removes the archives:

| archive | size | expands to | content |
|---|---:|---|---|
| `out/r7_confirmatory_kernel_ranking_20260917/selection_evidence.zip` | 4.2 MB | `selection/` | 586 files: the full 22-candidate rule search, degree calibration, generator QA and executor dry run of the R7 selection stage |
| `out/handoff_r11_remaining6/R11_CORE_evidence.zip` | 17.7 MB | `core/` | 730 files: the six R11 pending-item forensic evidence sets |
| `out/handoff_r11_remaining6/R11_E1_evidence.zip` | 2.4 MB | `e1/` | 241 files: the R11 E1 input/closure evidence set |

```bash
python unpack_release.py                # expand + verify, then delete the archives
python unpack_release.py --keep-zips    # expand + verify, keep the archives
python unpack_release.py --only r11-core
python unpack_release.py --verify-only  # verify what is on disk, expand nothing
```

All three are `testzip()`-clean, and each expands to exactly the member set recorded in
`RELEASE_MANIFEST.json`. `--only` narrows the expansion if you want just one trail.

> Archive hashes are recorded in `RELEASE_ARCHIVES.md` for reference only. ZIP stores member
> modification times, so rebuilding an archive from the same tree yields a *different* archive hash
> with *identical* member content. The per-file hashes in `RELEASE_MANIFEST.json` and
> `RELEASE_SHA256SUMS.txt` are authoritative, not the archive hashes.

### 4.3 Not shipped — and how to get it

See **`EXTERNAL_DATA_MANIFEST.md`** for the complete list with sizes, SHA256 of the originals where
recorded, and regeneration commands. In summary:

* **Development-cell cost/transport arrays (≈133 MB `.npz`)** — required only to *re-run* R5/R6.
  They are deterministic outputs of the released generator pipeline
  (`scripts/multi_bridge/run_build_dev_plans.py`), not primary data.
* **Raw on-chain corpora** — bridge candidate/flow tables and provider exports used to build the real
  flow features. **Not released.** Address-level entity labels are withheld; released flow-level
  labels keep only the anonymised identifiers required for evaluation, and no provider API key is
  released. This is a deliberate restriction, stated in the manuscript.
* **Derived transport/cost matrices (>100 MB each)** from the paper-main pipeline — excluded as
  regenerable intermediates.

**There is no DOI / external download link in this release.** No deposit has been made to Zenodo or
OSF, so no DOI is claimed here. If a deposit is made later, the DOI and the SHA256 of the deposited
archive will be added to `EXTERNAL_DATA_MANIFEST.md`.

---

## 5. License and citation

### License

* **Code** (`src/`, `scripts/`, `tests/`, `config/`, `schemas/`) — **MIT**, see `LICENSE`.
* **Manuscript, figures, tables, result artifacts** (`paper/`, `docs/`, `out/`, `audit/`) —
  **CC BY 4.0**, see `LICENSE-DOCS`.

### Citation

```bibtex
@article{uotkr2026crosschain,
  title   = {Non-One-to-One Cross-Chain Forensic Fund Flow Correspondence:
             Transport Representation and Conditional Decoding},
  author  = {UOT\_KR authors},
  journal = {Manuscript under review},
  year    = {2026},
  note    = {Artifact release v1.0-vldb-submission,
             \url{https://github.com/3074669057/UOT_KR}}
}
```

A machine-readable `CITATION.cff` is included. **The author list, ORCIDs and venue are placeholders
pending the final submission metadata** — update `CITATION.cff` and this block before publication.
