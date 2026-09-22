# R7 BLOCKER REPORT

> **STATUS: RESOLVED (2026-09-17).** The protocol owner approved option **A** —
> selection block = `206-211 + 112-115`. The R7 run then completed end to end:
> selection stage frozen, one-shot confirmatory execution on a fresh untouched block,
> independent validation, statistics and manuscript patches all produced.
> Final classification: **`CONFIRMATORY_METHOD_SUPPORT`**.
> See `FINAL_EXPERIMENT_REPORT.md` and `analysis/DECISION.json`.
> This report is retained unchanged as the record of the blocker as it stood.

* experiment id: `r7_confirmatory_kernel_ranking_20260917`
* report generated: 2026-09-17 (UTC)
* git HEAD at time of report: `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (branch `master`)
* output root: `out/r7_confirmatory_kernel_ranking_20260917/`
* status: **BLOCKED** (hard blocker, specification §1 / §71)

---

## 1. Exact blocker

**`SELECTION_BLOCK_ALREADY_USED` — the R7 selection seed block `206–215` is not fresh.**

Seeds **212, 213, 214, 215** already have fully produced experiment output on disk from a
pre-existing (paper-main-pipeline) run:

```
out/paper_full_pipeline_run/synthetic/synthetic_eval_seed_212/   (27 files, 29,025,832 bytes)
out/paper_full_pipeline_run/synthetic/synthetic_eval_seed_213/   (27 files, 21,286,768 bytes)
out/paper_full_pipeline_run/synthetic/synthetic_eval_seed_214/   (27 files, 28,660,318 bytes)
out/paper_full_pipeline_run/synthetic/synthetic_eval_seed_215/   (27 files, 25,824,285 bytes)
```

Each of these directories contains a *completed* generation + evaluation run, including

* `labels/synthetic_flow_labels.csv`      — the generated synthetic flow labels
* `labels/synthetic_uot_eval_metrics.json`— the generated truth/decoy/unmatched hints
* `uot/uot_cost_matrix.npz`               — the built cost matrix
* `uot/uot_transport_matrix.npz`          — the solved transport plan
* `eval/uot_evaluation_metrics.json`      — the evaluation result
* `synthetic_seed_stratification.json`    — the generator's own stratification record

This is produced data, not a reserved constant. Under the definition fixed in
`00_preflight/seed_freshness_audit.json` —

> A seed is CONTAMINATED iff produced experiment output for it exists on disk (a generated
> data/metric/plan artifact recording a completed run). Plain seed constants in code or
> reserved-ID lists are NOT contamination.

— seeds 212–215 are **contaminated**.

The specification states the requirement directly:

> `要求：206–215 = UNUSED`
> `401–410 = UNUSED`

and lists the failure case as a hard blocker:

> # 71. 唯一允许提前停止的硬阻塞
> * `206–215` 或 `401–410` 已被历史使用；

Seeds **206, 207, 208, 209, 210, 211 are fresh** (no output anywhere in the repository), so the
intended block is *partially* contaminated, not wholly.

---

## 2. Evidence

### 2.1 Machine-readable audit

| artefact | path |
|---|---|
| audit JSON | `00_preflight/seed_freshness_audit.json` |
| audit evidence rows | `00_preflight/seed_freshness_evidence.csv` |
| audit script | `scripts/run_r7_preflight.py` |

Reproduce with:

```
python scripts/run_r7_preflight.py
```

Audit result (verbatim from `seed_freshness_audit.json`):

```json
"selection_block_206_215": {
  "seeds": [206,207,208,209,210,211,212,213,214,215],
  "produced_output_seeds": [212,213,214,215],
  "status": "USED",
  "contaminated": true
},
"confirmatory_block_401_410": {
  "seeds": [401,...,410],
  "produced_output_seeds": [],
  "status": "UNUSED",
  "contaminated": false
},
"HARD_BLOCKER": "SELECTION_BLOCK_ALREADY_USED"
```

### 2.2 Direct filesystem probe (independent of the audit script)

```
206: ABSENT
207: ABSENT
208: ABSENT
209: ABSENT
210: ABSENT
211: ABSENT
212: EXISTS files=27 bytes=29025832 labels=True cost=True eval=True
213: EXISTS files=27 bytes=21286768 labels=True cost=True eval=True
214: EXISTS files=27 bytes=28660318 labels=True cost=True eval=True
215: EXISTS files=27 bytes=25824285 labels=True cost=True eval=True
401..410: ABSENT (all ten)
```

### 2.3 Full occupied-seed map (repository-wide directory scan)

Distinct seed-numbered directories found repo-wide: **176**, spanning **4 … 311**.
Occupied numbers in the 100–420 window:

```
100–111, 201–205, 212–311
```

Therefore the historically *fresh* numbers in that window are:

```
112–200   (fresh)
206–211   (fresh — the only fresh part of the intended selection block)
312–400   (fresh)
401–410   (fresh, and reserved as the R7 confirmatory holdout)
411+      (fresh)
```

The author's stated rationale — "`201–205`：R5/R6 已反复观察，污染" and "`206–215` = USED
requirement" — assumed `206–215` sat in an untouched gap. In fact the historical gap is
`112–211`; the pre-existing run resumes at `212`.

---

## 3. Failing path / command

```
path    : out/paper_full_pipeline_run/synthetic/synthetic_eval_seed_{212,213,214,215}/
command : python scripts/run_r7_preflight.py
          (audit stage: produced-output existence probe + seed-token content scan)
result  : HARD_BLOCKER = SELECTION_BLOCK_ALREADY_USED
```

The blocker is detected **before any R7 data generation**, so no R7 artefact exists yet.

---

## 4. Holdout touch status

> **`401–410 HAVE NOT BEEN GENERATED OR READ`. `HOLDOUT_TOUCHED = NO`.**

* the confirmatory block `401–410` is **UNUSED** (audit + direct probe, both agree)
* `confirmatory/CONFIRMATORY_TOUCH_ONCE.json` **does not exist**
* `confirmatory/raw/` **does not exist**
* no R7 confirmatory executor has been written to disk or run
* no R7 generator has been run on any seed

Because the holdout was **not** touched, a continued R7 execution remains scientifically
possible — but only after the selection-block question is resolved. The one-shot confirmatory
opportunity is **still intact**.

---

## 5. Steps completed before the blocker

| step | status |
|---|---|
| specification §2 task-state snapshot (`00_preflight/git_*`, `relevant_diff_before.patch`) | **done** |
| specification §2 software check (`00_preflight/software_check.json`) | **done** |
| specification §1 seed freshness audit (`00_preflight/seed_freshness_audit.json`) | **done** |
| repository mapping: real generator, decoders, cost pipeline, evaluation, provenance | **done** |
| location of real degree-distribution source (`v4_final_report.json` etc.) | **done** |
| location of prior implementations (Threshold-MM, Hungarian/LSAP path, SUPPORT masking, dual-softmax record) | **done** |
| no destructive git operation performed; pre-existing dirty tree (3,148 entries) left untouched | **confirmed** |

No R7 experiment data was generated. No historical seed was run.

---

## 6. Steps not completed (all gated on the blocker)

Everything downstream of the selection block:

* Stage 0A degree calibration from the frozen v4/v5 audit artifacts
* generator degree-distribution extension + 24-family expansion + structural QA
* `206–215`-equivalent selection run (rule candidates, Threshold-MM calibration, Dual-Softmax calibration)
* protocol freeze / hash lock / `locked_spec.json`
* one-shot `401–410` confirmatory execution
* independent validation, H1/H2/S1/S2 analysis, gates A–E, figures, manuscript patches

---

## 7. Why this is reported rather than self-repaired

The specification forbids the agent from repairing a hard blocker by relaxing a scientific
constraint:

> 不要通过放宽科学约束"修复"硬阻塞。

Substituting a different seed block for the selection stage is a **protocol amendment**: it
changes a pre-registered input of the study. That decision belongs to the protocol owner, not
to the executor, even when the substitute is a strictly *stronger* choice (a genuinely fresh
block). The `R7 selection runner 只接受：206–215` rule is explicit and is not the executor's
to rewrite.

Two candidate resolutions were identified and are put to the protocol owner:

| # | selection block | fresh seeds preserved | note |
|---|---|---|---|
| **A** | `206–211 + 112–115` | 6 of 10 from the intended block | smallest deviation from the written spec; non-contiguous |
| **B** | `112–121` | 0 | fully contiguous, fully fresh, maximum distance from every used block |
| **C** | `312–321` | 0 | fully contiguous, fully fresh, sits between the used `212–311` block and the reserved `401–410` holdout |
| **D** | keep `206–215`, treat `212–215` as selection-only | n/a | requires accepting that 4/10 selection seeds were historically observed; the executor cannot authorise this |

For all options the confirmatory holdout remains `401–410`, untouched and available.

---

## 8. Blocker classification

```
BLOCKER            : SELECTION_BLOCK_ALREADY_USED
SEVERITY           : hard (specification §71)
HOLDOUT_TOUCHED    : NO
R7_DATA_GENERATED  : NO
R7_RESULT_PACKAGE  : NOT PRODUCED
REPAIRABLE_BY_AGENT: NO  (requires a protocol-owner decision on the selection seed block)
```
