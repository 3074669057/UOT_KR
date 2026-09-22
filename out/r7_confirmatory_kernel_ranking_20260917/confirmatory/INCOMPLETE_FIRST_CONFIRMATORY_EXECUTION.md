# R7 confirmatory first execution — INCIDENT RECORD

* classification: **`INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION`**
* experiment: `r7_confirmatory_kernel_ranking_20260917`
* first touch (ledger written): **2026-09-17T07:32:40Z**
* locked spec sha256: `392f88f654d8bc97c1f3d42d77ae506991940f798d266854ff54ee26a89b0bd6`
* executor sha256: `65f0feef440d70c74888645ba8984dc1ef6475926b18e1c8cc2f6991e758341b`
* status: **confirmatory stage NOT completed; no confirmatory result package exists**

---

## 1. What happened

The one-shot confirmatory executor passed all eight pre-execution verifications, then, in
the required order:

1. atomically exclusive-created `confirmatory/CONFIRMATORY_TOUCH_ONCE.json`
   (`O_CREAT|O_EXCL`) — `HOLDOUT_TOUCHED = YES`;
2. created `confirmatory/raw/`;
3. began the unit loop and **generated the first confirmatory unit**,
   `Celer / seed 401`;
4. crashed on a plumbing defect **before writing any unit result file**.

## 2. Exact failure

```
[2026-09-17T07:32:40Z] [one-shot] touch ledger written -> CONFIRMATORY_TOUCH_ONCE.json; HOLDOUT_TOUCHED=YES
Traceback (most recent call last):
  File "scripts/run_r7_confirmatory_kernel_ranking.py", line 556, in main
    return execute()
  File "scripts/run_r7_confirmatory_kernel_ranking.py", line 257, in execute
    a, b = ctx.a, ctx.b
AttributeError: 'CellContext' object has no attribute 'a'
```

`CellContext` stores the marginals on the wrapped cell object, not as direct attributes.
The fix is a one-line change (`ctx.cell["a"]`, `ctx.cell["b"]`). **No method, threshold,
generator parameter, metric, gate or cost setting is implicated.**

## 3. Scientific state at the moment of the crash

| question | answer |
|---|---|
| Was the holdout touched? | **YES** — the ledger exists and `Celer/401` was generated |
| Were any confirmatory *results* produced? | **NO** |
| Was any unit result file written? | **NO** — `raw/units/*.json` does not exist |
| Was `INDEX.json` written? | **NO** |
| Were executor metric tables written? | **NO** |
| Was Gate D evaluated? | **NO** |
| Was any method metric observed or reported? | **NO** — the crash precedes the first metric log line |
| Was any parameter changed after seeing holdout data? | **NO** — no holdout data was seen |

Evidence on disk (`confirmatory/`):

```
CONFIRMATORY_TOUCH_ONCE.json                 1385 bytes   (ledger, created 07:32:40Z)
gate_pre_execution.json                      2371 bytes   (8/8 PASS)
raw/units/_scratch/Celer/seed_401/           generated data for ONE unit:
    cost.npz                  5,957,774 bytes
    transport_uot.npz           755,448 bytes
    ids.npz                      35,290 bytes
    labels.csv                   79,464 bytes
    flow_segments_eth_synth.csv 167,203 bytes
    flow_segments_bnb_synth.csv 189,465 bytes
    solver.json                   1,500 bytes
    labels/synthetic_flow_labels.csv        79,092 bytes
    labels/synthetic_uot_eval_metrics.json 215,838 bytes
INDEX.json                                  ABSENT
VALIDITY_GATE_D.json                        ABSENT
execution_summary.json                      ABSENT
```

## 4. Why the executor now refuses to run

The executor checks `one_shot_ledger_absent` before anything else. The ledger exists, so
a second `--execute` refuses. This is by construction, not a bug:

> 一旦该文件存在：confirmatory executor 永久拒绝第二次执行。

The only way to re-run `401-410` would be to delete the ledger, which the specification
forbids unconditionally:

> 不得删除 ledger 后重跑。

## 5. Root cause and process lesson

The freeze-time verification of the executor tested **hashes only**
(`--verify-only`), not the execution path. The executor hard-codes its confirmatory seed
block (correctly, so that no seed interface exists), which meant its *logic* could not be
exercised on selection seeds before the freeze without adding the very interface the
specification forbids. The defect was therefore only reachable on first real use.

**This is a defect in the executor's plumbing and in the pre-freeze test strategy, not in
the scientific protocol.**

## 6. Options (protocol-owner decision required)

| # | option | effect on confirmatory validity |
|---|---|---|
| **A** | Stop. Publish the selection stage only; classify the confirmatory stage `INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION`. | no confirmatory claim at all |
| **B** | Fix the plumbing defect and resume the one-shot on `401-410`. | requires overriding the "no second execution" rule; `Celer/401` would be a pre-generated unit |
| **C** | Retire `401-410` as spent and run the **unchanged frozen protocol** on a fresh, never-touched confirmatory block. | preserves a genuinely untouched holdout; consistent with the specification's own rule for a spent holdout ("应使用新的后续协议") |

Option C is consistent with the specification's stated principle that a touched holdout is
never re-used and that a **new subsequent protocol** is used instead, and it keeps the
frozen protocol (rule, thresholds, cost, gates, statistics) completely unmodified. It
requires only replacing the confirmatory *seed block* and re-freezing, which is not a
result-driven change because no result exists.

## 7. What is preserved

* the ledger — never deleted
* `gate_pre_execution.json` (8/8 PASS)
* the partially generated unit data for `Celer/401` — never deleted
* this incident record
* the full locked spec, frozen manifest and selection package, all intact

## 8. What is true regardless of the decision

* the R7 **selection** stage is complete, frozen and unaffected
* `TRUNCATION_BOUNDARY_DEPENDENCE` and every selection-stage number are unaffected
* no fix to any part of the protocol is required or permitted by this incident
* `401-410` can never again serve as an untouched confirmatory holdout
