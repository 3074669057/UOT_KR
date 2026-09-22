# R7 git change report

* git HEAD: `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (branch `master`)
* pre-existing dirty entries before R7: **3148**
* dirty entries now: **3153**
* newly appearing entries: **5**

## Safeguards honoured

No `git reset --hard`, `git clean -fd`, `git clean -fdx`, `git restore .`,
`git checkout .`, full stash, or whole-repo commit was performed. The pre-existing
dirty working tree was left exactly as found and was never reverted.

## Files R7 created

### Code

| file | bytes |
|---|---:|
| `scripts/run_r7_preflight.py` | 14216 |
| `scripts/run_r7_selection.py` | 76190 |
| `scripts/run_r7_confirmatory_kernel_ranking.py` | 27821 |
| `scripts/validate_r7_confirmatory_results.py` | 19267 |
| `scripts/r7/__init__.py` | 772 |
| `scripts/r7/r7_common.py` | 14079 |
| `scripts/r7/r7_degree.py` | 22933 |
| `scripts/r7/r7_generator.py` | 20860 |
| `scripts/r7/r7_methods.py` | 19164 |
| `scripts/r7/r7_pipeline.py` | 18569 |
| `scripts/r7/r7_qa.py` | 16563 |
| `scripts/r7/selftest_generator_equivalence.py` | 3524 |
| `scripts/r7/dryrun_executor_paths.py` | 9293 |
| `scripts/r7/r7_analysis.py` | 23366 |
| `scripts/r7/r7_figures.py` | 14787 |
| `scripts/r7/r7_reporting.py` | 83701 |
| `scripts/r7/finalize_r7.py` | 2270 |
| `scripts/r7/audit_deliverables.py` | 5078 |

### Experiment output (all under `out/r7_confirmatory_kernel_ranking_20260917/`)

* files: **1225**
* total bytes: **785,830,960**
* these are the ONLY experiment artefacts R7 produced; no R5/R6/R9/R10 output
  directory was written to

## Files R7 modified

| file | change |
|---|---|
| `src/cross/domain/evaluation/semi_synthetic_flows.py` | generator extended **in place** (degree sampling + 24-family grid) |

The extension is strictly additive: with the R7 options unused the generator is
**byte-identical** to the frozen generator, proven on 5 replay cases in
`selection/generator/generator_equivalence.json` (`ALL_PASS = true`).

## Task-specific diff (modified source only)

```diff
diff --git a/src/cross/domain/evaluation/semi_synthetic_flows.py b/src/cross/domain/evaluation/semi_synthetic_flows.py
index dcb02a3..d1313ce 100644
--- a/src/cross/domain/evaluation/semi_synthetic_flows.py
+++ b/src/cross/domain/evaluation/semi_synthetic_flows.py
@@ -118,6 +118,136 @@ def pick_semi_synthetic_seeds(
     return rows, {"mode": "head_confidence", "sampled_total": len(rows)}
 
 
+# --------------------------------------------------------------------------- #
+# R7 family grid
+# --------------------------------------------------------------------------- #
+# The frozen design stratifies anchors on 4 amount quartiles x 3 delay tertiles
+# = 12 cells.  R7 refines the delay axis to sextiles, which splits every frozen cell
+# into two halves:
+#   * 12 "original" families = the FIRST half of each frozen cell  (delay sextiles 0,2,4)
+#   * 12 "new"      families = the SECOND half of each frozen cell (delay sextiles 1,3,5)
+# Every family gets its own distinct base anchor, so no base-anchor cluster is shared.
+R7_AMOUNT_QUARTILES = 4
+R7_DELAY_SEXTILES = 6
+R7_INSTANCES_PER_FAMILY = 2
+
+
+def _r7_family_layout() -> list[dict[str, Any]]:
+    """The frozen 24-family layout: (amount quartile, delay sextile) -> family index."""
+    out: list[dict[str, Any]] = []
+    for a in range(R7_AMOUNT_QUARTILES):          # 0..3
+        for d in range(R7_DELAY_SEXTILES):        # 0..5
+            out.append({
+                "family_index": a * R7_DELAY_SEXTILES + d,
+                "amount_quartile": a,
+                "delay_sextile": d,
+                "delay_tertile": d // 2,
+                "half_of_frozen_cell": "first" if d % 2 == 0 else "second",
+                "original": (d % 2 == 0),
+            })
+    return out
+
+
+def pick_r7_family_seeds(
+    fl: pd.DataFrame,
+    *,
+    rng: random.Random,
+    instances_per_family: int = R7_INSTANCES_PER_FAMILY,
+) -> tuple[list[dict[str, Any]], dict[str, Any]]:
+    """Pick the R7 template rows: one DISTINCT base anchor per (family, instance).
+
+    Returns rows carrying ``_r7_family`` (0..23) and ``_r7_instance`` (0..k-1).  Anchors
+    are drawn without replacement, so the base-anchor clusters of the 24 families are
+    disjoint by construction.
+    """
+    diag: dict[str, Any] = {"mode": "r7_family_grid", "cells": [], "backfill": []}
+    if fl.empty:
+        return [], diag
+
+    pool = fl[fl.get("pattern_type", "").astype(str).str.lower() == "one_to_one"].copy()
+    if pool.empty:
+        pool = fl.copy()
+    pool["_amt"] = pd.to_numeric(pool.get("src_amount_usd"), errors="coerce").fillna(0.0)
+    pool["_delay"] = pd.to_numeric(pool.get("median_delay_sec"), errors="coerce").fillna(0.0)
+    pool["_amt_q"] = _assign_quantile_labels(pool["_amt"], R7_AMOUNT_QUARTILES, prefix="Q")
+    pool["_delay_q"] = _assign_quantile_labels(pool["_delay"], R7_DELAY_SEXTILES, prefix="S")
+    pool["_cell"] = pool["_amt_q"].astype(str) + "|" + pool["_delay_q"].astype(str)
+
+    layout = _r7_family_layout()
+    wanted = {f"{('Q' + str(c['amount_quartile']))}|{'S' + str(c['delay_sextile'])}": c
+              for c in layout}
+
+    used: set[int] = set()
+    chosen: list[tuple[int, dict[str, Any]]] = []
+    for cell, meta in sorted(wanted.items(), key=lambda kv: kv[1]["family_index"]):
+        grp = pool[pool["_cell"] == cell]
+        idxs = [int(i) for i in grp.index]
+        rng.shuffle(idxs)
+        take = idxs[:instances_per_family]
+        diag["cells"].append({
+            "cell": cell, "family_index": meta["family_index"],
+            "available": len(idxs), "sampled": len(take),
+            "deficit": instances_per_family - len(take),
+        })
+        for inst, i in enumerate(take):
+            used.add(i)
+            chosen.append((meta["family_index"], {**pool.loc[i].to_dict(),
+                                                  "_r7_family": meta["family_index"],
+                                                  "_r7_instance": inst,
+                                                  "_r7_cell": cell}))
+
+    need = len(layout) * instances_per_family - len(chosen)
+    if need > 0:
+        remaining = [int(i) for i in pool.index if int(i) not in used]
+        rng.shuffle(remaining)
+        # backfill deterministically into the families that came up short
+        counts: dict[int, int] = {}
+        for f, _ in chosen:
+            counts[f] = counts.get(f, 0) + 1
+        for f in [c["family_index"] for c in layout]:
+            while counts.get(f, 0) < instances_per_family and remaining:
+                i = remaining.pop()
+                counts[f] = counts.get(f, 0) + 1
+                chosen.append((f, {**pool.loc[i].to_dict(), "_r7_family": f,
+                                   "_r7_instance": counts[f] - 1,
+                                   "_r7_cell": str(pool.loc[i, "_cell"])}))
+                diag["backfill"].append({"index": i, "family_index": f})
+
+    diag["sampled_total"] = len(chosen)
+    diag["family_count"] = len({f for f, _ in chosen})
+    diag["instances_per_family"] = instances_per_family
+    diag["unique_base_anchors"] = len({str(r.get("src_flow_id")) for _, r in chosen})
+    diag["layout"] = layout
+    return [r for _, r in chosen], diag
+
+
+def r7_split_suffix(i: int) -> str:
+    """Deterministic leg suffix; index 0/1 reproduce the frozen ``a``/``b`` names."""
+    if i < 26:
+        return chr(ord("a") + i)
+    return f"z{i}"
+
+
+def r7_template_prefix(base_flow_id: str, family_index: int, instance: int) -> str:
+    """R7 instance prefix.
+
+    ``<base_flow_id>__r7fam<FF>__inst<II>``.  ``tpl_of`` of any synthetic id built on
+    this prefix returns the prefix itself, so a template id is exactly the instance
+    prefix; the family key is the part before ``__inst``.
+    """
+    return f"{base_flow_id}__r7fam{int(family_index):02d}__inst{int(instance):02d}"
+
+
+def r7_family_key(template_id: str) -> str:
+    s = str(template_id)
+    return s.split("
```

## Newly appearing dirty entries

Note: `scripts/run_r7_preflight.py` and the `out/r7_.../` output directory were
created *before* the pre-task status snapshot was captured, so they already
appear in the `before` snapshot and are not listed as newly appearing. They are
still R7-created files and are enumerated explicitly above.

```text
 M src/cross/domain/evaluation/semi_synthetic_flows.py
?? scripts/r7/
?? scripts/run_r7_confirmatory_kernel_ranking.py
?? scripts/run_r7_selection.py
?? scripts/validate_r7_confirmatory_results.py
```

## Entries that were dirty BEFORE R7

These are NOT attributable to R7 and were not touched.

```text
```

## `git diff --stat` (tracked changes, whole repository)

```text
 .../src/cross/domain/token/route_registry.py       |      82 -
 review_pack/src/cross/domain/uot/__init__.py       |      26 -
 .../domain/uot/candidate_recall_diagnostics.py     |     824 -
 review_pack/src/cross/domain/uot/cost_matrix.py    |     286 -
 .../src/cross/domain/uot/decode_transport.py       |     368 -
 .../domain/uot/flow_uot_candidate_subgraph.py      |     909 -
 review_pack/src/cross/domain/uot/uot_solver.py     |     136 -
 .../src/cross/domain/uot/uot_solver_numpy.py       |      44 -
 .../src/cross/domain/validation/__init__.py        |       5 -
 .../domain/validation/paper_artifact_validator.py  |     196 -
 .../src/cross/domain/validation/rc_uot_exports.py  |     299 -
 review_pack/src/cross/experiments/__init__.py      |       1 -
 .../src/cross/experiments/ablation_suite.py        |      65 -
 .../src/cross/experiments/export_paper_tables.py   |     135 -
 .../src/cross/experiments/paper_finalize_bundle.py |     656 -
 review_pack/src/cross/infrastructure/__init__.py   |       1 -
 .../src/cross/infrastructure/config/__init__.py    |       1 -
 .../src/cross/infrastructure/config/loader.py      |      73 -
 .../src/cross/infrastructure/config/service.py     |      12 -
 .../src/cross/infrastructure/evidence_validate.py  |      89 -
 .../src/cross/infrastructure/logging_bootstrap.py  |       7 -
 .../src/cross/infrastructure/online/__init__.py    |      10 -
 .../infrastructure/online/bnb_window_fetch.py      |    1614 -
 .../src/cross/infrastructure/online/eth_rpc.py     |      23 -
 .../infrastructure/online/evm_json_rpc_client.py   |     240 -
 .../online/receipt_transfer_verify.py              |     448 -
 review_pack/src/cross/infrastructure/run_init.py   |     127 -
 review_pack/src/cross/interfaces/__init__.py       |       1 -
 review_pack/src/cross/interfaces/cli.py            |     374 -
 review_pack/src/cross/legacy/__init__.py           |      19 -
 review_pack/src/cross/legacy/abct_tx_adapter.py    |     102 -
 review_pack/src/cross/legacy/locator_path.py       |      31 -
 review_pack/src/cross/reporting/__init__.py        |       1 -
 review_pack/src/cross/reporting/audit_report.py    |     540 -
 review_pack/src/cross/shared/__init__.py           |       1 -
 review_pack/src/cross/shared/amount_hints.py       |      92 -
 review_pack/src/cross/shared/amount_normalizer.py  |      61 -
 review_pack/src/cross/shared/bnb_pick.py           |      74 -
 review_pack/src/cross/shared/connector_decimals.py |      12 -
 review_pack/src/cross/shared/decimals_registry.py  |     113 -
 review_pack/src/cross/shared/label_availability.py |      21 -
 review_pack/src/cross/shared/label_index.py        |      22 -
 review_pack/src/cross/shared/label_split.py        |      49 -
 review_pack/src/cross/shared/load_csv.py           |      32 -
 review_pack/src/cross/shared/normalize.py          |       8 -
 review_pack/src/cross/shared/token_map.py          |      47 -
 review_pack/src/cross/shared/transfers.py          |      79 -
 review_pack/src/cross/shared/tx_hash.py            |      30 -
 review_pack/src/cross/utils/__init__.py            |       5 -
 review_pack/src/cross/utils/safe_cast.py           |     185 -
 .../application/experiments/run_uot_matching.py    |     107 +-
 src/cross/application/pipeline.py                  |      46 +
 src/cross/domain/evaluation/flow_metrics.py        |      30 +-
 .../domain/evaluation/semi_synthetic_flows.py      |     263 +-
 src/cross/domain/path_b/runner.py                  |     114 +
 src/cross/domain/uot/cost_matrix.py                |      65 +-
 src/cross/domain/uot/decode_transport.py           |      31 +-
 src/cross/interfaces/cli.py                        |      83 +-
 tests/test_cost_matrix.py                          |      10 +
 2820 files changed, 1715 insertions(+), 8737865 deletions(-)
```

