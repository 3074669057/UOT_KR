# PENDING1_WEIGHT_VALUES.md

**Machine-extracted values of the cost weights `w_k`, with exact provenance.**
Produced for PENDING-1 of the R11 remaining-evidence review.

This is a **fact sheet**, not manuscript text. Nothing was chosen, interpreted or reconciled
here: every number below is read from a named file at a named line, and every open question is
left open. No manuscript, DOCX, TeX or BibTeX file was modified; no experiment was run.

Origin traces: `SOURCE_TRACE.md` TRACE A. Packaged sources: `core/pending1_cost_weights/`.
All hashes are SHA-256 of the origin file as it exists in the working tree.

---

## 0. The one sentence that matters

There is **no identifier named `w_k`** anywhere in the repository, and the two files named by
the manuscript's own marker (`dev_candidate/af_common.py`, `decoder_audit/da_common.py`) carry
**five** cost weights and **zero** cost weights respectively. The full cost-weight object lives
in a **third** file, `src/cross/domain/uot/cost_matrix.py`, and it has **seven** keys of which
**six** are non-zero after the frozen graph merge.

---

## 1. Set F — the full frozen pairwise cost (used by the structural stress test, §4.2 line)

**Definition:** `src/cross/domain/uot/cost_matrix.py:32-41`
sha256 `189dcddc157ac47479344ff092f8272af3b1111a83d45864be7d27036e52e813` (10,023 B)

```python
def default_cost_weights() -> dict[str, float]:
    return {
        "amount": 0.35,
        "time": 0.25,
        "route": 0.15,
        "risk": 0.15,
        "graph": 0.05,
        "evidence": 0.05,
        "novelty": 0.05,
    }
```

**Applied at:** `cost_matrix.py:207-215` — the cost is the plain weighted sum of the seven
component arrays, clamped to `[0, 2]` at `:216`.

**Missing-key behaviour:** `_normalize_weights()` `cost_matrix.py:44-50` copies the caller's
dict, renames a legacy `bridge` key to `route`, and `setdefault`s every key of
`default_cost_weights()` back in. A caller that supplies a partial dict therefore does **not**
get zero-filled components — it silently gets the defaults.

### 1.1 Graph merge (this is what makes it "six")

**`cost_matrix.py:77-82`**

```python
def merge_graph_weight_into_amount(weights: dict[str, float]) -> dict[str, float]:
    w = dict(weights)
    g = float(w.pop("graph", 0.0))
    w["amount"] = float(w.get("amount", 0.35)) + g
    w["graph"] = 0.0
    return w
```

**Called when:** `build_cost_matrix_decomposed(..., use_graph=False)` → `cost_matrix.py:121-122`.
`uot_use_graph_embedding = False` in every frozen parameter dict, and
`graph_cost` is forced to `0.0` at `:194` in that branch.

### 1.2 Set F after the merge — the six non-zero absolute weights

| # | Component (code key) | Component array | Absolute weight (Σ = 1.05) | Share of 1.05 |
|---|---|---|---|---|
| 1 | `amount` (incl. merged `graph`) | `amount_cost` | **0.40** | 0.380952 |
| 2 | `time` | `time_cost` | **0.25** | 0.238095 |
| 3 | `route` | `route_cost` | **0.15** | 0.142857 |
| 4 | `risk` | `risk_cost` | **0.15** | 0.142857 |
| 5 | `evidence` | `evidence_cost` | **0.05** | 0.047619 |
| 6 | `novelty` | `address_novelty_cost` | **0.05** | 0.047619 |

Component construction sites, all in `cost_matrix.py`:
`amount_cost` `:166-167` (relative USD error, capped at 1.0) · `time_cost` `:148-155` (via
`populate_delay_cost_arrays`, delay window `max_delay_sec`, causal penalty
`causal_violation_penalty`) · `risk_cost` `:179` (`|aml_score − evidence_quality_score|`,
capped) · `route_cost` `:185` (via `_route_type_cost` `:85-105`) · `evidence_cost` `:190`
(via `evidence_penalty_from_level` `:53-74`) · `address_novelty_cost` `:197-205`
(`1 − Jaccard(address_set)`, default `0.5` when either side has no address set).

---

## 2. Set D4 — the amount-free cost actually used by the confirmatory experiment (§4.3 line)

**Definition:** `scripts/multi_bridge/dev_candidate/af_common.py:31-47`
sha256 `a09c386722cacc8ae1df2a175f08de17be2209fbed1a63f024f0e396b007dcf4` (3,564 B)

```python
KEPT_ABS_WEIGHTS = {"time": 0.25, "route": 0.15, "risk": 0.15, "evidence": 0.05,
                    "novelty": 0.05}
COMPONENT_OF = {"time": "time_cost", "route": "route_cost", "risk": "risk_cost",
                "evidence": "evidence_cost", "novelty": "address_novelty_cost"}
RENORM_DENOM = float(sum(KEPT_ABS_WEIGHTS.values()))   # 0.65, per the spec

def primary_weights() -> dict[str, float]:
    """PRIMARY amount-free renormalized weights (exact fractions, spec-locked)."""
    return {k: v / RENORM_DENOM for k, v in KEPT_ABS_WEIGHTS.items()}

def ablation_weights() -> dict[str, float]:
    """ABLATION unrenormalized weights (exact LOCO condition, spec-locked)."""
    return dict(KEPT_ABS_WEIGHTS)
```

**Applied at:** `af_common.py:50-64` (`build_amount_free_costs`). The amount component is
**removed from the pairwise cost only**; the amount-derived marginals (`a` = risk-weighted USD,
`b` = evidence-weighted USD) are preserved unchanged (`af_common.py:9-10`).

### 2.1 Set D4 — five weights, two normalisations

| # | Component (code key) | Absolute (`ablation_weights()`) = LOCO | Renormalised (`primary_weights()`) = PRIMARY |
|---|---|---|---|
| 1 | `time` | 0.25 | **5/13 = 0.384615384615** |
| 2 | `route` | 0.15 | **3/13 = 0.230769230769** |
| 3 | `risk` | 0.15 | **3/13 = 0.230769230769** |
| 4 | `evidence` | 0.05 | **1/13 = 0.076923076923** |
| 5 | `novelty` | 0.05 | **1/13 = 0.076923076923** |
| — | sum | 0.65 | 1.000000000000 |

Both are exact rationals because `RENORM_DENOM` is computed as `sum(KEPT_ABS_WEIGHTS.values())`,
not written as a literal.

### 2.2 The hash-locked spec that fixes Set D4

`ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/NEXT_CANDIDATE_SPEC.md`
sha256 `979e5931ced5e070c1d071a9c0199de85fc41d9e22cdbd54b4e00347119b72ec` (7,200 B)

Its §"PRIMARY CANDIDATE" states the renormalised fractions
`time 0.3846, route 0.2308, risk 0.2308, evidence 0.0769, novelty 0.0769` and the frozen
absolutes `0.25 / 0.15 / 0.15 / 0.05 / 0.05`, summing to `0.65`. The file declares its own lock
hash `45f58395d05e95b556ce918fd249a7d7044ee7e428c8b2305a6d2c41aeaa625b`; the companion
`next_candidate_spec_lock.json` and `preaudit_hashes.json` are packaged alongside it.

### 2.3 Calibration question — answered by the source itself

The manuscript marker asks "是否使用开发集标签校准". The code answer is **no**:

| Evidence | Location |
|---|---|
| Weights documented as "from the locked spec; do not re-derive" | `af_common.py:3-8` |
| `primary_weights()` docstring: "exact fractions, spec-locked" | `af_common.py:41` |
| `ablation_weights()` docstring: "exact LOCO condition, spec-locked" | `af_common.py:46` |
| Verifier asserts `abs(pw[k] − v/0.65) < 1e-12` for every kept weight | `verify_amount_free_dev.py:52-57` — sha256 `20e8d2fb278966e27ef66c976f41d2b045b50a74506c0228a83c249bddba82e3` |
| Verifier re-asserts the same identity | `verify_conditional_plan_dev.py:77-82` — sha256 `e9f844a2a58ede078f229174de957250b824e57906c45fc4981ef7a37d53d293` |
| Verifier asserts the UOT parameters are frozen | `verify_amount_free_dev.py:121-123` |
| Spec is derived from dev seeds 201–205 and the holdout 301–305 was never read at that point | `NEXT_CANDIDATE_SPEC.md:1-10, 84-99` |

No code path in the repository fits `w_k` to development-set labels; the weights are hard-coded
constants. (Contrast: `THRESHOLD_MM_CUTOFF = 0.47762288884480164`,
`decoder_audit/da_common.py:41`, sha256 `fff26071d54e813cf087122399b8dee4ad4c92c7652b4d3d6a7ba1e2f8bfd77c`,
**is** described in the manuscript as determined by a calibration protocol with no
development-selection step.)

---

## 3. What `decoder_audit/da_common.py` actually contains for PENDING-1

`scripts/multi_bridge/decoder_audit/da_common.py:33-41`
sha256 `fff26071d54e813cf087122399b8dee4ad4c92c7652b4d3d6a7ba1e2f8bfd77c` (15,328 B)

```python
FROZEN_PARAMS = {
    "uot_reg": 0.05, "uot_reg_m": 0.5, "uot_lambda_risk": 0.25,
    "uot_decode_threshold": 1e-9, "uot_max_delay_sec": 21600.0,
    "uot_causal_violation_penalty": 5.0, "uot_backend": "pot",
    "uot_allow_unmatched": True, "uot_use_graph_embedding": False,
}
THRESHOLD_MM_CUTOFF = 0.47762288884480164
```

These are **solver / decoding / risk parameters**, not cost weights. Two further, independent
copies of the same dict exist (`baseline_mechanism/common.py:29-39`,
sha256 `68b0beda9a6c2f29cd96cb569958f817e222093f0f80f2f79c75c13739aa0781`; and
`run_faithful_flow_structural.py:46-57`, sha256
`0c5928bdd80397dea3ddf056b421f89e8373e1a7ab06544d2fe29e5ad9130191`, which adds
`"cost_weights": default_cost_weights()`). The confirmatory identity check binds the weights
through `af_common` and these parameters through `da_common`
(`holdout_common.py:90-97`).

---

## 4. A second, unrelated six-weight set that must not be conflated

The R11 §3.6 coverage-tier paragraph describes an **evidence score** built from six weighted
inputs with weights `0.25 / 0.20 / 0.20 / 0.15 / 0.10 / 0.10`. Those are **not** the cost
weights. Source: `tools/cross_aml/cross_aml/feature_builder.py:26-31`
sha256 `c0acc1d08c3ea461f952e79d3fcd00a347831cfec8560593de5d436ee1b0b389` (3,470 B)

```python
evidence = (
    0.25 * bridge_s      # bridge_event_score
    + 0.20 * tk_match    # transfer_key_match
    + 0.20 * amount_c    # amount_consistency
    + 0.15 * time_c      # time_causality_score
    + 0.10 * token_c     # token_consistency
    + 0.10 * addr_ov     # address_overlap_score
)
```

Recorded here only so that the two six-weight families are not merged. The relation between
this open-source evidence-score rule and the frozen confirmatory tier rule is a separate open
item (`SOURCE_TRACE.md` TRACE B; the R11 §3.6 paragraph carries its own `【待核实】`).

---

## 5. Facts that prevent this marker from being closed by evidence alone

1. **Pointer mismatch.** The marker names `dev_candidate/af_common.py` and
   `decoder_audit/da_common.py`. The first provides **5** cost weights (Set D4), the second
   provides **0**. The 6/7-key weight object is in a third file, not named by the marker.
2. **Seven keys vs "six".** `default_cost_weights()` has 7 keys. The count becomes 6 only after
   the frozen `merge_graph_weight_into_amount()` folds `graph` into `amount` — i.e. under
   `use_graph=False`. Whether the manuscript's "六项" refers to this folded form is not stated
   anywhere in the repository.
3. **The enumerated names do not match the code keys.** R11 §3.2 lists "金额误差、时间因果与延迟、
   桥/路径一致性、AML 风险、图上下文与证据质量". Mapped to code keys this is
   `amount`, `time`, `route`, `risk`, `graph`, `evidence` — six names, in which `graph` is the
   very term that is zero after the merge, while **`novelty`**, the sixth non-zero component, is
   **not** in the list. The two readings give different members of the weight vector.
4. **No `w_k` symbol.** The manuscript uses a continuous index `k` over six items; the code uses
   named keys. There is no place in the repository where the six are numbered 1…6, so no source
   order can be cited for `w_1 … w_6`.

**Author decision required (not made here):** which of the following the manuscript's "六项权重
w_k" denotes —
(a) Set F without the graph term, i.e. `amount 0.40, time 0.25, route 0.15, risk 0.15,
evidence 0.05, novelty 0.05` (Σ 1.05), or
(b) the six names as enumerated in §3.2 (`amount, time, route, risk, graph, evidence`, with
`graph = 0.05` unmerged), or
(c) Set D4, the five amount-free weights, if §3.2 is meant only to introduce the cost family
while §4.3 states the weights actually used.

Each reading is fully supported by packaged, hash-verified source; the evidence package does not
choose among them.

---

## 6. File-hash index for this sheet

| File | SHA-256 |
|---|---|
| `src/cross/domain/uot/cost_matrix.py` | `189dcddc157ac47479344ff092f8272af3b1111a83d45864be7d27036e52e813` |
| `scripts/multi_bridge/dev_candidate/af_common.py` | `a09c386722cacc8ae1df2a175f08de17be2209fbed1a63f024f0e396b007dcf4` |
| `scripts/multi_bridge/decoder_audit/da_common.py` | `fff26071d54e813cf087122399b8dee4ad4c92c7652b4d3d6a7ba1e2f8bfd77c` |
| `scripts/multi_bridge/baseline_mechanism/common.py` | `68b0beda9a6c2f29cd96cb569958f817e222093f0f80f2f79c75c13739aa0781` |
| `scripts/multi_bridge/run_faithful_flow_structural.py` | `0c5928bdd80397dea3ddf056b421f89e8373e1a7ab06544d2fe29e5ad9130191` |
| `scripts/multi_bridge/verify_amount_free_dev.py` | `20e8d2fb278966e27ef66c976f41d2b045b50a74506c0228a83c249bddba82e3` |
| `scripts/multi_bridge/verify_conditional_plan_dev.py` | `e9f844a2a58ede078f229174de957250b824e57906c45fc4981ef7a37d53d293` |
| `.../cost_diag/NEXT_CANDIDATE_SPEC.md` | `979e5931ced5e070c1d071a9c0199de85fc41d9e22cdbd54b4e00347119b72ec` |
| `tools/cross_aml/cross_aml/feature_builder.py` | `c0acc1d08c3ea461f952e79d3fcd00a347831cfec8560593de5d436ee1b0b389` |

All nine files are packaged under `core/pending1_cost_weights/` and `core/pending2_coverage_tiers/`
and are listed in `SHA256SUMS.txt`.
