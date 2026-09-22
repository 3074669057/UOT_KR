# S10 feasibility report

* experiment: `r7_posthoc_unmatched_mass_localization_20260919`
* classification: **POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS**
* provenance mode: **`A_ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS`**
* locked spec sha256: `b65ce71527534d811fce2dbd84878d6c2cb0407365577ef1ca7cd7f94bfdb6b9`

## F1 — is per-node unmatched mass recoverable?

availability tier: **Tier A**

| item | value |
|---|---|
| confirmatory units | 30 |
| seeds | `[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]` |
| templates | **1440** |
| every unit has per-node `delta_S` and `delta_T` | **True** |
| forbidden seeds used | `[]` |

The archive additionally stores the frozen source/target id lists, the frozen marginals `a`/`b`, the realised masses, the totals, and the full truth structure, so localization can be computed **without re-solving anything**.

## Delta definition — located in the frozen source, never assumed

* delta^S_i = a_i - sum_j P_ij and delta^T_j = b_j - sum_i P_ij, where a is the NORMALISED risk-weighted source mass and b the NORMALISED evidence-weighted target mass of the frozen UOT solve

```text
delta_S_i = a_i - sum_j P_ij
delta_T_j = b_j - sum_i P_ij
```

* source: `5a125184154213c736a41bff8597237d01444372c0b805f35cb6ce4ca552f5bb` (`build_unit` lines 90-91); normalization in `r7_generator.py` lines 258-259
* units: both marginals are normalised to unit total mass, so delta^S and delta^T are in units of total source / target mass and lie in [0, 1] up to the over-delivery case
* normalization: `a_rw / sum(a_rw) and b_ev / sum(b_ev)  (frozen r7_generator.py build_cell lines 258-259)`
* sign convention: delta > 0 = mass requested by the marginal but NOT realised by the plan (unmatched / marginal deficit); delta < 0 would mean the plan over-delivers relative to the marginal
* total relation: sum_i delta^S_i = 1 - sum_ij P_ij = sum_j delta^T_j, i.e. the source-side and target-side total deficits are equal by construction of the unbalanced solve

### Independent re-derivation from the archived frozen plan

* units checked: 30
* worst absolute difference (vectors, totals, source/target gap, mass conservation): **4.441e-16** (tolerance 1e-12)
* ALL MATCH: **True**

## Truth labels

| item | value |
|---|---|
| templates with exactly 1 true unmatched source | **1440/1440** |
| templates with exactly 2 decoy targets | **1440/1440** |
| all ids unique | True |
| unmatched source carries a positive truth edge | 0 (expected 0) |
| all matched sources have >= 1 positive truth edge | True |

### Structural deviation from the task's assumed template (recorded, not forced)

* task section 6 assumed: decoy targets should have NO positive truth edge
* actual R7 generator: a decoy pair is a labelled positive truth edge: truth_structure defines positive = split UNION merge UNION decoy, so each of the two injected decoy targets DOES carry a positive (decoy-labelled) truth edge from its decoy source
* resolution: the real generator structure is used and reported; the task's assumed QA criterion is recorded as not applicable rather than forcing a template that the generator does not implement

## Join

| item | value |
|---|---|
| join key | `['bridge', 'seed', 'template_id']` |
| truth rows | 1440 |
| representation rows | 1440 |
| duplicate keys | **0** |
| unmatched truth | **0** |
| unmatched representation records | **0** |
| join cardinality | 1:1 |

## Frozen asset integrity (before and after)

| asset | unchanged |
|---|---|
| locked_spec | True |
| frozen_manifest | True |
| decision | True |
| gate_d | True |
| gate_e | True |
| raw_units | True |
| raw_index | True |
| s9_manifest | True |

**ALL UNCHANGED = True**

## Verdict

* F1: Tier A — PASS
* delta definition verified against the frozen plan — PASS
* truth labels complete for all 1440 templates — PASS
* join 1:1, zero duplicates, zero unmatched — PASS
* frozen assets unchanged — PASS

**FEASIBLE = True** — Mode A_ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS.

No prediction method was executed on this path; a runtime guard aborts on any solver/decoder/pipeline import. Block `401-410` was never read and was never regenerated.

