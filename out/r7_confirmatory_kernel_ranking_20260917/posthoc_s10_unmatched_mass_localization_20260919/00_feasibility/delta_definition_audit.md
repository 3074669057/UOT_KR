# S10 delta definition audit

**The definition was located in the frozen source; it was NOT invented, and it was NOT assumed to be `a - P.sum(1)` without verification.**

* answer: delta^S_i = a_i - sum_j P_ij and delta^T_j = b_j - sum_i P_ij, where a is the NORMALISED risk-weighted source mass and b the NORMALISED evidence-weighted target mass of the frozen UOT solve

## Exact formula

```text
delta_S: delta_s = a - P.sum(axis=1)
delta_T: delta_t = b - P.sum(axis=0)
```

## Source provenance

| file | sha256 | function | frozen manifest hash |
|---|---|---|---|
| `scripts/run_r7_confirmatory_kernel_ranking.py` | `5a125184154213c736a41bff8597237d01444372c0b805f35cb6ce4ca552f5bb` | `build_unit` | `5a125184154213c736a41bff8597237d01444372c0b805f35cb6ce4ca552f5bb` |
| `scripts/r7/r7_pipeline.py` | `9e749e0d48dc8ad3da081e4d0dd61541f890f81e01821b512e701e321daf1d58` | `CellContext.__init__` | `n/a` |
| `scripts/r7/r7_generator.py` | `9994fe2df397479df3b750b1e8996cd9192d7a3ed80a295ae5fec7b15a50d8f0` | `build_cell` | `n/a` |

## Frozen source lines

```text
90:     delta_s = a - P.sum(axis=1)        # unmatched / marginal-deficit mass, source side
91:     delta_t = b - P.sum(axis=0)        # unmatched / marginal-deficit mass, target side
129:             "delta_S": delta_s.tolist(), "delta_T": delta_t.tolist(),
130:             "delta_S_total": float(delta_s.sum()),
131:             "delta_T_total": float(delta_t.sum()),
```

* units: both marginals are normalised to unit total mass, so delta^S and delta^T are in units of total source / target mass and lie in [0, 1] up to the over-delivery case
* normalization: `a_rw / sum(a_rw) and b_ev / sum(b_ev)  (frozen r7_generator.py build_cell lines 258-259)`
* sign convention: delta > 0 = mass requested by the marginal but NOT realised by the plan (unmatched / marginal deficit); delta < 0 would mean the plan over-delivers relative to the marginal
* total relation: sum_i delta^S_i = 1 - sum_ij P_ij = sum_j delta^T_j, i.e. the source-side and target-side total deficits are equal by construction of the unbalanced solve

## Independent re-derivation from the archived frozen plan

* units checked: 30
* worst absolute difference across delta vectors, totals, source/target total gap and mass conservation: **4.441e-16** (tolerance 1e-12)
* ALL MATCH: **True**

This proves the archived `margin_mass.delta_S` / `delta_T` really are `a - P.sum(axis=1)` and `b - P.sum(axis=0)` of the frozen solve, so the analysis may use them directly (Tier A) with the frozen definition.
