# S10 provenance — frozen R7 artifacts read by this analysis

## Read-only inputs (never modified)

| artifact | how used |
|---|---|
| `confirmatory/raw/units/unit__<bridge>__s<seed>.json` (30 files) | `margin_mass.delta_S` / `delta_T` / `a` / `b` / realised masses, canonical source/target id lists, and the full truth structure (`positive`, `split`, `merge`, `decoy`, `unmatched_src`, `hidden_dst`) |
| `confirmatory/raw/units/_scratch/<bridge>/seed_<seed>/transport_uot.npz` | frozen plan `P` and marginals, used ONLY to re-derive delta and verify it against the archived vectors (Tier-B-style cross-check) |
| `config/locked_spec.json` | read for provenance; not modified |
| `config/FROZEN_PROTOCOL_MANIFEST.json` | frozen executor hash used in the delta audit; not modified |
| `analysis/DECISION.json`, `confirmatory/VALIDITY_GATE_D/E.json` | hash-guarded integrity checks only |

## Facts

* provenance mode **A** (`ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS`), tier **A**
* block `411-420` only; 30 units; 1440 templates
* block `401-410` (`INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION`): never read, never used, never regenerated
* `42-46 / 201-205 / 301-305`: no method executed, no data read
* **the UOT solver, Sinkhorn, the cost builder and every decoder were never called**; a runtime guard aborts on any such import
* `P` hashes unchanged; all frozen asset hashes verified identical before and after

## Not touched

* nothing under `confirmatory/retired_401_410/`
* the R7 `MANIFEST.json` and the S9 package were not modified
* no R7 or S9 file was written, renamed or deleted by S10

