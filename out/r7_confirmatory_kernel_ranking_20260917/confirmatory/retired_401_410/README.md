# Retired confirmatory block 401-410

This directory preserves, unmodified, every artefact produced by the FIRST R7
confirmatory execution attempt (block 401-410), which crashed on a plumbing defect
after its one-shot ledger was written but before any unit result was written.

* `../../CONFIRMATORY_TOUCH_ONCE.json` — the original first-touch ledger, still in
  its original location and never deleted.
* `gate_pre_execution.json` — the 8/8 PASS pre-execution verification of that attempt.
* `partial_raw_units_scratch/` — the partially generated unit data (`Celer` seed
  `401` only), moved out of the active `raw/` package so active and retired data
  can never be confused. Content unmodified.

Block 401-410 is **spent** and is never re-run. The active confirmatory block is
411-420. See `../INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md`.
