# R7 validator packaging correction

* classification: **packaging defect corrected after holdout execution; fully disclosed**
* holdout re-execution: **NO** — the frozen raw package was only re-read
* affected artefacts before correction: `scripts/validate_r7_confirmatory_results.py`,
  `config/locked_spec.json` (one documentation field)
* affected metrics / methods / thresholds / decoders / data: **NONE**

---

## 1. What was wrong

Two stale block constants survived the confirmatory-block substitution from `401-410` to
`411-420`. Both were written while the block was still `401-410`:

| artefact | field | value before correction | correct value |
|---|---|---|---|
| `scripts/validate_r7_confirmatory_results.py` | private `CONFIRMATORY_SEEDS` | `401-410` | `411-420` |
| `config/locked_spec.json` | `seed_blocks.confirmatory` | `401-410` | `411-420` |

`config/locked_spec.json` inherited the stale value from the **immutable** pre-selection
protocol snapshot (`config/operational_protocol_preselection.json`), which was locked
before the incident that forced the block substitution.

## 2. Effect of the defect

The first post-execution validation run produced a false `GATE_E = FAIL`.

* `metric_agreement_within_tolerance` — **PASS**
  max abs diff `2.220e-16` against a `1e-9` tolerance, over 30 units x 7 methods
  (210 method rows x 7 metric fields)
* `oracle_ceiling_agreement` — **PASS**
* `raw_unit_hashes_match_index` — **PASS**
* `executor_and_validator_cover_same_cells` — **PASS**
* `families_per_unit_is_24`, `all_methods_in_every_unit`, `no_duplicate_units`,
  `no_null_edge_endpoints` — **PASS**
* `index_declares_frozen_confirmatory_block` — **FAIL** (validator expected `401-410`)
* `unit_set_complete` — **FAIL** (same cause)

The two failures were therefore purely a seed-block bookkeeping mismatch in the
validators own private constant. No metric, decoder, threshold, cost, gate definition,
solver setting or data file was implicated.

The failure is preserved, unmodified, at:

```
confirmatory/VALIDITY_GATE_E_frozen_validator_precorrection.json
  sha256 68b5e5c73c5d4dd141b9bf7ff12fe45ef9ed45504477805bade924fe13737c94
  GATE_E = FAIL
  validator_sha256 59dbe87b636b4c0e4d5568b4d5549de2db51a86507d7a63c1cc9dcdbdf83dad0
```

## 3. What was corrected

1. **The validator no longer trusts a private constant.** It reads the expected
   confirmatory block from `config/FROZEN_PROTOCOL_MANIFEST.json` ->
   `confirmatory_seed_block`, which is the artefact the confirmatory executor itself
   verified itself against (`confirmatory_seed_block_only`, PASS). A stale private
   constant can therefore no longer silently disagree with the frozen protocol.
2. The stale `locked_spec.seed_blocks.confirmatory` field is now **read, cross-checked and
   reported as an explicit inconsistency** in the Gate E output
   (`expected_seed_block_source.inconsistencies`). It is deliberately **not** rewritten:
   the freeze guard refuses to rewrite the locked spec after the holdout touch, which is
   the correct behaviour.
3. `unit_set_complete` and `index_declares_frozen_confirmatory_block` now compare against
   the authoritative frozen record.

## 4. Verification that the block actually used was 411-420

Every *operative* frozen artefact agrees:

| artefact | recorded block |
|---|---|
| `config/FROZEN_PROTOCOL_MANIFEST.json` -> `confirmatory_seed_block` | `411-420` |
| `config/locked_spec.json` -> `executor.seed_block` | `411-420` |
| `confirmatory/raw/INDEX.json` -> `seed_block` | `411-420` |
| `confirmatory/CONFIRMATORY_TOUCH_ONCE__411_420.json` -> `seed_block` | `411-420` |
| executor run log (30 units, 3 bridges x 10 seeds) | `411-420` |

The executor hard-codes `CONFIRMATORY_SEEDS = (411..420)` and its seed guard rejects every
other block before any data access; `non_confirmatory_seeds_rejected` PASSED.

## 5. What this does NOT do

* it does **not** re-run `411-420`; the holdout was only re-read (specification section 45
  explicitly permits repeated reads of the frozen raw package by the validator);
* it does **not** change any metric, decoder, threshold, cost weight, gate definition or
  statistical procedure;
* it does **not** alter the frozen raw package, the ledger, or the executor;
* it does **not** delete or overwrite the pre-correction failure record.

## 6. Result after correction

The corrected validator passes all 11 checks, `GATE_E = PASS`, with the packaging
inconsistency reported in-band rather than suppressed.
