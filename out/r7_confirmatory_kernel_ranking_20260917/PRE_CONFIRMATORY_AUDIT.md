# R7 pre-confirmatory audit

* generated: 2026-09-17T07:51:46Z
* git HEAD: `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (branch `master`)

## Statement

> **THE ACTIVE CONFIRMATORY BLOCK HAS NOT BEEN GENERATED OR READ.**

* active confirmatory block: `[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]`
* active ledger `confirmatory/CONFIRMATORY_TOUCH_ONCE__411_420.json` exists: **False**
* `confirmatory/raw/` exists: **False**
* no R7 generator run has touched any active confirmatory seed
* no confirmatory executor invocation has occurred for this block

### Retired block 401-410 (disclosed)

* block `401-410` is **spent** and is never re-run: INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION (plumbing defect, no result produced)
* its ledger is retained, unmodified, at `confirmatory/CONFIRMATORY_TOUCH_ONCE.json`
* its partial artefacts are archived under `confirmatory/retired_401_410/`
* full incident record: `confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md`

## Frozen protocol

| item | value |
|---|---|
| freeze type | hash-locked protocol freeze |
| freeze time | 2026-09-17T07:51:46Z |
| locked spec | `config/locked_spec.json` |
| locked spec sha256 | `9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d` |
| code manifest | `config/FROZEN_PROTOCOL_MANIFEST.json` (42 hashed files) |
| selection block | `[206, 207, 208, 209, 210, 211, 112, 113, 114, 115]` |
| confirmatory block | `[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]` |
| selected rule | `R-const@k3` |
| rule parameters | `{"family": "R-const", "k": 3}` |
| Threshold-MM cutoff | `0.230150` |
| Dual-Softmax tau | `0.0` |
| epsilon / lambda | `0.05` / `0.5` |
| support threshold | `1e-09` |
| bootstrap | B=4000, RNG=20240101 |
| permutation | n_perm=20000, RNG=20240102 |
| Holm family | H1, H2 (FWER alpha = 0.05) |

## Pre-freeze verification evidence

* selection preflight: **ALL PASS**
* frozen reproduction anchors: {"AMOUNT_FREE_COST_D4": {"r7_value": 0.3171931003584229, "frozen_value": 0.3172, "abs_diff": 6.899641577073901e-06, "tolerance": 0.0005, "pass": true}, "CONDITIONAL_UOT_D4": {"r7_value": 0.3129093352883675, "frozen_value": 0.3129, "abs_diff": 9.335288367495753e-06, "tolerance": 0.0005, "pass": true}}
* independent validator self-test on selection raw outputs: **PASS**
* validator max |diff| vs executor: 1.6653345369377348e-16 (tolerance 1e-09)

## Wording

* this freeze may be described as a **hash-locked protocol freeze**
* it must NOT be described as an externally timestamped preregistration
