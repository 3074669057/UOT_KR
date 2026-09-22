# S9 feasibility report

* experiment: `r7_posthoc_degree_stratification_20260918`
* classification: **POST-HOC / NON-PREREGISTERED / ARCHIVED-PREDICTION RE-STRATIFICATION ANALYSIS**
* locked post-hoc spec sha256: `38b4bbbd1e6d3101a9aa6ed8b1a8fbe3751e4a9fbd0db9e64f56270f5b5518cb`

## F1 — is per-template prediction information retained?

* availability tier: **Tier A**

| item | value |
|---|---|
| R7 archive files (recounted) | **1231** |
| R7 MANIFEST declared files | 510 (excludes regenerable scratch/cells) |
| confirmatory raw units | 30 |
| seeds present | `[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]` |
| bridges present | `['Celer', 'Multi', 'Poly']` |
| templates per unit | [48] |
| total template instances | **1440** |
| every unit has UOT_KR + HUNGARIAN predictions | True |
| every unit has full truth edge lists | True |
| per-template oracle present | True |
| sampled degrees present | True |
| forbidden-seed units present | `[]` |
| only the successful 411–420 block used | True |

The archive retains, for every template instance: `bridge`, `seed`, family / template / instance id, the **truth edge lists**, and the **UOT_KR and HUNGARIAN_1TO1 prediction edge sets**. Both F1 values are therefore recomputed independently from archived prediction edges — no archived metric is reused.

## F2 — can template -> d_max be reconstructed unambiguously?

| item | value |
|---|---|
| generator sha256 (frozen manifest) | `efe435471ddad437d95265b132dfb29f7f29616ef562c47587e0d60a37c433dd` |
| generator sha256 (current worktree) | `efe435471ddad437d95265b132dfb29f7f29616ef562c47587e0d60a37c433dd` |
| generator sha256 (archived in unit) | `efe435471ddad437d95265b132dfb29f7f29616ef562c47587e0d60a37c433dd` |
| all three equal | **True** |
| truth-only deterministic replay used | True |
| units with bit-identical replayed truth | **30/30** |
| sampled split/merge degrees reproduced | True |

The replay is **truth-only**: it calls the frozen generator and the truth structure serialiser and nothing else. A runtime guard asserts that no solver, decoder or method-pipeline module is imported on this path (recorded in `truth_replay_audit.json`).

## Join

| item | value |
|---|---|
| join key | `['bridge', 'seed', 'template_id']` |
| truth rows | 1440 |
| prediction rows | 1440 |
| matched rows | 1440 |
| unmatched truth | **0** |
| unmatched predictions | **0** |
| duplicate keys | **0** |
| join cardinality | 1:1 |

## Unstratified reconstruction check (before any stratification)

| quantity | recomputed from archive | frozen R7 | abs diff |
|---|---:|---:|---:|
| UOT_KR macro edge F1 | `0.4292104171878746` | `0.4292104171878746` | 0 |
| HUNGARIAN macro edge F1 | `0.40367874250839936` | `0.4036787425083993` | 5.55e-17 |
| H1 effect | `0.025531674679475258` | `0.025531674679475275` | 1.73e-17 |

Tolerance `1e-09` — all reproduced: **True**.

## Verdict

* F1: **Tier A** (≥ Tier B) — PASS
* F2: generator hash match **True**, truth replay exact **True** — PASS
* join: 1:1, zero unmatched, zero duplicates — PASS
* unstratified reconstruction — PASS

**FEASIBLE = True** — proceeding to the stratified analysis.

