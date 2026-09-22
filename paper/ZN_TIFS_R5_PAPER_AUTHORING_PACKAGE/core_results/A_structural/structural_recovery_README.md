# Three-bridge structural recovery (split/merge)

The semi-synthetic split/merge stress test is seeded from **real 1-to-1 anchors**
of each bridge (dev split), then cloned into split / merge / unmatched / noise
(48 templates x 5 seeds).

| method | Celer | Multichain | PolyNetwork |
| --- | ---: | ---: | ---: |
| RC-UOT-Q (full flow pipeline) | 0.946 / 0.967 | (needs Multi flow pipeline) | (needs Poly flow pipeline) |
| Connector-style (one-to-one) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| ABCTracer-style (one-to-one) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

Cells are `split_recovery / merge_recovery`.

## Why the baselines are 0 on every bridge (bridge-invariant)

Connector-style and ABCTracer-style are one-to-one hard matchers: each source
emits at most one destination. A one-to-one matcher cannot emit a 1->2 split or a
2->1 merge, so its split/merge recovery is mathematically capped at <=0.5 and, in
the amount-driven setting used here, is exactly 0 on all three bridges. This is a
property of the matching paradigm, not of any particular bridge's data.

## RC-UOT-Q scope note

The quantitative RC-UOT-Q reference (split 0.946 / merge 0.967) comes from the
**full Celer flow-level pipeline** (flow segmentation + rich cost + unbalanced
transport + decode). It is currently Celer-only. The *capability* to represent
split/merge is inherent to the many-to-many transport and is bridge-independent,
but reproducing the exact per-bridge numbers for Multichain / PolyNetwork
requires building their flow-level pipeline (segmentation + cost), which is a
separate engineering task.
