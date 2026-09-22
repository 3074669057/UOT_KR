# V5_DATA_ADEQUACY_REPORT.md

```json
{
  "v5_start": 1747180800,
  "stop_block": null,
  "final_block": 12,
  "G_full": 8,
  "G_conservative": 2,
  "gates_full": {
    "G_ge_8": true,
    "max_share_lt_0.5": false,
    "n_fanout_ge_30": true,
    "spread_ge_2_months": true
  },
  "gates_conservative": {
    "G_ge_8": false,
    "max_share_lt_0.5": true,
    "n_fanout_ge_30": true,
    "spread_ge_2_months": true
  },
  "DATA_ADEQUACY_STATUS": "FAIL",
  "verdict_text": "DATA ADEQUACY FAIL under the conservative counting"
}
```
Conservative counting = OVERLAP_FAMILIAR clusters excluded (V5_DISJOINTNESS_PROTOCOL.md).
DATA-ONLY: no method prediction, no performance, no solver exists for this corpus.
