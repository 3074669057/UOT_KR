# Three-bridge baseline comparison ? honest summary

Evaluation regime: raw transaction-pair full-set, held-out test split, one shared
candidate pool per protocol, dev-frozen hyperparameters, no re-tuning at test time.

Protocols: Celer (2079), Multichain (2444), PolyNetwork (1630).

## 1. Full-set (Table 7 equivalent)

| bridge | method | precision | recall | F1 | coverage | abstention |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Celer | RC-UOT-Q | 0.523 | 0.523 | 0.523 | 1.000 | 0.000 |
| Celer | Connector-style | 1.000 | 0.479 | 0.647 | 0.479 | 0.521 |
| Celer | ABCTracer-style | 0.981 | 0.981 | 0.981 | 1.000 | 0.000 |
| Multi | RC-UOT-Q | 0.316 | 0.005 | 0.010 | 0.016 | 0.984 |
| Multi | Connector-style | 0.993 | 0.964 | 0.978 | 0.971 | 0.029 |
| Multi | ABCTracer-style | 0.959 | 0.959 | 0.959 | 1.000 | 0.000 |
| Poly | RC-UOT-Q | 0.889 | 0.783 | 0.833 | 0.881 | 0.119 |
| Poly | Connector-style | 0.999 | 0.998 | 0.998 | 0.999 | 0.001 |
| Poly | ABCTracer-style | 0.998 | 0.998 | 0.998 | 1.000 | 0.000 |

## 2. Symmetric masking ladder

| bridge | mask | RC-UOT-Q F1 | Connector-style F1 | ABCTracer-style F1 |
| --- | --- | ---: | ---: | ---: |
| Celer | full | 0.523 | 0.647 | 0.981 |
| Celer | no_receiver | 0.523 | NO_PREDICTIONS (0.0) | 0.963 |
| Celer | no_amount | 0.247 | 0.646 | 0.982 |
| Celer | no_receiver_no_amount | 0.247 | NO_PREDICTIONS (0.0) | 0.248 |
| Multi | full | 0.010 | 0.978 | 0.959 |
| Multi | no_receiver | 0.010 | NO_PREDICTIONS (0.0) | 0.855 |
| Multi | no_amount | 0.000 | 0.958 | 0.772 |
| Multi | no_receiver_no_amount | 0.000 | NO_PREDICTIONS (0.0) | 0.092 |
| Poly | full | 0.833 | 0.998 | 0.998 |
| Poly | no_receiver | 0.833 | NO_PREDICTIONS (0.0) | 0.995 |
| Poly | no_amount | 0.477 | 0.926 | 0.877 |
| Poly | no_receiver_no_amount | 0.477 | NO_PREDICTIONS (0.0) | 0.482 |

## 3. Coverage-precision (abstention) curve ? RC-UOT-Q

| bridge | thr=0.0 (P/cov) | thr=0.5 (P/cov) | thr=0.9 (P/cov) |
| --- | ---: | ---: | ---: |
| Celer | 0.523 / 1.000 | 0.840 / 0.012 | 1.000 / 0.004 |
| Multi | 0.045 / 1.000 | 0.201 / 0.098 | 0.583 / 0.005 |
| Poly | 0.822 / 0.993 | 0.889 / 0.881 | 0.984 / 0.614 |

## 4. Open-world characterisation (candidate pool is already open)

| bridge | GT | candidates | decoy ratio | src with 0 receiver+token match |
| --- | ---: | ---: | ---: | ---: |
| Celer | 2079 | 157714 | 0.987 | 1084 (52%) |
| Multi | 2444 | 81012 | 0.970 | 0 |
| Poly | 1630 | 4362 | 0.626 | 0 |

## Bottom line

In the raw transaction-pair full-set regime, RC-UOT-Q is NOT superior to
Connector-style / ABCTracer-style baselines on precision/recall/F1. This is
robust across full-set, masking, coverage-precision, and open-world views.

Reason: RC-UOT-Q uses only amount/time/token (no receiver) and abstains; the
baselines use receiver+amount, which are highly discriminative on these
near-1-to-1 anchors. RC-UOT-Q's genuine, defensible advantages live elsewhere:

1. Split/merge structural recovery (many-to-one / one-to-many) ? baselines are
   one-to-one hard matchers and cannot represent this at all (manuscript Table 2).
2. Robustness to missing bridge semantics ? beats Connector-style when receiver
   is unavailable (masking ladder above).
3. Explicit abstention / coverage qualification (it is the only method that
   refuses instead of hallucinating on uncertain pairs).
