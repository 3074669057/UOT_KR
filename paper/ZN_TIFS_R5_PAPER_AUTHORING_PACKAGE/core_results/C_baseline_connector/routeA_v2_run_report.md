# Route A v2 run report

Generated: 2026-06-10T15:33:51.767790+00:00

**Phase 1 acceptance:** PASS

## Degradation curve

- `full_native`: Connector raw=0.9736140350877193, RC-UOT-Q joint=0.7084536082474227, status=ACCEPTED
- `id_anchor_masked`: Connector raw=0.9736140350877193, RC-UOT-Q joint=0.7086185567010309, status=ACCEPTED
- `no_receiver`: Connector raw=None, RC-UOT-Q joint=0.7000993048659385, status=BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS
- `no_amount`: Connector raw=0.0, RC-UOT-Q joint=0.37140891106141405, status=ZERO_PREDICTIONS_AFTER_AMOUNT_MASK
- `no_receiver_no_amount`: Connector raw=None, RC-UOT-Q joint=0.37524341715350096, status=BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS

## Notes

- full_native / id_anchor_masked: Connector may exceed RC-UOT-Q under native bridge semantics (closed-set).
- no_receiver / no_receiver_no_amount: Connector blocked by receiver semantics.
- no_amount: Connector may collapse (ZERO_PREDICTIONS_AFTER_AMOUNT_MASK).

