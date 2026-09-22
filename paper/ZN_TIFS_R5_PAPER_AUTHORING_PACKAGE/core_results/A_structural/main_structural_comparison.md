# Main structural comparison — three bridges x five methods

48 templates x seeds 42-46 per bridge. Mean +/- std over the 5 seeds (per-seed value = mean over 48 templates); 95% bootstrap CI (seed resampling).
RC-UOT-Q = frozen faithful run decoded at 1e-9. Balanced-OT = same C, same marginals,
strictly balanced, same decode. Threshold-MM = same C, calibration GLOBAL tau. Connector/ABCTracer = project's existing per-source top-1 rules.

| Bridge | Method | Split exact | Merge exact | Edge P | Edge R | Edge F1 | Degree acc | FP edges | Pred edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Celer | ABCTracer-style | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.000 ± 0.000 | 5.8 ± 0.0 | 6.0 ± 0.0 |
| Celer | Balanced-OT | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.016 ± 0.002 | 0.968 ± 0.036 | 0.030 ± 0.005 | 0.002 ± 0.046 | 680.8 ± 128.9 | 686.6 ± 128.9 |
| Celer | Connector-style | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.000 ± 0.000 | 5.8 ± 0.0 | 6.0 ± 0.0 |
| Celer | RC-UOT-Q | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.022 ± 0.005 | 0.961 ± 0.035 | 0.040 ± 0.009 | 0.002 ± 0.046 | 539.0 ± 135.3 | 544.8 ± 135.5 |
| Celer | Threshold-MM | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.094 ± 0.009 | 0.971 ± 0.006 | 0.168 ± 0.016 | 0.004 ± 0.064 | 67.7 ± 9.7 | 73.5 ± 9.7 |
| Multi | ABCTracer-style | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.034 ± 0.006 | 0.034 ± 0.006 | 0.034 ± 0.006 | 0.000 ± 0.000 | 5.8 ± 0.0 | 6.0 ± 0.0 |
| Multi | Balanced-OT | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.009 ± 0.003 | 0.994 ± 0.008 | 0.018 ± 0.005 | 0.000 ± 0.000 | 1069.6 ± 142.4 | 1075.5 ± 142.4 |
| Multi | Connector-style | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 6.0 ± 0.0 | 6.0 ± 0.0 |
| Multi | RC-UOT-Q | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.015 ± 0.004 | 0.983 ± 0.020 | 0.027 ± 0.008 | 0.000 ± 0.000 | 925.4 ± 140.7 | 931.3 ± 140.8 |
| Multi | Threshold-MM | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.075 ± 0.005 | 0.844 ± 0.030 | 0.136 ± 0.009 | 0.000 ± 0.000 | 71.0 ± 2.9 | 76.0 ± 2.8 |
| Poly | ABCTracer-style | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 6.0 ± 0.0 | 6.0 ± 0.0 |
| Poly | Balanced-OT | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.013 ± 0.001 | 1.000 ± 0.000 | 0.026 ± 0.002 | 0.000 ± 0.000 | 852.8 ± 55.4 | 858.8 ± 55.4 |
| Poly | Connector-style | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 6.0 ± 0.0 | 6.0 ± 0.0 |
| Poly | RC-UOT-Q | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.017 ± 0.004 | 1.000 ± 0.000 | 0.034 ± 0.007 | 0.000 ± 0.000 | 743.6 ± 63.4 | 749.6 ± 63.4 |
| Poly | Threshold-MM | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.065 ± 0.006 | 0.997 ± 0.006 | 0.120 ± 0.010 | 0.000 ± 0.000 | 107.5 ± 8.1 | 113.5 ± 8.1 |

*Bootstrap CIs per metric are stored in `main_structural_comparison.csv` (seed-level and hierarchical).*
