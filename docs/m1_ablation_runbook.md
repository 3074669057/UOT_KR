# M1 Solver Ablation Runbook

## Overview

These experiments address Reviewer M1's concern: is the 0.589 → 0.889 precision jump mainly
due to the shared RC-UOT-Q temporal admissibility filter rather than RC-UOT itself?

The experiments **freeze all non-solver factors** (cost matrix, temporal filter, decoder, data split,
operation point) and only swap the transport/assignment solver.

## Quick Start

### Prerequisites

- Python 3.11+
- Data files in expected locations (see "Data Paths" below)
- Dependencies from `requirements.txt`

### Smoke Test (no data needed)

```bash
python tests/test_m1_ablation.py
```

This runs 11 unit/smoke tests on synthetic data.

### Dry Run on Synthetic Data

```bash
python scripts/run_m1_solver_ablation.py --dry-run --output-dir results/m1_solver_ablation
```

## Full Experiments

### 1. Solver Ablation

Compares all five solvers (thresholded_cost, greedy_nn, hungarian, balanced_ot, rc_uot)
under the identical RC-UOT-Q decoder.

```bash
python scripts/run_m1_solver_ablation.py \
    --config config/experiments/m1_solver_ablation.yaml \
    --output-dir results/m1_solver_ablation
```

### 2. Factorial Experiment (2×3)

2 causal conditions (no_causal_mask / with_causal_mask) × 3 solvers (rc_uot, hungarian, greedy_nn).

```bash
python scripts/run_m1_factorial.py \
    --config config/experiments/m1_factorial.yaml \
    --output-dir results/m1_factorial
```

## Data Paths

The experiments expect these files:

| File | Path |
|------|------|
| ETH transactions | `in/Celer_ETH_cun.csv` |
| BNB transactions | `label/tx/Celer_BNB_qu.csv` |
| Ground truth pairs | `out/baseline_compare/labels/gt_tx_pairs.csv` |
| UOT production run | `out/uot_delay_fixed_production/` |

If data is missing, use `--dry-run` for a synthetic smoke test.

## Output Files

### `results/m1_solver_ablation/`

| File | Description |
|------|-------------|
| `report.md` | Markdown report with tables and auto-generated interpretation |
| `solver_ablation.csv` | Main ablation table |
| `by_structure.csv` | Stratified by correspondence structure |
| `precision_coverage_auc.csv` | AUC for each solver |
| `ablation_summary.json` | Full JSON summary |
| `precision_coverage_curves/` | Per-solver PC curve CSVs and PNG/PDF plots |

### `results/m1_factorial/`

| File | Description |
|------|-------------|
| `factorial_report.md` | Markdown report answering the M1 question |
| `factorial_summary.csv` | 6-condition summary table |
| `factorial_by_structure.csv` | Per-structure metrics for all 6 conditions |

## Configuration

All experiment parameters are in `config/experiments/`:

- `m1_solver_ablation.yaml` — Solver ablation with operation point target
- `m1_factorial.yaml` — 2×3 factor experiment
- `m1_precision_coverage.yaml` — Precision-coverage curve sweep

Key configuration options:

```yaml
operation_point:
  mode: coverage      # "coverage" or "abstention"
  target: 0.90        # null = infer from main experiment

decoder:
  strategy: joint_time_admissible_filter
  delay_policy: tx_if_available_else_flow_representative

transport:
  temperature: 1.0
  apply_feasible_mask_inside_solver: true
```

## Architecture

```
src/cross/domain/uot/m1_ablation/
├── __init__.py              # Public API
├── transport_solver.py      # 5 solver implementations + TransportSolver protocol
├── fixed_decoder.py         # RC-UOT-Q decoder shared by all solvers
├── evaluation.py            # Metrics, bootstrap CI, structure labeling
├── calibration.py           # Operation point threshold calibration
└── precision_coverage.py    # Precision-coverage curves and plotting

scripts/
├── run_m1_solver_ablation.py  # Main ablation experiment
└── run_m1_factorial.py        # 2×3 factorial experiment

config/experiments/
├── m1_solver_ablation.yaml
├── m1_factorial.yaml
└── m1_precision_coverage.yaml

tests/
└── test_m1_ablation.py     # 11 smoke/unit tests
```

## Design Principles

1. **Freeze all non-solver factors**: same C, same decoder, same filter, same data split
2. **Only swap the transport/assignment solver**: thresholded_cost, greedy_nn, hungarian, balanced_ot, rc_uot
3. **Fixed RC-UOT-Q decoder**: quotient grouping, joint temporal admissibility filter, coverage qualification
4. **Same operation point**: all solvers calibrated to the same coverage/abstention target on validation
5. **Honest reporting**: report all results, even if RC-UOT does not win
