#!/usr/bin/env python
import sys, csv, time, numpy as np
from pathlib import Path
sys.path.insert(0, 'src')
from cross.domain.uot.m1_ablation.data_adapter import discover_data
from cross.domain.uot.m1_ablation import (
    get_solver, decode_with_fixed_rc_uot_q, DecodeConfig,
    compute_metrics, compute_stratified_metrics, assign_structure_label,
)

t0 = time.perf_counter()
ds = discover_data(data_root='data', auto_discover=True)
all_truth = dict(ds.ground_truth)
print(f'Load: {time.perf_counter()-t0:.1f}s, truth={len(all_truth)}')

dc = DecodeConfig(strategy='joint_time_admissible_filter', tx_decode_policy='legacy')
bootstrap_cfg = {'n_bootstrap': 200, 'seed': 42}
out_dir = Path('results/m1_factorial')
out_dir.mkdir(parents=True, exist_ok=True)
structure_labels = assign_structure_label(ds.ground_truth, ds.source_flows, ds.target_flows)

conditions = [
    ('no_causal_mask', np.ones(ds.C.shape, dtype=bool)),
    ('with_causal_mask', ds.feasible_mask),
]
solvers = ['rc_uot', 'hungarian', 'greedy_nn']
all_rows = []
all_stratified = []

for causal_cond, fm in conditions:
    print(f'\n--- {causal_cond} ---')
    for sname in solvers:
        t1 = time.perf_counter()
        solver = get_solver(sname)
        result = solver.solve(C=ds.C, feasible_mask=fm,
                               source_mass=ds.source_mass, target_mass=ds.target_mass,
                               config={'temperature': 1.0})
        dr = decode_with_fixed_rc_uot_q(
            T=result.T, C=ds.C,
            source_flows=ds.source_flows, target_flows=ds.target_flows,
            src_all=ds.src_all, dst_norm=ds.dst_norm,
            truth=all_truth, eth_ts=ds.eth_ts, bnb_ts=ds.bnb_ts,
            config=dc, score_threshold=None,
        )
        m = compute_metrics(mapping=dr.mapping, truth=all_truth,
                            n_source_flows=ds.n_source, n_abstained=dr.n_abstained,
                            n_predicted=dr.n_predicted, solver_name=sname,
                            score_threshold=None, bootstrap_config=bootstrap_cfg)
        row = {
            'causal_condition': causal_cond, 'solver': sname,
            'precision': m.precision, 'recall': m.recall, 'f1': m.f1,
            'coverage': m.coverage, 'abstention_rate': m.abstention_rate,
            'tp': m.tp, 'fp': m.fp, 'fn': m.fn,
            'n_predicted': m.n_predicted, 'n_abstained': m.n_abstained,
        }
        if m.precision_ci:
            row['precision_ci_low'] = m.precision_ci.lower_95
            row['precision_ci_high'] = m.precision_ci.upper_95
        all_rows.append(row)
        print(f'  {sname}: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}')

        strat = compute_stratified_metrics(
            mapping=dr.mapping, truth=all_truth, structure_labels=structure_labels,
            n_source_flows=ds.n_source, n_abstained=dr.n_abstained,
            n_predicted=dr.n_predicted, solver_name=sname,
            score_threshold=None, bootstrap_config=bootstrap_cfg,
        )
        for label, sr in strat.items():
            all_stratified.append({
                'causal_condition': causal_cond, 'solver': sname, 'structure': label,
                'support': sr.support, 'precision': sr.precision, 'recall': sr.recall,
                'f1': sr.f1, 'coverage': sr.coverage, 'abstention_rate': sr.abstention_rate,
                'tp': sr.tp, 'fp': sr.fp, 'fn': sr.fn,
            })
        print(f'    [{time.perf_counter()-t1:.1f}s]')

# Write CSV
csv_path = out_dir / 'factorial_summary.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    keys = list(all_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
print(f'Written {csv_path}')

strat_csv = out_dir / 'factorial_by_structure.csv'
with open(strat_csv, 'w', newline='', encoding='utf-8') as f:
    keys2 = list(all_stratified[0].keys())
    w = csv.DictWriter(f, fieldnames=keys2); w.writeheader(); w.writerows(all_stratified)
print(f'Written {strat_csv}')
print(f'Total: {time.perf_counter()-t0:.1f}s')
