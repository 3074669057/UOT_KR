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
bcfg = {'n_bootstrap': 200, 'seed': 42}
out_dir = Path('results/m1_solver_ablation')
out_dir.mkdir(parents=True, exist_ok=True)
structure_labels = assign_structure_label(ds.ground_truth, ds.source_flows, ds.target_flows)

solvers = ['thresholded_cost', 'greedy_nn', 'hungarian', 'balanced_ot', 'rc_uot', 'rc_uot_production_loaded']
all_metrics = []
all_stratified = []
all_meta = []

for sname in solvers:
    t1 = time.perf_counter()
    print(f'--- {sname} ---')
    solver = get_solver(sname)
    result = solver.solve(C=ds.C, feasible_mask=ds.feasible_mask,
                           source_mass=ds.source_mass, target_mass=ds.target_mass,
                           config={'temperature': 1.0})
    rt = result.meta.get('runtime_sec', 0)
    print(f'  T: {result.T.shape}, sum={result.T.sum():.4f}, rt={rt:.1f}s')

    dr = decode_with_fixed_rc_uot_q(
        T=result.T, C=ds.C,
        source_flows=ds.source_flows, target_flows=ds.target_flows,
        src_all=ds.src_all, dst_norm=ds.dst_norm,
        truth=all_truth, eth_ts=ds.eth_ts, bnb_ts=ds.bnb_ts,
        config=dc, score_threshold=None,
    )
    print(f'  n_pred={dr.n_predicted}, n_abst={dr.n_abstained}')

    m = compute_metrics(mapping=dr.mapping, truth=all_truth,
                        n_source_flows=ds.n_source, n_abstained=dr.n_abstained,
                        n_predicted=dr.n_predicted, solver_name=sname,
                        score_threshold=None, bootstrap_config=bcfg)
    all_metrics.append(m)
    ci = m.precision_ci
    print(f'  P={m.precision:.4f} [{ci.lower_95:.4f},{ci.upper_95:.4f}] R={m.recall:.4f} F1={m.f1:.4f} cov={m.coverage:.4f}')

    strat = compute_stratified_metrics(
        mapping=dr.mapping, truth=all_truth, structure_labels=structure_labels,
        n_source_flows=ds.n_source, n_abstained=dr.n_abstained,
        n_predicted=dr.n_predicted, solver_name=sname,
        score_threshold=None, bootstrap_config=bcfg,
    )
    all_stratified.extend(strat.values())
    all_meta.append({'solver': sname, 'transport_mass': float(result.T.sum()), 'runtime': rt})
    print(f'  [{time.perf_counter()-t1:.1f}s]')

# solver_ablation.csv
with open(out_dir/'solver_ablation.csv','w',newline='',encoding='utf-8') as f:
    cols=['solver','precision','precision_ci_low','precision_ci_high',
          'recall','recall_ci_low','recall_ci_high','f1','f1_ci_low','f1_ci_high',
          'coverage','abstention_rate','tp','fp','fn','n_predicted','n_abstained']
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for m in all_metrics:
        w.writerow({
            'solver':m.solver,'precision':m.precision,
            'precision_ci_low':m.precision_ci.lower_95 if m.precision_ci else '',
            'precision_ci_high':m.precision_ci.upper_95 if m.precision_ci else '',
            'recall':m.recall,
            'recall_ci_low':m.recall_ci.lower_95 if m.recall_ci else '',
            'recall_ci_high':m.recall_ci.upper_95 if m.recall_ci else '',
            'f1':m.f1,
            'f1_ci_low':m.f1_ci.lower_95 if m.f1_ci else '',
            'f1_ci_high':m.f1_ci.upper_95 if m.f1_ci else '',
            'coverage':m.coverage,'abstention_rate':m.abstention_rate,
            'tp':m.tp,'fp':m.fp,'fn':m.fn,
            'n_predicted':m.n_predicted,'n_abstained':m.n_abstained,
        })
print(f'Written solver_ablation.csv')

# by_structure.csv
with open(out_dir/'by_structure.csv','w',newline='',encoding='utf-8') as f:
    cols2=['solver','structure','support','precision','precision_ci_low','precision_ci_high',
           'recall','recall_ci_low','recall_ci_high','f1','f1_ci_low','f1_ci_high',
           'coverage','abstention_rate','tp','fp','fn']
    w=csv.DictWriter(f,fieldnames=cols2); w.writeheader()
    for sr in sorted(all_stratified, key=lambda x:(x.solver,x.structure)):
        w.writerow({
            'solver':sr.solver,'structure':sr.structure,'support':sr.support,
            'precision':sr.precision,
            'precision_ci_low':sr.precision_ci.lower_95 if sr.precision_ci else '',
            'precision_ci_high':sr.precision_ci.upper_95 if sr.precision_ci else '',
            'recall':sr.recall,
            'recall_ci_low':sr.recall_ci.lower_95 if sr.recall_ci else '',
            'recall_ci_high':sr.recall_ci.upper_95 if sr.recall_ci else '',
            'f1':sr.f1,
            'f1_ci_low':sr.f1_ci.lower_95 if sr.f1_ci else '',
            'f1_ci_high':sr.f1_ci.upper_95 if sr.f1_ci else '',
            'coverage':sr.coverage,'abstention_rate':sr.abstention_rate,
            'tp':sr.tp,'fp':sr.fp,'fn':sr.fn,
        })
print(f'Written by_structure.csv')
print(f'Total: {time.perf_counter()-t0:.1f}s')
