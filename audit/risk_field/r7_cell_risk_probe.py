"""Audit probe: measure risk_cost / risk-marginal effect in actual R7 generated cells."""
import json
import pathlib

import numpy as np
import pandas as pd

base = pathlib.Path("out/r7_confirmatory_kernel_ranking_20260917/selection/generator/_cells")
rows = []
for br in ["Celer", "Multi", "Poly"]:
    bdir = base / br
    if not bdir.is_dir():
        print(br, "NO CELL DIRECTORY")
        continue
    for d in sorted(bdir.glob("seed_*"))[:3]:
        cz = np.load(d / "cost.npz", allow_pickle=True)
        uz = np.load(d / "transport_uot.npz", allow_pickle=True)
        rc, ec, Cp = cz["risk_cost"], cz["evidence_cost"], cz["C_primary"]
        a0, arw = uz["a0"], uz["a_rw"]
        b0, bev = uz["b0"], uz["b_ev"]
        eth = pd.read_csv(d / "flow_segments_eth_synth.csv", dtype=str, keep_default_na=False)
        bnb = pd.read_csv(d / "flow_segments_bnb_synth.csv", dtype=str, keep_default_na=False)
        aml = pd.to_numeric(eth["aml_score_mean"], errors="coerce")
        qs = pd.to_numeric(eth["evidence_quality_mean"], errors="coerce")
        qb = pd.to_numeric(bnb["evidence_quality_mean"], errors="coerce")
        ratio = arw / np.maximum(a0, 1e-300)
        rec = {
            "bridge": br, "cell": d.name, "n_src": int(Cp.shape[0]), "n_dst": int(Cp.shape[1]),
            "risk_cost_min": float(rc.min()), "risk_cost_max": float(rc.max()),
            "risk_cost_mean": float(rc.mean()), "risk_cost_std": float(rc.std(ddof=0)),
            "risk_cost_nuniq": int(len(np.unique(rc))),
            "evidence_cost_mean": float(ec.mean()), "evidence_cost_nuniq": int(len(np.unique(ec))),
            "C_mean": float(Cp.mean()), "C_min": float(Cp.min()), "C_max": float(Cp.max()),
            "aml_mean_min": float(aml.min()), "aml_mean_max": float(aml.max()),
            "aml_mean_frac_zero": float((aml == 0).mean()),
            "aml_unique": int(aml.nunique()),
            "eth_q_unique": int(qs.nunique()), "eth_q_min": float(qs.min()), "eth_q_max": float(qs.max()),
            "bnb_q_unique": int(qb.nunique()), "bnb_q_min": float(qb.min()), "bnb_q_max": float(qb.max()),
            "src_aml_score_min": float(min(f for f in [0] )),
            "a_rw_over_a0_min": float(ratio.min()), "a_rw_over_a0_max": float(ratio.max()),
            "a_rw_over_a0_nuniq": int(len(np.unique(np.round(ratio, 12)))),
            "corr_a0_arw": float(np.corrcoef(a0, arw)[0, 1]),
            "b_ev_over_b0_maxdev": float(np.max(np.abs(bev - b0) / np.maximum(b0, 1e-300))),
            "cost_weight_risk_primary": 0.15 / 0.65,
            "risk_share_of_C": float(np.mean(0.15 / 0.65 * rc) / max(Cp.mean(), 1e-30)),
        }
        rows.append(rec)
        print(json.dumps(rec, indent=1))

# cross-field range overlap: risk (0-1 aml_score) vs q
print("\n=== range overlap test on frozen pools (flow-level aml_score after /100 rule) ===")
root = pathlib.Path("out/multi_bridge_expansion/faithful_flow_structural_three_bridges/feature_stats")
for br in ["Celer", "Multi", "Poly"]:
    e = pd.read_csv(root / br / "flow_segments_eth.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(root / br / "flow_segments_bnb.csv", dtype=str, keep_default_na=False)
    aml_raw = pd.to_numeric(e["aml_score_mean"], errors="coerce")
    # replicate uot_flow_loader scale rule
    aml_norm = np.where(aml_raw > 1.0, aml_raw / 100.0, aml_raw)
    qe = pd.to_numeric(e["evidence_quality_mean"], errors="coerce")
    qb = pd.to_numeric(b["evidence_quality_mean"], errors="coerce")
    print(f"{br}: rho in [{aml_norm.min():.4f},{aml_norm.max():.4f}]  q_src in [{qe.min():.3f},{qe.max():.3f}]  "
          f"q_dst in [{qb.min():.3f},{qb.max():.3f}]  overlap={bool(aml_norm.max() >= qb.min())}")
    # risk_cost = |rho - q_dst| always == q_dst - rho ?
    d = np.abs(aml_norm[:, None] - qb.to_numpy()[None, :])
    print(f"    |rho-q| min={d.min():.4f} max={d.max():.4f}; "
          f"fraction where rho>q: {float((aml_norm[:,None] > qb.to_numpy()[None,:]).mean()):.6f}; "
          f"frac |rho-q|>0.4: {float((d>0.4).mean()):.6f}")
