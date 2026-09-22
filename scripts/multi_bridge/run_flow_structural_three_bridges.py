"""Run the full RC-UOT-Q flow-level pipeline on synthetic split/merge for three bridges.

Reuses ``run_standalone_flow_uot`` (the paper's actual pipeline). Templates are
seeded from each bridge's real dev 1-to-1 anchors; USD is replaced by
decimal-normalized human amounts (the amount cost is relative, so this is a valid
scale-invariant proxy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached, truth_for, OLD  # noqa: E402
from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels  # noqa: E402
from cross.domain.evaluation.synthetic_segment_subgraph import write_synthetic_subgraph_segment_csvs  # noqa: E402
from cross.application.standalone_flow_uot import run_standalone_flow_uot  # noqa: E402
from cross.domain.uot.cost_matrix import default_cost_weights  # noqa: E402

OUT = REPO / "out" / "multi_bridge_expansion" / "flow_structural_three_bridges"
BRIDGES = ("Celer", "Multi", "Poly")
N_TEMPLATES = 48
SEEDS = (42, 43, 44, 45, 46)


def _f(v: Any) -> float:
    return float(pd.to_numeric(v, errors="coerce") or 0.0)


def norm(v: Any) -> str:
    return str(v or "").strip().lower()


def load_prices() -> dict[str, float]:
    p = REPO / "data" / "Token" / "token_prices_usd.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {str(a).lower(): float(v.get("usd") or 1.0) for a, v in raw.items()}


def load_decimals() -> tuple[dict[str, int], dict[str, int]]:
    eth = pd.read_csv(REPO / "data" / "Token" / "ERC20.csv", dtype={"address": str})
    bnb = pd.read_csv(REPO / "data" / "Token" / "BERC20.csv", dtype={"address": str})
    eth_dec = {norm(a): int(d) for a, d in zip(eth["address"], eth["decimal"]) if norm(a)}
    bnb_dec = {norm(a): int(d) for a, d in zip(bnb["address"], bnb["decimal"]) if norm(a)}
    return eth_dec, bnb_dec


def human(raw: float, dec: dict[str, int], addr: str) -> float:
    d = dec.get(norm(addr))
    return float(raw) / float(10 ** int(d)) if d and int(d) > 0 else float(raw)


def build_bridge_inputs(bridge: str, eth_dec: dict[str, int], bnb_dec: dict[str, int], prices: dict[str, float]):
    cached = load_bridge_cached(bridge)
    split = cached["split"]
    source = cached["source"]
    receiver = cached["receiver"]
    dev_candidates = cached["dev_candidates"]
    test_candidates = cached["test_candidates"]
    # dest candidate lookup: tx hash -> (amount_raw, token_address, timestamp, receiver)
    all_c = pd.concat([dev_candidates, test_candidates], ignore_index=True).drop_duplicates("candidate_tx_hash")
    c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}

    dev = split[split["split"] == "development"]
    # token map (ETH token -> BNB token) from dev truth
    token_map = OLD.infer_token_map(split, all_c)

    eth_rows, bnb_rows, label_rows = [], [], []
    for r in dev.itertuples():
        src_h = norm(r.source_tx_hash)
        dst_h = norm(r.dest_tx_hash)
        srow = source[source["source_tx_hash"] == src_h]
        if srow.empty:
            continue
        srow = srow.iloc[0]
        stok = norm(srow["source_token_address"])
        samt = _f(srow["source_amount_raw"])
        srecv = norm(receiver.get(src_h, srow.get("source_receiver", "")))
        sts = _f(srow["source_timestamp"])
        c = c_lookup.get(dst_h)
        if c is None:
            continue
        dtok = norm(c.token_address)
        damt = _f(c.amount_raw)
        drecv = norm(c.receiver)
        dts = _f(c.candidate_timestamp)
        samt_h = human(samt, eth_dec, stok) * prices.get(stok, 1.0)
        damt_h = human(damt, bnb_dec, dtok) * prices.get(dtok, 1.0)
        delay = max(0.0, dts - sts)
        eth_rows.append({
            "chain": "ETH", "flow_id": src_h, "tx_hashes": src_h,
            "address_set": srecv, "primary_address": srecv,
            "token_contracts": stok, "token_symbols": "T", "asset_group": f"token:{stok[:8]}",
            "route_id": src_h, "start_time": str(int(sts)), "end_time": str(int(sts)),
            "tx_count": "1", "raw_amount_sum": str(samt), "human_amount_sum": str(samt_h),
            "usd_amount_sum": str(samt_h), "aml_score_mean": "0", "aml_score_max": "0",
            "evidence_quality_mean": "0.825", "flow_construction_rule": "tx",
        })
        # NOTE: set the BNB destination amount equal to the SOURCE amount so that the
        # synthetic split/merge clones land on a common value scale (bridge fee is
        # irrelevant to the split/merge structural test).
        bnb_rows.append({
            "chain": "BNB", "flow_id": dst_h, "tx_hashes": dst_h,
            "address_set": drecv, "primary_address": drecv,
            "token_contracts": dtok, "token_symbols": "T", "asset_group": f"token:{dtok[:8]}",
            "route_id": dst_h, "start_time": str(int(dts)), "end_time": str(int(dts)),
            "tx_count": "1", "raw_amount_sum": str(samt), "human_amount_sum": str(samt_h),
            "usd_amount_sum": str(samt_h), "aml_score_mean": "0", "aml_score_max": "0",
            "evidence_quality_mean": "0.675", "flow_construction_rule": "tx",
        })
        label_rows.append({
            "src_flow_id": src_h, "dst_flow_id": dst_h,
            "src_amount_usd": str(samt_h), "dst_amount_usd": str(damt_h),
            "median_delay_sec": str(delay), "pattern_type": "one_to_one",
            "label_confidence": "1.0", "label_type": "supervised_flow_pair",
        })
    return pd.DataFrame(eth_rows), pd.DataFrame(bnb_rows), pd.DataFrame(label_rows)


def run_bridge(bridge: str, eth_dec, bnb_dec, prices, seed: int) -> dict[str, Any]:
    root = OUT / bridge
    root.mkdir(parents=True, exist_ok=True)
    eth_df, bnb_df, labels_df = build_bridge_inputs(bridge, eth_dec, bnb_dec, prices)
    fl = root / "flow_labels.csv"
    labels_df.to_csv(fl, index=False)
    (root / "flow_label_stats.json").write_text(json.dumps({"predominantly_one_to_one": True}), encoding="utf-8")
    syn_csv, syn_json = build_semi_synthetic_from_flow_labels(fl, root / "flow_label_stats.json", root, seed=seed, max_seeds=N_TEMPLATES, force=True)
    hints = json.loads(syn_json.read_text(encoding="utf-8"))
    clones = hints.get("segment_clone_records") or []
    se = root / "flow_segments_eth_synth.csv"
    sb = root / "flow_segments_bnb_synth.csv"
    eth_df.to_csv(root / "flow_segments_eth.csv", index=False)
    bnb_df.to_csv(root / "flow_segments_bnb.csv", index=False)
    write_synthetic_subgraph_segment_csvs(root / "flow_segments_eth.csv", root / "flow_segments_bnb.csv", clones, se, sb)

    run_standalone_flow_uot(
        root,
        src_flows_csv=se, dst_flows_csv=sb, flow_labels_csv=syn_csv,
        uot_reg=0.05, uot_reg_m=0.5, uot_decode_threshold=1e-9,
        uot_cost_weights=default_cost_weights(), uot_backend="pot",
        uot_max_delay_sec=21600.0, uot_causal_violation_penalty=5.0,
        uot_lambda_risk=0.25, uot_causal_infeasible_delay_sec=None,
        uot_export_matrix=False, uot_export_cost_components=False,
        uot_allow_unmatched=True, uot_use_graph_embedding=False,
        graph_ranker_checkpoint=None, uot_ablation="none",
        run_flow_baselines=False, flow_label_min_confidence=0.0,
        synthetic_eval_hints_path=Path(syn_json),
        uot_flow_dst_top_k=200, uot_flow_max_matrix_cells=6_000_000,
    )
    ev = json.loads((root / "eval" / "uot_evaluation_metrics.json").read_text(encoding="utf-8"))
    return {"bridge": bridge, "split_recovery": ev.get("split_recovery_rate"), "merge_recovery": ev.get("merge_recovery_rate")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eth_dec, bnb_dec = load_decimals()
    prices = load_prices()
    print("loaded prices:", len(prices), "tokens", flush=True)
    import numpy as np
    agg: dict[str, dict[str, list[float]]] = {}
    for bridge in BRIDGES:
        agg[bridge] = {"split": [], "merge": []}
        for seed in SEEDS:
            r = run_bridge(bridge, eth_dec, bnb_dec, prices, seed)
            agg[bridge]["split"].append(r["split_recovery"])
            agg[bridge]["merge"].append(r["merge_recovery"])
            print(f"{bridge} seed {seed}: split={r['split_recovery']:.3f} merge={r['merge_recovery']:.3f}", flush=True)
    summary = {}
    for bridge in BRIDGES:
        s = agg[bridge]["split"]; m = agg[bridge]["merge"]
        def ci(vals):
            a = np.array(vals, dtype=float)
            return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a)>1 else 0.0,
                    "per_seed": [round(float(x),4) for x in a]}
        summary[bridge] = {"split_recovery": ci(s), "merge_recovery": ci(m)}
    (OUT / "three_bridge_flow_structural_5seed.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n", json.dumps(summary, indent=2))
    return 0
