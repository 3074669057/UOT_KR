"""Debug CSVs for semi-synthetic unmatched / delay-noise scenarios (eval under main out_root)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def write_synthetic_failure_debug_csvs(
    out_root: Path,
    *,
    synth_work: Path | None = None,
    hints_path: Path | None = None,
    labels_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write ``eval/synthetic_failure_debug.csv`` and ``eval/synthetic_cost_component_comparison.csv``."""
    out_root = Path(out_root)
    sw = Path(synth_work) if synth_work else out_root / "synthetic_uot_work"
    hp = Path(hints_path) if hints_path else out_root / "labels" / "synthetic_uot_eval_metrics.json"
    lp = Path(labels_path) if labels_path else out_root / "labels" / "synthetic_flow_labels.csv"
    uot_dir = sw / "uot"
    um_p = uot_dir / "uot_unmatched_mass.csv"
    plan_p = uot_dir / "uot_transport_plan.csv"
    cost_p = uot_dir / "uot_cost_components.csv"

    failure_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    hints: dict[str, Any] = {}
    if hp.is_file():
        hints = json.loads(hp.read_text(encoding="utf-8"))
    eh = hints.get("eval_hints") or {}
    truth_unmatched = set(str(x) for x in (eh.get("truth_unmatched_src_flows") or []))
    noise_pairs = []
    for pair in eh.get("noise_decoy_pairs") or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            noise_pairs.append((str(pair[0]), str(pair[1])))

    um_thr_used = 0.08

    if um_p.is_file():
        um = pd.read_csv(um_p, dtype=str, keep_default_na=False)
        eth = um[um.get("chain", "").astype(str).str.upper() == "ETH"].copy()
        for sid in sorted(truth_unmatched):
            sub = eth[eth.get("flow_id", "").astype(str) == sid]
            if sub.empty:
                failure_rows.append(
                    {
                        "scenario": "unmatched",
                        "synthetic_src_flow_id": sid,
                        "issue": "missing_from_uot_unmatched_mass_eth",
                        "unmatched_ratio": "",
                        "passes_um_ratio_threshold_0_08": False,
                        "note": "flow_id not found in ETH unmatched_mass export",
                    }
                )
                continue
            r = sub.iloc[0]
            ur = float(pd.to_numeric(r.get("unmatched_ratio"), errors="coerce") or 0.0)
            passes = ur >= um_thr_used
            failure_rows.append(
                {
                    "scenario": "unmatched",
                    "synthetic_src_flow_id": sid,
                    "issue": "unmatched_ratio_too_small_for_eval_threshold"
                    if not passes
                    else "unmatched_ratio_ok_but_other_eval_issue",
                    "unmatched_ratio": ur,
                    "passes_um_ratio_threshold_0_08": passes,
                    "unmatched_mass": float(pd.to_numeric(r.get("unmatched_mass"), errors="coerce") or 0.0),
                    "transported_mass": float(pd.to_numeric(r.get("transported_mass"), errors="coerce") or 0.0),
                    "flow_mass": float(pd.to_numeric(r.get("flow_mass"), errors="coerce") or 0.0),
                    "note": (
                        "Synthetic unmatched_src rows show near-zero unmatched_ratio in USD-normalized columns; "
                        "evaluator pred set uses unmatched_ratio>=0.08 so TP set is empty → F1=0. "
                        "Consider scenario-specific unmatched_mass in probability space or lower um_ratio_thr for synth."
                    ),
                }
            )

        hidden_dst_ids: set[str] = set()
        if lp.is_file():
            lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
            for _, row in lab.iterrows():
                if "hidden" in str(row.get("dst_flow_id", "")).lower() or "synth_hidden" in str(row.get("dst_flow_id", "")):
                    hidden_dst_ids.add(str(row.get("dst_flow_id") or ""))
        for hid in sorted(x for x in hidden_dst_ids if x):
            bsub = um[um.get("chain", "").astype(str).str.upper() == "BNB"]
            bsub = bsub[bsub.get("flow_id", "").astype(str) == hid]
            failure_rows.append(
                {
                    "scenario": "unmatched_hidden_dst",
                    "hidden_dst_flow_id": hid,
                    "bnb_unmatched_row_present": not bsub.empty,
                    "note": "hidden template dst may not appear as BNB row in unmatched_mass if not in active transport pool",
                }
            )

    pred: set[tuple[str, str]] = set()
    if plan_p.is_file():
        pl = pd.read_csv(plan_p, dtype=str, keep_default_na=False)
        pl["_m"] = pd.to_numeric(pl.get("transport_mass"), errors="coerce").fillna(0.0)
        pred = set(zip(pl["src_flow_id"].astype(str), pl["dst_flow_id"].astype(str))) if not pl.empty else set()

    for a, b in noise_pairs[:40]:
        in_pred = (a, b) in pred
        failure_rows.append(
            {
                "scenario": "delay_noise_decoy",
                "decoy_src": a,
                "decoy_dst": b,
                "decoy_edge_in_transport_plan": in_pred,
                "note": "decoy edge in plan means positive mass on decoy pair; rejection rate metric uses absence in pred set",
            }
        )

    if cost_p.is_file() and plan_p.is_file():
        cc = pd.read_csv(cost_p, dtype=str, keep_default_na=False)
        pl = pd.read_csv(plan_p, dtype=str, keep_default_na=False)
        pl["_m"] = pd.to_numeric(pl.get("transport_mass"), errors="coerce").fillna(0.0)
        for _, pr in pl.iterrows():
            if float(pr["_m"]) <= 1e-12:
                continue
            sid = str(pr.get("src_flow_id") or "")
            did = str(pr.get("dst_flow_id") or "")
            sub = cc[(cc.get("src_flow_id", "").astype(str) == sid) & (cc.get("dst_flow_id", "").astype(str) == did)]
            if sub.empty:
                continue
            c = sub.iloc[0]
            cost_rows.append(
                {
                    "src_flow_id": sid,
                    "dst_flow_id": did,
                    "transport_mass": float(pr["_m"]),
                    "is_decoy_pair": (sid, did) in set(noise_pairs),
                    "total_cost": float(pd.to_numeric(c.get("total_cost"), errors="coerce") or 0.0),
                    "amount_cost": float(pd.to_numeric(c.get("amount_cost"), errors="coerce") or 0.0),
                    "time_cost": float(pd.to_numeric(c.get("time_cost"), errors="coerce") or 0.0),
                    "route_cost": float(pd.to_numeric(c.get("route_cost"), errors="coerce") or 0.0),
                    "evidence_cost": float(pd.to_numeric(c.get("evidence_cost"), errors="coerce") or 0.0),
                    "risk_cost": float(pd.to_numeric(c.get("risk_cost"), errors="coerce") or 0.0),
                    "graph_cost": float(pd.to_numeric(c.get("graph_cost"), errors="coerce") or 0.0),
                    "causal_violation_flag": str(c.get("causal_violation_flag", "")),
                    "delay_sec": float(pd.to_numeric(c.get("delay_sec"), errors="coerce") or 0.0),
                }
            )

    ev = out_root / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    fp1 = ev / "synthetic_failure_debug.csv"
    fp2 = ev / "synthetic_cost_component_comparison.csv"
    pd.DataFrame(failure_rows).to_csv(fp1, index=False)
    pd.DataFrame(cost_rows).to_csv(fp2, index=False)
    logger.info("Wrote %s and %s", fp1, fp2)
    return fp1, fp2
