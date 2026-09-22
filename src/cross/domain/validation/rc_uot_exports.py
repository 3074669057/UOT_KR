"""Post-UOT paper sidecars: marginals, causal summary, traceability, token-route checks."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import output_file
from cross.shared.normalize import norm_addr


def write_uot_marginals_csv(
    out_dir: Path,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    solver_diag: dict[str, Any],
    *,
    lambda_risk: float,
) -> None:
    a0 = list(solver_diag.get("source_mass_original") or [])
    arw = list(solver_diag.get("source_mass_risk_weighted") or [])
    b0 = list(solver_diag.get("target_mass_original") or [])
    brw = list(solver_diag.get("target_mass_evidence_weighted") or [])
    rows: list[dict[str, Any]] = []
    lr = float(lambda_risk)
    rw_enabled = bool(solver_diag.get("risk_weighted_marginal_enabled", True))

    for i, f in enumerate(eth_flows):
        fid = str(f.get("flow_id") or f"eth_{i}")
        usd = float(f.get("amount_usd") or 0.0)
        base = float(a0[i]) if i < len(a0) else 0.0
        rwm = float(arw[i]) if i < len(arw) else base
        aml = float(f.get("aml_risk_score_raw", float(f.get("aml_score", 0.0)) * 100.0))
        eq = float(f.get("evidence_quality_score", 1.0) or 1.0)
        final_m = rwm if rw_enabled else base
        rows.append(
            {
                "flow_id": fid,
                "chain": "ETH",
                "amount_usd": usd,
                "base_mass": base,
                "aml_risk_score": aml,
                "lambda_risk": lr,
                "risk_weighted_mass": rwm,
                "evidence_quality_score": eq,
                "final_mass": final_m,
                "mass_normalized": final_m,
            }
        )
    for j, f in enumerate(bnb_flows):
        fid = str(f.get("flow_id") or f"bnb_{j}")
        usd = float(f.get("amount_usd") or 0.0)
        base = float(b0[j]) if j < len(b0) else 0.0
        evw = float(brw[j]) if j < len(brw) else base
        aml = float(f.get("aml_risk_score_raw", float(f.get("aml_score", 0.0)) * 100.0))
        eq = float(f.get("evidence_quality_score", 0.75) or 0.75)
        rows.append(
            {
                "flow_id": fid,
                "chain": "BNB",
                "amount_usd": usd,
                "base_mass": base,
                "aml_risk_score": aml,
                "lambda_risk": 0.0,
                "risk_weighted_mass": base,
                "evidence_quality_score": eq,
                "final_mass": evw,
                "mass_normalized": evw,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "uot_marginals.csv", index=False)


def write_causal_feasibility_summary_from_decomp(out_dir: Path, decomp: dict[str, Any] | None) -> None:
    """Vectorized causal summary without building a full per-cell ``DataFrame``."""
    if not decomp or not isinstance(decomp.get("delay_sec"), np.ndarray):
        write_causal_feasibility_summary_json(out_dir, None)
        return
    d = np.asarray(decomp["delay_sec"], dtype=float)
    md = decomp.get("max_delay_sec")
    md_arr = np.asarray(md, dtype=float) if isinstance(md, np.ndarray) and md.shape == d.shape else np.zeros_like(d)
    finite = np.isfinite(d) & np.isfinite(md_arr)
    neg = int(np.sum((d < 0) & finite))
    ok = int(np.sum((d >= 0) & (d <= md_arr) & finite))
    soft = int(np.sum((d > md_arr) & finite))
    total = int(d.size)
    viol = decomp.get("causal_violation_flag")
    if isinstance(viol, np.ndarray) and viol.shape == d.shape:
        cvr = float(np.mean(viol.astype(bool)))
    else:
        cvr = float(neg / max(total, 1))
    ir = decomp.get("infeasible_reason")
    reason_counts: dict[str, int] = {}
    if isinstance(ir, np.ndarray):
        flat = ir.astype(str).ravel()
        reason_counts = dict(Counter(flat.tolist()))
    max_delay_cfg = float(np.nanmax(md_arr)) if md_arr.size else None
    payload = {
        "total_pairs": total,
        "feasible_pairs": ok,
        "soft_feasible_pairs": soft,
        "infeasible_pairs": neg,
        "causal_violation_rate": cvr,
        "avg_delay_sec": float(np.nanmean(d)) if np.isfinite(d).any() else None,
        "median_delay_sec": float(np.nanmedian(d)) if np.isfinite(d).any() else None,
        "max_delay_sec": max_delay_cfg,
        "infeasible_reason_counts": reason_counts,
    }
    with open(out_dir / "causal_feasibility_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_causal_feasibility_summary_json(out_dir: Path, cost_df: pd.DataFrame | None) -> None:
    if cost_df is None or cost_df.empty or "delay_sec" not in cost_df.columns:
        payload = {
            "total_pairs": 0,
            "feasible_pairs": 0,
            "soft_feasible_pairs": 0,
            "infeasible_pairs": 0,
            "causal_violation_rate": 0.0,
            "avg_delay_sec": None,
            "median_delay_sec": None,
            "max_delay_sec": None,
        }
    else:
        delays = pd.to_numeric(cost_df["delay_sec"], errors="coerce")
        maxd = pd.to_numeric(cost_df.get("max_delay_sec"), errors="coerce")
        max_delay_cfg = float(maxd.max()) if maxd.notna().any() else None
        reasons = cost_df.get("infeasible_reason", pd.Series([""] * len(cost_df))).astype(str)
        neg = delays < 0
        soft = (delays > maxd) & (delays.notna()) & (maxd.notna())
        ok = (delays >= 0) & (delays <= maxd) & (delays.notna()) & (maxd.notna())
        infeasible_pairs = int(neg.sum())
        soft_feasible_pairs = int(soft.sum())
        feasible_pairs = int(ok.sum())
        total_pairs = int(len(cost_df))
        viol = cost_df.get("causal_violation_flag")
        if viol is not None:
            cvr = float(pd.Series(viol).astype(bool).mean())
        else:
            cvr = float(neg.mean()) if total_pairs else 0.0
        payload = {
            "total_pairs": total_pairs,
            "feasible_pairs": feasible_pairs,
            "soft_feasible_pairs": soft_feasible_pairs,
            "infeasible_pairs": infeasible_pairs,
            "causal_violation_rate": cvr,
            "avg_delay_sec": float(delays.mean()) if delays.notna().any() else None,
            "median_delay_sec": float(delays.median()) if delays.notna().any() else None,
            "max_delay_sec": max_delay_cfg,
            "infeasible_reason_counts": dict(Counter(reasons.fillna(""))),
        }
    with open(out_dir / "causal_feasibility_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _evidence_ids_for_flow(flow: dict[str, Any], chain: str) -> list[str]:
    out: list[str] = []
    for txh in flow.get("tx_hashes") or []:
        h = norm_addr(str(txh))
        if not h:
            continue
        if chain.upper() == "ETH":
            out.append(f"eth:{h}:0")
        else:
            out.append(f"bnb:{h}:0")
    return sorted(set(out))


def _tx_hashes_for_flow(flow: dict[str, Any]) -> list[str]:
    return [norm_addr(str(x)) for x in (flow.get("tx_hashes") or []) if norm_addr(str(x))]


def build_traceability_rows(
    tplan: list[dict[str, Any]],
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return rows for ``traceability_index.csv`` and validation warnings."""
    warnings: list[str] = []
    fid_eth = {str(f.get("flow_id")): f for f in eth_flows}
    fid_bnb = {str(f.get("flow_id")): f for f in bnb_flows}
    cost_index: dict[tuple[str, str], dict[str, Any]] = {}
    if cost_rows:
        for r in cost_rows:
            cost_index[(str(r.get("src_flow_id")), str(r.get("dst_flow_id")))] = r

    out_rows: list[dict[str, Any]] = []
    for tr in tplan:
        sid = str(tr.get("src_flow_id") or "")
        tid = str(tr.get("dst_flow_id") or "")
        sf = fid_eth.get(sid) or {}
        tf = fid_bnb.get(tid) or {}
        src_txs = _tx_hashes_for_flow(sf)
        tgt_txs = _tx_hashes_for_flow(tf)
        src_eids = _evidence_ids_for_flow(sf, "ETH")
        tgt_eids = _evidence_ids_for_flow(tf, "BNB")
        ck = (sid, tid)
        cr = cost_index.get(ck, {})
        out_rows.append(
            {
                "source_tx_hash": "|".join(src_txs),
                "source_evidence_id": "|".join(src_eids),
                "source_flow_id": sid,
                "target_flow_id": tid,
                "target_evidence_ids": "|".join(tgt_eids),
                "target_tx_hashes": "|".join(tgt_txs),
                "route_id": str(sf.get("route_id") or tf.get("route_id") or ""),
                "transport_mass": tr.get("transport_mass"),
                "source_share": tr.get("source_share"),
                "target_share": tr.get("target_share"),
                "amount_cost": cr.get("amount_cost", ""),
                "time_cost": cr.get("time_cost", ""),
                "route_cost": cr.get("route_cost", ""),
                "risk_cost": cr.get("risk_cost", ""),
                "graph_cost": cr.get("graph_cost", ""),
                "evidence_cost": cr.get("evidence_cost", ""),
                "total_cost": cr.get("total_cost", tr.get("total_cost")),
                "evidence_levels": str(tf.get("evidence_levels") or tf.get("evidence_level") or ""),
                "match_type": tr.get("match_type", ""),
            }
        )

    for tr in tplan:
        sid = str(tr.get("src_flow_id") or "")
        tid = str(tr.get("dst_flow_id") or "")
        if sid not in fid_eth:
            warnings.append(f"traceability: missing eth flow segment for src_flow_id={sid}")
        if tid not in fid_bnb:
            warnings.append(f"traceability: missing bnb flow segment for dst_flow_id={tid}")

    return out_rows, warnings


def validate_traceability(eth_flows: list[dict[str, Any]], bnb_flows: list[dict[str, Any]], tplan: list[dict[str, Any]]) -> list[str]:
    _, w = build_traceability_rows(tplan, eth_flows, bnb_flows, None)
    return w


def build_token_route_validation(
    pairs: pd.DataFrame | None,
    *,
    unlabeled_priors: dict[str, Any] | None,
    route_json_path: Path | None,
) -> dict[str, Any]:
    issues: dict[str, list[Any]] = {
        "missing_decimals": [],
        "unknown_routes": [],
        "cross_asset_without_price": [],
        "suspicious_static_ratio": [],
        "weth_to_wbnb_static_ratio": [],
        "default_ratio_is_one": [],
    }
    sev: list[str] = []

    if isinstance(unlabeled_priors, dict):
        dr = unlabeled_priors.get("default_ratio")
        if dr is not None and float(dr) == 1.0:
            issues["default_ratio_is_one"].append(float(dr))
            sev.append("default_ratio_is_one")

    if pairs is not None and not pairs.empty:
        if "is_static_ratio_used" in pairs.columns and "expected_raw_ratio" in pairs.columns:
            sub = pairs[pairs["is_static_ratio_used"].astype(bool)]
            for _, r in sub.iterrows():
                rt = str(r.get("route_id") or "").lower()
                er = r.get("expected_raw_ratio")
                st = str(r.get("src_token") or "").lower()
                dt = str(r.get("dst_token") or "").lower()
                weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
                wbnb = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"
                if st == weth and dt == wbnb and er is not None and float(er) == 1.0:
                    issues["weth_to_wbnb_static_ratio"].append({"src": st, "dst": dt})
                    sev.append("weth_to_wbnb_static_ratio")
        if "src_decimals" in pairs.columns:
            for _, r in pairs.iterrows():
                sd = r.get("src_decimals")
                if str(sd).strip() == "" or sd is None or (isinstance(sd, float) and np.isnan(sd)):
                    issues["missing_decimals"].append({"col": "src", "row": r.get("srcTxHash")})
        if "cross_asset" in pairs.get("route_id", pd.Series(dtype=str)).astype(str).str.lower().values:
            pass

    payload = {
        "issues": issues,
        "severity_flags": sorted(set(sev)),
        "paper_blocking": bool(sev),
    }
    _ = route_json_path
    return payload


def write_token_route_validation_json(out_dir: Path, payload: dict[str, Any]) -> None:
    with open(output_file(out_dir, "token_route_validation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
