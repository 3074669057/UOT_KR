"""Weak flow-level labels from accepted tx anchors + tx→flow map."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import output_file
from cross.shared.tx_hash import normalize_tx_hash


def _flow_amounts(flow_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, r in flow_df.iterrows():
        fid = str(r.get("flow_id") or "")
        out[fid] = float(pd.to_numeric(r.get("usd_amount_sum"), errors="coerce") or 0.0)
    return out


def _tx_to_flow(map_df: pd.DataFrame) -> dict[str, str]:
    m: dict[str, str] = {}
    for _, r in map_df.iterrows():
        inc = str(r.get("included_in_flow", "true")).strip().lower()
        if inc in ("0", "false", "no"):
            continue
        h = normalize_tx_hash(str(r.get("tx_hash") or ""))
        if h:
            m[h] = str(r.get("flow_id") or "")
    return m


def _write_validation_report(
    out_dir: Path,
    *,
    stats: dict[str, Any],
    diag_path: Path | None,
) -> Path:
    payload: dict[str, Any] = {"flow_label_stats": stats, "tx_anchor_diagnostics": None, "sanity_checks": {}}
    if diag_path is not None and Path(diag_path).is_file():
        try:
            payload["tx_anchor_diagnostics"] = json.loads(Path(diag_path).read_text(encoding="utf-8"))
        except OSError:
            payload["tx_anchor_diagnostics"] = {"error": "unreadable", "path": str(diag_path)}
    d = payload["tx_anchor_diagnostics"] or {}
    n_acc = int(d.get("num_accepted_anchor_pairs") or 0)
    n_cand = int(d.get("num_candidate_pairs") or 0)
    tot_e = int(d.get("total_eth_unique_tx") or 0)
    cov_amt = float(stats.get("flow_label_coverage_by_amount") or 0.0)
    n_lab = int(stats.get("num_flow_labels") or 0)
    m2m = int(stats.get("many_to_many_flow_count") or 0)
    payload["sanity_checks"] = {
        "accepted_pairs_leq_eth_tx": bool(n_acc <= max(tot_e, 1) + 1),
        "coverage_by_amount_in_0_1": bool(0.0 <= cov_amt <= 1.0 + 1e-9),
        "flow_labels_not_grossly_above_accepted": bool(n_lab <= max(n_acc * 50, 1) or n_acc == 0),
        "many_to_many_not_near_total_labels": bool(m2m <= max(n_lab, 1)),
        "candidates_much_larger_than_accepted": bool(n_cand >= n_acc) if n_cand else True,
    }
    p = output_file(out_dir, "validation_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return p


def build_flow_labels(
    tx_anchor_labels_path: Path,
    flow_segments_eth_path: Path,
    flow_segments_bnb_path: Path,
    tx_to_flow_map_path: Path,
    out_dir: Path,
    *,
    min_confidence: float = 0.0,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_csv(tx_anchor_labels_path, dtype=str, keep_default_na=False)
    eth_f = pd.read_csv(flow_segments_eth_path, dtype=str, keep_default_na=False)
    bnb_f = pd.read_csv(flow_segments_bnb_path, dtype=str, keep_default_na=False)
    tfm = pd.read_csv(tx_to_flow_map_path, dtype=str, keep_default_na=False)

    eth_amt = _flow_amounts(eth_f)
    bnb_amt = _flow_amounts(bnb_f)
    tx_flow = _tx_to_flow(tfm)

    eth_by_id = eth_f.set_index("flow_id", drop=False) if not eth_f.empty else pd.DataFrame()
    bnb_by_id = bnb_f.set_index("flow_id", drop=False) if not bnb_f.empty else pd.DataFrame()

    if "accepted" in anchors.columns:
        acc_ok = anchors["accepted"].astype(str).str.strip().str.lower().isin(("1", "true", "yes"))
        anchors = anchors.loc[acc_ok].copy()
    else:
        # Legacy CSVs mixed all candidates into one file: keep top-1 per src by confidence.
        anchors = anchors.copy()
        anchors["_conf_legacy"] = pd.to_numeric(anchors["pair_label_confidence"], errors="coerce").fillna(0.0)
        anchors["_shk"] = anchors["src_tx_hash"].astype(str).map(normalize_tx_hash)
        anchors = anchors.sort_values("_conf_legacy", ascending=False).drop_duplicates("_shk", keep="first")
        anchors = anchors.drop(columns=["_conf_legacy", "_shk"], errors="ignore")
    anchors["_conf"] = pd.to_numeric(anchors["pair_label_confidence"], errors="coerce").fillna(0.0)
    a = anchors.loc[anchors["_conf"] >= float(min_confidence)].copy()
    a["_sh"] = a["src_tx_hash"].astype(str).map(normalize_tx_hash)
    a["_dh"] = a["dst_tx_hash"].astype(str).map(normalize_tx_hash)
    a["_sf"] = a["_sh"].map(tx_flow)
    a["_df"] = a["_dh"].map(tx_flow)
    a = a.dropna(subset=["_sf", "_df"])
    if a.empty:
        pd.DataFrame().to_csv(output_file(out_dir, "flow_labels.csv"), index=False)
        stats = {
            "num_flow_labels": 0,
            "num_accepted_anchor_pairs": 0,
            "num_tx_anchor_pairs": 0,
            "predominantly_one_to_one": False,
            "flow_label_coverage_by_amount": 0.0,
            "flow_label_coverage_by_amount_percent": 0.0,
        }
        stats_path = output_file(out_dir, "flow_label_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        diag_p = output_file(out_dir, "tx_anchor_diagnostics.json")
        _write_validation_report(out_dir, stats=stats, diag_path=diag_p if diag_p.is_file() else None)
        return output_file(out_dir, "flow_labels.csv"), stats_path

    a["_su"] = pd.to_numeric(a["src_usd_amount"], errors="coerce").fillna(0.0)
    a["_du"] = pd.to_numeric(a["dst_usd_amount"], errors="coerce").fillna(0.0)
    a["_delay"] = pd.to_numeric(a["delay_sec"], errors="coerce").fillna(0.0)

    g = a.groupby(["_sf", "_df"], as_index=False).agg(
        support_tx_pair_count=("src_tx_hash", "count"),
        support_src_tx_hashes=("_sh", lambda s: "|".join(sorted(set(s)))),
        support_dst_tx_hashes=("_dh", lambda s: "|".join(sorted(set(s)))),
        matched_src_amount_usd=("_su", "sum"),
        matched_dst_amount_usd=("_du", "sum"),
        label_confidence=("_conf", "mean"),
        min_delay_sec=("_delay", "min"),
        median_delay_sec=("_delay", "median"),
        max_delay_sec=("_delay", "max"),
    )

    # Accepted flow graph: one undirected logical edge per (src_flow, dst_flow) pair.
    src_to_dst: dict[str, set[str]] = defaultdict(set)
    dst_to_src: dict[str, set[str]] = defaultdict(set)
    for sf, df in set(zip(g["_sf"].astype(str), g["_df"].astype(str))):
        src_to_dst[sf].add(df)
        dst_to_src[df].add(sf)

    rows: list[dict[str, Any]] = []
    for _, r in g.iterrows():
        sfid = str(r["_sf"])
        dfid = str(r["_df"])
        src_deg = len(src_to_dst.get(sfid, set()))
        dst_deg = len(dst_to_src.get(dfid, set()))

        if src_deg == 1 and dst_deg == 1:
            ptype = "one_to_one"
        elif src_deg == 1 and dst_deg > 1:
            ptype = "one_to_many"
        elif src_deg > 1 and dst_deg == 1:
            ptype = "many_to_one"
        else:
            ptype = "many_to_many"

        src_amt = float(eth_amt.get(sfid, 0.0))
        dst_amt = float(bnb_amt.get(dfid, 0.0))
        matched_src = float(r["matched_src_amount_usd"])
        matched_dst = float(r["matched_dst_amount_usd"])

        er = eth_by_id.loc[sfid] if sfid in eth_by_id.index else None
        br = bnb_by_id.loc[dfid] if dfid in bnb_by_id.index else None
        ag_s = str(er["asset_group"]) if er is not None else ""
        ag_d = str(br["asset_group"]) if br is not None else ""
        route_ok = bool(ag_s and ag_s == ag_d)

        stc = int(er["tx_count"]) if er is not None else int(r["support_tx_pair_count"])
        dtc = int(br["tx_count"]) if br is not None else int(r["support_tx_pair_count"])
        n_src_tx = len(str(r["support_src_tx_hashes"]).split("|")) if r["support_src_tx_hashes"] else 0
        n_dst_tx = len(str(r["support_dst_tx_hashes"]).split("|")) if r["support_dst_tx_hashes"] else 0

        rows.append(
            {
                "src_flow_id": sfid,
                "dst_flow_id": dfid,
                "support_tx_pair_count": int(r["support_tx_pair_count"]),
                "src_tx_count": stc,
                "dst_tx_count": dtc,
                "support_src_tx_hashes": str(r["support_src_tx_hashes"]),
                "support_dst_tx_hashes": str(r["support_dst_tx_hashes"]),
                "src_amount_usd": src_amt,
                "dst_amount_usd": dst_amt,
                "matched_src_amount_usd": matched_src,
                "matched_dst_amount_usd": matched_dst,
                "flow_mass_ratio_src": float(matched_src / max(src_amt, 1e-12)),
                "flow_mass_ratio_dst": float(matched_dst / max(dst_amt, 1e-12)),
                "src_coverage": float(n_src_tx / max(stc, 1)),
                "dst_coverage": float(n_dst_tx / max(dtc, 1)),
                "label_confidence": float(r["label_confidence"]),
                "pattern_type": ptype,
                "src_degree": src_deg,
                "dst_degree": dst_deg,
                "route_consistency": route_ok,
                "time_valid": bool(float(r["min_delay_sec"]) >= 0.0),
                "min_delay_sec": float(r["min_delay_sec"]),
                "median_delay_sec": float(r["median_delay_sec"]),
                "max_delay_sec": float(r["max_delay_sec"]),
                "label_source": "tx_anchor_aggregated",
            }
        )

    out_csv = output_file(out_dir, "flow_labels.csv")
    fl = pd.DataFrame(rows)
    fl.to_csv(out_csv, index=False)

    one_one = int((fl["pattern_type"] == "one_to_one").sum()) if not fl.empty else 0
    n_accepted_anchors = int(len(anchors))
    mapped_src_tx = int(a["_sh"].nunique()) if not a.empty else 0

    # Amount coverage: each src_flow counted once; cap matched mass per flow by total src flow USD.
    matched_by_src = fl.groupby("src_flow_id")["matched_src_amount_usd"].sum().to_dict() if not fl.empty else {}
    numer = 0.0
    for sfid, msum in matched_by_src.items():
        tot = float(eth_amt.get(str(sfid), 0.0))
        numer += float(min(tot, msum))
    denom_total_src = float(sum(eth_amt.values())) if eth_amt else 0.0
    cov_amt = float(numer / max(denom_total_src, 1e-12)) if denom_total_src > 0 else 0.0
    cov_amt = max(0.0, min(1.0, cov_amt))

    eth_rows_map = int(tfm[tfm["chain"].astype(str).str.upper() == "ETH"]["tx_hash"].astype(str).map(normalize_tx_hash).nunique()) if not tfm.empty and "chain" in tfm.columns else 0
    flow_label_coverage_by_tx = float(mapped_src_tx / max(eth_rows_map, 1)) if eth_rows_map else 0.0

    stats = {
        "num_src_flows": int(eth_f["flow_id"].nunique()) if not eth_f.empty else 0,
        "num_dst_flows": int(bnb_f["flow_id"].nunique()) if not bnb_f.empty else 0,
        "num_flow_labels": int(len(fl)),
        "num_accepted_anchor_pairs": n_accepted_anchors,
        "num_tx_anchor_pairs": n_accepted_anchors,
        "avg_src_tx_per_flow": float(eth_f["tx_count"].astype(float).mean()) if not eth_f.empty else 0.0,
        "avg_dst_tx_per_flow": float(bnb_f["tx_count"].astype(float).mean()) if not bnb_f.empty else 0.0,
        "median_src_tx_per_flow": float(np.median(eth_f["tx_count"].astype(float))) if not eth_f.empty else 0.0,
        "median_dst_tx_per_flow": float(np.median(bnb_f["tx_count"].astype(float))) if not bnb_f.empty else 0.0,
        "multi_tx_src_flow_ratio": float((eth_f["tx_count"].astype(int) > 1).mean()) if not eth_f.empty else 0.0,
        "multi_tx_dst_flow_ratio": float((bnb_f["tx_count"].astype(int) > 1).mean()) if not bnb_f.empty else 0.0,
        "one_to_one_flow_count": one_one,
        "one_to_many_flow_count": int((fl["pattern_type"] == "one_to_many").sum()) if not fl.empty else 0,
        "many_to_one_flow_count": int((fl["pattern_type"] == "many_to_one").sum()) if not fl.empty else 0,
        "many_to_many_flow_count": int((fl["pattern_type"] == "many_to_many").sum()) if not fl.empty else 0,
        "weak_flow_label_count": int(len(fl)),
        "low_confidence_label_count": int((fl["label_confidence"] < 0.35).sum()) if not fl.empty else 0,
        "singleton_flow_label_ratio": float(one_one / max(len(fl), 1)),
        "flow_label_coverage_by_tx": flow_label_coverage_by_tx,
        "flow_label_coverage_by_amount": cov_amt,
        "flow_label_coverage_by_amount_percent": float(cov_amt * 100.0),
        "predominantly_one_to_one": bool(one_one >= 0.7 * max(len(fl), 1)),
        "mapped_unique_src_tx": mapped_src_tx,
        "eth_unique_tx_in_flow_map": eth_rows_map,
    }
    stats_path = output_file(out_dir, "flow_label_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    diag_p = output_file(out_dir, "tx_anchor_diagnostics.json")
    _write_validation_report(out_dir, stats=stats, diag_path=diag_p if diag_p.is_file() else None)

    return out_csv, stats_path
