#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 7.5: Connector-style and ABCTracer-style external baselines + unified evaluation.

Read-only use of ../Connector and ../ABCTracer for feasibility audit only.
All outputs under out/paper_full_pipeline_run/baselines/.
Does not rebuild canonical, re-freeze label_layer_v1, or re-run RC-UOT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cross.domain.evaluation.flow_eval import _metrics_for_pairs, _pair_set, _top1_pair_set

RUN_ROOT = REPO_ROOT / "out" / "paper_full_pipeline_run"
BASELINES_DIR = RUN_ROOT / "baselines"
CONNECTOR_DELAY_PRIMARY = 1800
CONNECTOR_DELAY_SENS = 3600
FEE_RATIO_PRIMARY = 0.03
FEE_RATIO_SENS = 0.05


def _norm_hash(h: Any) -> str:
    s = str(h or "").strip().lower()
    if s and not s.startswith("0x"):
        s = "0x" + s
    return s


def _norm_addr(a: Any) -> str:
    s = str(a or "").strip().lower()
    if s and not s.startswith("0x") and len(s) == 40:
        s = "0x" + s
    return s


def _evidence_tx_table(ev: pd.DataFrame, *, chain: str) -> pd.DataFrame:
    """Per-tx observable features from evidence only (no GT labels / tx_to_flow_map)."""
    ev = ev.copy()
    ev["tx_hash"] = ev["tx_hash"].map(_norm_hash)
    ev["_eq"] = (
        ev.get("evidence_quality", "")
        .astype(str)
        .str.lower()
        .map({"high": 2, "medium": 1, "low": 0})
        .fillna(0)
    )
    ev["_ts"] = pd.to_numeric(ev.get("time_stamp"), errors="coerce").fillna(0.0)
    ev["_raw"] = pd.to_numeric(ev.get("raw_value_int", ev.get("raw_value")), errors="coerce").fillna(0.0)
    ev["_ag"] = ev.get("asset_group", "").astype(str).str.strip()
    ev["_route"] = ev.get("route_id", "").astype(str).str.strip()
    ev["_from"] = ev.get("from_address", "").map(_norm_addr)
    ev["_to"] = ev.get("to_address", "").map(_norm_addr)
    ev["_role"] = ev.get("direction_role", "").astype(str)
    ev["_et"] = (
        ev.get("event_type", "").astype(str)
        + "|"
        + ev["_role"]
        + "|"
        + ev.get("token_symbol", "").astype(str)
    )

    def _pick_addr(g: pd.DataFrame) -> str:
        g = g.sort_values("_eq", ascending=False)
        roles = g["_role"].str.lower()
        if chain.upper() == "ETH":
            m = roles.str.contains("eth", na=False) & roles.str.contains("bridge|deposit|transfer", na=False)
            if m.any():
                return str(g.loc[m, "_from"].iloc[0] or g.loc[m, "_to"].iloc[0] or "")
            return str(g["_from"].iloc[0] or g["_to"].iloc[0] or "")
        m = roles.str.contains("release|receive|auxiliary", na=False)
        if m.any():
            return str(g.loc[m, "_to"].iloc[0] or g.loc[m, "_from"].iloc[0] or "")
        return str(g["_to"].iloc[0] or g["_from"].iloc[0] or "")

    rows: list[dict[str, Any]] = []
    for tx_h, g in ev.groupby("tx_hash", sort=False):
        g2 = g.sort_values("_eq", ascending=False)
        rows.append(
            {
                "tx_hash": tx_h,
                "_ts": float(g2["_ts"].max()),
                "_amt": float(g2.loc[g2["_raw"] > 0, "_raw"].sum() or g2["_raw"].max() or 0.0),
                "_ag": str(g2["_ag"].iloc[0]),
                "_route": str(g2["_route"].iloc[0]),
                "_addr": _pick_addr(g2),
                "_et": str(g2["_et"].iloc[0]),
                "from_address": str(g2["_from"].iloc[0]),
                "to_address": str(g2["_to"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _load_tx_tables_at(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchors = pd.read_csv(
        run_root / "labels" / "tx_anchor_labels_from_celer.csv", dtype=str, keep_default_na=False
    )
    eth_ev = pd.read_csv(run_root / "evidence" / "evidence_eth.csv", dtype=str, keep_default_na=False)
    bnb_ev = pd.read_csv(run_root / "evidence" / "evidence_bnb.csv", dtype=str, keep_default_na=False)

    anchors["src_tx_hash"] = anchors["src_tx_hash"].map(_norm_hash)
    anchors["dst_tx_hash"] = anchors["dst_tx_hash"].map(_norm_hash)

    eth_tx = _evidence_tx_table(eth_ev, chain="ETH")
    bnb_tx = _evidence_tx_table(bnb_ev, chain="BNB")

    return anchors, eth_tx, bnb_tx, pd.read_csv(
        run_root / "labels" / "flow_labels.csv", dtype=str, keep_default_na=False
    )


def _connector_candidates(
    eth_tx: pd.DataFrame,
    bnb_tx: pd.DataFrame,
    *,
    delay_sec: int,
    fee_ratio: float,
    missing_log: list[str],
) -> pd.DataFrame:
    bnb = bnb_tx.copy()
    bnb = bnb.sort_values("_ts")
    bnb_ts = bnb["_ts"].to_numpy(dtype=float)
    bnb_hashes = bnb["tx_hash"].astype(str).to_numpy()
    bnb_ag = bnb["_ag"].astype(str).to_numpy()
    bnb_amt = bnb["_amt"].to_numpy(dtype=float)
    bnb_addr = bnb["_addr"].astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    has_receiver = bool((eth_tx["_addr"].astype(str) != "").any())

    for _, er in eth_tx.iterrows():
        src_h = str(er["tx_hash"])
        ts0 = float(er["_ts"])
        ag0 = str(er["_ag"])
        amt0 = float(er["_amt"])
        recv0 = str(er.get("_addr") or "")
        if not has_receiver:
            missing_log.append("receiver_hint:eth_primary_address_sparse")

        i0 = int(np.searchsorted(bnb_ts, ts0, side="right"))
        for j in range(i0, len(bnb_ts)):
            dt = float(bnb_ts[j] - ts0)
            if dt <= 0:
                continue
            if dt > delay_sec:
                break
            if ag0 and bnb_ag[j] != ag0:
                continue
            rules: list[str] = ["time_causality", f"delay<={delay_sec}s", "asset_group_match"]
            reason_parts: list[str] = []

            if recv0 and bnb_addr[j]:
                if recv0 != bnb_addr[j]:
                    continue
                rules.append("receiver_match")
            elif recv0 or bnb_addr[j]:
                rules.append("receiver_skipped_missing")
                reason_parts.append("receiver_partial")
            else:
                rules.append("receiver_unavailable")
                missing_log.append("receiver_match:degraded")

            if amt0 > 0 and bnb_amt[j] > 0:
                if bnb_amt[j] > amt0 * (1.0 + fee_ratio):
                    continue
                diff = abs(bnb_amt[j] - amt0) / max(amt0, 1e-12)
                rules.append(f"amount<={1+fee_ratio:.0%}_src")
                reason_parts.append(f"amount_diff_ratio={diff:.6f}")
            else:
                rules.append("amount_degraded")
                diff = float("nan")
                missing_log.append("amount_usd:degraded")

            score = 1.0 / (1.0 + dt / max(delay_sec, 1.0))
            if np.isfinite(diff):
                score += max(0.0, 1.0 - min(diff, 1.0)) * 0.25

            rows.append(
                {
                    "src_tx_hash": src_h,
                    "dst_tx_hash": bnb_hashes[j],
                    "score": score,
                    "matched_rules": "|".join(rules),
                    "delay_sec": dt,
                    "amount_diff_ratio": diff if np.isfinite(diff) else "",
                    "reason": "; ".join(reason_parts) if reason_parts else "connector_style_pass",
                }
            )

    return pd.DataFrame(rows)


def _abctracer_rankings(
    eth_tx: pd.DataFrame,
    bnb_tx: pd.DataFrame,
    *,
    delay_sec: int,
    top_k: int = 5,
) -> pd.DataFrame:
    bnb = bnb_tx.copy().sort_values("_ts")
    bnb_ts = bnb["_ts"].to_numpy(dtype=float)
    bnb_hashes = bnb["tx_hash"].astype(str).to_numpy()
    bnb_ag = bnb["_ag"].astype(str).to_numpy()
    bnb_amt = bnb["_amt"].to_numpy(dtype=float)
    bnb_addr = bnb["_addr"].astype(str).to_numpy()
    bnb_et = bnb.get("_et", pd.Series([""] * len(bnb))).astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    for _, er in eth_tx.iterrows():
        src_h = str(er["tx_hash"])
        ts0 = float(er["_ts"])
        ag0 = str(er["_ag"])
        amt0 = float(er["_amt"])
        recv0 = str(er.get("_addr") or "")
        et0 = str(er.get("_et") or "")

        cands: list[tuple[float, str, str]] = []
        i0 = int(np.searchsorted(bnb_ts, ts0, side="right"))
        for j in range(i0, len(bnb_ts)):
            dt = float(bnb_ts[j] - ts0)
            if dt <= 0 or dt > delay_sec:
                if dt > delay_sec:
                    break
                continue
            sim = 0.0
            parts: list[str] = []
            if ag0 and bnb_ag[j] == ag0:
                sim += 0.35
                parts.append("asset")
            if amt0 > 0 and bnb_amt[j] > 0:
                sim += 0.35 * max(0.0, 1.0 - abs(bnb_amt[j] - amt0) / max(amt0, bnb_amt[j], 1e-12))
                parts.append("amount")
            sim += 0.2 * max(0.0, 1.0 - dt / max(delay_sec, 1.0))
            parts.append("time")
            if recv0 and bnb_addr[j] and recv0 == bnb_addr[j]:
                sim += 0.1
                parts.append("receiver")
            if et0 and bnb_et[j]:
                a, b = set(et0.split("|")), set(str(bnb_et[j]).split("|"))
                overlap = len(a & b) / max(len(a | b), 1)
                sim += 0.1 * overlap
                parts.append("event_text")
            cands.append((sim, bnb_hashes[j], "+".join(parts)))

        cands.sort(key=lambda x: (-x[0], x[1]))
        for rank, (sim, dst_h, feat) in enumerate(cands[:top_k], start=1):
            rows.append(
                {
                    "src_tx_hash": src_h,
                    "rank": rank,
                    "dst_tx_hash": dst_h,
                    "score": sim,
                    "similarity_features": feat,
                    "delay_sec": "",
                    "candidate_window_sec": delay_sec,
                }
            )
    return pd.DataFrame(rows)


def _tx_pairs_to_flow(
    tx_pairs: pd.DataFrame,
    tx_map: pd.DataFrame,
    *,
    score_col: str = "score",
) -> pd.DataFrame:
    m = tx_map[["tx_hash", "flow_id", "chain"]].drop_duplicates("tx_hash")
    m["tx_hash"] = m["tx_hash"].map(_norm_hash)
    eth_f = m[m["chain"].astype(str).str.upper() == "ETH"].rename(columns={"flow_id": "src_flow_id"})
    bnb_f = m[m["chain"].astype(str).str.upper() == "BNB"].rename(columns={"flow_id": "dst_flow_id"})

    p = tx_pairs.copy()
    p["src_tx_hash"] = p["src_tx_hash"].map(_norm_hash)
    p["dst_tx_hash"] = p["dst_tx_hash"].map(_norm_hash)
    p = p.merge(eth_f[["tx_hash", "src_flow_id"]], left_on="src_tx_hash", right_on="tx_hash", how="left")
    p = p.merge(
        bnb_f[["tx_hash", "dst_flow_id"]],
        left_on="dst_tx_hash",
        right_on="tx_hash",
        how="left",
        suffixes=("_s", "_d"),
    )
    p = p[p["src_flow_id"].astype(str).str.len() > 0]
    p = p[p["dst_flow_id"].astype(str).str.len() > 0]
    p["_sc"] = pd.to_numeric(p[score_col], errors="coerce").fillna(0.0)

    agg = (
        p.groupby(["src_flow_id", "dst_flow_id"], sort=False)
        .agg(
            transport_mass=("_sc", "max"),
            tx_pair_count=("src_tx_hash", "count"),
            best_src_tx=("src_tx_hash", "first"),
            best_dst_tx=("dst_tx_hash", "first"),
        )
        .reset_index()
    )
    return agg


def _tx_retrieval_metrics(
    rankings: pd.DataFrame,
    truth: dict[str, str],
    *,
    top_k: int = 5,
) -> dict[str, float]:
    hits1: list[float] = []
    r3: list[float] = []
    r5: list[float] = []
    mrrs: list[float] = []

    for src, dst_true in truth.items():
        sub = rankings[rankings["src_tx_hash"] == src].sort_values("rank")
        if sub.empty:
            hits1.append(0.0)
            r3.append(0.0)
            r5.append(0.0)
            mrrs.append(0.0)
            continue
        preds = sub["dst_tx_hash"].astype(str).tolist()
        hits1.append(1.0 if preds[0] == dst_true else 0.0)
        r3.append(1.0 if dst_true in preds[:3] else 0.0)
        r5.append(1.0 if dst_true in preds[:5] else 0.0)
        rr = 0.0
        for i, p in enumerate(preds[:top_k], start=1):
            if p == dst_true:
                rr = 1.0 / i
                break
        mrrs.append(rr)

    n = max(len(truth), 1)
    return {
        "tx_hit_at_1": float(np.mean(hits1)) if hits1 else 0.0,
        "tx_recall_at_3": float(np.mean(r3)) if r3 else 0.0,
        "tx_recall_at_5": float(np.mean(r5)) if r5 else 0.0,
        "tx_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "tx_eval_queries": len(truth),
    }


def _flow_metrics_from_plan(plan: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    truth = _pair_set(labels)
    if plan.empty:
        return {"flow_pair_f1": 0.0, "flow_mass_recall": 0.0, "split_recovery": 0.0, "merge_recovery": 0.0}
    plan = plan.copy()
    plan["transport_mass"] = pd.to_numeric(plan.get("transport_mass"), errors="coerce").fillna(0.0)
    pred_pairs = plan[plan["transport_mass"] > 1e-9]
    pred_set = set(zip(pred_pairs["src_flow_id"].astype(str), pred_pairs["dst_flow_id"].astype(str)))
    soft = _metrics_for_pairs(truth, pred_set, pred_pairs)
    top1 = _top1_pair_set(plan, mass_thr=1e-9)
    hard = _metrics_for_pairs(truth, top1, pred_pairs)

    src_mass = (
        labels.groupby("src_flow_id")["src_amount_usd"]
        .apply(lambda s: float(pd.to_numeric(s, errors="coerce").max()))
        .to_dict()
    )
    row_sum = plan.groupby(plan["src_flow_id"].astype(str))["transport_mass"].sum()
    matched = 0.0
    for s, d in truth & pred_set:
        sub = plan[(plan["src_flow_id"].astype(str) == s) & (plan["dst_flow_id"].astype(str) == d)]
        p_sd = float(sub["transport_mass"].sum()) if not sub.empty else 0.0
        rs = float(row_sum.get(s, 0.0))
        sm = float(src_mass.get(s, 0.0))
        matched += (p_sd / (rs + 1e-18)) * sm
    tot = sum(src_mass.values()) or 1.0

    split_ok = split_tot = merge_ok = merge_tot = 0
    for _, r in labels.iterrows():
        sf, dfid = str(r.get("src_flow_id")), str(r.get("dst_flow_id"))
        hit = (sf, dfid) in pred_set
        if str(r.get("pattern_type")) == "one_to_many":
            split_tot += 1
            split_ok += int(hit)
        if str(r.get("pattern_type")) == "many_to_one":
            merge_tot += 1
            merge_ok += int(hit)

    return {
        "flow_pair_f1": soft["flow_pair_f1"],
        "pair_f1_top1": hard["flow_pair_f1"],
        "flow_mass_recall": float(matched / tot),
        "split_recovery": float(split_ok / max(split_tot, 1)),
        "merge_recovery": float(merge_ok / max(merge_tot, 1)),
        "top1_flow_accuracy": soft["top1_flow_correspondence_accuracy"],
        "top3_flow_accuracy": soft["top3_flow_correspondence_accuracy"],
        "top5_flow_accuracy": soft["top5_flow_correspondence_accuracy"],
        "real_split_label_rows": split_tot,
        "real_merge_label_rows": merge_tot,
    }


def _audit_abctracer(repo_parent: Path) -> str:
    abct = repo_parent / "ABCTracer"
    lines = ["# ABCTracer feasibility audit (read-only)\n"]
    lines.append(f"Path: `{abct}`\n")
    lines.append(f"Exists: {abct.is_dir()}\n\n")
    req = abct / "requirements.txt"
    lines.append(f"- requirements.txt: {'yes' if req.is_file() else 'no'}\n")
    if req.is_file():
        lines.append(f"  - deps sample: `{req.read_text(encoding='utf-8', errors='replace').splitlines()[:3]}`\n")
    ckpt = list(abct.rglob("*.pt")) + list(abct.rglob("*.pth")) + list(abct.rglob("*.ckpt"))
    ckpt = [p for p in ckpt if ".venv" not in str(p)]
    lines.append(f"- pretrained checkpoints under repo (excl. .venv): **{len(ckpt)}**\n")
    lines.append("- Can ingest current `evidence_eth.csv` / `evidence_bnb.csv` directly: **no** (expects AllenNLP IR/TIR training pipeline on custom CCT dataset).\n")
    lines.append("- GPU script `run_ir_gpu.ps1` present: **yes** if file exists.\n")
    lines.append(f"  - `run_ir_gpu.ps1`: {(abct / 'run_ir_gpu.ps1').is_file()}\n")
    lines.append("\n## Decision\n\n")
    lines.append(
        "**Did not run original ABCTracer IR/TIR** in Phase 7.5: no bundled checkpoint, "
        "environment/training coupling, and task format mismatch with frozen Celer evidence CSVs. "
        "Implemented **ABCTracer-style** reproducible transaction retrieval baseline instead.\n"
    )
    return "".join(lines)


def _audit_connector(repo_parent: Path) -> str:
    conn = repo_parent / "Connector"
    lines = [
        "# Connector feasibility note\n\n",
        f"Path: `{conn}` exists={conn.is_dir()}\n\n",
        "**Did not run original Connector-main pipeline** at scale in Phase 7.5 "
        "(external repo, separate venv/Celer CSV conventions). "
        "Implemented **Connector-style** rule baseline on evidence observables only.\n",
    ]
    return "".join(lines)


def _load_frozen_rc_uot() -> dict[str, Any]:
    syn = RUN_ROOT / "synthetic" / "synthetic_eval_aggregated.json"
    out: dict[str, Any] = {"source": "frozen_phase2_phase3", "note": "semi_synthetic_stress_test; not re-run"}
    if syn.is_file():
        agg = json.loads(syn.read_text(encoding="utf-8"))
        bag = agg.get("aggregated") or agg.get("means") or {}
        for k in ("split_recovery", "merge_recovery", "topk_recovery"):
            if k in bag and isinstance(bag[k], dict) and "clipped_t" in bag[k]:
                out[k] = bag[k]["clipped_t"].get("mean")
    abl = RUN_ROOT / "paper_tables" / "table_ablation_multi_seed.csv"
    if abl.is_file():
        df = pd.read_csv(abl)
        row = df[df["experiment"] == "full_rc_uot"]
        if not row.empty:
            r = row.iloc[0]
            out["pair_f1_semi_synthetic"] = float(r["pair_f1_mean"])
            out["flow_mass_recall_semi_synthetic"] = float(r["flow_mass_recall_mean"])
            out["topk_recovery_semi_synthetic"] = float(r["topk_recovery_mean"])
    p21 = RUN_ROOT / "synthetic" / "phase2_1_unmatched_decoy_metrics.json"
    if p21.is_file():
        d = json.loads(p21.read_text(encoding="utf-8"))
        seeds = [str(s) for s in d.get("seeds", [])]
        aurocs = []
        decoy_aucs = []
        for s in seeds:
            block = (d.get("per_seed") or {}).get(s) or {}
            um = (block.get("unmatched") or {}).get("scores") or {}
            if "unmatched_ratio" in um:
                aurocs.append(um["unmatched_ratio"].get("auroc"))
            dec = (block.get("decoy") or {}).get("scores") or {}
            if "decoy_mass" in dec:
                decoy_aucs.append(dec["decoy_mass"].get("auroc"))
        if aurocs:
            out["unmatched_auroc_unmatched_ratio"] = float(np.nanmean(aurocs))
        if decoy_aucs:
            out["decoy_auroc_mass"] = float(np.nanmean(decoy_aucs))
    if syn.is_file():
        bag = json.loads(syn.read_text(encoding="utf-8")).get("aggregated") or {}
        dr = bag.get("decoy_rejection_rate") or {}
        if isinstance(dr, dict) and "clipped_t" in dr:
            out["decoy_rejection_rate"] = dr["clipped_t"].get("mean")
    out["unmatched_detection_f1"] = 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    baselines_dir = run_root / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    repo_parent = REPO_ROOT.parent
    anchors, eth_tx, bnb_tx, flow_labels = _load_tx_tables_at(run_root)
    # Evaluate on supervised anchor src txs only (matches Celer tx GT).
    anchor_src = set(anchors["src_tx_hash"])
    eth_tx = eth_tx[eth_tx["tx_hash"].isin(anchor_src)].copy()
    tx_map = pd.read_csv(run_root / "labels" / "tx_to_flow_map.csv", dtype=str, keep_default_na=False)
    tx_map["tx_hash"] = tx_map["tx_hash"].map(_norm_hash)

    truth_tx = dict(zip(anchors["src_tx_hash"], anchors["dst_tx_hash"]))

    # --- Connector-style ---
    miss_log: list[str] = []
    conn_tx = _connector_candidates(
        eth_tx,
        bnb_tx,
        delay_sec=CONNECTOR_DELAY_PRIMARY,
        fee_ratio=FEE_RATIO_PRIMARY,
        missing_log=miss_log,
    )
    conn_tx.to_csv(baselines_dir / "connector_style_tx_pairs.csv", index=False)

    conn_flow = _tx_pairs_to_flow(conn_tx, tx_map)
    conn_flow.to_csv(baselines_dir / "connector_style_flow_pairs.csv", index=False)

    # sensitivity row (3600s / 5%) — summary only
    conn_tx_s = _connector_candidates(
        eth_tx, bnb_tx, delay_sec=CONNECTOR_DELAY_SENS, fee_ratio=FEE_RATIO_SENS, missing_log=[]
    )

    # --- ABCTracer-style ---
    (baselines_dir / "abctracer_feasibility_audit.md").write_text(
        _audit_abctracer(repo_parent) + "\n" + _audit_connector(repo_parent),
        encoding="utf-8",
    )

    rank = _abctracer_rankings(eth_tx, bnb_tx, delay_sec=CONNECTOR_DELAY_PRIMARY, top_k=5)
    rank.to_csv(baselines_dir / "abctracer_style_tx_rankings.csv", index=False)

    # top1 tx pairs from rankings
    top1 = rank[rank["rank"] == 1][["src_tx_hash", "dst_tx_hash", "score"]].copy()
    top1["matched_rules"] = "abctracer_style_retrieval"
    top1["delay_sec"] = ""
    top1["amount_diff_ratio"] = ""
    top1["reason"] = "top1_retrieval"
    abct_flow = _tx_pairs_to_flow(top1, tx_map)
    abct_flow.to_csv(baselines_dir / "abctracer_style_flow_pairs.csv", index=False)

    # --- Rankings for connector (top1 per src for tx metrics) ---
    conn_rank_rows = []
    for src, g in conn_tx.groupby("src_tx_hash", sort=False):
        g2 = g.sort_values("score", ascending=False)
        for i, (_, r) in enumerate(g2.head(5).iterrows(), start=1):
            conn_rank_rows.append(
                {
                    "src_tx_hash": src,
                    "rank": i,
                    "dst_tx_hash": r["dst_tx_hash"],
                    "score": r["score"],
                }
            )
    conn_rank = pd.DataFrame(conn_rank_rows)

    # --- Evaluation ---
    eval_rows: list[dict[str, Any]] = []

    def _add(method: str, scope: str, tx_m: dict, flow_m: dict, diag: dict) -> None:
        eval_rows.append(
            {
                "method": method,
                "eval_scope": scope,
                **tx_m,
                **flow_m,
                **diag,
            }
        )

    _meta = {
        "evaluation_scope": "celer_real_tx_to_flow",
        "directly_comparable": "false",
        "notes": "transaction-level adapted baseline; not directly comparable with RC-UOT semi-synthetic result",
    }

    conn_tx_m = _tx_retrieval_metrics(conn_rank, truth_tx)
    conn_flow_m = _flow_metrics_from_plan(conn_flow, flow_labels)
    _add(
        "connector_style",
        "celer_real",
        conn_tx_m,
        conn_flow_m,
        {
            **_meta,
            "baseline_type": "adapted_transaction_level",
            "unmatched_detection_f1": "N/A",
            "decoy_auroc": "N/A",
            "delay_window_sec": CONNECTOR_DELAY_PRIMARY,
            "fee_ratio": FEE_RATIO_PRIMARY,
            "missing_fields": "|".join(sorted(set(miss_log)))[:500],
        },
    )

    conn_tx_m_s = _tx_retrieval_metrics(
        pd.DataFrame(
            [
                {"src_tx_hash": s, "rank": i + 1, "dst_tx_hash": r["dst_tx_hash"]}
                for s, g in conn_tx_s.groupby("src_tx_hash", sort=False)
                for i, (_, r) in enumerate(g.sort_values("score", ascending=False).head(5).iterrows())
            ]
        ),
        truth_tx,
    )
    eval_rows.append(
        {
            "method": "connector_style",
            "eval_scope": "celer_real_sensitivity",
            "evaluation_scope": "celer_real_tx_to_flow",
            "directly_comparable": "false",
            "baseline_type": "adapted_transaction_level",
            "delay_window_sec": CONNECTOR_DELAY_SENS,
            "fee_ratio": FEE_RATIO_SENS,
            **{k: conn_tx_m_s[k] for k in conn_tx_m_s},
            "notes": "tx-only sensitivity (3600s/5%); flow metrics omitted",
        }
    )

    abct_tx_m = _tx_retrieval_metrics(rank, truth_tx)
    abct_flow_m = _flow_metrics_from_plan(abct_flow, flow_labels)
    _add(
        "abctracer_style",
        "celer_real",
        abct_tx_m,
        abct_flow_m,
        {
            **_meta,
            "baseline_type": "adapted_transaction_level",
            "unmatched_detection_f1": "N/A",
            "decoy_auroc": "N/A",
        },
    )

    rc = _load_frozen_rc_uot()
    eval_rows.append(
        {
            "method": "rc_uot_full",
            "eval_scope": "semi_synthetic_frozen",
            "evaluation_scope": "semi_synthetic_flow_stress",
            "directly_comparable": "false",
            "baseline_type": "native_flow_level",
            "tx_hit_at_1": "N/A",
            "tx_recall_at_3": "N/A",
            "tx_recall_at_5": "N/A",
            "tx_mrr": "N/A",
            "flow_pair_f1": rc.get("pair_f1_semi_synthetic"),
            "flow_mass_recall": rc.get("flow_mass_recall_semi_synthetic"),
            "split_recovery": rc.get("split_recovery"),
            "merge_recovery": rc.get("merge_recovery"),
            "top3_flow_accuracy": rc.get("topk_recovery_semi_synthetic"),
            "unmatched_detection_f1": rc.get("unmatched_detection_f1"),
            "unmatched_auroc_unmatched_ratio": rc.get("unmatched_auroc_unmatched_ratio"),
            "decoy_auroc_mass": rc.get("decoy_auroc_mass"),
            "decoy_rejection_rate": rc.get("decoy_rejection_rate"),
            "notes": "semi-synthetic flow-level stress; not directly comparable to Table A; RC-UOT full real-pool transport not frozen",
        }
    )

    results = {
        "phase": "7.5",
        "connector_original_run": False,
        "abctracer_original_run": False,
        "connector_style": {"tx_pairs": len(conn_tx), "flow_pairs": len(conn_flow), **conn_tx_m, **conn_flow_m},
        "abctracer_style": {"rank_rows": len(rank), "flow_pairs": len(abct_flow), **abct_tx_m, **abct_flow_m},
        "rc_uot_frozen": rc,
        "eval_rows": eval_rows,
        "fairness": {
            "same_tx_ground_truth": "labels/tx_anchor_labels_from_celer.csv (7296 pairs)",
            "same_flow_ground_truth": "labels/flow_labels.csv (7128 edges)",
            "baselines_are_transaction_level_adapted": True,
            "split_merge_task_mismatch": "transaction-level baselines emit near one-to-one tx/flow pairs; many_to_one edges are structurally harder",
            "comparability": "Table A (celer real adapted tx baselines) and Table B (RC-UOT semi-synthetic stress) are not directly comparable",
        },
        "leakage_audit": {
            "scoring_uses_gt_dst": False,
            "scoring_uses_flow_labels_pairs": False,
            "scoring_uses_tx_to_flow_map_features": False,
            "scoring_uses_evidence_only": True,
            "tx_to_flow_map_used_post_prediction_only": True,
            "anchors_used_for_eval_query_set_and_metrics_only": True,
        },
    }
    (baselines_dir / "external_baseline_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(baselines_dir / "external_baseline_table.csv", index=False)
    _write_split_tables(df_eval, rc, baselines_dir)

    _write_readme(results, baselines_dir)
    _write_manuscript_section(results, run_root)
    print(json.dumps({"ok": True, "baselines_dir": str(baselines_dir)}, indent=2))


def _write_split_tables(df: pd.DataFrame, rc: dict[str, Any], baselines_dir: Path) -> None:
    """Table A: external tx baselines on real Celer; Table B: RC-UOT semi-synthetic stress."""
    scope_a = df["evaluation_scope"].astype(str) == "celer_real_tx_to_flow"
    table_a = df[scope_a & (df["eval_scope"].astype(str) == "celer_real")].copy()
    cols_a = [
        "method",
        "evaluation_scope",
        "directly_comparable",
        "baseline_type",
        "tx_hit_at_1",
        "tx_recall_at_3",
        "tx_recall_at_5",
        "tx_mrr",
        "flow_pair_f1",
        "flow_mass_recall",
        "split_recovery",
        "merge_recovery",
        "real_split_label_rows",
        "real_merge_label_rows",
        "notes",
    ]
    table_a = table_a[[c for c in cols_a if c in table_a.columns]]
    table_a.to_csv(baselines_dir / "table_a_external_tx_baselines_celer_real.csv", index=False)

    table_b = pd.DataFrame(
        [
            {
                "method": "rc_uot_full",
                "evaluation_scope": "semi_synthetic_flow_stress",
                "directly_comparable": "false",
                "baseline_type": "native_flow_level",
                "split_recovery": rc.get("split_recovery"),
                "merge_recovery": rc.get("merge_recovery"),
                "topk_recovery": rc.get("topk_recovery_semi_synthetic"),
                "flow_pair_f1": rc.get("pair_f1_semi_synthetic"),
                "flow_mass_recall": rc.get("flow_mass_recall_semi_synthetic"),
                "unmatched_detection_f1": rc.get("unmatched_detection_f1"),
                "unmatched_auroc_unmatched_ratio": rc.get("unmatched_auroc_unmatched_ratio"),
                "decoy_rejection_rate": rc.get("decoy_rejection_rate"),
                "decoy_auroc_mass": rc.get("decoy_auroc_mass"),
                "notes": "semi-synthetic flow-level stress; not directly comparable to Table A",
            }
        ]
    )
    table_b.to_csv(baselines_dir / "table_b_rc_uot_semi_synthetic_stress.csv", index=False)


def _write_readme(results: dict[str, Any], baselines_dir: Path) -> None:
    c = results["connector_style"]
    a = results["abctracer_style"]
    rc = results["rc_uot_frozen"]
    text = f"""# Phase 7.5 External Baselines

## Methods

| Method | Original system run? | Implementation |
|--------|---------------------|----------------|
| Connector-style | **No** | Rule baseline (`scripts/run_phase7_5_external_baselines.py`) |
| ABCTracer-style | **No** | Retrieval baseline (see `abctracer_feasibility_audit.md`) |
| RC-UOT full | N/A (frozen) | Phase 2/3 semi-synthetic stress (not re-run) |

## Scoring inputs (no GT leakage)

- **Scoring:** `evidence_eth.csv`, `evidence_bnb.csv` observables only (time, amount proxy, asset_group, addresses, event text).
- **Post-prediction mapping:** `tx_to_flow_map.csv` → flow pairs.
- **Evaluation GT:** `tx_anchor_labels_from_celer.csv`, `flow_labels.csv` (metrics only).

## Primary tables (do not merge)

### Table A — `table_a_external_tx_baselines_celer_real.csv`

`evaluation_scope=celer_real_tx_to_flow`, `directly_comparable=false`

| Method | Tx Hit@1 | Flow Pair-F1 | Split | Merge |
|--------|----------|--------------|-------|-------|
| Connector-style | {c.get('tx_hit_at_1', 0):.3f} | {c.get('flow_pair_f1', 0):.3f} | {c.get('split_recovery', 0):.3f} | {c.get('merge_recovery', 0):.3f} |
| ABCTracer-style | {a.get('tx_hit_at_1', 0):.3f} | {a.get('flow_pair_f1', 0):.3f} | {a.get('split_recovery', 0):.3f} | {a.get('merge_recovery', 0):.3f} |

Adapted transaction-level baselines; **not** native flow-level soft correspondence.

### Table B — `table_b_rc_uot_semi_synthetic_stress.csv`

`evaluation_scope=semi_synthetic_flow_stress`, `directly_comparable=false`

| Metric | RC-UOT full (frozen) |
|--------|----------------------|
| Split recovery | {rc.get('split_recovery', '—')} |
| Merge recovery | {rc.get('merge_recovery', '—')} |
| Top-k recovery | {rc.get('topk_recovery_semi_synthetic', '—')} |
| Pair-F1 (semi-synthetic) | {rc.get('pair_f1_semi_synthetic', '—')} |

**Not directly comparable** to Table A. RC-UOT full real-pool transport is **not frozen**.

## Legacy wide table

`external_baseline_table.csv` retains all rows with `evaluation_scope`, `directly_comparable`, `notes`.

Unmatched/decoy: **N/A** for transaction-level baselines (diagnostic only for RC-UOT).
"""
    (baselines_dir / "README.md").write_text(text, encoding="utf-8")


def _write_manuscript_section(results: dict[str, Any], run_root: Path) -> None:
    c = results["connector_style"]
    a = results["abctracer_style"]
    rc = results["rc_uot_frozen"]
    path = run_root / "manuscript" / "external_baselines_section.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# External Baselines (Phase 7.5)

## Role of Connector and ABCTracer

Connector and ABCTracer are **transaction-level** tracing / retrieval baselines. They do **not** natively model:

- flow-level **soft transport** plans,
- **split / merge** mass allocation,
- **unmatched** mass,
- **risk-constrained unbalanced correspondence** (RC-UOT / CSFFC).

We map their transaction-level outputs to flow IDs **only for reviewer-facing external comparison** on the real Celer mapping (`tx_to_flow_map.csv` used **after** prediction).

## What we ran

- **Original Connector / ABCTracer**: not executed; **Connector-style** rules and **ABCTracer-style** retrieval in-repo on **evidence observables only** (leakage audit in `diagnosis_report.md`).
- **RC-UOT**: Phase 2/3 **frozen** semi-synthetic stress results only (not re-run on full real pool).

## Table A — External transaction-level baselines (real Celer mapping)

`evaluation_scope = celer_real_tx_to_flow` — **not directly comparable** to Table B.

| Method | Tx Hit@1 | Tx R@3 | Tx R@5 | Flow Pair-F1 (adapted) | Flow mass recall | Split | Merge |
|--------|----------|--------|--------|------------------------|------------------|-------|-------|
| Connector-style | {c.get('tx_hit_at_1', 0):.3f} | {c.get('tx_recall_at_3', 0):.3f} | {c.get('tx_recall_at_5', 0):.3f} | {c.get('flow_pair_f1', 0):.3f} | {c.get('flow_mass_recall', 0):.3f} | {c.get('split_recovery', 0):.3f} | {c.get('merge_recovery', 0):.3f} |
| ABCTracer-style | {a.get('tx_hit_at_1', 0):.3f} | {a.get('tx_recall_at_3', 0):.3f} | {a.get('tx_recall_at_5', 0):.3f} | {a.get('flow_pair_f1', 0):.3f} | {a.get('flow_mass_recall', 0):.3f} | {a.get('split_recovery', 0):.3f} | {a.get('merge_recovery', 0):.3f} |

High adapted Pair-F1 reflects that the canonical Celer mapping is **dominated by single-anchor flow pairs** (mostly one-to-one). It does **not** imply superiority on CSFFC split/merge/unmatched soft correspondence.

## Table B — RC-UOT semi-synthetic flow-level stress (frozen)

`evaluation_scope = semi_synthetic_flow_stress` — **not directly comparable** to Table A.

| Metric | RC-UOT full (frozen, seeds 42–46) |
|--------|-----------------------------------|
| Split recovery | {rc.get('split_recovery', '—')} |
| Merge recovery | {rc.get('merge_recovery', '—')} |
| Top-k recovery | {rc.get('topk_recovery_semi_synthetic', '—')} |
| Pair-F1 (semi-synthetic subgraph) | {rc.get('pair_f1_semi_synthetic', '—')} |
| Flow-mass recall (semi-synthetic) | {rc.get('flow_mass_recall_semi_synthetic', '—')} |

Semi-synthetic Pair-F1 (~0.017) is **not** comparable to Table A real-pool adapted Pair-F1.

## Recommended wording

> Connector-style and ABCTracer-style baselines are strong transaction-level references on the real Celer mapping, especially because the canonical mapping is dominated by single-anchor flow pairs. However, these baselines do not natively produce soft flow-level transport plans or explicit unmatched mass. We therefore report them as adapted external baselines and interpret RC-UOT's advantage primarily in the flow-level split/merge stress and component-sensitivity settings.

## RC-UOT positioning (allowed)

- CSFFC formulation and supervised flow-label dataset.
- Split/merge **semi-synthetic stress** (Table B).
- Time causality and unbalanced-mass **component sensitivity** (Phase 3 ablation).

## Prohibited claims

- RC-UOT **outperforms Connector/ABCTracer overall**.
- Connector / ABCTracer **fail**.
- RC-UOT is **best on all metrics**.
- **Strong** unmatched detection or **strong** decoy rejection.

## Main-text impact

Add external baselines as a **separate, scope-labeled** comparison. Do **not** rank Table A and Table B in one combined performance table.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
