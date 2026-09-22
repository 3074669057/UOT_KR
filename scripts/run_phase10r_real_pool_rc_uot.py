#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 10R-A: Real Celer pool RC-UOT ranking (candidate-pruned) vs external baselines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cross.application.paper_experiment_closure import _uot_pool_kwargs_from_uk
from cross.application.pipeline import uot_kwargs_from_config
from cross.application.standalone_flow_uot import run_standalone_flow_uot
from cross.config.output_layout import output_file
from cross.config.paths import CROSS_ROOT
from cross.domain.evaluation.flow_eval import run_flow_level_eval
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv
from cross.infrastructure.config.service import load_and_validate_config
from cross.interfaces.cli import build_parser

RUN_ROOT = _REPO / "out" / "paper_full_pipeline_run"
OUT_DIR = RUN_ROOT / "real_pool_rc_uot"

CANDIDATE_CONFIGS = [
    {"name": "primary_1800_tight_k50", "delay_sec": 1800, "amt_lo": 0.90, "amt_hi": 1.05, "top_k": 50},
    {"name": "sens_3600_tight_k50", "delay_sec": 3600, "amt_lo": 0.90, "amt_hi": 1.05, "top_k": 50},
    {"name": "sens_1800_loose_k50", "delay_sec": 1800, "amt_lo": 0.80, "amt_hi": 1.10, "top_k": 50},
    {"name": "recall_k20_1800", "delay_sec": 1800, "amt_lo": 0.90, "amt_hi": 1.05, "top_k": 20},
    {"name": "recall_k100_1800", "delay_sec": 1800, "amt_lo": 0.90, "amt_hi": 1.05, "top_k": 100},
]


def _candidates_for_src(
    eth: dict[str, Any],
    bnb_flows: list[dict[str, Any]],
    *,
    delay_sec: float,
    amt_lo: float,
    amt_hi: float,
    top_k: int,
    missing: dict[str, int],
) -> list[tuple[str, float]]:
    se = float(eth.get("end_time") or eth.get("start_time") or 0.0)
    ss = float(eth.get("start_time") or se)
    s_usd = float(eth.get("amount_usd") or 0.0)
    ag = str(eth.get("asset_group") or "").strip()
    rid = str(eth.get("route_id") or "").strip()
    cands: list[tuple[str, float]] = []
    for j, b in enumerate(bnb_flows):
        ds = float(b.get("start_time") or 0.0)
        de = float(b.get("end_time") or ds)
        if not (ds >= ss or de >= ss):
            continue
        delay = ds - se
        if delay < 0 or delay > delay_sec:
            continue
        bag = str(b.get("asset_group") or "").strip()
        brid = str(b.get("route_id") or "").strip()
        if ag and bag and ag != bag:
            if not (rid and brid and rid == brid):
                continue
        elif rid and brid and rid == brid:
            pass
        elif not ag and not rid:
            missing["asset_route_both_empty"] = missing.get("asset_route_both_empty", 0) + 1
        t_usd = float(b.get("amount_usd") or 0.0)
        if s_usd > 0 and t_usd > 0:
            ratio = t_usd / s_usd
            if ratio < amt_lo or ratio > amt_hi:
                continue
        elif s_usd <= 0 or t_usd <= 0:
            missing["amount_usd_zero"] = missing.get("amount_usd_zero", 0) + 1
        score = 1.0 / (1.0 + delay / max(delay_sec, 1.0))
        if s_usd > 0 and t_usd > 0:
            score += max(0.0, 1.0 - abs(ratio - 1.0)) * 0.25
        cands.append((str(b.get("flow_id") or ""), score))
    cands.sort(key=lambda x: (-x[1], x[0]))
    return cands[:top_k]


def _candidate_audit(
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    labels: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    missing: dict[str, int] = {}
    bnb_by_id = {str(b["flow_id"]): b for b in bnb_flows}
    truth_rows = labels[labels.get("label_source", pd.Series()).astype(str).str.lower() == "celer_source"]
    truth_pairs = set(
        zip(truth_rows["src_flow_id"].astype(str), truth_rows["dst_flow_id"].astype(str))
    )
    pat_map = dict(zip(truth_rows["src_flow_id"].astype(str), truth_rows.get("pattern_type", "").astype(str)))

    pair_rows: list[dict[str, Any]] = []
    hit = {20: 0, 50: 0, 100: 0}
    tot = 0
    by_pat: dict[str, dict[str, int]] = {}

    for e in eth_flows:
        sf = str(e.get("flow_id") or "")
        if sf not in {s for s, _ in truth_pairs}:
            continue
        tot += 1
        true_dsts = {d for s, d in truth_pairs if s == sf}
        for k in (20, 50, 100):
            cands = _candidates_for_src(
                e,
                bnb_flows,
                delay_sec=float(cfg["delay_sec"]),
                amt_lo=float(cfg["amt_lo"]),
                amt_hi=float(cfg["amt_hi"]),
                top_k=k,
                missing=missing,
            )
            cand_ids = {c[0] for c in cands}
            if true_dsts & cand_ids:
                hit[k] += 1
            if cfg["name"].startswith("primary") or cfg["name"].startswith("sens") or cfg["name"].startswith("recall"):
                if k == int(cfg["top_k"]):
                    for dst_h, sc in cands:
                        pair_rows.append(
                            {
                                "config": cfg["name"],
                                "src_flow_id": sf,
                                "dst_flow_id": dst_h,
                                "heuristic_score": sc,
                                "in_gt": int((sf, dst_h) in truth_pairs),
                            }
                        )
        pt = str(pat_map.get(sf, "unknown"))
        by_pat.setdefault(pt, {"total": 0, "in_k50": 0})
        by_pat[pt]["total"] += 1
        c50 = {c[0] for c in _candidates_for_src(e, bnb_flows, delay_sec=cfg["delay_sec"], amt_lo=cfg["amt_lo"], amt_hi=cfg["amt_hi"], top_k=50, missing=missing)}
        if true_dsts & c50:
            by_pat[pt]["in_k50"] += 1

    return {
        "config": cfg["name"],
        "delay_sec": cfg["delay_sec"],
        "amount_ratio": [cfg["amt_lo"], cfg["amt_hi"]],
        "top_k": cfg["top_k"],
        "supervised_src_flows": tot,
        "candidate_recall_at_20": float(hit[20] / max(tot, 1)),
        "candidate_recall_at_50": float(hit[50] / max(tot, 1)),
        "candidate_recall_at_100": float(hit[100] / max(tot, 1)),
        "pattern_type_recall_k50": {
            p: float(v["in_k50"] / max(v["total"], 1)) for p, v in by_pat.items()
        },
        "missing_field_counts": missing,
        "pair_rows": pair_rows,
    }


def _allowed_candidate_pairs(pairs_csv: Path, config_name: str) -> set[tuple[str, str]]:
    if not pairs_csv.is_file():
        return set()
    df = pd.read_csv(pairs_csv, dtype=str, keep_default_na=False)
    if "config" in df.columns:
        df = df[df["config"].astype(str) == config_name]
    return set(zip(df["src_flow_id"].astype(str), df["dst_flow_id"].astype(str)))


def _filter_plan_candidate_pruned(
    plan: pd.DataFrame,
    *,
    allowed: set[tuple[str, str]] | None,
    top_k: int,
    decode_threshold: float,
) -> pd.DataFrame:
    if plan.empty:
        return plan
    plan = plan.copy()
    plan["_m"] = pd.to_numeric(plan.get("transport_mass"), errors="coerce").fillna(0.0)
    plan = plan[plan["_m"] > float(decode_threshold)]
    if allowed:
        allow_df = pd.DataFrame(list(allowed), columns=["src_flow_id", "dst_flow_id"])
        plan = plan.merge(
            allow_df,
            on=["src_flow_id", "dst_flow_id"],
            how="inner",
        )
    rows: list[pd.DataFrame] = []
    for _, g in plan.groupby("src_flow_id", sort=False):
        rows.append(g.sort_values("_m", ascending=False).head(int(top_k)))
    out = pd.concat(rows, ignore_index=True) if rows else plan.iloc[0:0]
    return out.drop(columns=["_m"], errors="ignore")


def _merge_component_plans_raw(
    component_meta: list[dict[str, Any]],
) -> pd.DataFrame:
    """Concatenate per-component solver transport plans without candidate pruning."""
    parts: list[pd.DataFrame] = []
    for comp in component_meta:
        comp_dir = Path(comp["dir"])
        p_path = output_file(comp_dir, "uot_transport_plan.csv")
        if not p_path.is_file():
            continue
        parts.append(pd.read_csv(p_path, dtype=str, keep_default_na=False))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _component_coverage(
    component_meta: list[dict[str, Any]], *, n_eth: int, n_bnb: int
) -> tuple[float, float, int, int]:
    sum_eth = sum(int(c.get("n_eth", 0)) for c in component_meta)
    sum_bnb = sum(int(c.get("n_bnb", 0)) for c in component_meta)
    return sum_eth / max(n_eth, 1), sum_bnb / max(n_bnb, 1), sum_eth, sum_bnb


def _count_primary_candidate_pairs(pairs_csv: Path, config_name: str) -> int:
    if not pairs_csv.is_file():
        return 0
    df = pd.read_csv(pairs_csv, dtype=str, keep_default_na=False)
    if "config" in df.columns:
        df = df[df["config"].astype(str) == config_name]
    return int(len(df))


def _finalize_real_pool_audit_exports(
    *,
    out_dir: Path,
    run_root: Path,
    component_meta: list[dict[str, Any]],
    allowed: set[tuple[str, str]],
    primary: dict[str, Any],
    stats: dict[str, Any],
    decode_threshold: float,
    n_eth: int,
    n_bnb: int,
) -> dict[str, Any]:
    """Write full/filtered transport exports, run eval on full plan, attach audit metadata."""
    labels_path = run_root / "labels" / "flow_labels.csv"
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    truth = set(zip(labels["src_flow_id"].astype(str), labels["dst_flow_id"].astype(str)))

    plan_full = _merge_component_plans_raw(component_meta)
    plan_filtered = _merge_component_plans(
        out_dir / "uot_run",
        component_meta,
        allowed=allowed,
        top_k=int(primary["top_k"]),
        decode_threshold=decode_threshold,
    )

    full_path = out_dir / "real_pool_rc_uot_transport_full_unfiltered.csv"
    filt_path = out_dir / "real_pool_rc_uot_transport.csv"
    plan_full.to_csv(full_path, index=False)
    plan_filtered.to_csv(filt_path, index=False)

    uot_run = out_dir / "uot_run"
    plan_src = output_file(uot_run, "uot_transport_plan.csv")
    plan_src.parent.mkdir(parents=True, exist_ok=True)
    plan_full.to_csv(plan_src, index=False)

    um_src = output_file(uot_run, "uot_unmatched_mass.csv")
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_flow_level_eval(
        labels_path, plan_src, um_src if um_src.is_file() else None, eval_dir, min_label_confidence=0.0
    )
    metrics.update(_mrr_topk(plan_filtered, truth))
    eth_cov, bnb_cov, sum_eth, sum_bnb = _component_coverage(component_meta, n_eth=n_eth, n_bnb=n_bnb)
    primary_pairs = _count_primary_candidate_pairs(out_dir / "real_pool_candidate_pairs.csv", primary["name"])
    n_full = int(len(plan_full))
    n_filt = int(len(plan_filtered))

    metrics.update(
        {
            "evaluation_scope": "real_celer_flow_level",
            "candidate_config": primary["name"],
            "exported_transport_rows": n_filt,
            "exported_transport_file": "real_pool_rc_uot_transport.csv",
            "full_unfiltered_transport_rows": n_full,
            "full_unfiltered_transport_file": "real_pool_rc_uot_transport_full_unfiltered.csv",
            "eval_source": "full_unfiltered_transport_plan",
            "rank_eval_source": "candidate_pruned_transport_plan",
            "plan_rows_before_filter": n_full,
            "plan_rows_after_candidate_filter": n_filt,
            "allowed_heuristic_pairs": int(len(allowed)),
            "primary_candidate_pairs": int(primary_pairs),
            "component_eth_coverage": float(eth_cov),
            "component_bnb_coverage": float(bnb_cov),
            "component_eth_flows_in_solver": int(sum_eth),
            "component_bnb_flows_in_solver": int(sum_bnb),
            "n_eth": int(n_eth),
            "n_bnb": int(n_bnb),
            "transport_export_note": (
                f"Table C flow-level metrics are computed from the full component-merged transport plan "
                f"({n_full} rows; eval_source=full_unfiltered_transport_plan). "
                f"real_pool_rc_uot_transport.csv is the candidate-pruned filtered export ({n_filt} rows) "
                f"and is NOT the eval source. flow_recall@k / MRR use the candidate-pruned ranking view."
            ),
            "real_data_component_coverage_note": (
                f"n_eth={n_eth}, n_bnb={n_bnb}; component solver coverage ETH={eth_cov:.4f} "
                f"({sum_eth}/{n_eth}), BNB={bnb_cov:.4f} ({sum_bnb}/{n_bnb}). Flows outside "
                f"asset-group components were skipped for per-component solver sizing (not label mutation). "
                f"Candidate recall denominator is supervised_src_flows={stats.get(primary['name'], {}).get('supervised_src_flows', n_eth)}. "
                "real-pool split/merge coverage is limited; split/merge claims remain grounded in semi-synthetic stress tests."
            ),
        }
    )
    (out_dir / "real_pool_rc_uot_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_table_c(run_root, out_dir, stats, primary, metrics)
    return metrics


def _merge_component_plans(
    uot_run: Path,
    component_meta: list[dict[str, Any]],
    *,
    allowed: set[tuple[str, str]] | None,
    top_k: int,
    decode_threshold: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for comp in component_meta:
        comp_dir = Path(comp["dir"])
        p_path = output_file(comp_dir, "uot_transport_plan.csv")
        if not p_path.is_file():
            continue
        part = pd.read_csv(p_path, dtype=str, keep_default_na=False)
        parts.append(
            _filter_plan_candidate_pruned(
                part, allowed=allowed, top_k=top_k, decode_threshold=decode_threshold
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _mrr_topk(plan: pd.DataFrame, truth: set[tuple[str, str]]) -> dict[str, float]:
    if plan.empty:
        return {"flow_recall_at_1": 0.0, "flow_recall_at_3": 0.0, "flow_recall_at_5": 0.0, "flow_mrr": 0.0}
    plan = plan.copy()
    plan["_m"] = pd.to_numeric(plan.get("transport_mass"), errors="coerce").fillna(0.0)
    r1, r3, r5, mrrs = [], [], [], []
    srcs = {s for s, _ in truth}
    for sf in srcs:
        true_d = {d for s, d in truth if s == sf}
        g = plan[plan["src_flow_id"].astype(str) == sf].sort_values("_m", ascending=False)
        dsts = g["dst_flow_id"].astype(str).tolist()
        r1.append(1.0 if dsts and dsts[0] in true_d else 0.0)
        r3.append(1.0 if any(d in set(dsts[:3]) for d in true_d) else 0.0)
        r5.append(1.0 if any(d in set(dsts[:5]) for d in true_d) else 0.0)
        rr = 0.0
        for i, d in enumerate(dsts[:5], start=1):
            if d in true_d:
                rr = 1.0 / i
                break
        mrrs.append(rr)
    n = max(len(srcs), 1)
    return {
        "flow_recall_at_1": float(np.mean(r1)) if r1 else 0.0,
        "flow_recall_at_3": float(np.mean(r3)) if r3 else 0.0,
        "flow_recall_at_5": float(np.mean(r5)) if r5 else 0.0,
        "flow_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT)
    ap.add_argument("--skip-uot", action="store_true", help="Only candidate audit, skip UOT solve")
    ap.add_argument("--eval-only", action="store_true", help="Merge/filter component plans and re-run eval + Table C")
    ap.add_argument("--force", action="store_true", help="Recompute candidates and UOT")
    args = ap.parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = run_root / "real_pool_rc_uot"
    out_dir.mkdir(parents=True, exist_ok=True)

    eth_path = run_root / "labels" / "flow_segments_eth.csv"
    bnb_path = run_root / "labels" / "flow_segments_bnb.csv"
    labels_path = run_root / "labels" / "flow_labels.csv"
    eth_flows = flows_from_segment_export_csv(eth_path, chain="ETH")
    bnb_flows = flows_from_segment_export_csv(bnb_path, chain="BNB")
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)

    stats_path = out_dir / "real_pool_candidate_stats.json"
    if stats_path.is_file() and not args.force:
        stats_blob = json.loads(stats_path.read_text(encoding="utf-8"))
        stats = stats_blob.get("configs") or {}
    else:
        audits = [_candidate_audit(eth_flows, bnb_flows, labels, c) for c in CANDIDATE_CONFIGS]
        all_pairs = []
        for a in audits:
            all_pairs.extend(a.pop("pair_rows", []))
        stats = {a["config"]: {k: v for k, v in a.items() if k != "pair_rows"} for a in audits}
        stats_path.write_text(
            json.dumps({"configs": stats, "n_eth": len(eth_flows), "n_bnb": len(bnb_flows)}, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(all_pairs).to_csv(out_dir / "real_pool_candidate_pairs.csv", index=False)

    if args.skip_uot:
        print(json.dumps({"ok": True, "skipped_uot": True}, indent=2))
        return

    primary = CANDIDATE_CONFIGS[0]
    pairs_csv = out_dir / "real_pool_candidate_pairs.csv"
    allowed = _allowed_candidate_pairs(pairs_csv, primary["name"])
    meta_path = out_dir / "real_pool_component_meta.json"
    component_meta: list[dict[str, Any]] = []
    if meta_path.is_file():
        component_meta = json.loads(meta_path.read_text(encoding="utf-8")).get("components") or []

    if args.eval_only:
        cfg_yaml = load_and_validate_config(CROSS_ROOT / "config" / "defaults.json", CROSS_ROOT / "config" / "local.json")
        parser = build_parser()
        cli_args, _ = parser.parse_known_args(["--out", str(run_root)])
        uk = uot_kwargs_from_config(cli_args, cfg_yaml)
        stats_blob = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        n_eth = int(stats_blob.get("n_eth") or len(eth_flows))
        n_bnb = int(stats_blob.get("n_bnb") or len(bnb_flows))
        metrics = _finalize_real_pool_audit_exports(
            out_dir=out_dir,
            run_root=run_root,
            component_meta=component_meta,
            allowed=allowed,
            primary=primary,
            stats=stats,
            decode_threshold=float(uk["uot_decode_threshold"]),
            n_eth=n_eth,
            n_bnb=n_bnb,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "eval_only": True,
                    "flow_pair_f1": metrics.get("flow_pair_f1"),
                    "full_plan_rows": metrics.get("full_unfiltered_transport_rows"),
                    "filtered_plan_rows": metrics.get("exported_transport_rows"),
                },
                indent=2,
            )
        )
        return

    cfg_yaml = load_and_validate_config(CROSS_ROOT / "config" / "defaults.json", CROSS_ROOT / "config" / "local.json")
    parser = build_parser()
    cli_args, _ = parser.parse_known_args(["--out", str(run_root)])
    uk = uot_kwargs_from_config(cli_args, cfg_yaml)
    uot_run = out_dir / "uot_run"
    uot_run.mkdir(parents=True, exist_ok=True)
    import logging

    logging.getLogger("cross.application.standalone_flow_uot").setLevel(logging.INFO)
    eth_df = pd.read_csv(eth_path, dtype=str, keep_default_na=False)
    bnb_df = pd.read_csv(bnb_path, dtype=str, keep_default_na=False)
    component_meta: list[dict[str, Any]] = []
    plan_parts: list[pd.DataFrame] = []
    um_parts: list[pd.DataFrame] = []
    pool_kw = _uot_pool_kwargs_from_uk(uk)
    for ag, eth_g in eth_df.groupby("asset_group", sort=False):
        bnb_g = bnb_df[bnb_df["asset_group"].astype(str) == str(ag)]
        if eth_g.empty or bnb_g.empty:
            continue
        safe_ag = str(ag).replace(":", "_").replace("/", "_")[:80]
        comp_dir = uot_run / f"component_{safe_ag}"
        comp_dir.mkdir(parents=True, exist_ok=True)
        p_exist = output_file(comp_dir, "uot_transport_plan.csv")
        if p_exist.is_file() and not args.force:
            plan_parts.append(pd.read_csv(p_exist, dtype=str, keep_default_na=False))
            u_exist = output_file(comp_dir, "uot_unmatched_mass.csv")
            if u_exist.is_file():
                um_parts.append(pd.read_csv(u_exist, dtype=str, keep_default_na=False))
            component_meta.append(
                {"asset_group": str(ag), "n_eth": int(len(eth_g)), "n_bnb": int(len(bnb_g)), "dir": str(comp_dir), "skipped": True}
            )
            continue
        eth_tmp, bnb_tmp = comp_dir / "flow_segments_eth.csv", comp_dir / "flow_segments_bnb.csv"
        eth_g.to_csv(eth_tmp, index=False)
        bnb_g.to_csv(bnb_tmp, index=False)
        top_k = int(primary["top_k"])
        if len(eth_g) > 800:
            top_k = 20
        elif len(eth_g) > 300:
            top_k = 30
        max_cells = min(int(uk.get("uot_flow_max_matrix_cells") or 6_000_000), max(50_000, len(eth_g) * top_k * 4))
        run_standalone_flow_uot(
            comp_dir,
            src_flows_csv=eth_tmp,
            dst_flows_csv=bnb_tmp,
            flow_labels_csv=None,
            uot_reg=float(uk["uot_reg"]),
            uot_reg_m=float(uk["uot_reg_m"]),
            uot_decode_threshold=float(uk["uot_decode_threshold"]),
            uot_cost_weights=uk.get("uot_cost_weights"),
            uot_backend=str(uk["uot_backend"]),
            uot_max_delay_sec=float(primary["delay_sec"]),
            uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
            uot_lambda_risk=float(uk["uot_lambda_risk"]),
            uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
            uot_export_matrix=False,
            uot_export_cost_components=False,
            uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
            uot_use_graph_embedding=False,
            graph_ranker_checkpoint=None,
            uot_ablation="none",
            flow_label_min_confidence=0.0,
            uot_flow_dst_top_k=top_k,
            uot_flow_max_matrix_cells=max_cells,
            **pool_kw,
        )
        p_path = output_file(comp_dir, "uot_transport_plan.csv")
        u_path = output_file(comp_dir, "uot_unmatched_mass.csv")
        if p_path.is_file():
            plan_parts.append(pd.read_csv(p_path, dtype=str, keep_default_na=False))
        if u_path.is_file():
            um_parts.append(pd.read_csv(u_path, dtype=str, keep_default_na=False))
        component_meta.append(
            {"asset_group": str(ag), "n_eth": int(len(eth_g)), "n_bnb": int(len(bnb_g)), "dir": str(comp_dir)}
        )
    (out_dir / "real_pool_component_meta.json").write_text(
        json.dumps({"components": component_meta, "n_components": len(component_meta)}, indent=2),
        encoding="utf-8",
    )
    uot_run = out_dir / "uot_run"
    um_merged = pd.concat(um_parts, ignore_index=True) if um_parts else pd.DataFrame()
    um_src = output_file(uot_run, "uot_unmatched_mass.csv")
    um_merged.to_csv(um_src, index=False)
    stats_blob = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    metrics = _finalize_real_pool_audit_exports(
        out_dir=out_dir,
        run_root=run_root,
        component_meta=component_meta,
        allowed=allowed,
        primary=primary,
        stats=stats,
        decode_threshold=float(uk["uot_decode_threshold"]),
        n_eth=int(stats_blob.get("n_eth") or len(eth_flows)),
        n_bnb=int(stats_blob.get("n_bnb") or len(bnb_flows)),
    )
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "flow_pair_f1": metrics.get("flow_pair_f1")}, indent=2))


def _write_table_c(
    run_root: Path,
    out_dir: Path,
    stats: dict[str, Any],
    primary: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    table_a = run_root / "baselines" / "table_a_external_tx_baselines_celer_real.csv"
    rows = []
    if table_a.is_file():
        ta = pd.read_csv(table_a)
        for _, r in ta.iterrows():
            rows.append(
                {
                    "method": r["method"],
                    "evaluation_scope": "real_celer_flow_level",
                    "directly_comparable": "true",
                    "baseline_type": r.get("baseline_type", "adapted_transaction_level"),
                    "tx_hit_at_1": r.get("tx_hit_at_1"),
                    "flow_pair_f1": r.get("flow_pair_f1"),
                    "flow_mass_recall": r.get("flow_mass_recall"),
                    "split_recovery": r.get("split_recovery"),
                    "merge_recovery": r.get("merge_recovery"),
                    "flow_recall_at_1": "N/A",
                    "notes": "Phase 7.5 tx-level adapted baseline",
                }
            )
    rows.append(
        {
            "method": "rc_uot_real_pool",
            "evaluation_scope": "real_celer_flow_level",
            "directly_comparable": "true",
            "baseline_type": "native_flow_level_candidate_pruned",
            "tx_hit_at_1": "N/A",
            "flow_pair_f1": metrics.get("flow_pair_f1"),
            "flow_mass_recall": metrics.get("flow_mass_recall"),
            "split_recovery": metrics.get("split_recovery_rate"),
            "merge_recovery": metrics.get("merge_recovery_rate"),
            "flow_recall_at_1": metrics.get("flow_recall_at_1"),
            "flow_recall_at_3": metrics.get("flow_recall_at_3"),
            "flow_recall_at_5": metrics.get("flow_recall_at_5"),
            "flow_mrr": metrics.get("flow_mrr"),
            "top3_flow_accuracy": metrics.get("top3_flow_correspondence_accuracy"),
            "candidate_recall_at_50": stats.get(primary["name"], {}).get("candidate_recall_at_50"),
            "notes": "flow-level RC-UOT; heuristic candidate allowlist + top_k=50; tx metrics N/A",
        }
    )
    pd.DataFrame(rows).to_csv(out_dir / "table_c_real_celer_same_scope_comparison.csv", index=False)


if __name__ == "__main__":
    main()
