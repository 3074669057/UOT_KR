"""Hou-rule-backed AML risk scoring for ETH source transactions."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import pandas as pd

from cross.shared.normalize import norm_addr
from cross.shared.transfers import parse_transfer_value


def _rules_config_path() -> Path:
    root = Path(__file__).resolve().parents[4]
    config_level = root / "config" / "rules_config.json"
    if config_level.is_file():
        return config_level
    root_level = root / "rules_config.json"
    if root_level.is_file():
        return root_level
    return config_level


def _load_hou_rules() -> list[dict]:
    p = _rules_config_path()
    if not p.is_file():
        raise FileNotFoundError(f"Hou AML rules config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("rules_config.json must be a JSON array")
    return [x for x in data if isinstance(x, dict)]


def risk_level(score: float, *, medium_threshold: float, high_threshold: float) -> str:
    if score >= float(high_threshold):
        return "high"
    if score >= float(medium_threshold):
        return "medium"
    return "low"


def _build_node_features_by_address(eth_df: pd.DataFrame) -> dict[str, dict]:
    work = eth_df.copy()
    work["from_n"] = work.get("from", "").map(norm_addr)
    work["to_n"] = work.get("to", "").map(norm_addr)
    work["ts_f"] = pd.to_numeric(work.get("timeStamp", 0), errors="coerce").fillna(0.0).astype(float)
    work["val_f"] = work.get("value", 0).map(parse_transfer_value).astype(float)
    out: dict[str, dict] = {}
    by_addr = defaultdict(list)
    for _, r in work.iterrows():
        frm = norm_addr(r.get("from_n", ""))
        to = norm_addr(r.get("to_n", ""))
        ts = float(r.get("ts_f", 0.0) or 0.0)
        val = float(r.get("val_f", 0.0) or 0.0)
        if frm:
            by_addr[frm].append({"type": "out", "to": to, "from": frm, "amount": val, "timestamp": ts})
        if to:
            by_addr[to].append({"type": "in", "to": to, "from": frm, "amount": val, "timestamp": ts})

    for addr, txs in by_addr.items():
        in_txs = [t for t in txs if t["type"] == "in"]
        out_txs = [t for t in txs if t["type"] == "out"]
        in_amt = [t["amount"] for t in in_txs]
        out_amt = [t["amount"] for t in out_txs]
        timestamps = sorted(float(t["timestamp"]) for t in txs if float(t["timestamp"]) > 0)
        time_diff_variation = 0.0
        tx_frequency = 0.0
        if len(timestamps) > 1:
            diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            avg_diff = sum(diffs) / len(diffs)
            std_diff = pd.Series(diffs).std(ddof=0) if len(diffs) > 1 else 0.0
            time_diff_variation = float(std_diff / (avg_diff + 1))
            span = timestamps[-1] - timestamps[0]
            if span > 0:
                tx_frequency = float(len(timestamps) / (span / 86400.0))
        out_degree = len({t.get("to", "") for t in out_txs if t.get("to")})
        in_degree = len({t.get("from", "") for t in in_txs if t.get("from")})
        total_out = float(sum(out_amt))
        total_in = float(sum(in_amt))
        avg_out = float(sum(out_amt) / len(out_amt)) if out_amt else 0.0
        avg_in = float(sum(in_amt) / len(in_amt)) if in_amt else 0.0
        std_out = float(pd.Series(out_amt).std(ddof=0)) if len(out_amt) > 1 else 0.0
        std_in = float(pd.Series(in_amt).std(ddof=0)) if len(in_amt) > 1 else 0.0
        round_amounts = sum(1 for x in out_amt if x % 1_000_000_000_000_000_000 == 0) if out_amt else 0
        out_ratio = float(round_amounts / len(out_amt)) if out_amt else 0.0
        out[addr] = {
            "transactions": txs,
            "in_degree": int(in_degree),
            "out_degree": int(out_degree),
            "degree_ratio": float(out_degree / in_degree) if in_degree > 0 else (1000.0 if out_degree > 0 else 0.0),
            "total_in": total_in,
            "total_out": total_out,
            "balance": total_in - total_out,
            "tx_count": int(len(txs)),
            "in_tx_count": int(len(in_txs)),
            "out_tx_count": int(len(out_txs)),
            "avg_in_amount": avg_in,
            "avg_out_amount": avg_out,
            "std_in_amount": std_in,
            "std_out_amount": std_out,
            "max_in_amount": float(max(in_amt)) if in_amt else 0.0,
            "max_out_amount": float(max(out_amt)) if out_amt else 0.0,
            "min_in_amount": float(min(in_amt)) if in_amt else 0.0,
            "min_out_amount": float(min(out_amt)) if out_amt else 0.0,
            "in_amount_variation": float(std_in / (avg_in + 1.0)) if in_amt else 0.0,
            "out_amount_variation": float(std_out / (avg_out + 1.0)) if out_amt else 0.0,
            "tx_frequency": tx_frequency,
            "time_diff_variation": time_diff_variation,
            "round_amount_ratio": out_ratio,
            "clustering": 0.0,
            "pagerank": 0.0,
            "is_same": 1 if any(t.get("from") == t.get("to") and t.get("from") for t in txs) else 0,
        }
    return out


def _eval_hou_rule(condition: str, node_features: dict, threshold: float, params: dict) -> bool:
    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.threshold = float(threshold)
    try:
        return bool(eval(condition, {"__builtins__": {}}, {"node_features": node_features, "self": ctx, "params": params}))
    except Exception:
        return False


def _annotate_src_aml_rules(
    src_all: pd.DataFrame,
    eth_df: pd.DataFrame,
    *,
    medium_threshold: float = 40.0,
    high_threshold: float = 70.0,
) -> tuple[pd.DataFrame, dict]:
    hou_rules = _load_hou_rules()
    threshold_proxy = float(high_threshold) / 100.0
    node_features = _build_node_features_by_address(eth_df)
    risk_scores: list[float] = []
    risk_levels: list[str] = []
    rule_hits_col: list[str] = []
    hit_counter = Counter()
    coverage_counter = Counter()
    total_weight = float(sum(float(r.get("weight", 0.0) or 0.0) for r in hou_rules))
    for _, row in src_all.iterrows():
        addr = norm_addr(row.get("args.receiver", ""))
        feat = node_features.get(addr, {})
        matched = []
        matched_weight = 0.0
        for idx, rule in enumerate(hou_rules):
            name = str(rule.get("name") or f"rule_{idx+1}")
            cond = str(rule.get("condition") or "")
            params = rule.get("params") if isinstance(rule.get("params"), dict) else {}
            hit = _eval_hou_rule(cond, feat, threshold_proxy, params)
            if hit:
                matched.append(name)
                matched_weight += float(rule.get("weight", 0.0) or 0.0)
                hit_counter[name] += 1
            coverage_counter[name] += 1
        score_0_1 = (matched_weight / total_weight) if total_weight > 0 else 0.0
        score = float(max(0.0, min(100.0, score_0_1 * 100.0)))
        risk_scores.append(score)
        risk_levels.append(risk_level(score, medium_threshold=medium_threshold, high_threshold=high_threshold))
        rule_hits_col.append(";".join(matched))
    out = src_all.copy()
    out["aml_risk_score"] = risk_scores
    out["aml_risk_level"] = risk_levels
    out["aml_rule_hits"] = rule_hits_col
    hit_rows = []
    for i, rule in enumerate(hou_rules):
        name = str(rule.get("name") or f"rule_{i+1}")
        cov = int(coverage_counter.get(name, 0))
        hits = int(hit_counter.get(name, 0))
        hit_rows.append(
            {
                "rule": name,
                "weight": float(rule.get("weight", 0.0) or 0.0),
                "hits": hits,
                "coverage": cov,
                "hit_rate": float(hits / cov) if cov > 0 else 0.0,
            }
        )
    return out, {"rule_hits": hit_rows}


def score_src_all_with_rules(
    src_all: pd.DataFrame,
    eth_df: pd.DataFrame,
    *,
    bridge_addr: str,
    medium_threshold: float = 40.0,
    high_threshold: float = 70.0,
) -> pd.DataFrame:
    if src_all.empty:
        return src_all.copy()
    out, _meta = _annotate_src_aml_rules(
        src_all,
        eth_df,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    return out


def apply_rules_to_src(
    src_all: pd.DataFrame,
    eth_df: pd.DataFrame,
    *,
    bridge_addr: str,
    medium_threshold: float = 40.0,
    high_threshold: float = 70.0,
    keep_levels: tuple[str, ...] = ("medium", "high"),
    score_floor: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    levels = {str(x).strip().lower() for x in keep_levels if str(x).strip()} or {"medium", "high"}
    if src_all.empty:
        return src_all, {
            "aml_filter_applied": True,
            "aml_mode": "rules",
            "aml_engine": "hou",
            "aml_kept_levels": sorted(levels),
            "aml_medium_threshold": float(medium_threshold),
            "aml_high_threshold": float(high_threshold),
            "aml_score_floor": float(score_floor) if score_floor is not None else None,
            "aml_kept_rows": 0,
            "aml_dropped_rows": 0,
            "aml_note": "src_all was empty before AML rules",
            "aml_rule_hits_table": [],
        }
    out, ext = _annotate_src_aml_rules(
        src_all,
        eth_df,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    mask_level = out["aml_risk_level"].isin(levels)
    mask = mask_level.copy()
    if score_floor is not None:
        sf = float(score_floor)
        scores = pd.to_numeric(out["aml_risk_score"], errors="coerce").fillna(0.0)
        mask = mask_level | (scores >= sf)
    kept = out.loc[mask].reset_index(drop=True)
    level_counts = out["aml_risk_level"].value_counts().to_dict()
    hit_counter = Counter()
    for h in out["aml_rule_hits"].astype(str).tolist():
        for item in h.split(";"):
            if item:
                hit_counter[item] += 1
    top_hits = [{"rule": k, "count": int(v)} for k, v in hit_counter.most_common(20)]
    meta_extra: dict = {}
    if score_floor is not None:
        scores = pd.to_numeric(out["aml_risk_score"], errors="coerce").fillna(0.0)
        meta_extra["aml_score_floor"] = float(score_floor)
        meta_extra["aml_kept_by_level_only"] = int(mask_level.sum())
        meta_extra["aml_kept_by_score_floor"] = int((mask & ~mask_level).sum())

    return kept, {
        "aml_filter_applied": True,
        "aml_mode": "rules",
        "aml_engine": "hou",
        "aml_kept_levels": sorted(levels),
        "aml_medium_threshold": float(medium_threshold),
        "aml_high_threshold": float(high_threshold),
        "aml_kept_rows": int(mask.sum()),
        "aml_dropped_rows": int((~mask).sum()),
        "aml_risk_level_counts": {k: int(v) for k, v in level_counts.items()},
        "aml_top_rule_hits": top_hits,
        "aml_rule_hits_table": list(ext.get("rule_hits", [])),
        "aml_rules_config_path": str(_rules_config_path()),
        **meta_extra,
    }


__all__ = ["risk_level", "score_src_all_with_rules", "apply_rules_to_src"]
