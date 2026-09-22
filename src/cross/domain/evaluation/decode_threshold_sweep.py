"""Sweep RC-UOT decode rules on a fixed transport matrix (no re-solve)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import locate_output_file, output_file

logger = logging.getLogger(__name__)


def _load_P_and_ids(out_root: Path) -> tuple[np.ndarray, list[str], list[str]] | None:
    out_root = Path(out_root)
    for npz in (out_root / "uot" / "uot_transport_matrix.npz", out_root / "matching_transport_matrix.npz"):
        if npz.is_file():
            z = np.load(npz, allow_pickle=True)
            if "P" in z.files:
                p = np.asarray(z["P"], dtype=float)
            elif "transport_matrix" in z.files:
                p = np.asarray(z["transport_matrix"], dtype=float)
            else:
                logger.warning("decode_threshold_sweep: %s missing P/transport_matrix", npz)
                return None
            si = [str(x) for x in np.asarray(z["source_flow_ids"], dtype=object).ravel()]
            tj = [str(x) for x in np.asarray(z["target_flow_ids"], dtype=object).ravel()]
            return p, si, tj
    return None


def _pred_df_from_P_mask(p: np.ndarray, si: list[str], tj: list[str], mask: np.ndarray) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    active = mask & (p > 1e-18)
    if not np.any(active):
        return pd.DataFrame(columns=["src_flow_id", "dst_flow_id", "transport_mass", "source_share"])
    ii, jj = np.nonzero(active)
    rs = p.sum(axis=1, keepdims=True) + 1e-18
    mass = p[ii, jj].astype(float, copy=False)
    shares = mass / rs[ii, 0]
    src_ids = np.array([si[i] if i < len(si) else str(i) for i in ii], dtype=object)
    dst_ids = np.array([tj[j] if j < len(tj) else str(j) for j in jj], dtype=object)
    return pd.DataFrame(
        {
            "src_flow_id": src_ids,
            "dst_flow_id": dst_ids,
            "transport_mass": mass,
            "source_share": shares,
        }
    )


def _decode_masks_share_threshold(p: np.ndarray, thr: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    rs = p.sum(axis=1, keepdims=True) + 1e-18
    share = p / rs
    return (p > 1e-18) & (share >= float(thr))


def _decode_masks_topk(p: np.ndarray, k: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = np.zeros_like(p, dtype=bool)
    kk = max(1, int(k))
    for i in range(p.shape[0]):
        row = p[i]
        idx = np.argsort(-row)
        for j in idx[:kk]:
            if row[j] > 1e-18:
                m[i, int(j)] = True
    return m


def _decode_masks_cumulative_mass(p: np.ndarray, frac: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = np.zeros_like(p, dtype=bool)
    f = float(frac)
    for i in range(p.shape[0]):
        row = p[i]
        s = float(row.sum())
        if s <= 1e-18:
            continue
        idx = np.argsort(-row)
        cum = 0.0
        for j in idx:
            v = float(row[int(j)])
            if v <= 1e-18:
                break
            m[i, int(j)] = True
            cum += v
            if cum / s >= f:
                break
    return m


def _metrics_from_pred(
    pred_df: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    min_label_confidence: float,
) -> dict[str, Any]:
    from cross.domain.evaluation.flow_eval import _metrics_for_pairs, _pair_set, _top1_pair_set

    labels = labels.copy()
    labels["_lc"] = pd.to_numeric(labels.get("label_confidence"), errors="coerce").fillna(0.0)
    truth = _pair_set(labels[labels["_lc"] >= float(min_label_confidence)])
    if pred_df.empty:
        pred_pairs = pd.DataFrame(columns=["src_flow_id", "dst_flow_id", "transport_mass", "source_share"])
    else:
        pred_pairs = pred_df.copy()
        if "transport_mass" not in pred_pairs.columns:
            pred_pairs["transport_mass"] = 0.0
        if "source_share" not in pred_pairs.columns:
            pred_pairs["source_share"] = 0.0
    pred_set = set(zip(pred_pairs["src_flow_id"].astype(str), pred_pairs["dst_flow_id"].astype(str)))
    soft = _metrics_for_pairs(truth, pred_set, pred_pairs)
    plan_for_top1 = pred_pairs.copy()
    plan_for_top1["transport_mass"] = pd.to_numeric(plan_for_top1["transport_mass"], errors="coerce").fillna(0.0)
    top1_set = _top1_pair_set(plan_for_top1, mass_thr=1e-18)
    hard = _metrics_for_pairs(truth, top1_set, pred_pairs)

    src_mass = (
        labels.groupby("src_flow_id")["src_amount_usd"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").max()))
        if not labels.empty and "src_amount_usd" in labels.columns
        else pd.Series(dtype=float)
    ).to_dict()
    row_sum_m = pred_pairs.groupby(pred_pairs["src_flow_id"].astype(str), sort=False)["transport_mass"].sum()
    matched_usd_on_tp = 0.0
    for s, d in truth & pred_set:
        sub = pred_pairs[(pred_pairs["src_flow_id"].astype(str) == s) & (pred_pairs["dst_flow_id"].astype(str) == d)]
        p_sd = float(pd.to_numeric(sub["transport_mass"], errors="coerce").sum()) if not sub.empty else 0.0
        rs = float(row_sum_m.get(s, 0.0))
        sm = float(src_mass.get(s, 0.0))
        frac = (p_sd / (rs + 1e-18)) if rs > 1e-18 else 0.0
        matched_usd_on_tp += frac * sm
    tot_src = sum(float(v) for v in src_mass.values()) or 1.0
    flow_mass_recall = float(matched_usd_on_tp / tot_src)
    pred_assigned_usd = 0.0
    for _, r in pred_pairs.iterrows():
        s = str(r.get("src_flow_id") or "")
        p_sd = float(pd.to_numeric(r.get("transport_mass"), errors="coerce") or 0.0)
        rs = float(row_sum_m.get(s, 0.0))
        sm = float(src_mass.get(s, 0.0))
        frac = (p_sd / (rs + 1e-18)) if rs > 1e-18 else 0.0
        pred_assigned_usd += frac * sm
    flow_mass_precision = float(matched_usd_on_tp / max(pred_assigned_usd, 1e-18))

    pred_src_n = int(pred_pairs["src_flow_id"].astype(str).nunique()) if not pred_pairs.empty else 0
    avg_edges = float(len(pred_pairs) / max(pred_src_n, 1))

    return {
        "num_predicted_edges": int(len(pred_set)),
        "num_true_edges": int(len(truth)),
        "num_true_positive_edges": int(soft["tp"]),
        "flow_pair_precision": float(soft["flow_pair_precision"]),
        "flow_pair_recall": float(soft["flow_pair_recall"]),
        "flow_pair_f1": float(soft["flow_pair_f1"]),
        "flow_mass_recall": float(flow_mass_recall),
        "flow_mass_precision": float(flow_mass_precision),
        "top1_flow_accuracy": float(soft["top1_flow_correspondence_accuracy"]),
        "top3_flow_accuracy": float(soft["top3_flow_correspondence_accuracy"]),
        "average_edges_per_source": float(avg_edges),
    }


def _combined_decode_mask(p: np.ndarray, *, share_thr: float, topk: int, cum_mass: float) -> np.ndarray:
    m_share = _decode_masks_share_threshold(p, float(share_thr))
    m_topk = _decode_masks_topk(p, int(topk))
    m_cum = _decode_masks_cumulative_mass(p, float(cum_mass))
    return m_share & m_topk & m_cum


RECOMMENDED_DECODE_SELECTION_DOC = (
    "Primary metric: maximize ``flow_pair_f1`` among rows with "
    "``decode_rule == combined_share_topk_cumulative``. "
    "Tie-breakers (descending): ``flow_mass_recall``, then ``flow_pair_recall``."
)


def recommend_decode_rule_from_sweep_dataframe(df: pd.DataFrame) -> dict[str, Any] | None:
    """Return the same payload as ``recommended_decode_rule.json`` (without writing)."""
    if df.empty or "decode_rule" not in df.columns:
        return None
    sub = df[df["decode_rule"].astype(str) == "combined_share_topk_cumulative"].copy()
    if sub.empty:
        return None
    for c in ("flow_pair_f1", "flow_mass_recall", "flow_pair_recall"):
        if c not in sub.columns:
            return None
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.sort_values(
        by=["flow_pair_f1", "flow_mass_recall", "flow_pair_recall"],
        ascending=[False, False, False],
        na_position="last",
    )
    best = sub.iloc[0]
    thr_k = str(best.get("threshold_or_k") or "")
    f1 = float(best["flow_pair_f1"])
    r_edge = float(best["flow_pair_recall"])
    r_mass = float(best["flow_mass_recall"])
    avg_e = float(best["average_edges_per_source"])
    return {
        "recommended_rule": "combined_share_topk_cumulative",
        "recommended_threshold_or_k": thr_k,
        "reason": (
            f"Highest flow_pair_f1={f1:.6g} over the full decode grid "
            "(intersection of source_share_ge, topk_per_source, cumulative_row_mass on dense P)."
        ),
        "flow_pair_f1": f1,
        "flow_pair_recall": r_edge,
        "flow_mass_recall": r_mass,
        "average_edges_per_source": avg_e,
        "source_share_ge": float(best["source_share_ge"]) if pd.notna(best.get("source_share_ge")) else None,
        "topk_per_source": int(best["topk_per_source"]) if pd.notna(best.get("topk_per_source")) else None,
        "cumulative_row_mass": float(best["cumulative_row_mass"]) if pd.notna(best.get("cumulative_row_mass")) else None,
    }


def _write_recommended_decode_rule_json(out_root: Path, df: pd.DataFrame) -> Path | None:
    """Pick best combined rule by ``flow_pair_f1`` (tie-break: flow_mass_recall, flow_pair_recall)."""
    obj = recommend_decode_rule_from_sweep_dataframe(df)
    if obj is None:
        return None
    outp = out_root / "experiments" / "recommended_decode_rule.json"
    outp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s", outp)
    return outp


def _write_large_budget_setting_json(out_root: Path, cfg: dict | None) -> Path | None:
    u = (cfg or {}).get("uot") or {}
    lbs = u.get("large_budget_setting")
    if not isinstance(lbs, dict) or not lbs:
        return None
    outp = out_root / "experiments" / "large_budget_setting.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    doc = {"large_budget_setting": lbs, "note": "Diagnostic higher matrix budget (not the primary default in config/defaults.json)."}
    outp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s", outp)
    return outp


def run_decode_threshold_sweep(
    out_root: Path,
    *,
    min_label_confidence: float = 0.0,
    flow_labels_path: Path | None = None,
    cfg: dict | None = None,
) -> Path:
    """Write ``experiments/decode_threshold_sweep.csv`` using dense ``P`` from ``uot/uot_transport_matrix.npz``."""
    out_root = Path(out_root)
    lab_p = Path(flow_labels_path) if flow_labels_path else locate_output_file(out_root, "flow_labels.csv")
    labels = pd.read_csv(lab_p, dtype=str, keep_default_na=False)

    loaded = _load_P_and_ids(out_root)
    outp = out_root / "experiments" / "decode_threshold_sweep.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)

    if loaded is None:
        msg = (
            "missing_dense_transport_matrix: need uot/uot_transport_matrix.npz (written on next flow UOT export) "
            "or matching_transport_matrix.npz at run root"
        )
        logger.warning("decode_threshold_sweep: %s", msg)
        pd.DataFrame([{"decode_rule": "error", "threshold_or_k": msg}]).to_csv(outp, index=False)
        return outp

    p, si, tj = loaded
    rows_out: list[dict[str, Any]] = []
    shares = (1e-9, 1e-6, 1e-4, 1e-3, 0.005, 0.01)
    topks = (1, 3, 5)
    cum_masses = (0.8, 0.9, 0.95)
    for share_thr in shares:
        for topk in topks:
            for cum_mass in cum_masses:
                mask = _combined_decode_mask(p, share_thr=float(share_thr), topk=int(topk), cum_mass=float(cum_mass))
                pred_df = _pred_df_from_P_mask(p, si, tj, mask)
                m = _metrics_from_pred(pred_df, labels, min_label_confidence=float(min_label_confidence))
                thr_or_k = f"share_ge={share_thr};topk={topk};cumulative_row_mass={cum_mass}"
                row = {
                    "source_share_ge": float(share_thr),
                    "topk_per_source": int(topk),
                    "cumulative_row_mass": float(cum_mass),
                    "decode_rule": "combined_share_topk_cumulative",
                    "threshold_or_k": thr_or_k,
                    **m,
                }
                rows_out.append(row)

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(outp, index=False)
    logger.info("Wrote %s (%d rows)", outp, len(rows_out))
    _write_recommended_decode_rule_json(out_root, df_out)
    _write_large_budget_setting_json(out_root, cfg)
    return outp
