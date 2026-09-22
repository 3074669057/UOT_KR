"""Score-based ETH↔BNB tx anchor candidates vs accepted labels (no row-index alignment)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import output_file
from cross.shared.amount_normalizer import raw_to_human
from cross.shared.connector_decimals import decimals_for_eth_bnb
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_int

ZERO = "0x0000000000000000000000000000000000000000"


def _pick_numeric_int(x: Any) -> int | None:
    if x is None or x == "" or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return int(pd.to_numeric(x, errors="coerce"))
    except Exception:
        return None


def _best_evidence_row_per_tx(ev: pd.DataFrame, *, chain: str) -> pd.DataFrame:
    """Prefer high-quality, positive-amount rows per ``tx_hash``."""
    if ev.empty or "tx_hash" not in ev.columns:
        return pd.DataFrame()
    df = ev.copy()
    df["tx_hash"] = df["tx_hash"].astype(str).map(norm_addr)
    df = df[df["tx_hash"].astype(bool)]

    def _rank_row(r: pd.Series) -> int:
        et = str(r.get("event_type") or "")
        q = str(r.get("evidence_quality") or "").lower()
        raw_i = _pick_numeric_int(r.get("raw_value_int"))
        amt_ok = bool(raw_i and raw_i > 0)
        if chain.upper() == "BNB":
            if et == "token_transfer_log" and amt_ok:
                return 4
            if et == "native_auxiliary_zero":
                return 0
        if chain.upper() == "ETH":
            if et == "erc20_deposit_or_transfer" and amt_ok:
                return 4
            if et == "native_deposit_or_call" and amt_ok:
                return 3
        if amt_ok:
            return 2
        return 1

    df["_prio"] = df.apply(_rank_row, axis=1)
    df["_q"] = df["evidence_quality"].map({"high": 3, "medium": 2, "low": 1}).fillna(0)
    df["_raw"] = df["raw_value_int"].map(lambda x: _pick_numeric_int(x) or 0)
    df = df.sort_values(["tx_hash", "_prio", "_q", "_raw"], ascending=[True, False, False, False])
    return df.drop_duplicates("tx_hash", keep="first")


def _bnb_tx_receiver_aux(bnb_e: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per BNB tx: token-transfer receiver, native-aux to_address, and flags for scoring/debug."""
    out: dict[str, dict[str, Any]] = {}
    if bnb_e.empty or "tx_hash" not in bnb_e.columns:
        return out
    df = bnb_e.copy()
    df["tx_hash"] = df["tx_hash"].astype(str).map(norm_addr)
    df = df[df["tx_hash"].astype(bool)]
    df["_et"] = df["event_type"].astype(str)
    df["_raw"] = df["raw_value_int"].map(lambda x: _pick_numeric_int(x) or 0)
    df["_to"] = df["to_address"].astype(str).map(norm_addr)
    for txh, g in df.groupby("tx_hash"):
        token_g = g[g["_et"] == "token_transfer_log"]
        aux_g = g[g["_et"] == "native_auxiliary_zero"]
        token_to = ""
        if not token_g.empty:
            best = token_g.sort_values("_raw", ascending=False).iloc[0]
            token_to = str(best["_to"])
        native_to = ""
        if not aux_g.empty:
            native_to = str(aux_g.sort_values("_raw", ascending=False).iloc[0]["_to"])
        has_tt = bool((token_g["_raw"] > 0).any()) if not token_g.empty else False
        has_aux = not aux_g.empty
        out[str(txh)] = {
            "dst_token_transfer_to": token_to,
            "dst_native_to": native_to,
            "has_token_transfer": has_tt,
            "has_native_auxiliary_zero": has_aux,
        }
    return out


def _human_usd_for_row(
    r: pd.Series,
    *,
    chain: str,
    dec_e: dict[str, int],
    dec_b: dict[str, int],
) -> tuple[float, float]:
    raw_i = _pick_numeric_int(r.get("raw_value_int"))
    if raw_i is None or raw_i <= 0:
        return 0.0, 0.0
    ca = str(r.get("token_contract") or "").strip().lower()
    if chain.upper() == "ETH":
        d = dec_e.get(ca) if ca and ca != ZERO else 18
        if d is None:
            return 0.0, 0.0
        h = float(raw_to_human(raw_i, int(d)))
        return h, h
    d = dec_b.get(ca) if ca and ca != ZERO else 18
    if d is None:
        return 0.0, 0.0
    h = float(raw_to_human(raw_i, int(d)))
    return h, h


def build_tx_anchor_labels(
    evidence_eth_path: Path,
    evidence_bnb_path: Path,
    out_dir: Path,
    *,
    max_delay_sec: float = 86_400.0,
    topk_per_src: int = 24,
    accept_threshold: float = 0.70,
    score_gap_min: float = 0.05,
    no_causal_ablation: bool = False,
) -> tuple[Path, Path]:
    """Write candidates, accepted labels, low-confidence top1, diagnostics, receiver debug.

    Causal violations never become accepted labels (even under ``no_causal_ablation``); the flag is
    recorded for diagnostics only.

    Returns:
        ``(tx_anchor_labels.csv path, tx_anchor_diagnostics.json path)`` for pipeline compatibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    eth_e = pd.read_csv(evidence_eth_path, dtype=str, keep_default_na=False)
    bnb_e = pd.read_csv(evidence_bnb_path, dtype=str, keep_default_na=False)

    bnb_recv_map = _bnb_tx_receiver_aux(bnb_e)

    eth_tx = _best_evidence_row_per_tx(eth_e, chain="ETH")
    bnb_tx = _best_evidence_row_per_tx(bnb_e, chain="BNB")

    want_e = {norm_addr(str(x)) for x in eth_tx.get("token_contract", []) if str(x).strip() and norm_addr(str(x)) != ZERO}
    want_b = {norm_addr(str(x)) for x in bnb_tx.get("token_contract", []) if str(x).strip() and norm_addr(str(x)) != ZERO}
    dec_e, dec_b = decimals_for_eth_bnb(want_e, want_b)

    eth_ts = eth_tx.assign(_ts=eth_tx["time_stamp"].map(lambda x: safe_int(x, 0))).sort_values("_ts")
    bnb_ts = bnb_tx.assign(_ts=bnb_tx["time_stamp"].map(lambda x: safe_int(x, 0))).sort_values("_ts").reset_index(drop=True)
    bnb_ts_arr = bnb_ts["_ts"].to_numpy(dtype=np.int64, copy=False)
    eth_order = {norm_addr(str(h)): i for i, h in enumerate(eth_ts["tx_hash"].tolist())}
    bnb_order = {norm_addr(str(h)): i for i, h in enumerate(bnb_ts["tx_hash"].tolist())}

    cand_rows: list[dict[str, Any]] = []
    for _, er in eth_ts.iterrows():
        src_h = norm_addr(str(er.get("tx_hash") or ""))
        if not src_h:
            continue
        src_ts = safe_int(er.get("time_stamp"), 0)
        src_from = norm_addr(str(er.get("from_address") or ""))
        src_sym = str(er.get("token_symbol") or "")
        src_ca = norm_addr(str(er.get("token_contract") or ""))
        ag_e = str(er.get("asset_group") or "")
        rid_e = str(er.get("route_id") or "")
        hu_e, usd_e = _human_usd_for_row(er, chain="ETH", dec_e=dec_e, dec_b=dec_b)
        raw_e = _pick_numeric_int(er.get("raw_value_int")) or 0

        cand_scores: list[tuple[float, dict[str, Any]]] = []
        t0 = int(src_ts)
        t1 = int(src_ts + int(max_delay_sec))
        lo = int(np.searchsorted(bnb_ts_arr, t0, side="left"))
        hi = int(np.searchsorted(bnb_ts_arr, t1, side="right"))
        window = bnb_ts.iloc[lo:hi]
        for _, br in window.iterrows():
            dst_h = norm_addr(str(br.get("tx_hash") or ""))
            if not dst_h:
                continue
            dst_ts = safe_int(br.get("time_stamp"), 0)
            dst_to = norm_addr(str(br.get("to_address") or ""))
            dst_from = norm_addr(str(br.get("from_address") or ""))
            dst_sym = str(br.get("token_symbol") or "")
            dst_ca = norm_addr(str(br.get("token_contract") or ""))
            ag_b = str(br.get("asset_group") or "")
            rid_b = str(br.get("route_id") or "")
            hu_b, usd_b = _human_usd_for_row(br, chain="BNB", dec_e=dec_e, dec_b=dec_b)
            raw_b = _pick_numeric_int(br.get("raw_value_int")) or 0
            repr_et = str(br.get("event_type") or "")

            aux = bnb_recv_map.get(dst_h, {})
            token_recv = norm_addr(str(aux.get("dst_token_transfer_to") or ""))
            native_recv = norm_addr(str(aux.get("dst_native_to") or ""))
            recv_compare_to = token_recv if token_recv else dst_to

            flags: list[str] = []
            causal_ok = dst_ts >= src_ts
            if not causal_ok:
                flags.append("causal_violation")

            delay = float(dst_ts - src_ts)
            time_score = 0.0 if not causal_ok else max(0.0, 1.0 - min(delay / max(max_delay_sec, 1.0), 1.0))

            recv = 1.0 if src_from and recv_compare_to and src_from == recv_compare_to else 0.0
            if recv < 1.0 and src_from and dst_from and src_from == dst_from:
                recv = 0.35

            route_score = 0.0
            if ag_e and ag_b and ag_e == ag_b:
                route_score = 1.0
            elif src_sym and dst_sym and src_sym.upper() == dst_sym.upper():
                route_score = 0.85
            elif rid_e and rid_b and rid_e.split("_None_")[-1] == rid_b.split("_None_")[-1]:
                route_score = 0.65

            amt_score = 0.0
            if hu_e > 0 and hu_b > 0:
                err = abs(hu_e - hu_b) / max(hu_e, hu_b, 1e-12)
                amt_score = max(0.0, 1.0 - min(err, 1.0))
            elif usd_e > 0 and usd_b > 0:
                err = abs(usd_e - usd_b) / max(usd_e, usd_b, 1e-12)
                amt_score = max(0.0, 1.0 - min(err, 1.0))

            oe = float(eth_order.get(src_h, 0))
            ob = float(bnb_order.get(dst_h, 0))
            order_prior = 1.0 / (1.0 + 0.03 * abs(oe - ob))

            score = (
                0.30 * time_score
                + 0.30 * recv
                + 0.20 * route_score
                + 0.15 * amt_score
                + 0.05 * order_prior
            )
            cand_scores.append(
                (
                    score,
                    {
                        "dst_h": dst_h,
                        "dst_ts": dst_ts,
                        "delay": delay,
                        "dst_from": dst_from,
                        "dst_to": dst_to,
                        "recv_compare_to": recv_compare_to,
                        "dst_sym": dst_sym,
                        "dst_ca": dst_ca,
                        "ag_b": ag_b,
                        "rid_b": rid_b,
                        "hu_b": hu_b,
                        "usd_b": usd_b,
                        "raw_b": raw_b,
                        "flags": list(flags),
                        "time_score": time_score,
                        "recv": recv,
                        "route_score": route_score,
                        "amt_score": amt_score,
                        "order_prior": order_prior,
                        "causal_ok": causal_ok,
                        "repr_et": repr_et,
                        "token_recv": token_recv,
                        "native_recv": native_recv,
                    },
                )
            )

        cand_scores.sort(key=lambda x: -x[0])
        top = cand_scores[: max(1, int(topk_per_src))]
        for score, c in top:
            conf = max(0.0, min(1.0, float(score)))
            causal_violation = not bool(c["causal_ok"])
            hard_valid = bool(c["causal_ok"] and c["recv"] >= 0.35 and c["amt_score"] >= 0.25)
            inv = ""
            if causal_violation:
                inv = "causal_violation"
            elif conf < 0.25:
                inv = "low_anchor_confidence"
            receiver_match_flag = bool(src_from and c["recv_compare_to"] and src_from == c["recv_compare_to"])
            cand_rows.append(
                {
                    "src_tx_hash": src_h,
                    "dst_tx_hash": c["dst_h"],
                    "src_chain": "ETH",
                    "dst_chain": "BNB",
                    "src_time": src_ts,
                    "dst_time": c["dst_ts"],
                    "delay_sec": c["delay"],
                    "src_from": src_from,
                    "src_to": norm_addr(str(er.get("to_address") or "")),
                    "dst_from": c["dst_from"],
                    "dst_to": c["dst_to"],
                    "dst_token_transfer_to": c["token_recv"],
                    "dst_native_to": c["native_recv"],
                    "src_symbol": src_sym,
                    "dst_symbol": c["dst_sym"],
                    "src_token_contract": src_ca,
                    "dst_token_contract": c["dst_ca"],
                    "asset_group": ag_e if ag_e == c["ag_b"] else f"{ag_e}|{c['ag_b']}",
                    "route_id": rid_e,
                    "src_raw_amount": raw_e,
                    "dst_raw_amount": c["raw_b"],
                    "src_human_amount": hu_e,
                    "dst_human_amount": c["hu_b"],
                    "src_usd_amount": usd_e,
                    "dst_usd_amount": c["usd_b"],
                    "pair_label_confidence": conf,
                    "label_source": "celer_evidence_scored_candidates",
                    "match_evidence_flags": "|".join(c["flags"]) if c["flags"] else "",
                    "hard_valid": hard_valid,
                    "causal_violation": causal_violation,
                    "invalid_reason": inv,
                    "selected_bnb_representative_row_type": c["repr_et"],
                    "candidate_rank": 0,
                    "score_gap_vs_next": "",
                    "ambiguous_top1": "",
                    "accepted": "false",
                }
            )

    cand_df = pd.DataFrame(cand_rows)
    if cand_df.empty:
        cand_df = pd.DataFrame(
            columns=[
                "src_tx_hash",
                "dst_tx_hash",
                "pair_label_confidence",
                "hard_valid",
                "causal_violation",
                "candidate_rank",
                "score_gap_vs_next",
                "ambiguous_top1",
                "accepted",
            ]
        )

    # Per-src rank and score gap (among all candidates for that src, sorted by score then dst for stability)
    if not cand_df.empty:
        cand_df["_conf_n"] = pd.to_numeric(cand_df["pair_label_confidence"], errors="coerce").fillna(0.0)
        cand_df = cand_df.sort_values(["src_tx_hash", "_conf_n", "dst_tx_hash"], ascending=[True, False, True])
        cand_df["candidate_rank"] = cand_df.groupby("src_tx_hash", sort=False).cumcount() + 1
        cand_df["_next_conf"] = cand_df.groupby("src_tx_hash", sort=False)["_conf_n"].shift(-1)
        # Rank-1 vs next: no second candidate → large synthetic gap (non-ambiguous).
        cand_df["score_gap_vs_next"] = np.where(
            cand_df["candidate_rank"] == 1,
            np.where(
                cand_df["_next_conf"].notna(),
                (cand_df["_conf_n"] - cand_df["_next_conf"]).astype(float),
                1.0,
            ),
            np.nan,
        )
        cand_df["ambiguous_top1"] = np.where(
            (cand_df["candidate_rank"] == 1)
            & (cand_df["_next_conf"].notna())
            & (cand_df["score_gap_vs_next"] < float(score_gap_min)),
            "true",
            "false",
        )
        cand_df.drop(columns=["_conf_n", "_next_conf"], inplace=True, errors="ignore")

    p_cand = output_file(out_dir, "tx_anchor_candidates.csv")
    cand_df.to_csv(p_cand, index=False)

    labels_rows: list[dict[str, Any]] = []
    low_conf_rows: list[dict[str, Any]] = []
    top1_debug_rows: list[dict[str, Any]] = []

    if not cand_df.empty:
        top1 = cand_df[cand_df["candidate_rank"] == 1].copy()
        for _, r in top1.iterrows():
            conf = float(pd.to_numeric(r.get("pair_label_confidence"), errors="coerce") or 0.0)
            gap = r.get("score_gap_vs_next")
            gap_f = float(gap) if gap is not None and str(gap) != "" and not (isinstance(gap, float) and math.isnan(gap)) else 1.0
            ambiguous = str(r.get("ambiguous_top1", "false")).lower() in ("1", "true", "yes")
            causal_violation = str(r.get("causal_violation", "false")).lower() in ("1", "true", "yes")
            hard_valid = str(r.get("hard_valid", "false")).lower() in ("1", "true", "yes")

            accept = (not causal_violation) and hard_valid and conf >= float(accept_threshold) and not ambiguous

            row_dict = r.to_dict()
            row_dict["accepted"] = "true" if accept else "false"
            row_dict["reject_reason"] = ""
            if not accept:
                reasons: list[str] = []
                if causal_violation:
                    reasons.append("causal_violation")
                if ambiguous:
                    reasons.append("ambiguous_score_gap")
                if not hard_valid:
                    reasons.append("hard_valid_false")
                if conf < float(accept_threshold):
                    reasons.append("below_confidence_threshold")
                row_dict["reject_reason"] = "|".join(reasons)

            top1_debug_rows.append(
                {
                    "src_tx_hash": row_dict["src_tx_hash"],
                    "dst_tx_hash": row_dict["dst_tx_hash"],
                    "src_from": row_dict.get("src_from", ""),
                    "src_to": row_dict.get("src_to", ""),
                    "dst_from": row_dict.get("dst_from", ""),
                    "dst_to": row_dict.get("dst_to", ""),
                    "dst_token_transfer_to": row_dict.get("dst_token_transfer_to", ""),
                    "dst_native_to": row_dict.get("dst_native_to", ""),
                    "src_symbol": row_dict.get("src_symbol", ""),
                    "dst_symbol": row_dict.get("dst_symbol", ""),
                    "route_id": row_dict.get("route_id", ""),
                    "time_delay": row_dict.get("delay_sec", ""),
                    "pair_label_confidence": row_dict.get("pair_label_confidence", ""),
                    "receiver_match_flag": "true"
                    if norm_addr(str(row_dict.get("src_from") or ""))
                    == norm_addr(str(row_dict.get("dst_token_transfer_to") or row_dict.get("dst_to") or ""))
                    else "false",
                    "selected_bnb_representative_row_type": row_dict.get("selected_bnb_representative_row_type", ""),
                }
            )

            if accept:
                labels_rows.append(row_dict)
            else:
                low_conf_rows.append(row_dict)

    labels_df = pd.DataFrame(labels_rows)
    low_df = pd.DataFrame(low_conf_rows)
    _drop_label_noise = ("reject_reason", "candidate_rank", "score_gap_vs_next", "ambiguous_top1")
    if not labels_df.empty:
        labels_df = labels_df.drop(columns=[c for c in _drop_label_noise if c in labels_df.columns], errors="ignore")
    if not low_df.empty:
        pass

    p_labels = output_file(out_dir, "tx_anchor_labels.csv")
    p_low = output_file(out_dir, "tx_anchor_low_confidence.csv")
    if labels_df.empty:
        pd.DataFrame().to_csv(p_labels, index=False)
    else:
        labels_df["accepted"] = "true"
        labels_df.to_csv(p_labels, index=False)
    if low_df.empty:
        pd.DataFrame().to_csv(p_low, index=False)
    else:
        low_df.to_csv(p_low, index=False)

    p_recv_dbg = output_file(out_dir, "tx_anchor_receiver_mismatch_debug.csv")
    pd.DataFrame(top1_debug_rows).to_csv(p_recv_dbg, index=False)

    eth_u = set(eth_tx["tx_hash"].map(norm_addr))
    bnb_u = set(bnb_tx["tx_hash"].map(norm_addr))
    pair_df = cand_df
    hi_c = pair_df[pd.to_numeric(pair_df["pair_label_confidence"], errors="coerce") >= float(accept_threshold)] if not pair_df.empty else pair_df
    lo_c = pair_df[pd.to_numeric(pair_df["pair_label_confidence"], errors="coerce") < 0.35] if not pair_df.empty else pair_df
    matched_eth = set(pair_df["src_tx_hash"].map(norm_addr)) if not pair_df.empty else set()
    matched_bnb = set(pair_df["dst_tx_hash"].map(norm_addr)) if not pair_df.empty else set()
    delays = pair_df["delay_sec"].tolist() if not pair_df.empty and "delay_sec" in pair_df else []
    neg_delay = int(sum(1 for d in delays if float(d) < 0)) if delays else 0

    lab_df = labels_df
    amt_cov = (
        float(
            (
                (pd.to_numeric(lab_df["src_human_amount"], errors="coerce") > 0)
                & (pd.to_numeric(lab_df["dst_human_amount"], errors="coerce") > 0)
            ).mean()
        )
        if not lab_df.empty
        else 0.0
    )
    amt_cov_cand = (
        float(
            (
                (pd.to_numeric(pair_df["src_human_amount"], errors="coerce") > 0)
                & (pd.to_numeric(pair_df["dst_human_amount"], errors="coerce") > 0)
            ).mean()
        )
        if not pair_df.empty
        else 0.0
    )

    top1_df = cand_df[cand_df["candidate_rank"] == 1] if not cand_df.empty else cand_df
    num_ambiguous = int((top1_df["ambiguous_top1"].astype(str).str.lower().isin(("true", "1"))).sum()) if not top1_df.empty else 0

    if not lab_df.empty:
        sf = lab_df["src_from"].astype(str).map(norm_addr)
        tt = lab_df["dst_token_transfer_to"].astype(str).map(norm_addr)
        dt = lab_df["dst_to"].astype(str).map(norm_addr)
        recv_match = (sf == tt) | ((tt == "") & (sf == dt))
        recv_rate_labels = float(recv_match.mean())
    else:
        recv_rate_labels = 0.0

    diag = {
        "total_eth_unique_tx": len(eth_u),
        "total_bnb_unique_tx": len(bnb_u),
        "num_candidate_pairs": int(len(pair_df)),
        "num_accepted_anchor_pairs": int(len(lab_df)),
        "num_high_confidence_pairs": int(len(hi_c)),
        "num_low_confidence_pairs": int(len(lo_c)),
        "num_ambiguous_pairs": num_ambiguous,
        "num_low_confidence_top1_written": int(len(low_df)),
        "num_unmatched_eth_tx": int(len(eth_u - matched_eth)),
        "num_unmatched_bnb_tx": int(len(bnb_u - matched_bnb)),
        "receiver_match_rate": recv_rate_labels,
        "receiver_match_rate_candidates_all_pairs": float(
            (
                (pair_df["src_from"].astype(str).map(norm_addr) == pair_df["dst_token_transfer_to"].astype(str).map(norm_addr))
                | (
                    (pair_df["dst_token_transfer_to"].astype(str).map(norm_addr) == "")
                    & (pair_df["src_from"].astype(str).map(norm_addr) == pair_df["dst_to"].astype(str).map(norm_addr))
                )
            ).mean()
        )
        if not pair_df.empty
        else 0.0,
        "route_match_rate": float((lab_df["src_symbol"].str.upper() == lab_df["dst_symbol"].str.upper()).mean())
        if not lab_df.empty
        else 0.0,
        "median_delay_sec": float(np.median(delays)) if delays else 0.0,
        "p90_delay_sec": float(np.percentile(delays, 90)) if delays else 0.0,
        "negative_delay_count": neg_delay,
        "amount_normalization_coverage": amt_cov,
        "amount_normalization_coverage_candidates": amt_cov_cand,
        "accept_threshold": float(accept_threshold),
        "score_gap_min": float(score_gap_min),
        "no_causal_ablation": bool(no_causal_ablation),
    }
    diag_path = output_file(out_dir, "tx_anchor_diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    return p_labels, diag_path
