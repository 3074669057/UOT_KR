"""Enrich flow segments with amounts, AML scores, and optional graph embeddings."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cross.shared.connector_decimals import decimals_for_eth_bnb
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import first_non_null, safe_float, safe_int

ZERO = "0x0000000000000000000000000000000000000000"


def _human_eth_amount(raw: float, ca: str, dec_e: dict[str, int]) -> float:
    if raw <= 0:
        return 0.0
    ca = (ca or "").strip().lower()
    if not ca or ca == ZERO:
        return raw / (10**18)
    d = dec_e.get(ca)
    if d is None:
        return 0.0
    return raw / (10**d)


def _human_bnb_amount(raw: float, ca: str, dec_b: dict[str, int]) -> float:
    if raw <= 0:
        return 0.0
    ca = (ca or "").strip().lower()
    if not ca or ca == ZERO:
        return raw / (10**18)
    d = dec_b.get(ca)
    if d is None:
        return 0.0
    return raw / (10**d)


def enrich_flow_segments(
    flows: list[dict[str, Any]],
    *,
    chain: str,
    aml_score_by_tx: dict[str, float] | None = None,
    graph_embedding_by_tx: dict[str, list[float]] | None = None,
    embedding_dim: int = 16,
) -> list[dict[str, Any]]:
    """Populate ``amount_usd`` (proxy: human token sum), ``aml_score``, ``graph_embedding``, ``risk_features``."""
    if not flows:
        return []

    want_eth: set[str] = set()
    want_bnb: set[str] = set()
    for f in flows:
        for pt in f.get("_per_tx") or []:
            ca = str(pt.get("args.asset_s") or pt.get("contractAddress") or "").strip().lower()
            if (chain or "").upper() == "ETH":
                if ca and ca != ZERO:
                    want_eth.add(ca)
            else:
                if ca and ca != ZERO:
                    want_bnb.add(ca)

    dec_e, dec_b = decimals_for_eth_bnb(want_eth, want_bnb)
    aml_score_by_tx = aml_score_by_tx or {}

    for f in flows:
        per = f.get("_per_tx") or []
        human_sum = 0.0
        aml_vals: list[float] = []
        embs: list[np.ndarray] = []
        ts_list: list[float] = []
        uniq_to: set[str] = set()
        ev_levels: list[int] = []

        raw_sum = 0.0
        ev_strs: list[str] = []
        for pt in per:
            txh = norm_addr(str(pt.get("txhash", "")))
            ts_list.append(safe_float(first_non_null(pt.get("timestamp"), pt.get("timeStamp")), 0.0))
            if (chain or "").upper() == "ETH":
                raw = safe_float(pt.get("args.amount"), 0.0)
                raw_sum += raw
                ca = str(pt.get("args.asset_s") or "").strip().lower()
                human_sum += _human_eth_amount(raw, ca, dec_e)
            else:
                raw = safe_float(pt.get("value"), 0.0)
                raw_sum += raw
                ca = str(pt.get("contractAddress") or "").strip().lower()
                human_sum += _human_bnb_amount(raw, ca, dec_b)
                if pt.get("evidence_level") is not None:
                    ev_levels.append(safe_int(pt.get("evidence_level"), 0))
                es = str(pt.get("evidence_level_str") or "").strip()
                if es:
                    ev_strs.append(es)
            if txh in aml_score_by_tx:
                aml_vals.append(float(aml_score_by_tx[txh]))
            if graph_embedding_by_tx and txh in graph_embedding_by_tx:
                vec = np.asarray(graph_embedding_by_tx[txh], dtype=float)
                if vec.size:
                    embs.append(vec)
            to_a = norm_addr(str(pt.get("to", "")))
            if to_a:
                uniq_to.add(to_a)

        f["amount_token"] = float(human_sum)
        f["amount_usd"] = float(human_sum)
        f["raw_amount_sum"] = float(raw_sum)
        f["human_amount_sum"] = float(human_sum)
        f["usd_amount_sum"] = float(human_sum)
        if ev_levels:
            f["evidence_level"] = max(ev_levels)
        if (chain or "").upper() == "BNB" and ev_strs:
            joined = ",".join(sorted(set(ev_strs)))
            if str(f.get("evidence_levels") or "").strip():
                f["evidence_levels"] = str(f.get("evidence_levels")) + "," + joined
            else:
                f["evidence_levels"] = joined
        if aml_vals:
            f["aml_score"] = float(sum(aml_vals) / len(aml_vals)) / 100.0
            f["aml_risk_score"] = float(max(aml_vals))
            f["aml_risk_score_raw"] = float(max(aml_vals))
        else:
            f["aml_score"] = 0.0
            if (chain or "").upper() == "ETH":
                f["aml_risk_score_raw"] = float(f.get("aml_risk_score_raw") or 0.0)

        if embs:
            stacked = np.stack(embs, axis=0)
            pooled = np.mean(stacked, axis=0)
            f["graph_embedding"] = pooled.tolist()
        else:
            f["graph_embedding"] = [0.0] * int(embedding_dim)

        if (chain or "").upper() == "BNB":
            lev_blob = str(f.get("evidence_levels", "")).lower()
            parts = [p.strip() for p in lev_blob.split(",") if p.strip()]
            if any("native_transfer" in p for p in parts):
                f["evidence_quality_score"] = 0.62
            elif any("token_transfer" in p for p in parts):
                f["evidence_quality_score"] = 0.94
            elif any("bridge" in p for p in parts):
                f["evidence_quality_score"] = 0.90
            else:
                f["evidence_quality_score"] = float(f.get("evidence_quality_score") or 0.78)

        tss = [t for t in ts_list if t > 0]
        span = (max(tss) - min(tss)) if len(tss) > 1 else 0.0
        burst = 1.0 - min(span / 3600.0, 1.0) if span >= 0 else 0.0
        f["risk_features"] = {
            "fanout": int(len(uniq_to)) if (chain or "").upper() == "BNB" else safe_int(f.get("tx_count"), 1),
            "fanin": safe_int(f.get("tx_count"), 1),
            "burst_score": float(burst),
            "new_address_ratio": 0.0,
        }

    for f in flows:
        f.pop("_per_tx", None)

    return flows


def aml_scores_from_src_all(src_all: pd.DataFrame) -> dict[str, float]:
    """Map src txhash -> aml_risk_score (0-100 scale as in CSV)."""
    out: dict[str, float] = {}
    if src_all is None or src_all.empty or "txhash" not in src_all.columns:
        return out
    col = "aml_risk_score" if "aml_risk_score" in src_all.columns else None
    if not col:
        return out
    for _, r in src_all.iterrows():
        h = norm_addr(r.get("txhash", ""))
        if not h:
            continue
        out[h] = safe_float(r.get(col), 0.0)
    return out
