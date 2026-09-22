"""Rolling-window flow segments from Celer evidence (tx-aggregated, then segmented)."""
from __future__ import annotations

import uuid
from collections import Counter
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


def _int_raw(x: Any) -> int:
    try:
        return int(pd.to_numeric(x, errors="coerce") or 0)
    except Exception:
        return 0


def _eq_quality(s: str) -> float:
    return {"high": 1.0, "medium": 0.65, "low": 0.35}.get(str(s).lower(), 0.4)


def _human_for_raw(raw_i: int, ca: str, *, chain: str, dec_e: dict[str, int], dec_b: dict[str, int]) -> float:
    if raw_i <= 0:
        return 0.0
    ca_n = norm_addr(str(ca or ""))
    if chain.upper() == "ETH":
        d = dec_e.get(ca_n) if ca_n and ca_n != ZERO else 18
        if d is None:
            d = 18
    else:
        d = dec_b.get(ca_n) if ca_n and ca_n != ZERO else 18
        if d is None:
            d = 18
    return float(raw_to_human(raw_i, int(d)))


def _evidence_to_tx_summaries(
    ev: pd.DataFrame,
    *,
    chain: str,
    dec_e: dict[str, int],
    dec_b: dict[str, int],
) -> list[dict[str, Any]]:
    """One summary row per ``tx_hash`` (BNB native-aux-only txs omitted)."""
    if ev.empty or "tx_hash" not in ev.columns:
        return []
    df = ev.copy()
    df["tx_hash"] = df["tx_hash"].astype(str).map(norm_addr)
    df = df[df["tx_hash"].astype(bool)]
    df["_et"] = df["event_type"].astype(str)
    df["_raw"] = df["raw_value_int"].map(_int_raw)
    df["_from"] = df["from_address"].astype(str).map(norm_addr)
    df["_to"] = df["to_address"].astype(str).map(norm_addr)
    df["_ca"] = df["token_contract"].astype(str).map(norm_addr)
    df["_sym"] = df["token_symbol"].astype(str)
    df["_ag"] = df["asset_group"].astype(str)
    df["_rid"] = df["route_id"].astype(str)
    df["_ts"] = df["time_stamp"].map(lambda x: safe_int(x, 0))
    df["_bn"] = df["block_number"].map(lambda x: safe_int(x, 0))
    df["_evq"] = df["evidence_quality"].astype(str)

    summaries: list[dict[str, Any]] = []
    ch = chain.upper()

    for txh, g in df.groupby("tx_hash"):
        ts_min = int(g["_ts"].min())
        bn_min = int(g["_bn"].min()) if "_bn" in g else 0
        etypes = sorted(set(g["_et"].tolist()))
        evq_mean = float(np.mean([_eq_quality(x) for x in g["_evq"].tolist()]))

        if ch == "BNB":
            has_aux = bool((g["_et"] == "native_auxiliary_zero").any())
            tok = g[(g["_et"] == "token_transfer_log") & (g["_raw"] > 0)]
            nat_nz = g[(g["_et"] == "bnb_native_transfer") & (g["_raw"] > 0)]
            has_tt = not tok.empty
            if not has_tt and nat_nz.empty:
                # native_auxiliary_zero only (or zero-value noise): no dst flow segment
                continue
            amount_raw_sum = 0
            if has_tt:
                amount_raw_sum = int(tok["_raw"].sum())
                lead = tok.sort_values("_raw", ascending=False).iloc[0]
            else:
                amount_raw_sum = int(nat_nz["_raw"].sum())
                lead = nat_nz.sort_values("_raw", ascending=False).iloc[0]
            ca = str(lead["_ca"])
            sym = str(lead["_sym"])
            ag = str(lead["_ag"])
            rid = str(lead["_rid"])
            receiver = str(lead["_to"])
            counterparty = str(lead["_from"])
            human = _human_for_raw(amount_raw_sum, ca, chain=ch, dec_e=dec_e, dec_b=dec_b)
            summaries.append(
                {
                    "tx_hash": str(txh),
                    "chain": ch,
                    "primary_address": receiver,
                    "counterparty_address": counterparty,
                    "token_contract": ca,
                    "token_symbol": sym,
                    "asset_group": ag,
                    "route_id": rid,
                    "time_stamp": ts_min,
                    "block_number": bn_min,
                    "amount_raw_sum": amount_raw_sum,
                    "amount_human_sum": human,
                    "amount_usd_sum": human,
                    "event_types": "|".join(etypes),
                    "evidence_quality": float(evq_mean),
                    "has_token_transfer": has_tt,
                    "has_native_auxiliary_zero": has_aux,
                }
            )
        else:
            # ETH: aggregate all positive-value rows; prefer ERC20 row as lead metadata.
            pos = g[g["_raw"] > 0]
            if pos.empty:
                continue
            erc = pos[pos["_et"] == "erc20_deposit_or_transfer"]
            nat = pos[pos["_et"] == "native_deposit_or_call"]
            if not erc.empty:
                lead = erc.sort_values("_raw", ascending=False).iloc[0]
                amount_raw_sum = int(pos["_raw"].sum())
            elif not nat.empty:
                lead = nat.sort_values("_raw", ascending=False).iloc[0]
                amount_raw_sum = int(pos["_raw"].sum())
            else:
                lead = pos.sort_values("_raw", ascending=False).iloc[0]
                amount_raw_sum = int(pos["_raw"].sum())
            ca = str(lead["_ca"])
            sym = str(lead["_sym"])
            ag = str(lead["_ag"])
            rid = str(lead["_rid"])
            src_user = str(lead["_from"])
            counterparty = str(lead["_to"])
            human = float(
                sum(
                    _human_for_raw(int(r["_raw"]), str(r["_ca"]), chain=ch, dec_e=dec_e, dec_b=dec_b)
                    for _, r in pos.iterrows()
                )
            )
            has_aux_eth = bool((g["_et"] == "native_auxiliary_zero").any())
            summaries.append(
                {
                    "tx_hash": str(txh),
                    "chain": ch,
                    "primary_address": src_user,
                    "counterparty_address": counterparty,
                    "token_contract": ca,
                    "token_symbol": sym,
                    "asset_group": ag,
                    "route_id": rid,
                    "time_stamp": ts_min,
                    "block_number": bn_min,
                    "amount_raw_sum": amount_raw_sum,
                    "amount_human_sum": human,
                    "amount_usd_sum": human,
                    "event_types": "|".join(etypes),
                    "evidence_quality": float(evq_mean),
                    "has_token_transfer": bool(not erc.empty),
                    "has_native_auxiliary_zero": has_aux_eth,
                }
            )

    summaries.sort(key=lambda s: int(s["time_stamp"]))
    return summaries


def _segment_key_from_summary(s: dict[str, Any], *, chain: str) -> tuple[str, str, str]:
    """Return (flow_primary, bundle_dim, chain) for rolling-window grouping."""
    ch = chain.upper()
    ag = str(s.get("asset_group") or "")
    rid = str(s.get("route_id") or "")
    bundle = ag if ag else rid
    if ch == "ETH":
        primary = norm_addr(str(s.get("primary_address") or ""))
    else:
        primary = norm_addr(str(s.get("primary_address") or ""))
    return primary, bundle, ch


def _summaries_to_flows(
    summaries: list[dict[str, Any]],
    *,
    chain: str,
    window_sec: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flows: list[dict[str, Any]] = []
    tx_map_rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current and current.get("tx_hashes"):
            flows.append(current)
        current = None

    for s in summaries:
        if str(s.get("chain") or "").upper() != chain.upper():
            continue
        txh = str(s.get("tx_hash") or "")
        if not txh:
            continue
        pk = _segment_key_from_summary(s, chain=chain)
        ts = int(s.get("time_stamp") or 0)
        need_new = (
            current is None
            or tuple(current["segment_key"]) != pk
            or (ts - float(current["last_ts"])) > int(window_sec)
        )
        if need_new:
            flush()
            fid = f"{chain.lower()}_flow_{uuid.uuid4().hex[:14]}"
            current = {
                "flow_id": fid,
                "chain": chain.upper(),
                "segment_key": pk,
                "tx_hashes": [],
                "address_set": set(),
                "primary_address": pk[0],
                "token_contracts": set(),
                "token_symbols": set(),
                "asset_group": str(s.get("asset_group") or ""),
                "route_id": str(s.get("route_id") or ""),
                "start_time": ts,
                "end_time": ts,
                "last_ts": ts,
                "raw_amount_sum": 0,
                "human_amount_sum": 0.0,
                "usd_amount_sum": 0.0,
                "aml_scores": [],
                "_eq_vals": [],
                "flow_construction_rule": f"tx_agg_rolling_{window_sec}s_same_asset_group_or_route",
            }

        assert current is not None
        current["tx_hashes"].append(txh)
        pa = str(s.get("primary_address") or "")
        cp = str(s.get("counterparty_address") or "")
        if pa:
            current["address_set"].add(pa)
        if cp:
            current["address_set"].add(cp)
        ca = str(s.get("token_contract") or "")
        sym = str(s.get("token_symbol") or "")
        if ca:
            current["token_contracts"].add(norm_addr(ca))
        if sym:
            current["token_symbols"].add(sym)
        current["end_time"] = ts
        current["last_ts"] = ts

        raw_add = int(s.get("amount_raw_sum") or 0)
        hum_add = float(s.get("amount_human_sum") or 0.0)
        usd_add = float(s.get("amount_usd_sum") or 0.0)
        current["raw_amount_sum"] = int(current["raw_amount_sum"]) + raw_add
        current["human_amount_sum"] = float(current["human_amount_sum"]) + hum_add
        current["usd_amount_sum"] = float(current["usd_amount_sum"]) + usd_add
        current["_eq_vals"].append(float(s.get("evidence_quality") if s.get("evidence_quality") is not None else 0.4))

        tx_map_rows.append(
            {
                "chain": chain.upper(),
                "tx_hash": txh,
                "flow_id": current["flow_id"],
                "role": "tx_summary",
                "primary_address": pa,
                "route_id": str(s.get("route_id") or ""),
                "asset_group": str(s.get("asset_group") or ""),
                "time_stamp": ts,
                "amount_usd": usd_add,
                "included_in_flow": True,
                "exclude_reason": "",
            }
        )

    flush()

    out_flows: list[dict[str, Any]] = []
    for f in flows:
        q_mean = float(np.mean(f.get("_eq_vals") or [0.0]))
        uniq_tx = sorted(set(f["tx_hashes"]))
        out_flows.append(
            {
                "chain": f["chain"],
                "flow_id": f["flow_id"],
                "tx_hashes": "|".join(uniq_tx),
                "address_set": "|".join(sorted(x for x in f["address_set"] if x)),
                "primary_address": f["primary_address"],
                "token_contracts": "|".join(sorted(x for x in f["token_contracts"] if x)),
                "token_symbols": "|".join(sorted(x for x in f["token_symbols"] if x)),
                "asset_group": f["asset_group"],
                "route_id": f["route_id"],
                "start_time": int(f["start_time"]),
                "end_time": int(f["end_time"]),
                "tx_count": len(uniq_tx),
                "raw_amount_sum": int(f["raw_amount_sum"]),
                "human_amount_sum": float(f["human_amount_sum"]),
                "usd_amount_sum": float(f["usd_amount_sum"]),
                "aml_score_mean": 0.0,
                "aml_score_max": 0.0,
                "evidence_quality_mean": q_mean,
                "flow_construction_rule": f["flow_construction_rule"],
            }
        )

    return out_flows, tx_map_rows


def build_flow_segments_from_evidence(
    evidence_eth_path: Path,
    evidence_bnb_path: Path,
    tx_anchor_labels_path: Path,
    out_dir: Path,
    *,
    flow_window_sec: int = 1800,
) -> tuple[Path, Path, Path]:
    """``tx_anchor_labels_path`` kept for API symmetry; segmentation is evidence-only."""
    _ = tx_anchor_labels_path
    out_dir.mkdir(parents=True, exist_ok=True)
    eth_e = pd.read_csv(evidence_eth_path, dtype=str, keep_default_na=False)
    bnb_e = pd.read_csv(evidence_bnb_path, dtype=str, keep_default_na=False)

    ca_all = pd.concat(
        [eth_e["token_contract"], bnb_e["token_contract"]], ignore_index=True
    ).astype(str).map(norm_addr)
    want_e: set[str] = set()
    want_b: set[str] = set()
    for ca in ca_all:
        if ca and ca != ZERO:
            want_e.add(str(ca))
            want_b.add(str(ca))
    max_decimals_lookup = 256
    if len(want_e) + len(want_b) > max_decimals_lookup:
        c_e = Counter(eth_e["token_contract"].astype(str).map(norm_addr))
        c_b = Counter(bnb_e["token_contract"].astype(str).map(norm_addr))
        want_e = {t for t, _ in c_e.most_common(max_decimals_lookup) if t and t != ZERO}
        want_b = {t for t, _ in c_b.most_common(max_decimals_lookup) if t and t != ZERO}
    dec_e, dec_b = decimals_for_eth_bnb(want_e, want_b)

    eth_sum = _evidence_to_tx_summaries(eth_e, chain="ETH", dec_e=dec_e, dec_b=dec_b)
    bnb_sum = _evidence_to_tx_summaries(bnb_e, chain="BNB", dec_e=dec_e, dec_b=dec_b)

    eth_flows, eth_map = _summaries_to_flows(eth_sum, chain="ETH", window_sec=flow_window_sec)
    bnb_flows, bnb_map = _summaries_to_flows(bnb_sum, chain="BNB", window_sec=flow_window_sec)

    p_eth = output_file(out_dir, "flow_segments_eth.csv")
    p_bnb = output_file(out_dir, "flow_segments_bnb.csv")
    p_map = output_file(out_dir, "tx_to_flow_map.csv")
    pd.DataFrame(eth_flows).to_csv(p_eth, index=False)
    pd.DataFrame(bnb_flows).to_csv(p_bnb, index=False)
    pd.DataFrame(eth_map + bnb_map).to_csv(p_map, index=False)
    return p_eth, p_bnb, p_map
