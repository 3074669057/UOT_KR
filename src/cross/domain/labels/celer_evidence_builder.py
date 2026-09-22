"""Normalize Celer ETH/BNB export rows into traceable evidence events."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cross.config.output_layout import output_file
from cross.domain.labels.raw_value import parse_raw_value
from cross.shared.normalize import norm_addr

ZERO = "0x0000000000000000000000000000000000000000"


def _split_symbol_field(sym: str) -> tuple[str, str]:
    s = (sym or "").strip()
    if "_" in s:
        base, rest = s.split("_", 1)
        return base.strip().upper(), rest.strip().lower()
    return s.upper() if s else "", ""


def _asset_group_from_token(symbol_base: str) -> str:
    u = (symbol_base or "").upper()
    if u in ("USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD"):
        return f"stable:{u}"
    if u in ("ETH", "BNB"):
        return f"native:{u}"
    if u in ("WETH", "WBETH"):
        return f"wrapped:{u}"
    if not u:
        return "unknown"
    return f"token:{u}"


def _row_to_evidence(
    row: pd.Series,
    *,
    chain: str,
    row_id: int,
    bridge: str,
    source_csv_basename: str,
) -> dict[str, Any]:
    raw_hash = str(row.get("hash") or "").strip()
    raw_from = str(row.get("from") or "").strip()
    raw_to = str(row.get("to") or "").strip()
    raw_ts = str(row.get("timeStamp") or row.get("timestamp") or "").strip()
    raw_bn = str(row.get("blockNumber") or row.get("block_number") or "").strip()

    tx_hash = norm_addr(raw_hash)
    from_a = norm_addr(raw_from)
    to_a = norm_addr(raw_to)
    ca = norm_addr(str(row.get("contractAddress") or row.get("contract_address") or ""))
    sym_field = str(row.get("symbol") or "")
    token_symbol, token_tail = _split_symbol_field(sym_field)
    if token_tail.startswith("0x") and len(token_tail) >= 42:
        token_contract = norm_addr(token_tail[:42])
    else:
        token_contract = ca if ca and ca != ZERO else ""

    raw_s = row.get("value")
    raw_int, inv = parse_raw_value(raw_s)
    raw_value_str = str(raw_int) if raw_int is not None else ""

    ts = int(pd.to_numeric(row.get("timeStamp"), errors="coerce") or 0)
    bn = int(pd.to_numeric(row.get("blockNumber"), errors="coerce") or 0)

    is_native = (not ca) or ca == ZERO
    is_zero_value = raw_int == 0 if raw_int is not None else True

    event_type = ""
    direction_role = ""
    evidence_quality = "low"
    invalid_reason = inv

    chain_u = chain.upper()
    if chain_u == "ETH":
        if is_native:
            event_type = "native_deposit_or_call"
            direction_role = "eth_native_bridge_interaction"
            evidence_quality = "medium"
        else:
            event_type = "erc20_deposit_or_transfer"
            direction_role = "eth_token_bridge_interaction"
            evidence_quality = "high" if raw_int and raw_int > 0 else "medium"
    elif chain_u == "BNB":
        if is_native and is_zero_value:
            event_type = "native_auxiliary_zero"
            direction_role = "bnb_native_auxiliary"
            evidence_quality = "low"
        elif (not is_native) and raw_int is not None and raw_int > 0:
            event_type = "token_transfer_log"
            direction_role = "bnb_token_release"
            evidence_quality = "high"
        elif is_native and raw_int is not None and raw_int > 0:
            event_type = "bnb_native_transfer"
            direction_role = "bnb_native_nonzero"
            evidence_quality = "medium"
        else:
            event_type = "bnb_other"
            direction_role = "bnb_other"
            evidence_quality = "low"
    else:
        event_type = "unknown_chain"
        direction_role = ""

    bridge_contract = to_a if chain_u == "ETH" else norm_addr(str(row.get("Address") or row.get("address") or ""))
    if not bridge_contract:
        bridge_contract = to_a

    asset_group = _asset_group_from_token(token_symbol)
    route_id = str(row.get("id") or "")

    if invalid_reason:
        evidence_quality = "low"

    return {
        "chain": chain_u,
        "bridge": bridge or str(row.get("Bridge") or "CelerNetwork"),
        "tx_hash": tx_hash,
        "row_id": int(row_id),
        "source_csv": source_csv_basename,
        "csv_tx_hash": raw_hash,
        "csv_from": raw_from,
        "csv_to": raw_to,
        "csv_time_stamp": raw_ts,
        "csv_block_number": raw_bn,
        "event_type": event_type,
        "bridge_contract": bridge_contract,
        "token_symbol": token_symbol or (sym_field or ""),
        "token_contract": token_contract,
        "from_address": from_a,
        "to_address": to_a,
        "raw_value": raw_value_str,
        "raw_value_int": int(raw_int) if raw_int is not None else "",
        "time_stamp": ts,
        "block_number": bn,
        "is_native": bool(is_native),
        "is_zero_value": bool(is_zero_value),
        "direction_role": direction_role,
        "asset_group": asset_group,
        "route_id": route_id,
        "evidence_quality": evidence_quality,
        "invalid_reason": invalid_reason,
    }


def build_celer_evidence_from_csvs(
    eth_csv: Path,
    bnb_csv: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write ``evidence_eth.csv`` and ``evidence_bnb.csv`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eth_df = pd.read_csv(eth_csv, dtype=str, keep_default_na=False)
    bnb_df = pd.read_csv(bnb_csv, dtype=str, keep_default_na=False)

    eth_name = eth_csv.name
    bnb_name = bnb_csv.name
    eth_rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(eth_df.iterrows()):
        eth_rows.append(
            _row_to_evidence(row, chain="ETH", row_id=i, bridge=str(row.get("Bridge") or ""), source_csv_basename=eth_name)
        )

    bnb_rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(bnb_df.iterrows()):
        bnb_rows.append(
            _row_to_evidence(row, chain="BNB", row_id=i, bridge=str(row.get("Bridge") or ""), source_csv_basename=bnb_name)
        )

    eth_out = output_file(out_dir, "evidence_eth.csv")
    bnb_out = output_file(out_dir, "evidence_bnb.csv")
    pd.DataFrame(eth_rows).to_csv(eth_out, index=False)
    pd.DataFrame(bnb_rows).to_csv(bnb_out, index=False)
    return eth_out, bnb_out
