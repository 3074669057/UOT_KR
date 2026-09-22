"""Build minimal flow-segment CSVs for semi-synthetic UOT (cloned from real segments)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


def _scale_numeric(val: Any, scale: float) -> str:
    try:
        x = float(pd.to_numeric(val, errors="coerce") or 0.0) * float(scale)
    except Exception:
        x = 0.0
    if math.isnan(x):
        x = 0.0
    if abs(x) < 1e-18 and scale <= 0.0:
        x = 1e-9
    return str(x)


def clone_segment_row(
    row: pd.Series,
    *,
    new_flow_id: str,
    amount_scale: float,
    time_offset_sec: float = 0.0,
) -> dict[str, Any]:
    out = {str(k): str(v) for k, v in row.to_dict().items()}
    out["flow_id"] = str(new_flow_id)
    sc = max(float(amount_scale), 0.0)
    for k in ("usd_amount_sum", "human_amount_sum", "raw_amount_sum"):
        if k in out:
            out[k] = _scale_numeric(out.get(k), sc if sc > 0 else 0.0)
    if sc <= 0.0:
        out["usd_amount_sum"] = "1e-9"
        out["human_amount_sum"] = "0"
        out["raw_amount_sum"] = "0"
    off = float(time_offset_sec or 0.0)
    if off != 0.0:
        # Real delay/noise perturbation for decoy flows: shift BOTH boundary
        # timestamps so start/end stay consistent (single-tx flows: start == end).
        for k in ("start_time", "end_time"):
            if k in out and str(out.get(k, "")).strip():
                try:
                    t = float(str(out[k]))
                    out[k] = str(int(round(t + off)))
                except ValueError:
                    pass
    return out


def write_synthetic_subgraph_segment_csvs(
    eth_segments_csv: Path,
    bnb_segments_csv: Path,
    clone_records: list[dict[str, Any]],
    out_eth: Path,
    out_bnb: Path,
) -> tuple[Path, Path]:
    """Emit segment CSVs containing only synthetic ``flow_id`` rows (cloned from templates)."""
    eth_df = pd.read_csv(eth_segments_csv, dtype=str, keep_default_na=False) if eth_segments_csv.is_file() else pd.DataFrame()
    bnb_df = pd.read_csv(bnb_segments_csv, dtype=str, keep_default_na=False) if bnb_segments_csv.is_file() else pd.DataFrame()
    eth_idx = eth_df.set_index("flow_id", drop=False) if not eth_df.empty and "flow_id" in eth_df.columns else None
    bnb_idx = bnb_df.set_index("flow_id", drop=False) if not bnb_df.empty and "flow_id" in bnb_df.columns else None

    eth_rows: list[dict[str, Any]] = []
    bnb_rows: list[dict[str, Any]] = []
    for rec in clone_records:
        sid = str(rec.get("synthetic_flow_id") or "")
        tid = str(rec.get("template_flow_id") or "")
        chain = str(rec.get("chain") or "").upper()
        scale = float(rec.get("amount_scale") or 1.0)
        off = float(rec.get("time_offset_sec") or 0.0)
        if not sid or not tid:
            continue
        if chain == "ETH" and eth_idx is not None and tid in eth_idx.index:
            row = eth_idx.loc[tid]
            eth_rows.append(
                clone_segment_row(
                    row.iloc[0] if isinstance(row, pd.DataFrame) else row,
                    new_flow_id=sid,
                    amount_scale=scale,
                    time_offset_sec=off,
                )
            )
        elif chain == "BNB" and bnb_idx is not None and tid in bnb_idx.index:
            row = bnb_idx.loc[tid]
            bnb_rows.append(
                clone_segment_row(
                    row.iloc[0] if isinstance(row, pd.DataFrame) else row,
                    new_flow_id=sid,
                    amount_scale=scale,
                    time_offset_sec=off,
                )
            )

    out_eth.parent.mkdir(parents=True, exist_ok=True)
    out_bnb.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eth_rows).to_csv(out_eth, index=False)
    pd.DataFrame(bnb_rows).to_csv(out_bnb, index=False)
    return out_eth, out_bnb
