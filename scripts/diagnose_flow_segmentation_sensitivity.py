#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only flow segmentation sensitivity (Phase 1.5). Does not write to labels/ or label_layer_v1/."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cross.config.paths import CROSS_ROOT
from cross.domain.labels.celer_supervised_pipeline import resolve_celer_label_csv_path
from cross.domain.labels.flow_label_builder import _tx_to_flow
from cross.domain.labels.flow_segment_builder import ZERO, _evidence_to_tx_summaries
from cross.shared.connector_decimals import decimals_for_eth_bnb
from cross.shared.normalize import norm_addr
from cross.shared.tx_hash import normalize_tx_hash

CANONICAL_WINDOW = 1800
CANONICAL_KEY = "addr_ag_route"


def _segment_key_from_summary(
    s: dict[str, Any],
    *,
    chain: str,
    key_mode: str,
) -> tuple[str, str, str]:
    ch = chain.upper()
    primary = norm_addr(str(s.get("primary_address") or ""))
    ag = str(s.get("asset_group") or "")
    rid = str(s.get("route_id") or "")
    if key_mode == "addr_ag_only":
        bundle = ag
    else:
        bundle = ag if ag else rid
    return primary, bundle, ch


def _adaptive_gap_limit(*, last_gap_sec: float, window_sec: int) -> float:
    """Cap rolling gap: min(window, max(300s, 2.5 * previous inter-tx gap))."""
    if last_gap_sec <= 0:
        return float(window_sec)
    return float(min(window_sec, max(300.0, 2.5 * last_gap_sec)))


def _summaries_to_flows_variant(
    summaries: list[dict[str, Any]],
    *,
    chain: str,
    window_sec: int,
    key_mode: str,
    adaptive_gap: bool,
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
        pk = _segment_key_from_summary(s, chain=chain, key_mode=key_mode)
        ts = int(s.get("time_stamp") or 0)
        if current is None:
            need_new = True
        else:
            gap = ts - float(current["last_ts"])
            gap_lim = (
                _adaptive_gap_limit(last_gap_sec=gap, window_sec=window_sec)
                if adaptive_gap
                else float(window_sec)
            )
            need_new = tuple(current["segment_key"]) != pk or gap > gap_lim
        if need_new:
            flush()
            fid = f"{chain.lower()}_flow_{uuid.uuid4().hex[:14]}"
            rule = f"tx_agg_rolling_{window_sec}s"
            if key_mode == "addr_ag_only":
                rule += "_same_asset_group"
            else:
                rule += "_same_asset_group_or_route"
            if adaptive_gap:
                rule += "_adaptive_gap"
            current = {
                "flow_id": fid,
                "chain": chain.upper(),
                "segment_key": pk,
                "tx_hashes": [],
                "primary_address": pk[0],
                "asset_group": str(s.get("asset_group") or ""),
                "route_id": str(s.get("route_id") or ""),
                "start_time": ts,
                "end_time": ts,
                "last_ts": ts,
                "prev_ts": ts,
                "usd_amount_sum": 0.0,
                "flow_construction_rule": rule,
            }

        assert current is not None
        current["tx_hashes"].append(txh)
        current["end_time"] = ts
        current["last_ts"] = ts
        hum = float(s.get("amount_usd_sum") or 0.0)
        current["usd_amount_sum"] = float(current["usd_amount_sum"]) + hum
        tx_map_rows.append(
            {
                "chain": chain.upper(),
                "tx_hash": txh,
                "flow_id": current["flow_id"],
                "included_in_flow": True,
            }
        )

    flush()

    out_flows: list[dict[str, Any]] = []
    for f in flows:
        uniq_tx = sorted(set(f["tx_hashes"]))
        out_flows.append(
            {
                "flow_id": f["flow_id"],
                "chain": f["chain"],
                "tx_count": len(uniq_tx),
                "usd_amount_sum": float(f["usd_amount_sum"]),
                "asset_group": f.get("asset_group") or "",
                "route_id": f.get("route_id") or "",
                "flow_construction_rule": f["flow_construction_rule"],
            }
        )
    return out_flows, tx_map_rows


def _resolve_evidence_csv(run_root: Path, name: str) -> Path:
    for rel in (
        name,
        f"evidence/{name}",
        f"labels/{name}",
        f"label_layer_v1/{name}",
    ):
        p = run_root / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"{name} not found under {run_root}")


def _load_evidence_summaries(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eth_p = _resolve_evidence_csv(run_root, "evidence_eth.csv")
    bnb_p = _resolve_evidence_csv(run_root, "evidence_bnb.csv")
    eth_e = pd.read_csv(eth_p, dtype=str, keep_default_na=False)
    bnb_e = pd.read_csv(bnb_p, dtype=str, keep_default_na=False)

    ca_all = pd.concat([eth_e["token_contract"], bnb_e["token_contract"]], ignore_index=True).astype(str).map(norm_addr)
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
    return eth_sum, bnb_sum


def _build_celer_flow_labels(
    celer_tx: pd.DataFrame,
    eth_seg: pd.DataFrame,
    bnb_seg: pd.DataFrame,
    tx_map: pd.DataFrame,
) -> pd.DataFrame:
    tx_flow = _tx_to_flow(tx_map)
    eth_amt = {
        str(r["flow_id"]): float(pd.to_numeric(r.get("usd_amount_sum"), errors="coerce") or 0.0)
        for _, r in eth_seg.iterrows()
    }
    bnb_amt = {
        str(r["flow_id"]): float(pd.to_numeric(r.get("usd_amount_sum"), errors="coerce") or 0.0)
        for _, r in bnb_seg.iterrows()
    }
    eth_by_id = eth_seg.set_index("flow_id", drop=False) if not eth_seg.empty else pd.DataFrame()
    bnb_by_id = bnb_seg.set_index("flow_id", drop=False) if not bnb_seg.empty else pd.DataFrame()

    a = celer_tx.copy()
    a["_sh"] = a["src_tx_hash"].astype(str).map(normalize_tx_hash)
    a["_dh"] = a["dst_tx_hash"].astype(str).map(normalize_tx_hash)
    a["_sf"] = a["_sh"].map(tx_flow)
    a["_df"] = a["_dh"].map(tx_flow)
    a = a.dropna(subset=["_sf", "_df"])
    if a.empty:
        return pd.DataFrame()

    g = a.groupby(["_sf", "_df"], as_index=False).agg(
        support_tx_pair_count=("_sh", "count"),
        support_src_tx_hashes=("_sh", lambda s: "|".join(sorted(set(s)))),
        support_dst_tx_hashes=("_dh", lambda s: "|".join(sorted(set(s)))),
    )

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
        er = eth_by_id.loc[sfid] if sfid in eth_by_id.index else None
        br = bnb_by_id.loc[dfid] if dfid in bnb_by_id.index else None
        stc = int(er["tx_count"]) if er is not None else int(r["support_tx_pair_count"])
        dtc = int(br["tx_count"]) if br is not None else int(r["support_tx_pair_count"])
        rows.append(
            {
                "src_flow_id": sfid,
                "dst_flow_id": dfid,
                "support_tx_pair_count": int(r["support_tx_pair_count"]),
                "src_tx_count": stc,
                "dst_tx_count": dtc,
                "pattern_type": ptype,
                "src_degree": src_deg,
                "dst_degree": dst_deg,
            }
        )
    return pd.DataFrame(rows)


def _degree_distribution(series: pd.Series) -> dict[str, int]:
    vc = series.value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


def _summarize_labels(fl: pd.DataFrame) -> dict[str, Any]:
    if fl.empty:
        return {
            "flow_label_count": 0,
            "pattern_type": {},
            "src_degree": {},
            "dst_degree": {},
            "support_tx_pair_count_sum": 0,
            "one_to_one_ratio": 0.0,
        }
    pt = fl["pattern_type"].value_counts().to_dict()
    n = len(fl)
    return {
        "flow_label_count": int(n),
        "pattern_type": {str(k): int(v) for k, v in pt.items()},
        "src_degree": _degree_distribution(fl["src_degree"]),
        "dst_degree": _degree_distribution(fl["dst_degree"]),
        "support_tx_pair_count_sum": int(fl["support_tx_pair_count"].sum()),
        "one_to_one_ratio": float((fl["pattern_type"] == "one_to_one").mean()),
        "many_to_one_ratio": float((fl["pattern_type"] == "many_to_one").mean()),
        "one_to_many_ratio": float((fl["pattern_type"] == "one_to_many").mean()),
    }


def _diff_vs_canonical(row: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    return {
        "eth_flow_count_delta": int(row["eth_flow_count"]) - int(canonical["eth_flow_count"]),
        "bnb_flow_count_delta": int(row["bnb_flow_count"]) - int(canonical["bnb_flow_count"]),
        "flow_label_count_delta": int(row["flow_label_count"]) - int(canonical["flow_label_count"]),
        "one_to_one_ratio_delta_pp": round(
            100.0 * (float(row["one_to_one_ratio"]) - float(canonical["one_to_one_ratio"])), 2
        ),
        "support_tx_pair_count_sum_delta": int(row["support_tx_pair_count_sum"])
        - int(canonical["support_tx_pair_count_sum"]),
    }


def _run_variant(
    eth_sum: list[dict[str, Any]],
    bnb_sum: list[dict[str, Any]],
    celer_tx: pd.DataFrame,
    *,
    window_sec: int,
    key_mode: str,
    adaptive_gap: bool,
) -> dict[str, Any]:
    eth_flows, eth_map = _summaries_to_flows_variant(
        eth_sum, chain="ETH", window_sec=window_sec, key_mode=key_mode, adaptive_gap=adaptive_gap
    )
    bnb_flows, bnb_map = _summaries_to_flows_variant(
        bnb_sum, chain="BNB", window_sec=window_sec, key_mode=key_mode, adaptive_gap=adaptive_gap
    )
    eth_df = pd.DataFrame(eth_flows)
    bnb_df = pd.DataFrame(bnb_flows)
    tx_map = pd.DataFrame(eth_map + bnb_map)
    fl = _build_celer_flow_labels(celer_tx, eth_df, bnb_df, tx_map)
    out = {
        "window_sec": int(window_sec),
        "key_mode": key_mode,
        "adaptive_gap": bool(adaptive_gap),
        "eth_flow_count": int(len(eth_df)),
        "bnb_flow_count": int(len(bnb_df)),
    }
    out.update(_summarize_labels(fl))
    return out


def _load_canonical_metrics(run_root: Path) -> dict[str, Any]:
    fl = pd.read_csv(run_root / "labels" / "flow_labels.csv")
    eth = pd.read_csv(run_root / "labels" / "flow_segments_eth.csv")
    bnb = pd.read_csv(run_root / "labels" / "flow_segments_bnb.csv")
    base = {
        "source": "labels/flow_labels.csv (frozen canonical, not recomputed)",
        "window_sec": CANONICAL_WINDOW,
        "key_mode": CANONICAL_KEY,
        "adaptive_gap": False,
        "eth_flow_count": int(len(eth)),
        "bnb_flow_count": int(len(bnb)),
    }
    base.update(_summarize_labels(fl))
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1.5 flow segmentation sensitivity (read-only)")
    ap.add_argument(
        "--run-root",
        type=Path,
        default=_REPO / "out" / "paper_full_pipeline_run",
        help="Pipeline run root (reads evidence + celer; never writes labels/)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="JSON output path (default: <run-root>/experiments/flow_segmentation_sensitivity.json)",
    )
    args = ap.parse_args()
    run_root = Path(args.run_root).resolve()
    out_json = args.out_json or (run_root / "experiments" / "flow_segmentation_sensitivity.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    celer_path, _ = resolve_celer_label_csv_path(CROSS_ROOT, None)
    if celer_path is None or not celer_path.is_file():
        print("ERROR: celer_label.csv not found", file=sys.stderr)
        return 1
    celer_tx = pd.read_csv(celer_path, dtype=str, keep_default_na=False)
    if "src_tx_hash" not in celer_tx.columns:
        for alt in ("srcTxhash", "src_txhash"):
            if alt in celer_tx.columns:
                celer_tx = celer_tx.rename(columns={alt: "src_tx_hash"})
                break
    if "dst_tx_hash" not in celer_tx.columns:
        for alt in ("dstTxhash", "dst_txhash"):
            if alt in celer_tx.columns:
                celer_tx = celer_tx.rename(columns={alt: "dst_tx_hash"})
                break
    celer_tx["src_tx_hash"] = celer_tx["src_tx_hash"].astype(str).map(normalize_tx_hash)
    celer_tx["dst_tx_hash"] = celer_tx["dst_tx_hash"].astype(str).map(normalize_tx_hash)

    eth_sum, bnb_sum = _load_evidence_summaries(run_root)
    canonical = _load_canonical_metrics(run_root)

    variants: list[dict[str, Any]] = []
    for w in (600, 1800, 3600):
        row = _run_variant(
            eth_sum, bnb_sum, celer_tx, window_sec=w, key_mode=CANONICAL_KEY, adaptive_gap=False
        )
        row["variant_id"] = f"window_{w}s_{CANONICAL_KEY}"
        row["diff_vs_canonical"] = _diff_vs_canonical(row, canonical)
        variants.append(row)

    for key_mode in ("addr_ag_only", "addr_ag_route"):
        if key_mode == CANONICAL_KEY:
            # already covered by window_1800s; tag explicitly for ablation table
            pass
        row = _run_variant(
            eth_sum,
            bnb_sum,
            celer_tx,
            window_sec=CANONICAL_WINDOW,
            key_mode=key_mode,
            adaptive_gap=False,
        )
        row["variant_id"] = f"key_{key_mode}_window_{CANONICAL_WINDOW}s"
        row["diff_vs_canonical"] = _diff_vs_canonical(row, canonical)
        variants.append(row)

    row_adapt = _run_variant(
        eth_sum,
        bnb_sum,
        celer_tx,
        window_sec=CANONICAL_WINDOW,
        key_mode=CANONICAL_KEY,
        adaptive_gap=True,
    )
    row_adapt["variant_id"] = f"adaptive_gap_{CANONICAL_KEY}_window_{CANONICAL_WINDOW}s"
    row_adapt["diff_vs_canonical"] = _diff_vs_canonical(row_adapt, canonical)
    variants.append(row_adapt)

    # Recompute 1800s canonical key for reproducibility check
    row_repro = _run_variant(
        eth_sum, bnb_sum, celer_tx, window_sec=1800, key_mode=CANONICAL_KEY, adaptive_gap=False
    )
    repro_check = {
        "eth_flow_count_match": row_repro["eth_flow_count"] == canonical["eth_flow_count"],
        "bnb_flow_count_match": row_repro["bnb_flow_count"] == canonical["bnb_flow_count"],
        "flow_label_count_match": row_repro["flow_label_count"] == canonical["flow_label_count"],
        "recomputed": row_repro,
    }

    payload = {
        "run_root": str(run_root),
        "celer_label_csv": str(celer_path),
        "canonical_on_disk": canonical,
        "reproducibility_check_1800s": repro_check,
        "variants": variants,
        "code_refs": {
            "production_segmentation": "src/cross/domain/labels/flow_segment_builder.py",
            "segment_key": "_segment_key_from_summary (primary_address, asset_group or route_id, chain)",
            "window_param": "flow_window_sec / --flow-window-sec (default 1800)",
            "path_b_segment_builder": "src/cross/domain/flows/segment_builder.py (UOT Path B only, not label layer)",
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json}")
    for v in variants:
        print(
            f"{v['variant_id']}: eth={v['eth_flow_count']} bnb={v['bnb_flow_count']} "
            f"labels={v['flow_label_count']} o2o={v['one_to_one_ratio']:.2%} "
            f"delta_labels={v['diff_vs_canonical']['flow_label_count_delta']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
