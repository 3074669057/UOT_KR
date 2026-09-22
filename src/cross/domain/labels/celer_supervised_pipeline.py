"""Celer CSV tx-pair supervision + weak-evidence labels: split, merge, diagnostics, canonical copies."""
from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import output_file
from cross.config.paths import CROSS_ROOT
from cross.domain.labels.flow_label_builder import _tx_to_flow
from cross.shared.tx_hash import normalize_tx_hash, validate_tx_hash

logger = logging.getLogger(__name__)

LABEL_SOURCE_CELER = "celer_label"
LABEL_SOURCE_WEAK = "weak_evidence"
LABEL_SOURCE_MERGED = "merged"

_CELER_WEAK_FALLBACK_WARNING = (
    "Celer not available, falling back to weak labels — this will produce "
    "non-supervised labels and break paper claim 2"
)


def resolve_effective_label_source_mode(
    requested: str,
    *,
    celer_resolved: Path | None,
) -> tuple[str, str, bool, str]:
    """Map ``celer_only_if_available`` → ``celer_only`` or ``weak_only``; return (requested, effective, fallback, reason)."""
    req = (requested or "celer_only_if_available").strip().lower()
    if req != "celer_only_if_available":
        return req, req, False, ""
    if celer_resolved is not None and celer_resolved.is_file():
        return req, "celer_only", False, ""
    logger.warning("%s", _CELER_WEAK_FALLBACK_WARNING)
    return req, "weak_only", True, "celer_only_if_available: celer_label.csv not found after search"


def resolve_celer_label_csv_path(
    project_root: Path | str,
    explicit: Path | str | None,
) -> tuple[Path | None, list[str]]:
    """Resolve Celer tx-pair CSV: ``--celer-tx-labels`` then ``label/celer_label.csv`` then ``label/tx/``."""
    root = Path(project_root).resolve()
    tried: list[str] = []
    if explicit is not None and str(explicit).strip():
        p = Path(explicit).expanduser().resolve()
        tried.append(str(p))
        if p.is_file():
            return p, tried
    for rel in ("label/celer_label.csv", "label/tx/celer_label.csv"):
        p = (root / rel).resolve()
        tried.append(str(p))
        if p.is_file():
            return p, tried
    return None, tried


def _normalize_chain_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ("eth", "ethereum", "mainnet"):
        return "eth"
    if s in ("bnb", "bsc", "bnb chain", "binance smart chain", "bscc", "bsc-mainnet"):
        return "bnb"
    return s


def _lower_col_map(df: pd.DataFrame) -> dict[str, str]:
    return {c.lower(): c for c in df.columns}


def _pick_col(df: pd.DataFrame, *names: str) -> str | None:
    m = _lower_col_map(df)
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None


def load_celer_tx_labels(label_csv: str | Path, out_dir: str | Path) -> pd.DataFrame:
    """Parse ``celer_label.csv`` and write ``labels/tx_anchor_labels_from_celer.csv`` (no thresholding).

    Writes ``labels/celer_tx_load_diagnostics.json`` with row counts / dedupe / invalid stats.
    """
    out_root = Path(out_dir)
    path = Path(label_csv)
    if not path.is_file():
        raise FileNotFoundError(f"celer_label CSV not found: {path}")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    c_srcnet = _pick_col(raw, "srcnet", "src_net", "srcchain")
    c_srctx = _pick_col(raw, "srcTxhash", "src_tx_hash", "srctxhash", "eth_tx", "src_txhash")
    c_dstnet = _pick_col(raw, "dstnet", "dst_net", "dstchain")
    c_dsttx = _pick_col(raw, "dstTxhash", "dst_tx_hash", "dsttxhash", "bnb_tx", "dst_txhash")
    if not all([c_srcnet, c_srctx, c_dstnet, c_dsttx]):
        raise ValueError(
            f"celer_label.csv missing required columns (srcnet, srcTxhash, dstnet, dstTxhash). "
            f"Got columns: {list(raw.columns)}"
        )

    rows: list[dict[str, Any]] = []
    invalid = 0
    for _, r in raw.iterrows():
        rs = str(r.get(c_srcnet) or "").strip()
        rd = str(r.get(c_dstnet) or "").strip()
        sh = normalize_tx_hash(str(r.get(c_srctx) or ""))
        dh = normalize_tx_hash(str(r.get(c_dsttx) or ""))
        cn_s = _normalize_chain_name(rs)
        cn_d = _normalize_chain_name(rd)
        if cn_s != "eth" or cn_d != "bnb":
            invalid += 1
            continue
        if not (validate_tx_hash(sh) and validate_tx_hash(dh)):
            invalid += 1
            continue
        rows.append(
            {
                "src_tx_hash": sh,
                "dst_tx_hash": dh,
                "src_chain": "eth",
                "dst_chain": "bnb",
                "label_source": LABEL_SOURCE_CELER,
                "label_type": "supervised_tx_pair",
                "confidence": 1.0,
                "is_supervised": "true",
                "raw_srcnet": rs,
                "raw_dstnet": rd,
            }
        )

    df = pd.DataFrame(rows)
    before_dedupe = len(df)
    if not df.empty:
        df = df.drop_duplicates(subset=["src_tx_hash", "dst_tx_hash"], keep="first")
    dup_ct = max(0, before_dedupe - len(df))

    out_p = output_file(out_root, "tx_anchor_labels_from_celer.csv")
    df.to_csv(out_p, index=False)

    mini = {
        "celer_total_raw_rows": int(len(raw)),
        "celer_valid_tx_pairs": int(len(df)),
        "celer_duplicate_tx_pairs": int(dup_ct),
        "celer_invalid_rows": int(invalid),
    }
    output_file(out_root, "celer_tx_load_diagnostics.json").write_text(
        json.dumps(mini, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Celer supervised tx labels: raw=%d valid_pairs=%d duplicates_removed=%d invalid=%d -> %s",
        len(raw),
        len(df),
        dup_ct,
        invalid,
        out_p,
    )
    return df


def _flow_time_amount_maps(
    eth_seg: pd.DataFrame,
    bnb_seg: pd.DataFrame,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]], dict[str, float], dict[str, float]]:
    eth_t: dict[str, tuple[int, int]] = {}
    bnb_t: dict[str, tuple[int, int]] = {}
    eth_amt: dict[str, float] = {}
    bnb_amt: dict[str, float] = {}
    if not eth_seg.empty and "flow_id" in eth_seg.columns:
        for _, r in eth_seg.iterrows():
            fid = str(r.get("flow_id") or "")
            if not fid:
                continue
            st = int(pd.to_numeric(r.get("start_time"), errors="coerce") or 0)
            et = int(pd.to_numeric(r.get("end_time"), errors="coerce") or 0)
            eth_t[fid] = (st, et)
            eth_amt[fid] = float(pd.to_numeric(r.get("usd_amount_sum"), errors="coerce") or 0.0)
    if not bnb_seg.empty and "flow_id" in bnb_seg.columns:
        for _, r in bnb_seg.iterrows():
            fid = str(r.get("flow_id") or "")
            if not fid:
                continue
            st = int(pd.to_numeric(r.get("start_time"), errors="coerce") or 0)
            et = int(pd.to_numeric(r.get("end_time"), errors="coerce") or 0)
            bnb_t[fid] = (st, et)
            bnb_amt[fid] = float(pd.to_numeric(r.get("usd_amount_sum"), errors="coerce") or 0.0)
    return eth_t, bnb_t, eth_amt, bnb_amt


def build_flow_labels_from_celer(
    tx_anchor_labels_from_celer: pd.DataFrame,
    flow_segments_eth: pd.DataFrame,
    flow_segments_bnb: pd.DataFrame,
    tx_to_flow_map: pd.DataFrame,
    out_dir: str | Path,
    *,
    out_filename: str = "flow_labels_from_celer.csv",
) -> pd.DataFrame:
    """Map Celer tx pairs to (src_flow, dst_flow) and aggregate; writes ``labels/<out_filename>``."""
    out_root = Path(out_dir)
    tx_flow = _tx_to_flow(tx_to_flow_map)
    eth_t, bnb_t, eth_amt, bnb_amt = _flow_time_amount_maps(flow_segments_eth, flow_segments_bnb)

    if tx_anchor_labels_from_celer.empty:
        out_csv = output_file(out_root, out_filename)
        pd.DataFrame().to_csv(out_csv, index=False)
        return pd.DataFrame()

    a = tx_anchor_labels_from_celer.copy()
    a["_sh"] = a["src_tx_hash"].astype(str).map(normalize_tx_hash)
    a["_dh"] = a["dst_tx_hash"].astype(str).map(normalize_tx_hash)
    a["_sf"] = a["_sh"].map(tx_flow)
    a["_df"] = a["_dh"].map(tx_flow)
    a = a.dropna(subset=["_sf", "_df"])

    g = a.groupby(["_sf", "_df"], as_index=False).agg(
        matched_tx_count=("src_tx_hash", "count"),
        src_tx_hashes=("_sh", lambda s: "|".join(sorted(set(s)))),
        dst_tx_hashes=("_dh", lambda s: "|".join(sorted(set(s)))),
    )

    rows: list[dict[str, Any]] = []
    for _, r in g.iterrows():
        sfid = str(r["_sf"])
        dfid = str(r["_df"])
        st0, et0 = eth_t.get(sfid, (0, 0))
        st1, et1 = bnb_t.get(dfid, (0, 0))
        sa = float(eth_amt.get(sfid, 0.0))
        da = float(bnb_amt.get(dfid, 0.0))
        rows.append(
            {
                "src_flow_id": sfid,
                "dst_flow_id": dfid,
                "matched_tx_count": int(r["matched_tx_count"]),
                "src_tx_hashes": str(r["src_tx_hashes"]),
                "dst_tx_hashes": str(r["dst_tx_hashes"]),
                "label_source": LABEL_SOURCE_CELER,
                "label_type": "supervised_flow_pair",
                "label_confidence": 1.0,
                "confidence": 1.0,
                "is_supervised": "true",
                "total_amount_usd": float(sa + da),
                "src_amount_usd": sa,
                "dst_amount_usd": da,
                "first_src_time": float(st0),
                "last_src_time": float(et0),
                "first_dst_time": float(st1),
                "last_dst_time": float(et1),
            }
        )

    fl = pd.DataFrame(rows)
    out_csv = output_file(out_root, out_filename)
    fl.to_csv(out_csv, index=False)
    logger.info("Celer supervised flow labels: %d rows -> %s", len(fl), out_csv)
    return fl


def _enrich_weak_tx_anchors(weak_df: pd.DataFrame) -> pd.DataFrame:
    if weak_df.empty:
        return weak_df
    out = weak_df.copy()
    out["label_source"] = LABEL_SOURCE_WEAK
    out["label_type"] = "weak_tx_pair"
    out["is_supervised"] = "false"
    if "pair_label_confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["pair_label_confidence"], errors="coerce").fillna(0.0)
    else:
        out["confidence"] = 0.0
    return out


def _enrich_weak_flow_labels(weak_fl: pd.DataFrame) -> pd.DataFrame:
    if weak_fl.empty:
        return weak_fl
    out = weak_fl.copy()
    out["label_source"] = LABEL_SOURCE_WEAK
    out["label_type"] = "weak_flow_pair"
    out["is_supervised"] = "false"
    if "label_confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["label_confidence"], errors="coerce").fillna(0.0)
    else:
        out["confidence"] = 0.0
    return out


def merge_tx_anchor_labels(celer_tx: pd.DataFrame, weak_tx: pd.DataFrame) -> pd.DataFrame:
    celer_keys: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    if not celer_tx.empty:
        for _, r in celer_tx.iterrows():
            k = (normalize_tx_hash(str(r.get("src_tx_hash") or "")), normalize_tx_hash(str(r.get("dst_tx_hash") or "")))
            celer_keys.add(k)
            rows.append(dict(r))
    if not weak_tx.empty:
        for _, r in weak_tx.iterrows():
            k = (normalize_tx_hash(str(r.get("src_tx_hash") or "")), normalize_tx_hash(str(r.get("dst_tx_hash") or "")))
            if k in celer_keys:
                continue
            d = dict(r)
            d.setdefault("label_source", LABEL_SOURCE_WEAK)
            d.setdefault("label_type", "weak_tx_pair")
            d.setdefault("is_supervised", "false")
            rows.append(d)
    return pd.DataFrame(rows)


def merge_flow_labels(celer_fl: pd.DataFrame, weak_fl: pd.DataFrame) -> pd.DataFrame:
    weak_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if not weak_fl.empty:
        for _, r in weak_fl.iterrows():
            k = (str(r.get("src_flow_id") or "").strip(), str(r.get("dst_flow_id") or "").strip())
            if k[0] and k[1]:
                weak_by_key[k] = dict(r)

    rows: list[dict[str, Any]] = []
    if not celer_fl.empty:
        for _, r in celer_fl.iterrows():
            k = (str(r.get("src_flow_id") or "").strip(), str(r.get("dst_flow_id") or "").strip())
            base = dict(r)
            w = weak_by_key.pop(k, None)
            if w is not None:
                base["weak_confidence"] = float(pd.to_numeric(w.get("label_confidence"), errors="coerce") or 0.0)
                base["label_source"] = "celer_label+weak_evidence"
            else:
                base.setdefault("weak_confidence", "")
            rows.append(base)

    for k, w in weak_by_key.items():
        d = dict(w)
        d.setdefault("label_source", LABEL_SOURCE_WEAK)
        d.setdefault("label_type", "weak_flow_pair")
        d.setdefault("is_supervised", "false")
        d["weak_confidence"] = float(pd.to_numeric(w.get("label_confidence"), errors="coerce") or 0.0)
        rows.append(d)

    return pd.DataFrame(rows)


def _flow_pair_connected_components(edges: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Undirected connected components over flow_ids (edges are weak flow-pairs)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for sf, df in edges:
        if sf and df:
            union(sf, df)
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sf, df in edges:
        if sf and df:
            buckets[find(sf)].append((sf, df))
    return list(buckets.values())


def _split_flow_pair_keys_by_components(
    components: list[list[tuple[str, str]]],
    *,
    rng: np.random.Generator,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    comps = [list(c) for c in components]
    rng.shuffle(comps)
    comps.sort(key=len, reverse=True)
    total_e = sum(len(c) for c in comps)
    if total_e == 0:
        return set(), set(), set()
    tr_t = 0.70 * total_e
    va_t = 0.10 * total_e
    tr: set[tuple[str, str]] = set()
    va: set[tuple[str, str]] = set()
    te: set[tuple[str, str]] = set()
    tsz, vsz = 0, 0
    for c in comps:
        sset = set(c)
        n = len(c)
        if tsz < tr_t:
            tr |= sset
            tsz += n
        elif vsz < va_t:
            va |= sset
            vsz += n
        else:
            te |= sset
    return tr, va, te


def _train_val_test_overlap_stats(
    train_keys: set[tuple[str, str]],
    test_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    def srcs(keys: set[tuple[str, str]]) -> set[str]:
        return {a for a, _ in keys}

    def dsts(keys: set[tuple[str, str]]) -> set[str]:
        return {b for _, b in keys}

    ts, tss = srcs(train_keys), srcs(test_keys)
    td, tds = dsts(train_keys), dsts(test_keys)
    po = train_keys & test_keys
    leak = bool((ts & tss) or (td & tds) or po)
    return {
        "train_test_src_flow_overlap_count": int(len(ts & tss)),
        "train_test_dst_flow_overlap_count": int(len(td & tds)),
        "train_test_flow_pair_overlap_count": int(len(po)),
        "split_leakage_detected": bool(leak),
    }


def _annotate_celer_tx_with_flows(celer_tx: pd.DataFrame, tx_flow: dict[str, str]) -> pd.DataFrame:
    a = celer_tx.copy()
    a["_sh"] = a["src_tx_hash"].astype(str).map(normalize_tx_hash)
    a["_dh"] = a["dst_tx_hash"].astype(str).map(normalize_tx_hash)
    a["_sf"] = a["_sh"].map(tx_flow)
    a["_df"] = a["_dh"].map(tx_flow)
    return a


def _write_celer_heldout_from_flow_graph(
    out_root: Path,
    *,
    celer_tx: pd.DataFrame,
    celer_fl: pd.DataFrame,
    eth_seg: pd.DataFrame,
    bnb_seg: pd.DataFrame,
    tfm: pd.DataFrame,
    seed: int = 42,
) -> dict[str, Any]:
    """Split by flow-pair connected components; emit train/val/test tx + flow CSVs."""
    edges = [
        (str(r.get("src_flow_id") or "").strip(), str(r.get("dst_flow_id") or "").strip())
        for _, r in celer_fl.iterrows()
        if str(r.get("src_flow_id") or "").strip() and str(r.get("dst_flow_id") or "").strip()
    ]
    comps = _flow_pair_connected_components(edges)
    rng = np.random.default_rng(int(seed))
    tr_keys, va_keys, te_keys = _split_flow_pair_keys_by_components(comps, rng=rng)

    tx_flow = _tx_to_flow(tfm)
    mapped = _annotate_celer_tx_with_flows(celer_tx, tx_flow)
    ok = mapped.dropna(subset=["_sf", "_df"]).copy()
    ok["_k"] = list(zip(ok["_sf"].astype(str), ok["_df"].astype(str)))

    def _tx_for(keys: set[tuple[str, str]]) -> pd.DataFrame:
        sub = ok[ok["_k"].map(lambda x: x in keys)].copy()
        return sub.drop(columns=["_sh", "_dh", "_sf", "_df", "_k"], errors="ignore")

    tr_tx, va_tx, te_tx = _tx_for(tr_keys), _tx_for(va_keys), _tx_for(te_keys)
    tr_tx.to_csv(output_file(out_root, "celer_train_tx_labels.csv"), index=False)
    va_tx.to_csv(output_file(out_root, "celer_val_tx_labels.csv"), index=False)
    te_tx.to_csv(output_file(out_root, "celer_test_tx_labels.csv"), index=False)
    tr_tx.to_csv(output_file(out_root, "tx_anchor_labels_from_celer_train.csv"), index=False)
    va_tx.to_csv(output_file(out_root, "tx_anchor_labels_from_celer_val.csv"), index=False)
    te_tx.to_csv(output_file(out_root, "tx_anchor_labels_from_celer_test.csv"), index=False)

    build_flow_labels_from_celer(tr_tx, eth_seg, bnb_seg, tfm, out_root, out_filename="flow_labels_from_celer_train.csv")
    build_flow_labels_from_celer(va_tx, eth_seg, bnb_seg, tfm, out_root, out_filename="flow_labels_from_celer_val.csv")
    build_flow_labels_from_celer(te_tx, eth_seg, bnb_seg, tfm, out_root, out_filename="flow_labels_from_celer_test.csv")

    tr_fl = pd.read_csv(output_file(out_root, "flow_labels_from_celer_train.csv"), dtype=str, keep_default_na=False)
    va_fl = pd.read_csv(output_file(out_root, "flow_labels_from_celer_val.csv"), dtype=str, keep_default_na=False)
    te_fl = pd.read_csv(output_file(out_root, "flow_labels_from_celer_test.csv"), dtype=str, keep_default_na=False)

    def _fk(df: pd.DataFrame) -> set[tuple[str, str]]:
        if df.empty or "src_flow_id" not in df.columns:
            return set()
        return {
            (str(a).strip(), str(b).strip())
            for a, b in zip(df["src_flow_id"], df["dst_flow_id"])
            if str(a).strip() and str(b).strip()
        }

    tr_fk, te_fk = _fk(tr_fl), _fk(te_fl)
    va_fk = _fk(va_fl)
    ov_tt = _train_val_test_overlap_stats(tr_fk, te_fk)
    return {
        "train_flow_label_count": int(len(tr_fl)),
        "val_flow_label_count": int(len(va_fl)),
        "test_flow_label_count": int(len(te_fl)),
        "heldout_train_tx_rows": int(len(tr_tx)),
        "heldout_val_tx_rows": int(len(va_tx)),
        "heldout_test_tx_rows": int(len(te_tx)),
        "heldout_connected_components": int(len(comps)),
        **ov_tt,
        "train_val_src_flow_overlap_count": int(len({a for a, _ in tr_fk} & {a for a, _ in va_fk})),
        "train_val_flow_pair_overlap_count": int(len(tr_fk & va_fk)),
    }


def _copy_canonical(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def resolve_flow_labels_path_for_mode(out_root: Path, mode: str) -> Path:
    mode = (mode or "weak_only").strip().lower()
    if mode == "weak_only":
        p = output_file(out_root, "flow_labels_weak.csv")
        return p if p.is_file() else output_file(out_root, "flow_labels.csv")
    if mode == "celer_only":
        return output_file(out_root, "flow_labels_from_celer.csv")
    if mode == "merged":
        return output_file(out_root, "flow_labels_merged.csv")
    if mode == "heldout_celer":
        return output_file(out_root, "flow_labels_from_celer_test.csv")
    raise ValueError(f"Unknown label_source_mode: {mode}")


def resolve_tx_anchor_path_for_mode(out_root: Path, mode: str) -> Path | None:
    mode = (mode or "weak_only").strip().lower()
    if mode == "weak_only":
        p = output_file(out_root, "tx_anchor_labels_weak.csv")
        return p if p.is_file() else output_file(out_root, "tx_anchor_labels.csv")
    if mode == "celer_only":
        p = output_file(out_root, "tx_anchor_labels_from_celer.csv")
        return p if p.is_file() else None
    if mode == "merged":
        p = output_file(out_root, "tx_anchor_labels_merged.csv")
        return p if p.is_file() else None
    if mode == "heldout_celer":
        p = output_file(out_root, "tx_anchor_labels_from_celer_test.csv")
        return p if p.is_file() else None
    return None


def apply_label_source_mode_to_canonical(out_root: Path, mode: str) -> tuple[Path, Path | None, str, str | None]:
    """Install canonical ``labels/flow_labels.csv`` and ``labels/tx_anchor_labels.csv``; return source paths."""
    mode = (mode or "weak_only").strip().lower()
    flow_dst = output_file(out_root, "flow_labels.csv")
    tx_dst = output_file(out_root, "tx_anchor_labels.csv")
    flow_src_desc = ""
    tx_src_desc: str | None = None

    if mode == "weak_only":
        src = output_file(out_root, "flow_labels_weak.csv")
        if src.is_file():
            _copy_canonical(src, flow_dst)
            flow_src_desc = str(src.resolve())
        else:
            flow_src_desc = "labels/flow_labels_weak.csv missing; canonical flow_labels.csv not overwritten"
        tsrc = output_file(out_root, "tx_anchor_labels_weak.csv")
        if tsrc.is_file():
            _copy_canonical(tsrc, tx_dst)
            tx_src_desc = str(tsrc.resolve())
        return flow_dst, tx_dst if tx_dst.is_file() else None, flow_src_desc, tx_src_desc

    if mode == "celer_only":
        src = output_file(out_root, "flow_labels_from_celer.csv")
        if not src.is_file():
            raise FileNotFoundError(f"celer_only requires {src}")
        _copy_canonical(src, flow_dst)
        flow_src_desc = str(src.resolve())
        tsrc = output_file(out_root, "tx_anchor_labels_from_celer.csv")
        if tsrc.is_file():
            _copy_canonical(tsrc, tx_dst)
            tx_src_desc = str(tsrc.resolve())
        return flow_dst, tx_dst if tx_dst.is_file() else None, flow_src_desc, tx_src_desc

    if mode == "merged":
        src = output_file(out_root, "flow_labels_merged.csv")
        if not src.is_file():
            raise FileNotFoundError(f"merged requires {src}")
        _copy_canonical(src, flow_dst)
        flow_src_desc = str(src.resolve())
        tsrc = output_file(out_root, "tx_anchor_labels_merged.csv")
        if tsrc.is_file():
            _copy_canonical(tsrc, tx_dst)
            tx_src_desc = str(tsrc.resolve())
        return flow_dst, tx_dst if tx_dst.is_file() else None, flow_src_desc, tx_src_desc

    if mode == "heldout_celer":
        src = output_file(out_root, "flow_labels_from_celer_test.csv")
        if not src.is_file():
            raise FileNotFoundError(f"heldout_celer requires {src}")
        _copy_canonical(src, flow_dst)
        flow_src_desc = str(src.resolve())
        tsrc = output_file(out_root, "tx_anchor_labels_from_celer_test.csv")
        if tsrc.is_file():
            _copy_canonical(tsrc, tx_dst)
            tx_src_desc = str(tsrc.resolve())
        return flow_dst, tx_dst if tx_dst.is_file() else None, flow_src_desc, tx_src_desc

    raise ValueError(f"Unknown label_source_mode: {mode}")


def print_label_bundle_summary_to_log(out_root: Path) -> None:
    """Log one-line summary from ``labels/label_diagnostics.json`` (for ``--print-label-diagnostics``)."""
    p = output_file(out_root, "label_diagnostics.json")
    if not p.is_file():
        logger.warning("label_diagnostics.json missing: %s", p)
        return
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        logger.warning("label_diagnostics.json unreadable: %s", p)
        return
    cf = output_file(out_root, "flow_labels.csv")
    n_can = 0
    if cf.is_file():
        try:
            n_can = int(len(pd.read_csv(cf, dtype=str, keep_default_na=False)))
        except OSError:
            n_can = -1
    logger.info(
        "Label bundle summary | weak_flow_label_count=%s celer_valid_tx_pairs=%s celer_pairs_mapped_to_both_flows=%s "
        "celer_flow_label_count=%s merged_flow_label_count=%s canonical_flow_label_count=%s canonical_label_source=%s "
        "label_source_mode_applied=%s fallback_occurred=%s fallback_reason=%s",
        d.get("weak_flow_label_count"),
        d.get("celer_valid_tx_pairs"),
        d.get("celer_pairs_mapped_to_both_flows"),
        d.get("celer_flow_label_count"),
        d.get("merged_flow_label_count"),
        n_can,
        d.get("canonical_label_source"),
        d.get("label_source_mode_applied"),
        d.get("fallback_occurred"),
        d.get("fallback_reason"),
    )


def finalize_multi_source_label_bundle(
    out_root: Path,
    *,
    celer_label_csv: Path | str | None,
    label_source_mode: str,
    weak_flow_label_count: int | None = None,
    paper_mode: bool = False,
) -> dict[str, Any]:
    """Build celer / weak / merged artifacts, diagnostics, and canonical ``flow_labels.csv`` per mode."""
    out_root = Path(out_root)
    requested_mode = (label_source_mode or "celer_only_if_available").strip().lower()
    celer_resolved, celer_tried = resolve_celer_label_csv_path(CROSS_ROOT, celer_label_csv)
    requested_mode, mode, fallback_occurred, fallback_reason = resolve_effective_label_source_mode(
        requested_mode, celer_resolved=celer_resolved
    )

    if paper_mode and requested_mode in ("celer_only", "heldout_celer") and celer_resolved is None:
        raise RuntimeError(
            "paper_mode / paper_experiment_closure: Celer supervision CSV is required for "
            f"label_source_mode={requested_mode!r}. Tried paths: {celer_tried}"
        )

    weak_tx_p = output_file(out_root, "tx_anchor_labels_weak.csv")
    if not weak_tx_p.is_file():
        weak_tx_p = output_file(out_root, "tx_anchor_labels.csv")
    weak_fl_p = output_file(out_root, "flow_labels_weak.csv")
    if not weak_fl_p.is_file():
        weak_fl_p = output_file(out_root, "flow_labels.csv")
    if not weak_tx_p.is_file() or not weak_fl_p.is_file():
        raise FileNotFoundError(f"finalize_multi_source_label_bundle: need {weak_tx_p} and {weak_fl_p}")

    weak_tx = pd.read_csv(weak_tx_p, dtype=str, keep_default_na=False)
    weak_fl = pd.read_csv(weak_fl_p, dtype=str, keep_default_na=False)
    weak_tx_en = _enrich_weak_tx_anchors(weak_tx)
    weak_fl_en = _enrich_weak_flow_labels(weak_fl)
    output_file(out_root, "tx_anchor_labels_weak.csv").parent.mkdir(parents=True, exist_ok=True)
    weak_tx_en.to_csv(output_file(out_root, "tx_anchor_labels_weak.csv"), index=False)
    weak_fl_en.to_csv(output_file(out_root, "flow_labels_weak.csv"), index=False)

    celer_tx = pd.DataFrame()
    celer_fl = pd.DataFrame()
    celer_load_diag: dict[str, Any] = {}
    held_diag: dict[str, Any] = {}

    if celer_resolved is not None and celer_resolved.is_file():
        celer_tx = load_celer_tx_labels(celer_resolved, out_root)
        try:
            celer_load_diag = json.loads(
                output_file(out_root, "celer_tx_load_diagnostics.json").read_text(encoding="utf-8")
            )
        except OSError:
            celer_load_diag = {}
        eth_seg = pd.read_csv(output_file(out_root, "flow_segments_eth.csv"), dtype=str, keep_default_na=False)
        bnb_seg = pd.read_csv(output_file(out_root, "flow_segments_bnb.csv"), dtype=str, keep_default_na=False)
        tfm = pd.read_csv(output_file(out_root, "tx_to_flow_map.csv"), dtype=str, keep_default_na=False)
        celer_fl = build_flow_labels_from_celer(
            celer_tx, eth_seg, bnb_seg, tfm, out_root, out_filename="flow_labels_from_celer.csv"
        )
        held_diag = _write_celer_heldout_from_flow_graph(
            out_root, celer_tx=celer_tx, celer_fl=celer_fl, eth_seg=eth_seg, bnb_seg=bnb_seg, tfm=tfm, seed=42
        )
    elif not paper_mode or requested_mode not in ("celer_only", "heldout_celer"):
        logger.warning(
            "Celer tx label file missing or not provided (tried %s); supervised outputs empty for this run.",
            celer_tried,
        )

    if paper_mode and requested_mode in ("celer_only", "heldout_celer"):
        fl_path = output_file(out_root, "flow_labels_from_celer.csv")
        if not fl_path.is_file():
            raise RuntimeError(
                f"paper_mode: expected {fl_path} for label_source_mode={requested_mode!r}"
            )
        if celer_fl.empty or len(celer_fl) == 0:
            raise RuntimeError(
                f"paper_mode: Celer supervision produced zero mapped flow pairs for {requested_mode!r}"
            )
        if requested_mode == "heldout_celer":
            te_p = output_file(out_root, "flow_labels_from_celer_test.csv")
            if not te_p.is_file():
                raise RuntimeError(f"paper_mode: heldout_celer requires {te_p}")
            te_rows = pd.read_csv(te_p, dtype=str, keep_default_na=False)
            if te_rows.empty or len(te_rows) == 0:
                raise RuntimeError(
                    f"paper_mode: heldout_celer requires at least one test flow-pair row in {te_p}"
                )

    merged_tx = merge_tx_anchor_labels(celer_tx, weak_tx_en)
    merged_tx.to_csv(output_file(out_root, "tx_anchor_labels_merged.csv"), index=False)
    merged_fl = merge_flow_labels(celer_fl, weak_fl_en)
    merged_fl.to_csv(output_file(out_root, "flow_labels_merged.csv"), index=False)

    if requested_mode in ("celer_only", "heldout_celer") and mode != "weak_only":
        sup_ok = bool(celer_resolved and celer_resolved.is_file() and not celer_fl.empty)
        if not sup_ok and not paper_mode:
            mode = "weak_only"
            fallback_occurred = True
            if celer_resolved is None or not celer_resolved.is_file():
                fallback_reason = "celer_label.csv not found after search (dev weak_only fallback)"
            elif celer_tx.empty:
                fallback_reason = "celer_label.csv produced zero valid ETH↔BNB tx pairs (dev weak_only fallback)"
            else:
                fallback_reason = "zero Celer flow-pair labels after tx→flow mapping (dev weak_only fallback)"
            logger.warning("%s", _CELER_WEAK_FALLBACK_WARNING)
            logger.warning("label_source_mode %s unavailable: %s", requested_mode, fallback_reason)

    if (
        requested_mode == "heldout_celer"
        and not paper_mode
        and mode != "weak_only"
        and not celer_fl.empty
    ):
        te_p = output_file(out_root, "flow_labels_from_celer_test.csv")
        te_n = 0
        if te_p.is_file():
            try:
                te_n = len(pd.read_csv(te_p, dtype=str, keep_default_na=False))
            except OSError:
                te_n = 0
        if te_n == 0:
            mode = "weak_only"
            fallback_occurred = True
            fallback_reason = (
                "heldout_celer produced zero test flow-pair rows after component split (dev weak_only fallback)"
            )
            logger.warning("label_source_mode heldout_celer unavailable: %s", fallback_reason)

    mapped_src = mapped_dst = mapped_both = 0
    unmapped_src = unmapped_dst = 0
    if not celer_tx.empty:
        tfm2 = pd.read_csv(output_file(out_root, "tx_to_flow_map.csv"), dtype=str, keep_default_na=False)
        tx_flow = _tx_to_flow(tfm2)
        for _, r in celer_tx.iterrows():
            sh = normalize_tx_hash(str(r.get("src_tx_hash") or ""))
            dh = normalize_tx_hash(str(r.get("dst_tx_hash") or ""))
            hs, hd = sh in tx_flow, dh in tx_flow
            if hs:
                mapped_src += 1
            else:
                unmapped_src += 1
            if hd:
                mapped_dst += 1
            else:
                unmapped_dst += 1
            if hs and hd:
                mapped_both += 1

    tx_per: list[float] = []
    if not celer_fl.empty and "matched_tx_count" in celer_fl.columns:
        tx_per = pd.to_numeric(celer_fl["matched_tx_count"], errors="coerce").fillna(0).tolist()

    prev_weak = int(weak_flow_label_count) if weak_flow_label_count is not None else int(len(weak_fl_en))
    diag: dict[str, Any] = {
        "label_source_mode_requested": requested_mode,
        "label_source_mode_applied": mode,
        "celer_label_csv_resolved": str(celer_resolved) if celer_resolved else "",
        "celer_label_csv_search_tried": celer_tried,
        "paper_mode": bool(paper_mode),
        "celer_total_raw_rows": int(celer_load_diag.get("celer_total_raw_rows", 0)),
        "celer_valid_tx_pairs": int(len(celer_tx)),
        "celer_duplicate_tx_pairs": int(celer_load_diag.get("celer_duplicate_tx_pairs", 0)),
        "celer_invalid_rows": int(celer_load_diag.get("celer_invalid_rows", 0)),
        "celer_pairs_mapped_to_src_flow": int(mapped_src),
        "celer_pairs_mapped_to_dst_flow": int(mapped_dst),
        "celer_pairs_mapped_to_both_flows": int(mapped_both),
        "celer_pairs_unmapped_src_tx_count": int(unmapped_src),
        "celer_pairs_unmapped_dst_tx_count": int(unmapped_dst),
        "celer_pairs_unmapped_either_side_count": int(len(celer_tx) - mapped_both),
        "celer_flow_label_count": int(len(celer_fl)),
        "celer_avg_tx_per_flow_label": float(np.mean(tx_per)) if tx_per else 0.0,
        "celer_max_tx_per_flow_label": int(max(tx_per)) if tx_per else 0,
        "weak_tx_anchor_count": int(len(weak_tx_en)),
        "weak_flow_label_count": int(len(weak_fl_en)),
        "merged_tx_anchor_count": int(len(merged_tx)),
        "merged_flow_label_count": int(len(merged_fl)),
        "supervised_flow_label_ratio": float(len(celer_fl) / max(len(merged_fl), 1)),
        "weak_flow_label_ratio": float(len(weak_fl_en) / max(len(merged_fl), 1)),
        "flow_label_count_from_previous_pipeline": int(prev_weak),
        "flow_label_count_difference": int(len(merged_fl) - prev_weak),
        "evaluation_label_source": LABEL_SOURCE_CELER
        if mode == "celer_only"
        else (LABEL_SOURCE_MERGED if mode == "merged" else ("heldout_celer" if mode == "heldout_celer" else LABEL_SOURCE_WEAK)),
        "fallback_occurred": fallback_occurred,
        "fallback_reason": fallback_reason,
        "train_flow_label_count": int(held_diag.get("train_flow_label_count", 0)),
        "val_flow_label_count": int(held_diag.get("val_flow_label_count", 0)),
        "test_flow_label_count": int(held_diag.get("test_flow_label_count", 0)),
        "train_test_src_flow_overlap_count": int(held_diag.get("train_test_src_flow_overlap_count", 0)),
        "train_test_dst_flow_overlap_count": int(held_diag.get("train_test_dst_flow_overlap_count", 0)),
        "train_test_flow_pair_overlap_count": int(held_diag.get("train_test_flow_pair_overlap_count", 0)),
        "split_leakage_detected": bool(held_diag.get("split_leakage_detected", False)),
        "heldout_connected_components": int(held_diag.get("heldout_connected_components", 0)),
    }

    _, _, flow_src_desc, tx_src_desc = apply_label_source_mode_to_canonical(out_root, mode)
    cf = output_file(out_root, "flow_labels.csv")
    canonical_n = 0
    if cf.is_file():
        try:
            canonical_n = int(len(pd.read_csv(cf, dtype=str, keep_default_na=False)))
        except OSError:
            canonical_n = -1
    diag["canonical_flow_labels_source"] = flow_src_desc
    diag["canonical_tx_anchor_labels_source"] = tx_src_desc or ""
    diag["canonical_flow_label_count"] = int(canonical_n)
    diag["canonical_label_source"] = str(mode)

    diag_path = output_file(out_root, "label_diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    logger.info(
        "Label bundle: requested=%s applied=%s canonical_flow=%s | Celer tx=%d mapped_both=%d "
        "celer_flow=%d weak_flow=%d merged_flow=%d canonical_rows=%d fallback=%s",
        requested_mode,
        mode,
        flow_src_desc,
        int(len(celer_tx)),
        int(mapped_both),
        int(len(celer_fl)),
        int(len(weak_fl_en)),
        int(len(merged_fl)),
        int(canonical_n),
        fallback_occurred,
    )
    print_label_bundle_summary_to_log(out_root)
    return diag


def patch_uot_evaluation_label_source(out_root: Path, evaluation_label_source: str) -> None:
    """Attach ``evaluation_label_source`` to ``eval/uot_evaluation_metrics.json`` when present."""
    p = output_file(out_root, "uot_evaluation_metrics.json")
    if not p.is_file():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        return
    data["evaluation_label_source"] = str(evaluation_label_source)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
