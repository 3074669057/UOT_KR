"""Data adapter for M1 ablation experiments.

Auto-discovers data from data/ directory and loads:
  - Ground truth transaction pairs
  - Source and target transactions
  - Pre-computed cost matrix and flow segments (from out/ if available,
    or constructs from data/ if not)
  - Temporal/causal feasibility mask

Uses cached production outputs from out/uot_delay_fixed_production/ for
cost matrix and flow segments when available, since these require the
full pipeline (AML scoring, graph embeddings, route classification, etc.).
The raw transaction data and ground truth labels are always loaded from data/.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.shared.normalize import norm_addr
from cross.domain.uot.delay_policy import flow_pair_delay_sec, DEFAULT_TIME_DELAY_POLICY

_REPO = Path(__file__).resolve().parents[5]


@dataclass
class M1Dataset:
    """Standardized dataset for M1 solver ablation experiments."""
    # Core matrices
    C: np.ndarray                        # Cost matrix [n_src, n_dst]
    feasible_mask: np.ndarray            # Causal/temporal feasibility mask
    source_mass: np.ndarray | None       # Source marginals
    target_mass: np.ndarray | None       # Target marginals

    # Flow data
    source_flows: list[dict[str, Any]]
    target_flows: list[dict[str, Any]]

    # Transaction data for decode
    src_all: pd.DataFrame
    dst_norm: pd.DataFrame

    # Timestamps
    eth_ts: dict[str, float]
    bnb_ts: dict[str, float]

    # Ground truth
    ground_truth: dict[str, str]         # {src_tx: dst_tx}
    ground_truth_df: pd.DataFrame

    # Split indices (for calibration vs evaluation)
    train_indices: Any = None
    val_indices: Any = None
    test_indices: Any = None

    # Metadata
    data_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def n_source(self) -> int:
        return int(self.C.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.C.shape[1])

    @property
    def n_ground_truth(self) -> int:
        return len(self.ground_truth)


def discover_data(
    data_root: str | Path = "data",
    production_root: str | Path | None = None,
    *,
    auto_discover: bool = True,
    config: dict[str, Any] | None = None,
) -> M1Dataset:
    """Auto-discover and load M1 dataset from data/ and production outputs.

    Args:
        data_root: Root directory for raw data files.
        production_root: Root for production pipeline outputs (cached cost matrix, flows).
            If None, defaults to "out/uot_delay_fixed_production".
        auto_discover: If True, search for files automatically.
        config: Optional config overrides for specific paths.

    Returns:
        M1Dataset ready for solver ablation experiments.
    """
    data_root = Path(data_root)
    if production_root is None:
        prod_root = _REPO / "out" / "uot_delay_fixed_production"
    else:
        prod_root = Path(production_root)

    cfg = config or {}
    summary: dict[str, Any] = {"data_root": str(data_root.resolve())}

    # ---- Step 1: Discover ground truth labels ----
    gt_path = _resolve_path(cfg.get("ground_truth_path"), [
        data_root / "label" / "celer_label.csv",
        data_root / "label" / "tx" / "Celer_ETH_BNB_completed.csv",
    ], "ground truth labels")
    if gt_path is None:
        raise FileNotFoundError(
            f"Ground truth labels not found in {data_root}/label/. "
            f"Expected celer_label.csv or similar. "
            f"Set data.ground_truth_path in config."
        )
    gt_df = pd.read_csv(gt_path)
    # Normalize column names
    src_col = next((c for c in gt_df.columns if "src" in c.lower() and ("tx" in c.lower() or "hash" in c.lower())), None)
    dst_col = next((c for c in gt_df.columns if "dst" in c.lower() and ("tx" in c.lower() or "hash" in c.lower())), None)
    if src_col is None or dst_col is None:
        raise ValueError(f"Cannot find src/dst txhash columns in {gt_path}. Columns: {list(gt_df.columns)}")
    gt_df = gt_df.rename(columns={src_col: "srcTxHash", dst_col: "dstTxHash"})
    ground_truth = {norm_addr(str(r["srcTxHash"])): norm_addr(str(r["dstTxHash"]))
                    for _, r in gt_df.iterrows() if r["srcTxHash"] and r["dstTxHash"]}
    summary["ground_truth_path"] = str(gt_path.resolve())
    summary["n_ground_truth"] = len(ground_truth)

    # ---- Step 2: Load transaction data ----
    eth_path = _resolve_path(cfg.get("source_transactions_path"), [
        data_root / "in" / "Celer_ETH_cun.csv",
        data_root / "label" / "tx" / "Celer_ETH_cun.csv",
    ], "ETH transactions")
    bnb_path = _resolve_path(cfg.get("target_transactions_path"), [
        data_root / "label" / "tx" / "Celer_BNB_qu.csv",
    ], "BNB transactions")
    if eth_path is None or bnb_path is None:
        raise FileNotFoundError(
            f"Transaction files not found. ETH: {eth_path}, BNB: {bnb_path}"
        )
    eth_df = pd.read_csv(eth_path, low_memory=False)
    bnb_df = pd.read_csv(bnb_path, low_memory=False)
    summary["eth_transactions_path"] = str(eth_path.resolve())
    summary["bnb_transactions_path"] = str(bnb_path.resolve())
    summary["n_eth_transactions"] = len(eth_df)
    summary["n_bnb_transactions"] = len(bnb_df)

    # ---- Step 3: Load flow segments and cost matrix ----
    flow_eth_path = prod_root / "uot" / "uot_flow_segments_eth.csv"
    flow_bnb_path = prod_root / "uot" / "uot_flow_segments_bnb.csv"
    cost_path = prod_root / "uot" / "uot_cost_matrix.npz"
    transport_path = prod_root / "uot" / "uot_transport_matrix.npz"

    using_cached = (flow_eth_path.is_file() and flow_bnb_path.is_file() and cost_path.is_file())
    summary["cost_matrix_source"] = "cached" if using_cached else "construct"

    if using_cached:
        from cross.application.experiments.uot_cache_utils import load_flow_segments
        source_flows = load_flow_segments(flow_eth_path)
        target_flows = load_flow_segments(flow_bnb_path)
        cost_data = np.load(cost_path)
        C = np.asarray(cost_data["C_effective"] if "C_effective" in cost_data.files
                       else list(cost_data.values())[0], dtype=float)

        # Load cached source/target masses if available
        if transport_path.is_file():
            tm = np.load(transport_path)
            source_mass = np.asarray(tm["source_mass_risk_weighted"], dtype=float)
            target_mass = np.asarray(tm["target_mass_evidence_weighted"], dtype=float)
            summary["mass_source"] = "cached_transport"
        else:
            source_mass = np.ones(len(source_flows), dtype=float)
            target_mass = np.ones(len(target_flows), dtype=float)
            summary["mass_source"] = "uniform"

        summary["flow_source"] = f"cached ({prod_root})"
    else:
        # Construct flows from raw transactions (fallback - limited features)
        source_flows = _build_simple_flows(eth_df, "ETH")
        target_flows = _build_simple_flows(bnb_df, "BNB")
        from cross.domain.uot.cost_matrix import build_cost_matrix
        C = build_cost_matrix(source_flows, target_flows)
        source_mass = None
        target_mass = None
        summary["flow_source"] = "constructed_from_transactions"

    summary["n_source_flows"] = len(source_flows)
    summary["n_target_flows"] = len(target_flows)
    summary["C_shape"] = list(C.shape)

    # ---- Step 4: Build causal feasibility mask ----
    mask_path = None
    if cfg.get("feasible_mask_path"):
        mask_path = Path(cfg["feasible_mask_path"])
    if mask_path and mask_path.is_file():
        feasible_mask = np.load(mask_path) if mask_path.suffix == ".npy" or mask_path.suffix == ".npz" else np.ones(C.shape, dtype=bool)
        summary["feasible_mask_source"] = f"loaded ({mask_path})"
    else:
        feasible_mask = _build_causal_mask(source_flows, target_flows)
        summary["feasible_mask_source"] = "constructed_from_flow_delays"
    summary["feasible_mask_density"] = float(feasible_mask.sum()) / float(feasible_mask.size)

    # ---- Step 5: Build timestamp lookup ----
    eth_ts = _build_timestamp_map(eth_df)
    bnb_ts = _build_timestamp_map(bnb_df)
    summary["eth_ts_entries"] = len(eth_ts)
    summary["bnb_ts_entries"] = len(bnb_ts)

    # ---- Step 6: Build src_all and dst_norm DataFrames ----
    from cross.shared.transfers import eth_df_to_src_txs
    flow_txs = {norm_addr(str(h)) for f in source_flows for h in (f.get("tx_hashes") or [])}
    src_all = eth_df_to_src_txs(eth_df)
    if flow_txs and "txhash" in src_all.columns:
        src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)

    # ---- Step 7: Build split if needed ----
    seed = int(cfg.get("split_seed", 2026))
    val_frac = float(cfg.get("val_fraction", 0.2))
    test_frac = float(cfg.get("test_fraction", 0.2))
    gt_srcs = list(ground_truth.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(gt_srcs)
    n_test = int(len(gt_srcs) * test_frac)
    n_val = int(len(gt_srcs) * val_frac)
    test_srcs = set(gt_srcs[:n_test])
    val_srcs = set(gt_srcs[n_test:n_test + n_val])
    summary["split"] = f"seed={seed}, val={val_frac}, test={test_frac}"
    summary["n_val"] = len(val_srcs)
    summary["n_test"] = len(test_srcs)

    # ---- Step 8: Structure distribution ----
    structure_dist = _compute_structure_distribution(source_flows, target_flows, ground_truth)
    summary["structure_distribution"] = structure_dist

    return M1Dataset(
        C=C,
        feasible_mask=feasible_mask,
        source_mass=source_mass,
        target_mass=target_mass,
        source_flows=source_flows,
        target_flows=target_flows,
        src_all=src_all,
        dst_norm=dst_norm,
        eth_ts=eth_ts,
        bnb_ts=bnb_ts,
        ground_truth=ground_truth,
        ground_truth_df=gt_df,
        val_indices=val_srcs,
        test_indices=test_srcs,
        data_summary=summary,
    )


def _resolve_path(explicit: Any, candidates: list[Path], label: str) -> Path | None:
    """Resolve a file path from explicit config or candidates."""
    if explicit and str(explicit) not in ("null", "None", ""):
        p = Path(explicit)
        if p.is_file():
            return p
    for c in candidates:
        if c.is_file():
            return c
    return None


def _build_timestamp_map(df: pd.DataFrame) -> dict[str, float]:
    """Build {txhash: timestamp} from transaction DataFrame."""
    ts_map: dict[str, float] = {}
    ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "time_stamp", "block_timestamp")), None)
    hash_col = next((c for c in df.columns if "hash" in str(c).lower()), None)
    if ts_col and hash_col:
        for _, r in df.iterrows():
            h = norm_addr(str(r.get(hash_col, "")))
            ts = float(r.get(ts_col, 0.0))
            if h and ts > 0:
                ts_map[h] = ts
    return ts_map


def _build_causal_mask(
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    policy: str = DEFAULT_TIME_DELAY_POLICY,
) -> np.ndarray:
    """Build causal feasibility mask from flow-pair delays."""
    n_src, n_dst = len(source_flows), len(target_flows)
    mask = np.zeros((n_src, n_dst), dtype=bool)
    for i in range(n_src):
        for j in range(n_dst):
            d = flow_pair_delay_sec(source_flows[i], target_flows[j], policy=policy)
            mask[i, j] = float(d) >= 0.0
    return mask


def _build_simple_flows(df: pd.DataFrame, chain: str) -> list[dict[str, Any]]:
    """Build minimal flow dicts from raw transactions (fallback when no cached flows)."""
    hash_col = next((c for c in df.columns if "hash" in str(c).lower()), None)
    ts_col = next((c for c in df.columns if "timestamp" in str(c).lower()), None)
    val_col = next((c for c in df.columns if "value" in str(c).lower()), None)

    if hash_col is None:
        raise ValueError(f"No hash column found in {list(df.columns)}")

    flows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        h = str(r.get(hash_col, ""))
        if not h:
            continue
        ts = float(r.get(ts_col, 0.0)) if ts_col else 0.0
        try:
            val = float(r.get(val_col, 0.0)) if val_col else 0.0
        except (ValueError, TypeError):
            val = 0.0
        flows.append({
            "flow_id": h,
            "chain": chain,
            "tx_hashes": [h],
            "start_time": ts,
            "end_time": ts,
            "amount_usd": val,
            "aml_score": 0.0,
            "evidence_quality_score": 0.65,
            "route_type": "unknown",
            "bridge_contract_hit": False,
            "address_set": [],
            "graph_embedding": None,
            "evidence_level": None,
            "price_snapshot_ok": True,
        })
    return flows


def _compute_structure_distribution(
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    ground_truth: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Compute ground-truth structure distribution at multiple levels.

    Returns multi-level: {"tx": {...}, "flow": {...}}.
    tx-level matches paper (72.32% 1-1, 27.51% m-1, 0.17% 1-m).
    """
    from cross.domain.uot.m1_ablation.evaluation import (
        assign_structure_label_tx,
        assign_structure_label_flow,
    )
    from collections import Counter

    # tx-level (paper definition)
    tx_labels = assign_structure_label_tx(ground_truth)
    tx_dist = dict(Counter(tx_labels.values()))
    tx_out = {k: tx_dist.get(k, 0) for k in ["1-1", "m-1", "1-m", "m-n", "unmatched", "other"]}

    # flow-level
    flow_labels = assign_structure_label_flow(ground_truth, source_flows, target_flows)
    flow_dist = dict(Counter(flow_labels.values()))
    flow_out = {k: flow_dist.get(k, 0) for k in ["1-1", "m-1", "1-m", "m-n", "unmatched", "other"]}

    # quotient-group-level: group by source flow index
    tx_to_src_flow: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            h = str(txh).strip().lower()
            if h and h != "0x" and len(h) >= 10:
                tx_to_src_flow[h] = i

    tx_to_dst_flow: dict[str, set[int]] = {}
    for j, tf in enumerate(target_flows):
        for txh in tf.get("tx_hashes") or []:
            h = str(txh).strip().lower()
            if h and h != "0x" and len(h) >= 10:
                tx_to_dst_flow.setdefault(h, set()).add(j)

    # quotient group: source flow -> set of target flows (via gt tx pairs)
    qg_pairs: dict[int, set[int]] = {}
    for s, d in ground_truth.items():
        si = tx_to_src_flow.get(s, -1)
        djs = tx_to_dst_flow.get(d, set())
        if si >= 0 and djs:
            qg_pairs.setdefault(si, set()).update(djs)

    qg_adj: dict[int, set[int]] = {}
    for si, djs in qg_pairs.items():
        qg_adj[si] = djs
    # Compute in-degree (how many source flows map to each target flow)
    dst_to_srcs: dict[int, set[int]] = {}
    for si, djs in qg_adj.items():
        for dj in djs:
            dst_to_srcs.setdefault(dj, set()).add(si)

    qg_dist = {"1-1": 0, "m-1": 0, "1-m": 0, "m-n": 0, "unmatched": 0, "other": 0}
    for si, djs in qg_adj.items():
        out_deg = len(djs)
        if out_deg == 0:
            qg_dist["unmatched"] += 1
            continue
        in_degs = [len(dst_to_srcs.get(dj, set())) for dj in djs]
        max_in = max(in_degs) if in_degs else 1
        if out_deg == 1 and max_in == 1:
            qg_dist["1-1"] += 1
        elif out_deg == 1 and max_in > 1:
            qg_dist["m-1"] += 1
        elif out_deg > 1 and max_in == 1:
            qg_dist["1-m"] += 1
        elif out_deg > 1 and max_in > 1:
            qg_dist["m-n"] += 1
        else:
            qg_dist["other"] += 1

    return {"tx": tx_out, "flow": flow_out, "quotient_group": qg_dist}

def print_data_summary(dataset: M1Dataset) -> str:
    """Generate a printable data summary string."""
    s = dataset.data_summary
    lines = [
        "# M1 Solver Ablation - Data Summary",
        "",
        f"**Data root**: `{s.get('data_root', '?')}`",
        "",
        "## File Paths",
        f"- Ground truth: `{s.get('ground_truth_path', '?')}`",
        f"- ETH transactions: `{s.get('eth_transactions_path', '?')}`",
        f"- BNB transactions: `{s.get('bnb_transactions_path', '?')}`",
        "",
        "## Counts",
        f"- Ground truth pairs: {s.get('n_ground_truth', 0)}",
        f"- ETH transactions: {s.get('n_eth_transactions', 0)}",
        f"- BNB transactions: {s.get('n_bnb_transactions', 0)}",
        f"- Source flows: {s.get('n_source_flows', 0)}",
        f"- Target flows: {s.get('n_target_flows', 0)}",
        "",
        "## Cost Matrix",
        f"- Shape: {s.get('C_shape', '?')}",
        f"- Source: {s.get('cost_matrix_source', '?')}",
        f"- Flow source: {s.get('flow_source', '?')}",
        "",
        "## Feasibility Mask",
        f"- Source: {s.get('feasible_mask_source', '?')}",
        f"- Density: {s.get('feasible_mask_density', 0):.4f}",
        "",
        "## Split",
        f"- {s.get('split', '?')}",
        f"- Validation src txs: {s.get('n_val', 0)}",
        f"- Test src txs: {s.get('n_test', 0)}",
        "",
        "## Structure Distribution",
    ]
    struct = s.get("structure_distribution", {})
    if isinstance(struct, dict) and "tx" in struct:
        lines.append("")
        lines.append("### Transaction-level (paper: 72.32% 1-1, 27.51% m-1, 0.17% 1-m)")
        for label in ["1-1", "m-1", "1-m", "m-n", "unmatched", "other"]:
            tx_n = struct["tx"].get(label, 0)
            pct = 100.0 * tx_n / max(s.get("n_ground_truth", 1), 1)
            lines.append("- {}: {} ({:.2f}%)".format(label, tx_n, pct))
        lines.append("")
        lines.append("### Flow-level")
        flow_d = struct["flow"]
        for label in ["1-1", "m-1", "1-m", "m-n", "unmatched", "other"]:
            lines.append("- {}: {}".format(label, flow_d.get(label, 0)))
        lines.append("")
        lines.append("### Quotient-group-level")
        qg_d = struct["quotient_group"]
        for label in ["1-1", "m-1", "1-m", "m-n", "unmatched", "other"]:
            lines.append("- {}: {}".format(label, qg_d.get(label, 0)))
    else:
        for label in ["1-1", "m-1", "1-m", "m-n", "unmatched", "other"]:
            lines.append("- {}: {}".format(label, struct.get(label, 0)))
    return "\n".join(lines)
