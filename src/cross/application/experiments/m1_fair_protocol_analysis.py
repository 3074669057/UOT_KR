"""M1 Fair Protocol Analysis.

Protocols A-E, significance tests, topology-stratified diagnosis,
mechanism metrics, leakage audit, decision gate.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.run_admissible_decoding import (
    _build_predictions, _evaluate_strategy, _truth_from_labels,
)
from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.domain.uot.delay_policy import DEFAULT_TIME_DELAY_POLICY, flow_pair_delay_sec
from cross.domain.uot.tx_decode_policy import pick_dst_tx_in_flow
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs
from cross.utils.safe_cast import safe_float

RC_STRATEGY = "joint_time_admissible_filter"
N_GT = 7296
BUDGETS = [1000, 2000, 3000, 4000, 4829, 6000, N_GT]

@dataclass
class ProtocolResult:
    solver: str; protocol: str; threshold_or_budget: float | None
    n_pred: int; precision: float | None; recall: float | None; f1: float | None
    top3_recall: float | None; tx_cvr: float | None; tp: int; fp: int; fn: int
    abstention_rate: float | None

def _utc(): return datetime.now(timezone.utc).isoformat()

def _load_paper_context():
    repo = Path(__file__).resolve().parents[4]
    uot_prod = repo / "out" / "uot_delay_fixed_production"
    eth_flows = load_flow_segments(uot_prod / "uot" / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(uot_prod / "uot" / "uot_flow_segments_bnb.csv")
    label_csv = repo / "out" / "baseline_compare" / "labels" / "gt_tx_pairs.csv"
    label_df = pd.read_csv(label_csv)
    label_df = label_df.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    truth = _truth_from_labels(label_df)
    eth_csv = repo / "in" / "Celer_ETH_cun.csv"
    bnb_csv = repo / "label" / "tx" / "Celer_BNB_qu.csv"
    eth_df = pd.read_csv(eth_csv, low_memory=False)
    bnb_df = pd.read_csv(bnb_csv, low_memory=False)
    from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)
    flow_txs = {norm_addr(str(h)) for f in eth_flows for h in (f.get("tx_hashes") or [])}
    src_all = eth_df_to_src_txs(eth_df)
    if flow_txs:
        src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)
    tx_to_j: dict[str, set[int]] = {}
    for j, tf in enumerate(bnb_flows):
        for txh in tf.get("tx_hashes") or []:
            h = norm_addr(str(txh)); tx_to_j.setdefault(h, set()).add(j)
    return {
        "eth_flows": eth_flows, "bnb_flows": bnb_flows,
        "label_df": label_df, "truth": truth,
        "eth_df": eth_df, "bnb_df": bnb_df,
        "eth_ts": eth_ts, "bnb_ts": bnb_ts,
        "src_all": src_all, "dst_norm": dst_norm, "tx_to_j": tx_to_j,
    }

print("Module header written successfully")
import csv, json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np; import pandas as pd
from cross.application.experiments.run_admissible_decoding import _build_predictions, _evaluate_strategy, _truth_from_labels
from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.domain.uot.tx_decode_policy import pick_dst_tx_in_flow
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs
from cross.utils.safe_cast import safe_float
RC_STRATEGY,N_GT,BUDGETS="joint_time_admissible_filter",7296,[1000,2000,3000,4000,4829,6000,7296]
def _utc():return datetime.now(timezone.utc).isoformat()
def _load_ctx():
    r=Path(__file__).resolve().parents[4];u=r/"out"/"uot_delay_fixed_production"
    ef=load_flow_segments(u/"uot"/"uot_flow_segments_eth.csv");bf=load_flow_segments(u/"uot"/"uot_flow_segments_bnb.csv")
    ld=pd.read_csv(r/"out"/"baseline_compare"/"labels"/"gt_tx_pairs.csv");ld=ld.rename(columns={"src_tx_hash":"srcTxHash","dst_tx_hash":"dstTxHash"})
    t=_truth_from_labels(ld);ed=pd.read_csv(r/"in"/"Celer_ETH_cun.csv",low_memory=False);bd=pd.read_csv(r/"label"/"tx"/"Celer_BNB_qu.csv",low_memory=False)
    from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
    ets,bts,_=_tx_timestamp_lookup(ed,bd)
    ftxs={norm_addr(str(h))for f in ef for h in(f.get("tx_hashes")or[])}
    sa=eth_df_to_src_txs(ed)
    if ftxs:sa=sa[sa["txhash"].astype(str).map(norm_addr).isin(ftxs)].reset_index(drop=True)
    dn=bd.copy()
    if"hash"in dn.columns:dn["hash"]=dn["hash"].astype(str).map(norm_addr)
    t2j={}
    for j,tf in enumerate(bf):
        for txh in tf.get("tx_hashes")or[]:h=norm_addr(str(txh));t2j.setdefault(h,set()).add(j)
    return{"eth_flows":ef,"bnb_flows":bf,"label_df":ld,"truth":t,"eth_df":ed,"bnb_df":bd,"eth_ts":ets,"bnb_ts":bts,"src_all":sa,"dst_norm":dn,"tx_to_j":t2j}
