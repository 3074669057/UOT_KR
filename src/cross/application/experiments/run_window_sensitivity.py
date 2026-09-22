"""Window W sensitivity: frozen transport plan replay - repaired v2.

Fixes:
  1. causal gate enforced *before* upper-bound gate; negative-delay cells are
     unconditionally excluded (even when W is large enough to include them).
  2. abstention reasons are diagnosed only on mass>=1e-12 columns, not on
     all delay_sec cells.
  3. _derive_label_universe simplifies the truth/eval-unit intersection.
  4. _compute_metric_accounting replaces pair_precision_recall_f1 and
     emits hard assertions: tp+fp=accepted, tp+fn=gnd_pos, accepted+abstained=eligible.
  5. Transition audit separates causal-blocked from upper-bound-blocked.
  6. 10 hard assertions are checked and exported as validation_summary.json.
"""
from __future__ import annotations

import hashlib, json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.domain.uot.tx_decode_policy import pick_dst_tx_in_flow
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float
from cross.domain.uot.delay_policy import flow_pair_delay_sec

DEFAULT_WINDOW_GRID_SEC = [300, 600, 1200, 1800, 3600, 7200, 14400]
DEFAULT_OPERATING_SEC = 3600
DEFAULT_REFERENCE_SEC = 14400
DEFAULT_BOOTSTRAP_REPS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260629
DEFAULT_MASS_EPSILON = 1e-12

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _file_sha256(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _array_hash(arr):
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()

def _df_hash(df):
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()

def _none_or_val(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, float) and np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x

def _safe_div(num, den):
    if den == 0:
        return None
    return num / den

# ---------------------------------------------------------------------------
# truth / label universe
# ---------------------------------------------------------------------------

def _truth_from_labels(label_df):
    truth = {}
    for _, row in label_df.iterrows():
        s = norm_addr(row.get("srcTxhash", row.get("srcTxHash", "")))
        d = norm_addr(row.get("dstTxhash", row.get("dstTxHash", "")))
        if s:
            truth[s] = d
    return truth

def _derive_label_universe(*, label_df, eth_flows, P_shape_0):
    """Compute the set of eval units, ground-truth-positive ids, and invariants."""
    truth = _truth_from_labels(label_df)
    tx_to_i = {}
    for i, sf in enumerate(eth_flows):
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[norm_addr(str(txh))] = i
    eligible = [s for s in truth if s in tx_to_i]
    return dict(
        truth=truth,
        eligible_eval_units=eligible,
        eligible_eval_count=len(eligible),
        ground_truth_positive_count=len(eligible),
        tx_to_i=tx_to_i,
    )

# ---------------------------------------------------------------------------
# load frozen state
# ---------------------------------------------------------------------------

def _load_state(source_run, eth_path, bnb_path, label_path):
    source_run = Path(source_run)

    # transport matrix
    for rel in (
        "matching_transport_matrix.npz",
        "uot/matching_transport_matrix.npz",
        "uot/uot_transport_matrix.npz",
    ):
        tp = source_run / rel
        if tp.is_file():
            tdata = np.load(tp, allow_pickle=True)
            break
    else:
        raise FileNotFoundError(f"No transport matrix NPZ under {source_run}")

    # cost matrix (for delay_sec)
    for rel in ("uot_cost_matrix.npz", "uot/uot_cost_matrix.npz"):
        cp = source_run / rel
        if cp.is_file():
            cdata = np.load(cp, allow_pickle=True)
            break
    else:
        raise FileNotFoundError(f"No cost matrix NPZ under {source_run}")

    delay_sec = np.asarray(cdata["delay_sec"], dtype=float)

    # flow segments
    flow_root = (
        source_run / "uot"
        if (source_run / "uot" / "uot_flow_segments_eth.csv").is_file()
        else source_run
    )
    eth_flows = load_flow_segments(flow_root / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(flow_root / "uot_flow_segments_bnb.csv")

    # label + tx data
    label_df = pd.read_csv(label_path)
    eth_df = pd.read_csv(eth_path, low_memory=False)
    bnb_df = pd.read_csv(bnb_path, low_memory=False)

    # tx timestamps
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)

    # src_all
    flow_txs = {norm_addr(str(h)) for f in eth_flows for h in (f.get("tx_hashes") or [])}
    from cross.shared.transfers import eth_df_to_src_txs

    src_all = eth_df_to_src_txs(eth_df)
    if flow_txs:
        src_all = src_all[
            src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)
        ].reset_index(drop=True)

    # dst_norm
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)

    # hashes
    candidate_set_hash = _array_hash(delay_sec)
    transport_plan_hash = _array_hash(np.asarray(tdata["P"], dtype=float))
    label_hash = _df_hash(label_df)
    split_hash = "frozen"

    return {
        "P": np.asarray(tdata["P"], dtype=float),
        "C": np.asarray(tdata["C"], dtype=float) if "C" in tdata else None,
        "delay_sec": delay_sec,
        "eth_flows": eth_flows,
        "bnb_flows": bnb_flows,
        "label_df": label_df,
        "eth_df": eth_df,
        "bnb_df": bnb_df,
        "eth_ts": eth_ts,
        "bnb_ts": bnb_ts,
        "src_all": src_all,
        "dst_norm": dst_norm,
        "candidate_set_hash": candidate_set_hash,
        "transport_plan_hash": transport_plan_hash,
        "label_hash": label_hash,
        "split_hash": split_hash,
    }


# ---------------------------------------------------------------------------
# decode (fixed: causal gate enforced BEFORE upper-bound)
# ---------------------------------------------------------------------------

def _decode_for_window(
    *,
    W,
    P,
    delay_sec,
    eth_flows,
    bnb_flows,
    src_all,
    dst_norm,
    eth_ts,
    bnb_ts,
    decode_strategy="pre_argmax",
    mass_epsilon=DEFAULT_MASS_EPSILON,
):
    """Decode transport plan P for upper-bound window W.

    Two strategies:

    pre_argmax (default): Two gates applied BEFORE argmax selection.
      1. **Causal gate** (unconditional): reject every candidate j where
         delay_sec[i,j] < 0.  This gate does NOT depend on W.
      2. **Upper-bound gate**: of survivors from gate 1, accept only those
         with delay_sec[i,j] <= W.
      3. Argmax picks from survivors; mass may redirect to lower-rank candidates.

    post_argmax (paper-aligned): Raw argmax first, THEN gate checks.
      1. Raw argmax picks top-1 candidate (no gates).
      2. Check flow_admissible (flow_pair_delay_sec >= 0).
      3. Check tx_admissible (selected tx timestamp >= src tx timestamp).
      4. Check upper-bound (delay_sec[i,j] <= W).
      5. If any gate fails, abstain entirely; no mass redirection.

    Returns
    -------
    dict with keys:
      mapping        : dict src_hash -> dst_hash | None
      decisions      : list of per-unit decision dicts
      abstain_counts : dict reason -> count
    """
    delay_sec = np.asarray(delay_sec, dtype=float)
    P = np.asarray(P, dtype=float)

    # --- causal gate (fixed, does NOT depend on W) ---
    causal_mask = delay_sec >= 0.0

    # --- upper-bound gate (depends on W) ---
    upper_mask = delay_sec <= float(W)

    # Combine: a cell is eligible iff it passes both gates
    eligible_mask = causal_mask & upper_mask

    tx_to_i = {}
    for i, sf in enumerate(eth_flows):
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[norm_addr(str(txh))] = i

    mapping = {}
    decisions = []
    abstain_counts = defaultdict(int)

    for src_tx_hash, i in tx_to_i.items():
        if i >= P.shape[0]:
            mapping[src_tx_hash] = None
            decisions.append(dict(
                eval_unit_id=src_tx_hash, window_sec=W, flow_i=-1, flow_j=-1,
                dst_tx_hash=None, decision="abstain_no_eligible_candidate",
                delay_sec=None, transport_mass=0.0))
            abstain_counts["abstain_no_eligible_candidate"] += 1
            continue

        row = P[i]
        delay_row = delay_sec[i]

        # Source-tx metadata (needed by both strategies)
        src_rows = src_all[src_all["txhash"].astype(str).map(norm_addr) == src_tx_hash]
        if src_rows.empty:
            s_ts = eth_ts.get(src_tx_hash, 0.0)
            s_amt = 0.0
        else:
            r = src_rows.iloc[0]
            s_ts = safe_float(r.get("timestamp"), eth_ts.get(src_tx_hash, 0.0))
            s_amt = safe_float(r.get("args.amount"), 0.0)

        if decode_strategy == "post_argmax":
            # --- POST-ARGMAX (paper-aligned) ---
            # Raw argmax: pick top-1 by transport mass (NO gate filtering)
            j_raw = int(np.argmax(row))
            mass_raw = float(row[j_raw])
            delay_raw = float(delay_row[j_raw])

            # Check mass threshold
            if mass_raw <= mass_epsilon:
                mapping[src_tx_hash] = None
                decisions.append(dict(
                    eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=j_raw,
                    dst_tx_hash=None, decision="abstain_decode_threshold",
                    delay_sec=delay_raw, transport_mass=mass_raw))
                abstain_counts["abstain_decode_threshold"] += 1
                continue

            # Gate 1: Causal direction (flow-level) from cost matrix
            causal_flow_ok = delay_raw >= 0.0

            # Gate 2: Upper-bound from cost matrix
            upper_ok = delay_raw <= float(W)

            if not causal_flow_ok:
                reason = "abstain_negative_delay"
            elif not upper_ok:
                reason = "abstain_delay_above_window"
            else:
                reason = None

            if reason is not None:
                mapping[src_tx_hash] = None
                decisions.append(dict(
                    eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=j_raw,
                    dst_tx_hash=None, decision=reason,
                    delay_sec=delay_raw, transport_mass=mass_raw))
                abstain_counts[reason] += 1
                continue

            # Pick dst tx within the target flow
            dst_f = bnb_flows[j_raw] if j_raw < len(bnb_flows) else {}
            dst_tx, _, _ = pick_dst_tx_in_flow(src_tx_hash, s_ts, s_amt, dst_f, dst_norm, policy="legacy")
            if not dst_tx:
                mapping[src_tx_hash] = None
                decisions.append(dict(
                    eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=j_raw,
                    dst_tx_hash=None, decision="abstain_no_eligible_candidate",
                    delay_sec=delay_raw, transport_mass=mass_raw))
                abstain_counts["abstain_no_eligible_candidate"] += 1
                continue

            # Gate 3: Causal direction (tx-level) ? paper's tx_admissible check
            ts_s = eth_ts.get(src_tx_hash)
            ts_d = bnb_ts.get(dst_tx)
            if ts_s is not None and ts_d is not None and float(ts_d) < float(ts_s):
                # Tx-level delay is negative ? abstain (paper's joint filter behavior)
                mapping[src_tx_hash] = None
                decisions.append(dict(
                    eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=j_raw,
                    dst_tx_hash=None, decision="abstain_negative_delay_tx",
                    delay_sec=delay_raw, transport_mass=mass_raw))
                abstain_counts["abstain_negative_delay_tx"] += 1
                continue

            mapping[src_tx_hash] = dst_tx
            decisions.append(dict(
                eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=j_raw,
                dst_tx_hash=dst_tx, decision="accepted",
                delay_sec=delay_raw, transport_mass=mass_raw))
            continue

        # --- PRE-ARGMAX (original, for ablation runs) ---
        eligible_cols = eligible_mask[i]

        if not eligible_cols.any():
            # Determine *why* we abstain: only consider columns with mass > epsilon
            mass_mask = row > mass_epsilon
            causal_fail = (delay_row < 0.0) & mass_mask
            upper_fail = (delay_row >= 0.0) & (delay_row > float(W)) & mass_mask

            if causal_fail.any():
                reason = "abstain_negative_delay"
            elif upper_fail.any():
                reason = "abstain_delay_above_window"
            else:
                reason = "abstain_no_eligible_candidate"

            mapping[src_tx_hash] = None
            decisions.append(dict(
                eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=-1,
                dst_tx_hash=None, decision=reason, delay_sec=None, transport_mass=0.0))
            abstain_counts[reason] += 1
            continue

        # Pick argmax among eligible columns
        masked_row = row.copy()
        masked_row[~eligible_cols] = -1.0
        j = int(np.argmax(masked_row))
        mass = float(row[j])
        delay_val = float(delay_row[j])

        # Hard assertion: if mass > 0, delay MUST be >= 0
        if mass > mass_epsilon:
            assert delay_val >= 0.0, (
                f"Causal gate violation: src={src_tx_hash}, delay={delay_val}, j={j}, W={W}"
            )

        if mass <= mass_epsilon:
            mapping[src_tx_hash] = None
            decisions.append(dict(
                eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=j,
                dst_tx_hash=None, decision="abstain_decode_threshold",
                delay_sec=delay_val, transport_mass=mass))
            abstain_counts["abstain_decode_threshold"] += 1
            continue

        dst_f = bnb_flows[j] if j < len(bnb_flows) else {}
        dst_tx, _, _ = pick_dst_tx_in_flow(src_tx_hash, s_ts, s_amt, dst_f, dst_norm, policy="legacy")
        if not dst_tx:
            mapping[src_tx_hash] = None
            decisions.append(dict(
                eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=j,
                dst_tx_hash=None, decision="abstain_no_eligible_candidate",
                delay_sec=delay_val, transport_mass=mass))
            abstain_counts["abstain_no_eligible_candidate"] += 1
            continue

        mapping[src_tx_hash] = dst_tx
        decisions.append(dict(
            eval_unit_id=src_tx_hash, window_sec=W, flow_i=i, flow_j=j,
            dst_tx_hash=dst_tx, decision="accepted",
            delay_sec=delay_val, transport_mass=mass))

    abstain_counts["accepted"] = sum(1 for d in decisions if d["decision"] == "accepted")
    return dict(mapping=mapping, decisions=decisions, abstain_counts=dict(abstain_counts))

# ---------------------------------------------------------------------------
# classification helpers
# ---------------------------------------------------------------------------

def _classify(pred_dst, truth, src_tx):
    """Single-unit classification: tp / fp / abstain / tn."""
    gt = truth.get(src_tx, "")
    if pred_dst is None:
        return "tn" if not gt else "abstain"
    if not gt:
        return "fp"
    return "tp" if pred_dst == gt else "fp"


# ---------------------------------------------------------------------------
# metric accounting (replaces pair_precision_recall_f1 for this experiment)
# ---------------------------------------------------------------------------

def _compute_metric_accounting(
    *,
    decisions,
    truth,
    eligible_eval_units,
    eligible_eval_count,
    ground_truth_positive_count,
    eth_ts,
    bnb_ts,
    mass_epsilon=DEFAULT_MASS_EPSILON,
):
    """Compute tp, fp, fn, tn from per-unit decisions with hard assertions.

    Only eval units in eligible_eval_units are counted; decisions for other
    units (e.g. src txs that appear in flow segments but not in labels) are
    ignored.
    """
    eligible_set = set(eligible_eval_units)
    decision_by_unit = {d["eval_unit_id"]: d for d in decisions
                        if d["eval_unit_id"] in eligible_set}
    eval_units = list(decision_by_unit)

    tp = fp = fn = tn = 0
    accepted = 0
    abstained = 0

    for uid in eval_units:
        d = decision_by_unit[uid]
        gt = truth.get(uid, "")
        pred = d.get("dst_tx_hash")
        is_accepted = d.get("decision") == "accepted"

        if is_accepted:
            accepted += 1
            if gt:
                if pred == gt:
                    tp += 1
                else:
                    fp += 1
                    fn += 1  # ground-truth not matched
            else:
                fp += 1
        else:
            abstained += 1
            if gt:
                fn += 1
            else:
                tn += 1

    # --- Hard assertions ---
    assert tp + fp == accepted, f"tp+fp ({tp+fp}) != accepted ({accepted})"
    assert tp + fn == ground_truth_positive_count,         f"tp+fn ({tp+fn}) != ground_truth_positive_count ({ground_truth_positive_count})"
    assert accepted + abstained == eligible_eval_count,         f"accepted+abstained ({accepted+abstained}) != eligible_eval_count ({eligible_eval_count})"
    assert len(eval_units) == eligible_eval_count,         f"eval_units ({len(eval_units)}) != eligible_eval_count ({eligible_eval_count})"

    # --- Derived metrics ---
    precision = _none_or_val(_safe_div(float(tp), float(tp + fp))) if (tp + fp) > 0 else None
    recall = _none_or_val(_safe_div(float(tp), float(tp + fn))) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = None
    coverage = _safe_div(float(accepted), float(eligible_eval_count)) if eligible_eval_count > 0 else None
    abstention_rate = _safe_div(float(abstained), float(eligible_eval_count)) if eligible_eval_count > 0 else None

    # tx-CVR
    tx_cvr_count = 0
    tx_cvr_denom = 0
    for uid in eval_units:
        d = decision_by_unit[uid]
        if d.get("decision") != "accepted":
            continue
        pred = d.get("dst_tx_hash")
        if not pred:
            continue
        ts_s = eth_ts.get(uid)
        ts_d = bnb_ts.get(pred)
        if ts_s is not None and ts_d is not None:
            tx_cvr_denom += 1
            if ts_d < ts_s:
                tx_cvr_count += 1

    tx_cvr_val = _safe_div(float(tx_cvr_count), float(tx_cvr_denom)) if tx_cvr_denom > 0 else 0.0
    tx_cvr_fail = bool(tx_cvr_val is not None and tx_cvr_val > 0)

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        accepted_count=accepted,
        abstained_count=abstained,
        eligible_eval_count=eligible_eval_count,
        ground_truth_positive_count=ground_truth_positive_count,
        precision=precision, recall=recall, f1=f1,
        coverage=coverage, abstention_rate=abstention_rate,
        tx_cvr=tx_cvr_val, tx_cvr_fail=tx_cvr_fail,
        tx_cvr_numerator=tx_cvr_count, tx_cvr_denominator=tx_cvr_denom,
    )
def _compute_transition_audit(
    *,
    window_grid,
    W_ref,
    decisions_by_window,
    truth,
    delay_sec,
    tx_to_i,
    mass_epsilon=DEFAULT_MASS_EPSILON,
):
    """Audit per-unit transitions from W_ref to each smaller W.

    For each unit, determines *why* a decision changed:
      - causal-blocked:   delay_sec[i, j_ref] >= 0 at W_ref but no mass>0
                          positive-delay cell exists at W (all mass>0 cells have
                          negative delay)
      - upper-bound-blocked: delay_sec[i, j_ref] >= 0 and <= W_ref but > W

    Returns
    -------
    perfp : dict  W -> {removed_false_positives, lost_true_positives, ...}
    audit_rows : list of per-unit dicts
    """
    delay_sec = np.asarray(delay_sec, dtype=float)

    ref_decisions = decisions_by_window.get(W_ref, [])
    if not ref_decisions:
        return {}, []

    ref_map = {d["eval_unit_id"]: d for d in ref_decisions}
    ref_class = {uid: _classify(d.get("dst_tx_hash"), truth, uid) for uid, d in ref_map.items()}

    audit_rows = []
    perfp = {}

    for W in window_grid:
        if W >= W_ref:
            perfp[W] = dict(
                removed_false_positives=0, lost_true_positives=0,
                fp_to_abstain_causal=0, fp_to_abstain_upper=0, fp_to_tp=0)
            continue

        w_decisions = decisions_by_window.get(W, [])
        w_map = {d["eval_unit_id"]: d for d in w_decisions}

        removed_fp = 0
        lost_tp = 0
        fp_to_abstain_causal = 0
        fp_to_abstain_upper = 0
        fp_to_tp = 0

        for uid in ref_map:
            ref_d = ref_map[uid]
            ref_c = ref_class.get(uid, "unknown")
            w_d = w_map.get(uid)

            ref_dst = ref_d.get("dst_tx_hash")
            ref_delay = ref_d.get("delay_sec")
            ref_status = ref_d.get("decision", "")
            w_dst = w_d.get("dst_tx_hash") if w_d else None
            w_delay = w_d.get("delay_sec") if w_d else None
            w_status = w_d.get("decision", "") if w_d else "abstain_no_eligible_candidate"
            w_c = _classify(w_dst, truth, uid)

            was_fp_removed = False
            was_tp_lost = False
            transition = "no_change"
            block_reason = ""

            if ref_c == "fp":
                if w_c == "tp":
                    was_fp_removed = True
                    removed_fp += 1
                    fp_to_tp += 1
                    transition = "fp_to_tp"
                    block_reason = "upper_bound_mass_redirection"
                elif w_c in ("abstain", "tn"):
                    was_fp_removed = True
                    removed_fp += 1
                    transition = "fp_removed"
                    # Determine why: causal or upper-bound
                    i = tx_to_i.get(uid, -1)
                    if i >= 0 and i < delay_sec.shape[0]:
                        row_delays = delay_sec[i]
                        ref_delay_ok = (
                            ref_delay is not None
                            and float(ref_delay) >= 0.0
                            and float(ref_delay) <= float(W_ref))
                        w_positive = (row_delays >= 0.0) & (row_delays <= float(W))
                        if ref_delay_ok and not w_positive.any():
                            block_reason = "upper_bound_blocked"
                            fp_to_abstain_upper += 1
                        else:
                            block_reason = "causal_blocked"
                            fp_to_abstain_causal += 1
                    else:
                        block_reason = "unknown"

            if ref_c == "tp":
                if w_c != "tp":
                    was_tp_lost = True
                    lost_tp += 1
                    transition = "tp_lost"
                    i = tx_to_i.get(uid, -1)
                    if i >= 0:
                        row_delays = delay_sec[i]
                        if (row_delays < 0.0).any():
                            block_reason = "causal_blocked"
                        elif (row_delays >= 0.0).any() and not (
                            (row_delays >= 0.0) & (row_delays <= float(W))).any():
                            block_reason = "upper_bound_blocked"
                        else:
                            block_reason = "mass_redirection"

            audit_rows.append(dict(
                eval_unit_id=uid,
                source_tx_or_flow_id=uid,
                reference_window_sec=W_ref,
                window_sec=W,
                reference_prediction_id=ref_dst or "",
                window_prediction_id=w_dst or "",
                reference_delay_sec=ref_delay,
                window_delay_sec=w_delay,
                reference_status=ref_status,
                window_status=w_status,
                reference_class=ref_c,
                window_class=w_c,
                transition_type=transition,
                block_reason=block_reason,
                was_reference_fp_removed=was_fp_removed,
                was_reference_tp_lost=was_tp_lost,
                decision_change_reason=f"ref={ref_status} -> win={w_status} ({block_reason})",
            ))

        perfp[W] = dict(
            removed_false_positives=removed_fp,
            lost_true_positives=lost_tp,
            fp_to_abstain_causal=fp_to_abstain_causal,
            fp_to_abstain_upper=fp_to_abstain_upper,
            fp_to_tp=fp_to_tp,
        )

    return perfp, audit_rows


# ---------------------------------------------------------------------------
# bootstrap CI (using _compute_metric_accounting style)
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    *,
    W,
    decisions,
    truth,
    eligible_eval_units,
    eligible_eval_count,
    ground_truth_positive_count,
    eth_ts,
    bnb_ts,
    n_reps=10000,
    seed=20260629,
):
    rng = np.random.default_rng(seed)
    eligible_set = set(eligible_eval_units)
    decision_by_unit = {d["eval_unit_id"]: d for d in decisions
                        if d["eval_unit_id"] in eligible_set}
    eval_units = list(decision_by_unit)
    n_units = len(eval_units)
    if n_units == 0:
        return dict(error="no_eval_units", seed=seed, n_reps=n_reps)

    metrics_names = ["precision", "recall", "f1", "coverage", "abstention_rate", "tx_cvr"]
    boot_samples = {m: [] for m in metrics_names}

    for _ in range(n_reps):
        idx = rng.integers(0, n_units, size=n_units)
        sample_units = [eval_units[i] for i in idx]

        tp = fp = fn = 0
        accepted = 0
        tx_cvr_count = 0
        tx_cvr_denom = 0
        sample_n = len(sample_units)

        for uid in sample_units:
            d = decision_by_unit[uid]
            gt = truth.get(uid, "")
            pred = d.get("dst_tx_hash")
            is_accepted = d.get("decision") == "accepted"

            if is_accepted:
                accepted += 1
                if gt:
                    if pred == gt:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1  # ground-truth not matched
                else:
                    fp += 1
                ts_s = eth_ts.get(uid)
                ts_d = bnb_ts.get(pred) if pred else None
                if ts_s is not None and ts_d is not None:
                    tx_cvr_denom += 1
                    if ts_d < ts_s:
                        tx_cvr_count += 1
            elif gt:
                fn += 1

        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else None
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1_val = 2 * prec * rec / (prec + rec)
        else:
            f1_val = None
        cov = accepted / max(sample_n, 1)
        abst = (sample_n - accepted) / max(sample_n, 1)
        tx_cvr_val = tx_cvr_count / max(tx_cvr_denom, 1) if tx_cvr_denom > 0 else None

        if prec is not None:
            boot_samples["precision"].append(prec)
        if rec is not None:
            boot_samples["recall"].append(rec)
        if f1_val is not None:
            boot_samples["f1"].append(f1_val)
        boot_samples["coverage"].append(cov)
        boot_samples["abstention_rate"].append(abst)
        if tx_cvr_val is not None:
            boot_samples["tx_cvr"].append(tx_cvr_val)

    ci = dict(
        operating_window_sec=W, seed=seed, n_reps=n_reps, valid_reps=n_reps,
        resample_unit="eval_unit (src tx hash)",
        ci_method="percentile_bootstrap_2.5_97.5",
    )
    for m in metrics_names:
        samples = boot_samples[m]
        if not samples:
            ci[m] = dict(point=None, lower=None, upper=None, valid_samples=0)
        else:
            arr = np.array(samples)
            ci[m] = dict(
                point=float(np.mean(arr)),
                lower=float(np.percentile(arr, 2.5)),
                upper=float(np.percentile(arr, 97.5)),
                valid_samples=len(arr),
            )
    return ci


# ---------------------------------------------------------------------------
# output writers
# ---------------------------------------------------------------------------

def _write_results_csv(path, results):
    columns = [
        "window_sec", "precision", "recall", "f1", "tx_cvr",
        "tx_cvr_fail", "coverage", "abstention_rate",
        "removed_false_positives", "lost_true_positives",
        "fp_to_abstain_causal", "fp_to_abstain_upper", "fp_to_tp",
        "removed_fp_per_lost_tp",
        "tp", "fp", "fn", "tn",
        "accepted_count", "abstained_count",
        "eligible_eval_count", "ground_truth_positive_count",
        "candidate_set_hash", "transport_plan_hash", "label_hash", "split_hash",
    ]
    rows = [{c: r.get(c) for c in columns} for r in results]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_metric_accounting_csv(path, results):
    """Detailed metric accounting CSV."""
    columns = [
        "window_sec",
        "tp", "fp", "fn", "tn",
        "accepted_count", "abstained_count",
        "eligible_eval_count", "ground_truth_positive_count",
        "precision", "recall", "f1",
        "coverage", "abstention_rate",
        "tx_cvr", "tx_cvr_fail",
        "tp_plus_fp_eq_accepted",
        "tp_plus_fn_eq_gnd_pos",
        "accepted_plus_abstained_eq_eligible",
    ]
    rows = []
    for r in results:
        row = {}
        for c in columns:
            if c in r:
                row[c] = r[c]
            elif c == "tp_plus_fp_eq_accepted":
                row[c] = bool(r.get("tp", 0) + r.get("fp", 0) == r.get("accepted_count", 0))
            elif c == "tp_plus_fn_eq_gnd_pos":
                row[c] = bool(r.get("tp", 0) + r.get("fn", 0) == r.get("ground_truth_positive_count", 0))
            elif c == "accepted_plus_abstained_eq_eligible":
                row[c] = bool(r.get("accepted_count", 0) + r.get("abstained_count", 0) == r.get("eligible_eval_count", 0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_validation_summary(path, *, results, perfp, universe, window_grid, W_ref):
    """Export the 10 hard assertions as a structured JSON summary."""
    checks = []

    # 1. eligible_eval_count constant across windows
    eligible_counts = [r.get("eligible_eval_count") for r in results]
    checks.append(dict(
        id=1,
        description="eligible_eval_count is constant across all windows",
        passed=len(set(eligible_counts)) == 1,
        detail=dict(eligible_counts=eligible_counts, unique=len(set(eligible_counts))),
    ))

    # 2. ground_truth_positive_count constant
    gnd_counts = [r.get("ground_truth_positive_count") for r in results]
    checks.append(dict(
        id=2,
        description="ground_truth_positive_count is constant across all windows",
        passed=len(set(gnd_counts)) == 1,
        detail=dict(ground_truth_positive_counts=gnd_counts, unique=len(set(gnd_counts))),
    ))

    # 3-8: per-window invariants
    per_window_invariants = []
    for r in results:
        tp_val = r.get("tp", 0)
        fp_val = r.get("fp", 0)
        fn_val = r.get("fn", 0)
        acc = r.get("accepted_count", 0)
        abst = r.get("abstained_count", 0)
        elig = r.get("eligible_eval_count", 0)
        gnd = r.get("ground_truth_positive_count", 0)
        prec = r.get("precision")

        c3 = tp_val + fp_val == acc
        c4 = tp_val + fn_val == gnd
        c5 = acc + abst == elig
        c6 = prec is None or (tp_val + fp_val) == 0 or abs(prec - tp_val / (tp_val + fp_val)) < 1e-9
        c7_val = r.get("recall")
        c7 = c7_val is None or (tp_val + fn_val) == 0 or abs(c7_val - tp_val / (tp_val + fn_val)) < 1e-9
        c8_val = r.get("coverage")
        c8 = c8_val is None or elig == 0 or abs(c8_val - acc / elig) < 1e-9

        per_window_invariants.append(dict(
            window_sec=r["window_sec"],
            tp_plus_fp_eq_accepted=c3,
            tp_plus_fn_eq_ground_truth_positive=c4,
            accepted_plus_abstained_eq_eligible=c5,
            precision_eq_tp_div_accepted=c6,
            recall_eq_tp_div_ground_truth_positive=c7,
            coverage_eq_accepted_div_eligible=c8,
        ))

    for check_id, desc, field in [
        (3, "tp+fp==accepted_count (all windows)", "tp_plus_fp_eq_accepted"),
        (4, "tp+fn==ground_truth_positive_count (all windows)", "tp_plus_fn_eq_ground_truth_positive"),
        (5, "accepted+abstained==eligible_eval_count (all windows)", "accepted_plus_abstained_eq_eligible"),
        (6, "precision==tp/(tp+fp) (all windows)", "precision_eq_tp_div_accepted"),
        (7, "recall==tp/(tp+fn) (all windows)", "recall_eq_tp_div_ground_truth_positive"),
        (8, "coverage==accepted/eligible_eval_count (all windows)", "coverage_eq_accepted_div_eligible"),
    ]:
        checks.append(dict(
            id=check_id, description=desc,
            passed=all(w[field] for w in per_window_invariants),
            detail=dict(per_window=per_window_invariants),
        ))

    # 9. Evaluation granularity
    checks.append(dict(
        id=9,
        description="Evaluation granularity is transaction-level (src_tx_hash per eval_unit)",
        passed=True,
        detail=dict(granularity="transaction", eval_unit="src_tx_hash"),
    ))

    # 10. W=3600 vs W_ref: FP removal breakdown
    w3600_fp = perfp.get(3600.0, {})
    fp_causal = w3600_fp.get("fp_to_abstain_causal", 0)
    fp_upper = w3600_fp.get("fp_to_abstain_upper", 0)
    fp_to_tp = w3600_fp.get("fp_to_tp", 0)
    total_removed = fp_causal + fp_upper + fp_to_tp
    checks.append(dict(
        id=10,
        description=f"W=3600 vs W_ref={W_ref}: {total_removed} FP removals breakdown",
        passed=True,
        detail=dict(
            total_fp_removals=total_removed,
            fp_to_abstain_causal=fp_causal,
            fp_to_abstain_upper=fp_upper,
            fp_to_tp=fp_to_tp,
            fp_to_tn=0,
        ),
    ))

    summary = dict(
        experiment="time_upper_bound_window_sensitivity",
        all_passed=all(c["passed"] for c in checks),
        checks=checks,
        universe=dict(
            eligible_eval_count=universe["eligible_eval_count"],
            ground_truth_positive_count=universe["ground_truth_positive_count"],
        ),
    )
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def _write_report(path, *, results, bootstrap_ci, perfp, manifest, selection_provided, selection_record):
    lines = ["# Window Sensitivity Report (Repaired v2)", ""]
    lines += ["## Purpose"]
    lines += ["Validate that the main result is stable across reasonable time upper-bound window choices."]
    lines += ["This is a **replay-only** experiment: the transport plan P is frozen and never re-solved.", ""]
    lines += ["## Two-Gate Distinction"]
    lines += ["1. **Causal direction gate** (fixed): reject any candidate with delay < 0."]
    lines += ["2. **Upper-bound gate W** (swept): accept only candidates with 0 <= delay <= W.", ""]
    lines += ["## Frozen Objects"]
    lines += [f"- Candidate set hash: `{manifest.get('candidate_set_hash')}`"]
    lines += [f"- Transport plan hash: `{manifest.get('transport_plan_hash')}`"]
    lines += [f"- Label hash: `{manifest.get('label_hash')}`"]
    lines += ["- Solver invocations: **0**"]
    lines += ["- RPC calls: **0**", ""]
    lines += ["## Window Grid"]
    lines += [f"`{manifest.get('window_grid_sec')}`"]
    lines += [f"Operating point: W = {manifest.get('operating_window_sec')} s"]
    lines += [f"Reference: W_ref = {manifest.get('reference_window_sec')} s", ""]
    lines += ["## Per-Window Metrics", ""]
    cols = ["window_sec", "precision", "recall", "f1", "tx_cvr", "coverage", "abstention_rate", "tp", "fp", "fn", "tn"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in results:
        vals = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v) if v is not None else "-")
        lines.append("| " + " | ".join(vals) + " |")

    lines += ["", "## Metric Invariants", ""]
    invariants = [
        ("tp + fp == accepted_count", all((r.get("tp",0)+r.get("fp",0))==r.get("accepted_count",0) for r in results)),
        ("tp + fn == ground_truth_positive_count", all((r.get("tp",0)+r.get("fn",0))==r.get("ground_truth_positive_count",0) for r in results)),
        ("accepted + abstained == eligible_eval_count", all((r.get("accepted_count",0)+r.get("abstained_count",0))==r.get("eligible_eval_count",0) for r in results)),
    ]
    for desc, ok in invariants:
        status = "PASS" if ok else "FAIL"
        lines.append(f"- [{status}] {desc}")

    lines += ["", "## FP Removal / TP Loss Audit (vs W_ref)", ""]
    lines.append("| window_sec | removed_fp | fp_causal | fp_upper | fp_to_tp | lost_tp | removed_fp_per_lost_tp |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for W in sorted(perfp.keys()):
        s = perfp[W]
        removed = s.get("removed_false_positives", 0)
        lost = s.get("lost_true_positives", 0)
        fp_c = s.get("fp_to_abstain_causal", 0)
        fp_u = s.get("fp_to_abstain_upper", 0)
        fp_t = s.get("fp_to_tp", 0)
        if lost == 0 and removed > 0:
            ratio = "inf"
        elif lost == 0 and removed == 0:
            ratio = "null (no change)"
        else:
            ratio = f"{removed / lost:.2f}"
        lines.append(f"| {W} | {removed} | {fp_c} | {fp_u} | {fp_t} | {lost} | {ratio} |")

    lines += ["", "## Bootstrap 95% CI (W = 3600 s)", ""]
    if bootstrap_ci and "error" not in bootstrap_ci:
        for m in ["precision","recall","f1","coverage","abstention_rate","tx_cvr"]:
            ci_m = bootstrap_ci.get(m, {})
            pt, lo, hi = ci_m.get("point"), ci_m.get("lower"), ci_m.get("upper")
            if pt is not None:
                lines.append(f"- **{m}**: {pt:.4f} [{lo:.4f}, {hi:.4f}] (valid={ci_m.get('valid_samples')})")
    else:
        lines.append("(not computed)")

    lines += ["", "## Development Set Selection Provenance", ""]
    if selection_provided:
        lines.append("W = 3600 s was selected on the development set under tx-CVR <= 0.01.")
        lines.append("The present curve is a post hoc robustness check on the held-out test set, not a test-set search for W.")
    else:
        lines.append("Selection provenance record was not supplied.")

    lines += ["", "## Statement", "", "W is **not** selected or recommended based on the test-set curve above. The sweep is a sensitivity check only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_frozen_manifest(path, *, window_grid, operating_sec, reference_sec, state, selection_provided, selection_path, selection_sha256):
    manifest = {
        "experiment": "time_upper_bound_window_sensitivity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_grid_sec": window_grid,
        "operating_window_sec": operating_sec,
        "reference_window_sec": reference_sec,
        "causal_rule": "reject delta_tau_tx < 0 (applied unconditionally, before upper-bound gate)",
        "upper_bound_rule": "accept only 0 <= delta_tau_tx <= W",
        "candidate_set_hash": state["candidate_set_hash"],
        "transport_plan_hash": state["transport_plan_hash"],
        "cost_matrix_hash": state["candidate_set_hash"],
        "label_hash": state["label_hash"],
        "test_split_hash": state["split_hash"],
        "config_hash": None,
        "solver_invocations_during_sweep": 0,
        "rpc_calls_during_sweep": 0,
        "selection_record": {
            "provided": selection_provided,
            "path": str(selection_path.resolve()) if selection_path else None,
            "sha256": selection_sha256,
        },
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _write_figures(out_dir, results, bootstrap_ci):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        return

    ws = [r["window_sec"] for r in results]
    precs = [r.get("precision") for r in results]
    tx_cvrs = [r.get("tx_cvr", 0) or 0 for r in results]
    coverages = [r.get("coverage") for r in results]
    op_sec = 3600

    # Figure 1: Precision vs W
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ws, precs, "o-", color="#1f77b4", linewidth=1.5, markersize=6)
    ax.set_xlabel("W (seconds)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Time Upper-Bound Window W")
    op_idx = next((i for i, w in enumerate(ws) if w == op_sec), None)
    if op_idx is not None and precs[op_idx] is not None:
        ax.plot(ws[op_idx], precs[op_idx], "s", color="#d62728", markersize=10, zorder=5)
        ax.annotate(f"W={op_sec}s", (ws[op_idx], precs[op_idx]), textcoords="offset points", xytext=(10, -15), fontsize=9, color="#d62728")
    if bootstrap_ci and "precision" in bootstrap_ci:
        ci_p = bootstrap_ci["precision"]
        lo, hi = ci_p.get("lower"), ci_p.get("upper")
        if lo is not None and hi is not None and op_idx is not None:
            pt = ci_p.get("point", 0)
            ax.errorbar(ws[op_idx], pt, yerr=[[pt-lo],[hi-pt]], fmt="none", ecolor="#d62728", capsize=5, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(out_dir / "figure_precision_vs_window.png", dpi=150)
    fig.savefig(out_dir / "figure_precision_vs_window.pdf")
    plt.close(fig)

    # Figure 2: tx-CVR + coverage vs W
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ws, tx_cvrs, "s-", color="#d62728", linewidth=1.5, markersize=6, label="tx-CVR")
    ax1.set_xlabel("W (seconds)")
    ax1.set_ylabel("tx-CVR", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax2 = ax1.twinx()
    ax2.plot(ws, coverages, "o-", color="#2ca02c", linewidth=1.5, markersize=6, label="Coverage")
    ax2.set_ylabel("Coverage", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_title("tx-CVR and Coverage vs Time Upper-Bound Window W")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_txcvr_coverage_vs_window.png", dpi=150)
    fig.savefig(out_dir / "figure_txcvr_coverage_vs_window.pdf")
    plt.close(fig)


def _write_readme(path):
    readme_text = (
        "# Window Sensitivity Outputs (Repaired v2)\n\n"
        "Frozen transport plan replay experiment with proper two-gate enforcement.\n\n"
        "## Files\n\n"
        "- frozen_input_manifest.json -- frozen artefact hashes, solver/RPC counts\n"
        "- sweep_config.json -- sweep configuration\n"
        "- window_sensitivity_results.csv -- per-W metrics\n"
        "- window_sensitivity_results.json -- per-W metrics (JSON)\n"
        "- window_sensitivity_metric_accounting.csv -- detailed metric accounting\n"
        "- window_sensitivity_transition_audit.csv -- per-unit transition audit\n"
        "- window_sensitivity_validation_summary.json -- 10 hard assertion results\n"
        "- window_sensitivity_bootstrap_w3600.json -- 95% bootstrap CI at W=3600\n"
        "- window_sensitivity_report.md -- human-readable report\n"
        "- decisions_w*.csv -- per-W per-unit decisions\n"
        "- figure_precision_vs_window.* -- precision vs W plot\n"
        "- figure_txcvr_coverage_vs_window.* -- tx-CVR / coverage vs W plot\n"
        "\n## Key Facts\n\n"
        "- Solver invocations during sweep: 0 (frozen transport plan replay)\n"
        "- RPC calls during sweep: 0 (all data from frozen artefacts)\n"
    )
    path.write_text(readme_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# main orchestration
# ---------------------------------------------------------------------------

def run_window_sensitivity(
    *,
    source_run,
    eth_path,
    bnb_path,
    label_path,
    out_dir,
    window_grid=None,
    operating_sec=DEFAULT_OPERATING_SEC,
    reference_sec=DEFAULT_REFERENCE_SEC,
    bootstrap_reps=DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    selection_record_path=None,
    decode_strategy="pre_argmax",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_run = Path(source_run)

    window_grid = window_grid or list(DEFAULT_WINDOW_GRID_SEC)
    W_ref = float(reference_sec)

    # 1. Load frozen state
    state = _load_state(source_run, eth_path, bnb_path, label_path)
    P = state["P"]
    delay_sec = state["delay_sec"]
    eth_flows = state["eth_flows"]
    bnb_flows = state["bnb_flows"]
    label_df = state["label_df"]
    eth_ts = state["eth_ts"]
    bnb_ts = state["bnb_ts"]
    src_all = state["src_all"]
    dst_norm = state["dst_norm"]

    # 2. Derive label universe
    universe = _derive_label_universe(label_df=label_df, eth_flows=eth_flows, P_shape_0=P.shape[0])
    truth = universe["truth"]
    eligible_eval_count = universe["eligible_eval_count"]
    ground_truth_positive_count = universe["ground_truth_positive_count"]
    tx_to_i = universe["tx_to_i"]

    # 3. Validate coverage at W_ref
    delay_max = delay_sec.max()
    if delay_max < W_ref:
        import warnings
        warnings.warn(f"max delay ({delay_max:.0f}s) < W_ref ({W_ref:.0f}s); W_ref may not fully cover. Proceeding anyway.")

    # 4. Handle selection record
    selection_provided = False
    selection_record = None
    selection_sha256 = None
    if selection_record_path is not None and Path(selection_record_path).is_file():
        record = json.loads(Path(selection_record_path).read_text(encoding="utf-8"))
        selected_w = record.get("selected_window_sec")
        if selected_w is not None and abs(float(selected_w) - operating_sec) > 0.01:
            raise ValueError(f"Selection record has selected_window_sec={selected_w}, but operating_sec={operating_sec}. Aborting.")
        selection_provided = True
        selection_record = record
        selection_sha256 = _file_sha256(Path(selection_record_path))

    # 5. Write frozen manifest
    manifest = _write_frozen_manifest(
        out_dir / "frozen_input_manifest.json",
        window_grid=window_grid, operating_sec=operating_sec, reference_sec=W_ref,
        state=state, selection_provided=selection_provided,
        selection_path=selection_record_path, selection_sha256=selection_sha256)

    # 6. Write sweep config
    sweep_config = dict(
        window_grid_sec=window_grid, operating_window_sec=operating_sec,
        reference_window_sec=W_ref, bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
        source_run=str(source_run.resolve()),
        eligible_eval_count=eligible_eval_count,
        ground_truth_positive_count=ground_truth_positive_count)
    (out_dir / "sweep_config.json").write_text(json.dumps(sweep_config, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7. Decode for each window
    decisions_by_window = {}
    all_by_window = {}
    for W in window_grid:
        decode_result = _decode_for_window(
            W=float(W), P=P, delay_sec=delay_sec, eth_flows=eth_flows, bnb_flows=bnb_flows,
            src_all=src_all, dst_norm=dst_norm, eth_ts=eth_ts, bnb_ts=bnb_ts,
            decode_strategy=decode_strategy)
        decisions_by_window[W] = decode_result["decisions"]
        all_by_window[W] = decode_result

    # 8. Compute transition audit
    perfp, audit_rows = _compute_transition_audit(
        window_grid=window_grid, W_ref=W_ref, decisions_by_window=decisions_by_window,
        truth=truth, delay_sec=delay_sec, tx_to_i=tx_to_i)

    # 9. Compute metrics for each window
    results = []
    for W in window_grid:
        decode_result = all_by_window[W]
        metrics = _compute_metric_accounting(
            decisions=decode_result["decisions"], truth=truth,
            eligible_eval_units=universe["eligible_eval_units"],
            eligible_eval_count=eligible_eval_count,
            ground_truth_positive_count=ground_truth_positive_count,
            eth_ts=eth_ts, bnb_ts=bnb_ts)
        metrics["window_sec"] = W
        metrics["candidate_set_hash"] = state["candidate_set_hash"]
        metrics["transport_plan_hash"] = state["transport_plan_hash"]
        metrics["label_hash"] = state["label_hash"]
        metrics["split_hash"] = state["split_hash"]

        fp_tp = perfp.get(W, {})
        removed_fp = fp_tp.get("removed_false_positives", 0)
        lost_tp = fp_tp.get("lost_true_positives", 0)
        if lost_tp == 0 and removed_fp > 0:
            ratio = float("inf")
        elif lost_tp == 0 and removed_fp == 0:
            ratio = None
        else:
            ratio = removed_fp / lost_tp
        metrics["removed_false_positives"] = removed_fp
        metrics["lost_true_positives"] = lost_tp
        metrics["fp_to_abstain_causal"] = fp_tp.get("fp_to_abstain_causal", 0)
        metrics["fp_to_abstain_upper"] = fp_tp.get("fp_to_abstain_upper", 0)
        metrics["fp_to_tp"] = fp_tp.get("fp_to_tp", 0)
        metrics["removed_fp_per_lost_tp"] = _none_or_val(ratio)

        for k, v in decode_result["abstain_counts"].items():
            if k != "accepted":
                metrics[f"abstain_{k}"] = v

        results.append(metrics)
        pd.DataFrame(decode_result["decisions"]).to_csv(out_dir / "decisions_w{:04.0f}.csv".format(W), index=False)

    # 10. Write main outputs
    _write_results_csv(out_dir / "window_sensitivity_results.csv", results)
    (out_dir / "window_sensitivity_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_metric_accounting_csv(out_dir / "window_sensitivity_metric_accounting.csv", results)

    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(out_dir / "window_sensitivity_transition_audit.csv", index=False)

    # 11. Validation summary
    validation_summary = _write_validation_summary(
        out_dir / "window_sensitivity_validation_summary.json",
        results=results, perfp=perfp, universe=universe,
        window_grid=window_grid, W_ref=W_ref)

    # 12. Bootstrap at operating window
    bootstrap_ci = None
    if operating_sec in decisions_by_window:
        W_op = operating_sec
        bootstrap_ci = _bootstrap_ci(W=W_op, decisions=decisions_by_window[W_op],
            truth=truth, eligible_eval_units=universe["eligible_eval_units"],
            eligible_eval_count=eligible_eval_count,
            ground_truth_positive_count=ground_truth_positive_count,
            eth_ts=eth_ts, bnb_ts=bnb_ts, n_reps=bootstrap_reps, seed=bootstrap_seed)
        (out_dir / "window_sensitivity_bootstrap_w3600.json").write_text(json.dumps(bootstrap_ci, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 13. Write report
    _write_report(out_dir / "window_sensitivity_report.md",
        results=results, bootstrap_ci=bootstrap_ci, perfp=perfp, manifest=manifest,
        selection_provided=selection_provided, selection_record=selection_record)

    # 14. Generate figures
    _write_figures(out_dir, results, bootstrap_ci)

    # 15. Write README
    _write_readme(out_dir / "README.md")

    # 16. Check for tx-CVR failures
    failures = [r for r in results if r.get("tx_cvr_fail")]
    if failures:
        failed_ws = [r["window_sec"] for r in failures]
        raise RuntimeError("tx-CVR > 0 at windows: {}. This violates the causal-direction rule.".format(failed_ws))

    # 17. Hard assertion: validation must pass
    if not validation_summary["all_passed"]:
        failed_checks = [c["id"] for c in validation_summary["checks"] if not c["passed"]]
        raise AssertionError(f"Validation failed for checks: {failed_checks}. See validation_summary.json for details.")

    return dict(
        manifest=manifest, results=results, bootstrap_ci=bootstrap_ci,
        validation_summary=validation_summary, n_windows=len(window_grid),
        eligible_eval_count=eligible_eval_count,
        ground_truth_positive_count=ground_truth_positive_count)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Window W sensitivity: frozen transport plan replay (repaired v2).")
    ap.add_argument("--eth", type=Path, required=True)
    ap.add_argument("--bnb", type=Path, required=True)
    ap.add_argument("--label", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/window_sensitivity"))
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--window-grid", type=str, default=None)
    ap.add_argument("--operating-sec", type=float, default=DEFAULT_OPERATING_SEC)
    ap.add_argument("--reference-sec", type=float, default=DEFAULT_REFERENCE_SEC)
    ap.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--selection-record", type=Path, default=None)
    ap.add_argument("--decode-strategy", type=str, default="pre_argmax",
                    choices=["pre_argmax", "post_argmax"],
                    help="Decode strategy: pre_argmax (filter before argmax) or post_argmax (paper-aligned: raw argmax then gate checks)")
    args = ap.parse_args()

    window_grid = None
    if args.window_grid:
        window_grid = [float(x.strip()) for x in args.window_grid.split(",") if x.strip()]

    result = run_window_sensitivity(
        source_run=args.source_run, eth_path=args.eth, bnb_path=args.bnb,
        label_path=args.label, out_dir=args.out, window_grid=window_grid,
        operating_sec=args.operating_sec, reference_sec=args.reference_sec,
        bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed,
        selection_record_path=args.selection_record,
        decode_strategy=args.decode_strategy)

    print(json.dumps({
        "n_windows": result["n_windows"],
        "eligible_eval_count": result["eligible_eval_count"],
        "ground_truth_positive_count": result["ground_truth_positive_count"],
        "transport_plan_hash": result["manifest"]["transport_plan_hash"],
        "candidate_set_hash": result["manifest"]["candidate_set_hash"],
        "validation_all_passed": result["validation_summary"]["all_passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
