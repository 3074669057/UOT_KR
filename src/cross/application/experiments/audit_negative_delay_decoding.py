"""Audit negative-delay decoded tx pairs: flow vs tx-level attribution."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup, delay_distribution
from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.domain.evaluation.flow_metrics import pair_precision_recall_f1, topk_flow_accuracy
from cross.domain.uot.delay_policy import flow_pair_delay_sec
from cross.domain.uot.tx_decode_policy import (
    TX_DECODE_POLICIES,
    derive_tx_pairs,
    infer_legacy_selection_rule,
    pick_dst_tx_in_flow,
)
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs, parse_transfer_value
from cross.utils.safe_cast import safe_float

DIAGNOSTIC_POLICIES: tuple[str, ...] = (
    "legacy",
    "choose_earliest_positive_delay_tx",
    "choose_nearest_positive_delay_tx",
    "choose_amount_nearest_positive_delay_tx",
    "choose_latest_tx",
    "choose_label_oracle_if_in_flow",
)


def _load_transport(run_dir: Path) -> np.ndarray:
    run_dir = Path(run_dir)
    for rel in ("matching_transport_matrix.npz", "uot/uot_transport_matrix.npz"):
        p = run_dir / rel
        if p.is_file():
            data = np.load(p)
            return np.asarray(data["P"], dtype=float)
    raise FileNotFoundError(f"No transport matrix under {run_dir}")


def _truth_from_labels(label_df: pd.DataFrame) -> dict[str, str]:
    truth: dict[str, str] = {}
    for _, row in label_df.iterrows():
        s = norm_addr(row.get("srcTxhash", row.get("srcTxHash", "")))
        d = norm_addr(row.get("dstTxhash", row.get("dstTxHash", "")))
        if s:
            truth[s] = d
    return truth


def _tx_to_flow_index(source_flows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            out[norm_addr(str(txh))] = i
    return out


def _tx_to_target_flows(target_flows: list[dict[str, Any]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for j, tf in enumerate(target_flows):
        for txh in tf.get("tx_hashes") or []:
            out[norm_addr(str(txh))].add(j)
    return out


def _prepare_src_all(eth_path: Path, eth_flows: list[dict[str, Any]]) -> pd.DataFrame:
    eth_df = pd.read_csv(eth_path, low_memory=False)
    src = eth_df_to_src_txs(eth_df)
    flow_txs = {norm_addr(str(h)) for f in eth_flows for h in (f.get("tx_hashes") or [])}
    if flow_txs:
        src = src[src["txhash"].astype(str).map(norm_addr).isin(flow_txs)]
    return src.reset_index(drop=True)


def _flow_pair_level_metrics(
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Layer A: argmax target flow per source flow, representative delay."""
    delays: list[float] = []
    violations = 0
    for i in range(p.shape[0]):
        if p.shape[1] == 0:
            continue
        j = int(np.argmax(p[i]))
        d = flow_pair_delay_sec(eth_flows[i], bnb_flows[j], policy="tx_if_available_else_flow_representative")
        delays.append(d)
        if d < 0:
            violations += 1
    n = len(delays)
    dist = delay_distribution(delays)
    return {
        "flow_pair_cvr": float(violations / max(n, 1)),
        "flow_pair_median_delay": dist.get("median"),
        "flow_pair_p05_sec": dist.get("p05"),
        "flow_pair_p95_sec": dist.get("p95"),
        "flow_pair_negative_ratio": dist.get("negative_ratio"),
        "n_source_flows_with_argmax": n,
        **{f"flow_pair_{k}": v for k, v in dist.items() if k not in ("median", "p05", "p95", "negative_ratio")},
    }


def _labeled_tx_within_flow_pair(
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    mapping: dict[str, str],
) -> dict[str, Any]:
    """Layer B: labeled tx pairs within decoded flow pairs."""
    truth = _truth_from_labels(label_df)
    tx_to_i = _tx_to_flow_index(eth_flows)
    tx_to_j = _tx_to_target_flows(bnb_flows)

    n_flow_pairs = 0
    any_positive = 0
    gt_in_target = 0
    flow_ok_tx_wrong = 0

    seen_pairs: set[tuple[int, int]] = set()
    for s, gt_d in truth.items():
        i = tx_to_i.get(s, -1)
        if i < 0:
            continue
        j = int(np.argmax(p[i])) if p.shape[1] else -1
        if j < 0:
            continue
        key = (i, j)
        if key not in seen_pairs:
            seen_pairs.add(key)
            n_flow_pairs += 1

            labeled_src_in_flow = [tx for tx, fi in tx_to_i.items() if fi == i and tx in truth]
            labeled_dst_in_tgt = [d for d, jset in tx_to_j.items() if j in jset and d in truth.values()]

            has_pos = False
            for ls in labeled_src_in_flow:
                ts_s = eth_ts.get(ls)
                if ts_s is None:
                    continue
                for ld in labeled_dst_in_tgt:
                    ts_d = bnb_ts.get(ld)
                    if ts_d is not None and float(ts_d - ts_s) > 0:
                        has_pos = True
                        break
                if has_pos:
                    break
            if has_pos:
                any_positive += 1

            gt_j_set = tx_to_j.get(gt_d, set())
            if j in gt_j_set:
                gt_in_target += 1

        pred_d = mapping.get(s, "")
        gt_j_set = tx_to_j.get(gt_d, set())
        if j in gt_j_set and pred_d and pred_d != gt_d:
            flow_ok_tx_wrong += 1

    n_labeled = len(truth)
    return {
        "fraction_matched_flow_pairs_with_any_positive_labeled_tx_pair": float(any_positive / max(n_flow_pairs, 1)),
        "fraction_matched_flow_pairs_with_gt_dst_inside_target_flow": float(gt_in_target / max(n_labeled, 1)),
        "fraction_matched_flow_pairs_where_flow_match_correct_but_tx_decode_wrong": float(
            flow_ok_tx_wrong / max(n_labeled, 1)
        ),
        "n_unique_decoded_flow_pairs": n_flow_pairs,
        "n_labeled_src": n_labeled,
        "n_flow_correct_tx_wrong": flow_ok_tx_wrong,
    }


def _tx_level_metrics(
    mapping: dict[str, str],
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
) -> dict[str, Any]:
    truth = _truth_from_labels(label_df)
    delays: list[float] = []
    violations = 0
    total = 0
    for s in truth:
        d_pred = mapping.get(s, "")
        if not d_pred:
            continue
        ts_s = eth_ts.get(s)
        ts_d = bnb_ts.get(d_pred)
        if ts_s is None or ts_d is None:
            continue
        delay = float(ts_d - ts_s)
        delays.append(delay)
        total += 1
        if delay < 0:
            violations += 1
    dist = delay_distribution(delays)
    return {
        "tx_cvr": float(violations / max(total, 1)),
        "median_tx_delay": dist.get("median"),
        "tx_p05_sec": dist.get("p05"),
        "tx_p95_sec": dist.get("p95"),
        "n_evaluated": total,
        "negative_ratio": dist.get("negative_ratio"),
    }


def _current_decode_diagnostics(
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    mapping: dict[str, str],
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rule_counts: Counter[str] = Counter()
    for _, r in src_all.iterrows():
        txh = norm_addr(r.get("txhash", ""))
        if not txh or txh not in mapping:
            continue
        m = meta.get(txh) or {}
        j = int(m.get("flow_j", -1))
        if j < 0:
            continue
        dst_f = bnb_flows[j]
        picked = mapping.get(txh, "")
        rule = infer_legacy_selection_rule(
            safe_float(r.get("timestamp"), 0.0),
            safe_float(r.get("args.amount"), 0.0),
            dst_f,
            dst_norm,
            picked,
        )
        rule_counts[rule] += 1
    dominant = rule_counts.most_common(1)[0][0] if rule_counts else "unknown"
    return {
        "current_tx_decode_selection_rule": dominant,
        "current_tx_decode_selection_rule_counts": dict(rule_counts),
        "current_tx_decode_rule_description": (
            "legacy: min(amount_diff + time_distance/86400) with 1e6 penalty when dst_ts < src_ts; "
            "falls back to amount/time among all txs when no positive-delay candidate wins"
        ),
    }


def _evaluate_policy(
    policy: str,
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
) -> dict[str, Any]:
    truth = _truth_from_labels(label_df)
    mapping, meta = derive_tx_pairs(
        p, eth_flows, bnb_flows, src_all, dst_norm, policy=policy, label_truth=truth if policy == "choose_label_oracle_if_in_flow" else None
    )
    pairs_df = pd.DataFrame([{"srcTxHash": s, "dstTxHash": d} for s, d in mapping.items() if d])
    pr = pair_precision_recall_f1(pairs_df, label_df)
    txm = _tx_level_metrics(mapping, label_df, eth_ts, bnb_ts)
    fallbacks = sum(1 for m in meta.values() if m.get("tx_decode_fallback"))
    covered = sum(1 for s in truth if mapping.get(s))
    return {
        "policy": policy,
        "pair_f1": pr.get("pair_f1"),
        "top1_recall": pr.get("pair_recall"),
        "tx_cvr": txm["tx_cvr"],
        "median_tx_delay": txm["median_tx_delay"],
        "coverage": float(covered / max(len(truth), 1)),
        "selection_available_ratio": float((covered - fallbacks) / max(covered, 1)) if policy != "choose_label_oracle_if_in_flow" else txm.get("n_evaluated", 0) / max(len(truth), 1),
        "tx_decode_fallback_count": fallbacks,
        "tp": pr.get("tp"),
    }


def _classify_bad_case(
    *,
    src_tx: str,
    gt_dst: str,
    pred_dst: str,
    flow_i: int,
    flow_j: int,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    tx_to_j: dict[str, set[int]],
) -> str:
    ts_s = eth_ts.get(src_tx)
    ts_p = bnb_ts.get(pred_dst) if pred_dst else None
    if ts_s is None or ts_p is None:
        return "metadata_missing_timestamp"

    gt_j_set = tx_to_j.get(gt_dst, set())
    flow_correct = flow_j in gt_j_set
    gt_in_same_flow = gt_dst in {norm_addr(h) for h in (bnb_flows[flow_j].get("tx_hashes") or [])} if 0 <= flow_j < len(bnb_flows) else False

    if flow_correct and pred_dst != gt_dst and gt_in_same_flow:
        return "flow_match_correct_tx_selected_wrong"

    if not flow_correct:
        sf = eth_flows[flow_i] if 0 <= flow_i < len(eth_flows) else {}
        tf = bnb_flows[flow_j] if 0 <= flow_j < len(bnb_flows) else {}
        d_flow = flow_pair_delay_sec(sf, tf, policy="tx_if_available_else_flow_representative")
        if d_flow < 0:
            return "flow_match_wrong_target_before_source"

    if gt_in_same_flow and pred_dst != gt_dst:
        return "gt_target_in_same_flow_but_not_selected"

    # any positive delay tx in target flow?
    if 0 <= flow_j < len(bnb_flows):
        pos = False
        for h in bnb_flows[flow_j].get("tx_hashes") or []:
            hn = norm_addr(str(h))
            ts_d = bnb_ts.get(hn)
            if ts_d is not None and ts_s is not None and float(ts_d - ts_s) > 0:
                pos = True
                break
        if not pos:
            return "no_positive_delay_tx_in_target_flow"

    src_end = float(eth_flows[flow_i].get("end_time", 0.0)) if 0 <= flow_i < len(eth_flows) else 0.0
    if ts_s is not None and abs(float(ts_s) - src_end) < 300:
        return "source_tx_time_near_flow_end_boundary_issue"

    if ts_p is not None and ts_s is not None and float(ts_p) < float(ts_s):
        return "candidate_pool_contains_pre_source_target"

    return "flow_match_correct_tx_selected_wrong"


def run_negative_delay_decoding_audit(
    *,
    run_dir: Path,
    base_run: Path,
    eth_path: Path,
    bnb_path: Path,
    label_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    base_run = Path(base_run)

    p = _load_transport(run_dir)
    flow_root = run_dir / "uot" if (run_dir / "uot" / "uot_flow_segments_eth.csv").is_file() else base_run / "uot"
    eth_flows = load_flow_segments(flow_root / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(flow_root / "uot_flow_segments_bnb.csv")
    label_df = pd.read_csv(label_path)
    truth = _truth_from_labels(label_df)
    src_all = _prepare_src_all(eth_path, eth_flows)
    bnb_df = pd.read_csv(bnb_path, low_memory=False)
    eth_df = pd.read_csv(eth_path, low_memory=False)
    eth_ts, bnb_ts, ts_missing = _tx_timestamp_lookup(eth_df, bnb_df)

    layer_a = _flow_pair_level_metrics(p, eth_flows, bnb_flows)

    mapping, meta = derive_tx_pairs(p, eth_flows, bnb_flows, src_all, bnb_df, policy="legacy")
    layer_b = _labeled_tx_within_flow_pair(p, eth_flows, bnb_flows, label_df, eth_ts, bnb_ts, mapping)
    layer_c_tx = _tx_level_metrics(mapping, label_df, eth_ts, bnb_ts)
    layer_c_rule = _current_decode_diagnostics(p, eth_flows, bnb_flows, src_all, bnb_df, mapping, meta)

    policy_rows: list[dict[str, Any]] = []
    for pol in DIAGNOSTIC_POLICIES:
        policy_rows.append(
            _evaluate_policy(pol, p, eth_flows, bnb_flows, src_all, bnb_df, label_df, eth_ts, bnb_ts)
        )

    tx_to_j = _tx_to_target_flows(bnb_flows)
    taxonomy_rows: list[dict[str, Any]] = []
    flow_correct_tx_wrong_examples: list[dict[str, Any]] = []
    flow_wrong_examples: list[dict[str, Any]] = []

    for s, gt_d in truth.items():
        pred_d = mapping.get(s, "")
        if not pred_d:
            continue
        ts_s = eth_ts.get(s)
        ts_p = bnb_ts.get(pred_d)
        if ts_s is None or ts_p is None:
            continue
        delay = float(ts_p - ts_s)
        if delay >= 0:
            continue
        m = meta.get(s, {})
        i = int(m.get("flow_i", -1))
        j = int(m.get("flow_j", -1))
        tax = _classify_bad_case(
            src_tx=s,
            gt_dst=gt_d,
            pred_dst=pred_d,
            flow_i=i,
            flow_j=j,
            eth_flows=eth_flows,
            bnb_flows=bnb_flows,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            tx_to_j=tx_to_j,
        )
        row = {
            "srcTxHash": s,
            "gtDstTxHash": gt_d,
            "predDstTxHash": pred_d,
            "delay_tx_sec": delay,
            "taxonomy": tax,
            "source_flow_id": m.get("source_flow_id"),
            "target_flow_id": m.get("target_flow_id"),
        }
        taxonomy_rows.append(row)
        if tax == "flow_match_correct_tx_selected_wrong" and len(flow_correct_tx_wrong_examples) < 15:
            flow_correct_tx_wrong_examples.append(row)
        if tax in ("flow_match_wrong_target_before_source",) and len(flow_wrong_examples) < 15:
            flow_wrong_examples.append(row)

    tax_counts = Counter(r["taxonomy"] for r in taxonomy_rows)
    top3 = topk_flow_accuracy(p, eth_flows, bnb_flows, label_df, k=3)

    # Case determination
    flow_cvr = float(layer_a["flow_pair_cvr"])
    tx_cvr = float(layer_c_tx["tx_cvr"])
    if flow_cvr < 0.15 and tx_cvr > 0.3:
        audit_case = "case_1_flow_ok_tx_decode_bad"
    elif flow_cvr >= 0.15:
        audit_case = "case_2_flow_match_also_bad"
    else:
        audit_case = "case_3_mixed"

    best_non_oracle = max(
        (r for r in policy_rows if r["policy"] != "choose_label_oracle_if_in_flow"),
        key=lambda r: (-float(r.get("tx_cvr") or 1), float(r.get("pair_f1") or 0)),
    )
    oracle_row = next((r for r in policy_rows if r["policy"] == "choose_label_oracle_if_in_flow"), None)
    legacy_row = next((r for r in policy_rows if r["policy"] == "legacy"), None)

    recommend_tx_fix = (
        audit_case == "case_1_flow_ok_tx_decode_bad"
        and float(best_non_oracle.get("tx_cvr") or 1) < float(legacy_row.get("tx_cvr") or 1) * 0.7
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "base_run": str(base_run.resolve()),
        "audit_case": audit_case,
        "recommend_tx_decode_fix": recommend_tx_fix,
        "recommended_policy": best_non_oracle.get("policy") if recommend_tx_fix else None,
        "layer_a_flow_pair": layer_a,
        "layer_b_labeled_within_flow": layer_b,
        "layer_c_current_tx_decode": {**layer_c_tx, **layer_c_rule},
        "flow_level_top3_recall": top3,
        "taxonomy_counts": dict(tax_counts),
        "n_negative_delay_labeled_pairs": len(taxonomy_rows),
        "tx_timestamp_lookup_missing": ts_missing,
        "oracle_in_flow_diagnostic": oracle_row,
        "legacy_policy": legacy_row,
    }

    (out_dir / "negative_delay_decoding_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Negative delay decoding audit",
        "",
        f"**Audit case:** `{audit_case}`",
        "",
        "## Layer A — flow-pair representative",
        "",
        f"- flow_pair_cvr: **{layer_a['flow_pair_cvr']:.4f}**",
        f"- flow_pair_median_delay: **{layer_a['flow_pair_median_delay']}** s",
        f"- flow_pair_p05/p95: {layer_a['flow_pair_p05_sec']} / {layer_a['flow_pair_p95_sec']} s",
        "",
        "## Layer C — current tx hard decode",
        "",
        f"- current_tx_decode_cvr: **{layer_c_tx['tx_cvr']:.4f}**",
        f"- current_tx_decode_median_delay: **{layer_c_tx['median_tx_delay']}** s",
        f"- selection rule (dominant): **{layer_c_rule['current_tx_decode_selection_rule']}**",
        "",
        "## Layer B — labeled tx within matched flow",
        "",
        f"- fraction flow pairs with any positive labeled tx pair: {layer_b['fraction_matched_flow_pairs_with_any_positive_labeled_tx_pair']:.4f}",
        f"- fraction GT dst inside predicted target flow: {layer_b['fraction_matched_flow_pairs_with_gt_dst_inside_target_flow']:.4f}",
        f"- fraction flow correct but tx wrong: {layer_b['fraction_matched_flow_pairs_where_flow_match_correct_but_tx_decode_wrong']:.4f}",
        "",
        "## Taxonomy (top)",
        "",
    ]
    for k, v in tax_counts.most_common(10):
        md_lines.append(f"- {k}: {v}")
    (out_dir / "negative_delay_decoding_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    pd.DataFrame(policy_rows).to_csv(out_dir / "tx_selection_policy_comparison.csv", index=False)
    pol_md = ["# Tx selection policy comparison", "", "| policy | pair_f1 | tx_cvr | median_tx_delay | coverage |", "|--------|--------:|-------:|----------------:|---------:|"]
    for r in policy_rows:
        pol_md.append(
            f"| {r['policy']} | {r.get('pair_f1', '')} | {r.get('tx_cvr', '')} | {r.get('median_tx_delay', '')} | {r.get('coverage', '')} |"
        )
    (out_dir / "tx_selection_policy_comparison.md").write_text("\n".join(pol_md) + "\n", encoding="utf-8")

    pd.DataFrame(taxonomy_rows).to_csv(out_dir / "bad_case_taxonomy.csv", index=False)
    tax_md = ["# Bad case taxonomy", ""]
    for k, v in tax_counts.most_common():
        tax_md.append(f"- **{k}**: {v}")
    (out_dir / "bad_case_taxonomy.md").write_text("\n".join(tax_md) + "\n", encoding="utf-8")

    ex1 = ["# Examples: flow correct, tx wrong", ""]
    for r in flow_correct_tx_wrong_examples[:10]:
        ex1.append(f"- `{r['srcTxHash'][:16]}…` pred `{r['predDstTxHash'][:16]}…` gt `{r['gtDstTxHash'][:16]}…` delay={r['delay_tx_sec']:.0f}s")
    (out_dir / "examples_flow_correct_tx_wrong.md").write_text("\n".join(ex1) + "\n", encoding="utf-8")

    ex2 = ["# Examples: flow wrong", ""]
    for r in flow_wrong_examples[:10]:
        ex2.append(f"- `{r['srcTxHash'][:16]}…` taxonomy={r['taxonomy']} delay={r['delay_tx_sec']:.0f}s")
    (out_dir / "examples_flow_wrong.md").write_text("\n".join(ex2) + "\n", encoding="utf-8")

    if recommend_tx_fix:
        paper_text = f"""# Decoding fix recommendation

Audit case **Case 1**: flow-level representative CVR is low ({flow_cvr:.1%}) while tx-level CVR is high ({tx_cvr:.1%}).
The UOT flow correspondence is temporally consistent; negative delays arise from tx selection within matched target flows.

Recommended production tx decode policy: `{best_non_oracle.get('policy')}`.
Expected tx CVR: {best_non_oracle.get('tx_cvr'):.4f} (legacy: {legacy_row.get('tx_cvr'):.4f}).
Expected pair F1: {best_non_oracle.get('pair_f1'):.4f} (legacy: {legacy_row.get('pair_f1'):.4f}).
"""
    elif audit_case == "case_2_flow_match_also_bad":
        paper_text = """# Decoding limitation (Case 2)

Flow-level representative CVR remains high. Do not patch tx decoder alone; investigate cost weights and candidate universe.
"""
    else:
        paper_text = f"""# Decoding limitation or fix

The UOT flow-level correspondence is temporally consistent under representative flow delay (flow_pair_cvr={flow_cvr:.4f}),
but tx-level compatibility decoding can select pre-source transactions within a matched target flow (tx_cvr={tx_cvr:.4f}).
We therefore report flow-level correspondence as the primary object and treat tx-level hard pairs as a compatibility artifact.

Oracle-in-flow diagnostic pair_f1={oracle_row.get('pair_f1') if oracle_row else 'n/a'} shows flow-level match carries label information;
non-oracle tx selection remains the bottleneck.
"""
    (out_dir / "paper_decoding_limitation_or_fix.md").write_text(paper_text, encoding="utf-8")

    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Audit negative-delay tx decoding attribution.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--base-run", type=Path, default=Path("out/leave_anchor_out_real"))
    p.add_argument("--eth", type=Path, required=True)
    p.add_argument("--bnb", type=Path, required=True)
    p.add_argument("--label", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("out/negative_delay_decoding_audit"))
    args = p.parse_args()
    result = run_negative_delay_decoding_audit(
        run_dir=args.run_dir,
        base_run=args.base_run,
        eth_path=args.eth,
        bnb_path=args.bnb,
        label_path=args.label,
        out_dir=args.out,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("layer_a_flow_pair", "taxonomy_counts")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
