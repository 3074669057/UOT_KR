#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 0.2: projection-policy audit for baseline_compare.

Audits tx-flow membership multiplicity and freezes baseline comparison policy.
All outputs under cross/out/baseline_compare/projection_audit/.
Does not enter Phase 1.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.application.experiments.uot_cache_utils import load_flow_segments  # noqa: E402
from cross.shared.normalize import norm_addr  # noqa: E402

OUT = REPO / "out" / "baseline_compare"
AUDIT = OUT / "projection_audit"
LABELS = OUT / "labels"

LAO_ETH = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_eth.csv"
LAO_BNB = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_bnb.csv"
GT_TX = REPO / "label" / "celer_label.csv"
ADM_SUMMARY = REPO / "out" / "admissible_decoding" / "admissible_decoding_summary.json"


def _norm(h: Any) -> str:
    return norm_addr(str(h or ""))


def _stats(counts: list[int]) -> dict[str, float | int]:
    if not counts:
        return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0, "n": 0}
    return {
        "min": int(min(counts)),
        "median": float(statistics.median(counts)),
        "mean": float(sum(counts) / len(counts)),
        "max": int(max(counts)),
        "n": len(counts),
    }


def _membership_from_segments(path: Path) -> tuple[dict[str, set[str]], dict[str, int]]:
    """tx_hash -> set(flow_id); flow_id -> tx count."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    tx_to_flows: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        fid = str(row["flow_id"])
        for tx in str(row.get("tx_hashes", "")).split("|"):
            tx = tx.strip()
            if tx:
                tx_to_flows[_norm(tx)].add(fid)
    per_tx = {tx: len(flows) for tx, flows in tx_to_flows.items()}
    return dict(tx_to_flows), per_tx


def _audit_membership() -> dict[str, Any]:
    eth_tx_to_flows, eth_per_tx = _membership_from_segments(LAO_ETH)
    bnb_tx_to_flows, bnb_per_tx = _membership_from_segments(LAO_BNB)

    gt = pd.read_csv(GT_TX, dtype=str, keep_default_na=False)
    gt_src = [_norm(x) for x in gt["srcTxhash"]]
    gt_dst = [_norm(x) for x in gt["dstTxhash"]]

    eth_gt_counts = [eth_per_tx.get(s, 0) for s in gt_src]
    bnb_gt_counts = [bnb_per_tx.get(d, 0) for d in gt_dst]

    eth_multi = sum(1 for c in eth_per_tx.values() if c > 1)
    bnb_multi = sum(1 for c in bnb_per_tx.values() if c > 1)
    eth_gt_multi = sum(1 for c in eth_gt_counts if c > 1)
    bnb_gt_multi = sum(1 for c in bnb_gt_counts if c > 1)

    # GT tx-pair -> possible flow-pairs (Cartesian of src_flows x dst_flows)
    flow_pair_counts: list[int] = []
    flow_pair_examples: list[dict[str, Any]] = []
    unique_flow_pairs: set[tuple[str, str]] = set()
    collapse_counter: Counter[tuple[str, str]] = Counter()

    for s, d in zip(gt_src, gt_dst):
        sf = eth_tx_to_flows.get(s, set())
        df = bnb_tx_to_flows.get(d, set())
        n_fp = len(sf) * len(df) if sf and df else 0
        flow_pair_counts.append(n_fp)
        for a in sf:
            for b in df:
                pair = (a, b)
                unique_flow_pairs.add(pair)
                collapse_counter[pair] += 1
        if n_fp > 1 and len(flow_pair_examples) < 5:
            flow_pair_examples.append(
                {
                    "src_tx": s,
                    "dst_tx": d,
                    "src_flows": sorted(sf),
                    "dst_flows": sorted(df),
                    "n_possible_flow_pairs": n_fp,
                }
            )

    collapsed = {k: v for k, v in collapse_counter.items() if v > 1}
    collapse_hist = Counter(collapsed.values())

    # Cross-check exported label files
    eth_cand = pd.read_csv(LABELS / "candidate_eth_universe_all_txs.csv", dtype=str)
    bnb_cand = pd.read_csv(LABELS / "candidate_bnb_universe_all_txs.csv", dtype=str)
    bnb_mem = pd.read_csv(LABELS / "candidate_bnb_tx_flow_memberships.csv", dtype=str)

    eth_cand_mismatch = int(
        sum(
            1
            for _, r in eth_cand.iterrows()
            if eth_per_tx.get(_norm(r["tx_hash"]), 0) != int(r.get("n_src_flow_memberships", 0))
        )
    )
    bnb_cand_mismatch = int(
        sum(
            1
            for _, r in bnb_cand.iterrows()
            if bnb_per_tx.get(_norm(r["tx_hash"]), 0) != int(r.get("n_dst_flow_memberships", 0))
        )
    )

    # RC-UOT-Q _tx_to_flow_index: last flow index wins for ETH src
    eth_flows = load_flow_segments(LAO_ETH)
    tx_to_i: dict[str, int] = {}
    for i, sf in enumerate(eth_flows):
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[_norm(str(txh))] = i

    eth_ambiguous_for_rc_uot = sum(1 for s in gt_src if eth_per_tx.get(s, 0) > 1)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "lao_eth_segments": str(LAO_ETH.relative_to(REPO)),
            "lao_bnb_segments": str(LAO_BNB.relative_to(REPO)),
            "gt_tx_pairs": str(GT_TX.relative_to(REPO)),
        },
        "eth_src_tx_to_src_flow": {
            "stats_all_txs_in_segments": _stats(list(eth_per_tx.values())),
            "stats_gt_src_txs": _stats(eth_gt_counts),
            "n_tx_with_multiple_src_flows": eth_multi,
            "n_gt_src_tx_with_multiple_src_flows": eth_gt_multi,
            "all_gt_src_tx_map_to_exactly_one_src_flow": eth_gt_multi == 0,
            "distribution_gt_src_flow_count": dict(Counter(eth_gt_counts)),
        },
        "bnb_dst_tx_to_dst_flow": {
            "stats_all_txs_in_segments": _stats(list(bnb_per_tx.values())),
            "stats_gt_dst_txs": _stats(bnb_gt_counts),
            "n_tx_with_multiple_dst_flows": bnb_multi,
            "n_gt_dst_tx_with_multiple_dst_flows": bnb_gt_multi,
            "all_gt_dst_tx_map_to_exactly_one_dst_flow": bnb_gt_multi == 0,
            "distribution_gt_dst_flow_count": dict(Counter(bnb_gt_counts)),
        },
        "gt_tx_pair_to_flow_pair_projection": {
            "n_tx_pairs": len(gt_src),
            "stats_possible_flow_pairs_per_tx_pair": _stats(flow_pair_counts),
            "n_unique_flow_pairs": len(unique_flow_pairs),
            "n_collapsed_flow_pairs": len(collapsed),
            "collapse_histogram_tx_pairs_per_flow_pair": {str(k): v for k, v in sorted(collapse_hist.items())},
            "max_tx_pairs_per_flow_pair": max(collapse_counter.values()) if collapse_counter else 0,
            "has_one_to_many": any(c > 1 for c in flow_pair_counts),
            "has_many_to_many": any(
                eth_per_tx.get(s, 0) > 1 and bnb_per_tx.get(d, 0) > 1 for s, d in zip(gt_src, gt_dst)
            ),
            "examples_multi_flow_pair": flow_pair_examples,
        },
        "label_file_crosscheck": {
            "candidate_eth_universe_all_txs_mismatch_rows": eth_cand_mismatch,
            "candidate_bnb_universe_all_txs_mismatch_rows": bnb_cand_mismatch,
            "candidate_bnb_tx_flow_membership_rows": len(bnb_mem),
        },
        "rc_uot_q_production_mapping": {
            "_tx_to_flow_index_used_for_src": True,
            "_tx_to_target_flows_used_for_dst": True,
            "eth_src_tx_in_multiple_flows_count": eth_multi,
            "gt_src_tx_would_be_ambiguous_under_last_wins_index": eth_ambiguous_for_rc_uot,
            "note": (
                "RC-UOT-Q maps each src_tx to one source flow index via _tx_to_flow_index (last segment wins if duplicate). "
                "For dst evaluation, top3_recall uses _target_flow_indices_for_tx (all flows containing dst tx)."
            ),
        },
        "gt_flow_pairs_csv_caveat": {
            "phase0_file": "labels/gt_flow_pairs.csv",
            "n_unique_flow_pairs_last_wins_bnb": 3934,
            "n_unique_flow_pairs_membership_aware": len(unique_flow_pairs),
            "explanation": (
                "Phase 0 gt_flow_pairs.csv assigned one dst_flow per dst_tx via dict last-wins on tx_to_flow_map. "
                "Membership-aware projection yields more unique flow-pairs. "
                "Do not use gt_flow_pairs.csv for strict flow-level TP."
            ),
        },
    }
    return audit


def _write_membership_md(audit: dict[str, Any]) -> None:
    e = audit["eth_src_tx_to_src_flow"]
    b = audit["bnb_dst_tx_to_dst_flow"]
    p = audit["gt_tx_pair_to_flow_pair_projection"]
    lines = [
        "# Tx-flow membership audit (Phase 0.2)",
        "",
        f"Generated: {audit['generated_at_utc']}",
        "",
        "## ETH: src_tx → src_flow",
        "",
        f"| Stat | All ETH txs | GT src txs |",
        f"|------|------------:|-----------:|",
    ]
    for key in ("min", "median", "mean", "max"):
        lines.append(
            f"| {key} | {e['stats_all_txs_in_segments'][key]} | {e['stats_gt_src_txs'][key]} |"
        )
    lines.extend(
        [
            "",
            f"- Tx with multiple src flows (all): **{e['n_tx_with_multiple_src_flows']}**",
            f"- GT src tx with multiple src flows: **{e['n_gt_src_tx_with_multiple_src_flows']}**",
            f"- All GT src tx uniquely map to one src flow: **{e['all_gt_src_tx_map_to_exactly_one_src_flow']}**",
            "",
            "## BNB: dst_tx → dst_flow",
            "",
            f"| Stat | All BNB txs | GT dst txs |",
            f"|------|------------:|-----------:|",
        ]
    )
    for key in ("min", "median", "mean", "max"):
        lines.append(
            f"| {key} | {b['stats_all_txs_in_segments'][key]} | {b['stats_gt_dst_txs'][key]} |"
        )
    lines.extend(
        [
            "",
            f"- Tx with multiple dst flows (all): **{b['n_tx_with_multiple_dst_flows']}**",
            f"- GT dst tx with multiple dst flows: **{b['n_gt_dst_tx_with_multiple_dst_flows']}**",
            f"- All GT dst tx uniquely map to one dst flow: **{b['all_gt_dst_tx_map_to_exactly_one_dst_flow']}**",
            "",
            "## GT tx-pair → possible flow-pair",
            "",
            f"- tx-pairs: **{p['n_tx_pairs']}**",
            f"- unique flow-pairs (membership-aware): **{p['n_unique_flow_pairs']}**",
            f"- Phase 0 `gt_flow_pairs.csv` (BNB last-wins): **3934** unique flow-pairs — not membership-aware",
            f"- collapsed flow-pairs (>1 tx-pair): **{p['n_collapsed_flow_pairs']}**",
            f"- max tx-pairs per flow-pair: **{p['max_tx_pairs_per_flow_pair']}**",
            "",
            f"| Stat | possible flow-pairs per tx-pair |",
            f"|------|--------------------------------:|",
        ]
    )
    s = p["stats_possible_flow_pairs_per_tx_pair"]
    for key in ("min", "median", "mean", "max"):
        lines.append(f"| {key} | {s[key]} |")
    lines.extend(
        [
            "",
            f"- one-to-many projection exists: **{p['has_one_to_many']}**",
            f"- many-to-many projection exists: **{p['has_many_to_many']}**",
            "",
            "## RC-UOT-Q mapping note",
            "",
            audit["rc_uot_q_production_mapping"]["note"],
            "",
        ]
    )
    (AUDIT / "tx_flow_membership_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metric_unit_audit() -> None:
    summary = json.loads(ADM_SUMMARY.read_text(encoding="utf-8"))
    text = f"""# RC-UOT-Q metric unit audit (Phase 0.2)

Generated: {datetime.now(timezone.utc).isoformat()}

Sources:
- `src/cross/application/experiments/run_admissible_decoding.py` (`_build_predictions`, `_evaluate_strategy`, `_decode_src_tx`)
- `src/cross/domain/evaluation/flow_metrics.py` (`pair_precision_recall_f1`, `topk_flow_accuracy`)
- `src/cross/domain/uot/tx_decode_policy.py` (`pick_dst_tx_in_flow`)
- `out/admissible_decoding/admissible_decoding_summary.json`

## 1. Main-table pair precision / recall / F1

**Unit: tx-pair exact match** (one predicted dst tx hash per labeled src tx hash).

Pipeline:
1. UOT transport matrix `P[i,j]` is **flow-level** (3258 × 5226).
2. Each labeled **src_tx** maps to one source flow index `i` via `_tx_to_flow_index(eth_flows)`.
3. `_decode_src_tx` selects target flow(s) `j` from row `P[i,:]` (argmax or top-k rescue + admissibility).
4. `pick_dst_tx_in_flow(...)` decodes a concrete **dst_tx** inside target flow `j`.
5. `_evaluate_strategy` builds `pairs_df` with columns `srcTxHash`, `dstTxHash`.
6. `pair_precision_recall_f1(pairs_df, label_df)` compares against `celer_label.csv` (**tx-pair GT**).

**Not** primary flow-pair match. Flow-pair is intermediate; final headline P/R/F1 is **tx-pair exact match**.

Flow-level diagnostic: `flow_level_recall` in summary (fraction of predictions where predicted target flow index contains GT dst tx).

## 2. coverage

**Denominator: `n_source_flows` = 3258** (all ETH flow segments in transport plan).

Definition (from admissible_decoding_summary.json):
> {summary.get('coverage_definition', '')}

Computed as: `len(covered_flows) / len(eth_flows)` where `covered_flows` = source flow indices with ≥1 **non-abstained** tx projection among evaluated labeled pairs.

**Not** tx-pair coverage. **Not** expected to complement abstention_rate.

## 3. abstention_rate

**Denominator: `n_ground_truth_pairs` = 7296** (labeled tx-pairs).

Definition:
> {summary.get('abstention_rate_definition', '')}

Computed as: `n_abstained / n_ground_truth_pairs`.

Abstention = no dst tx projected for that labeled src tx.

## 4. tx_level_cvr (tx-CVR)

**Denominator: non-abstained predicted tx-pairs only** (`n_tx_pairs` with both timestamps available).

From `_evaluate_strategy`:
- For each non-abstained prediction, compute `tx_delay = bnb_ts[pred_dst] - eth_ts[src]`.
- `tx_level_cvr = count(tx_delay < 0) / count(evaluated tx pairs with timestamps)`.

CVR is **not** computed on abstained pairs.

## 5. top3_recall

**Unit: tx-labeled examples, flow-ranked retrieval.**

From `topk_flow_accuracy` docstring:
- For each labeled **src_tx** → source flow row `P[i,:]`.
- Rank **target flows** by descending transport mass.
- Hit if **any** target flow containing the **true dst tx hash** appears in top-k.
- Uses `_target_flow_indices_for_tx` because a dst tx may belong to **multiple** target flows.

**Not** tx-ranking top-3. **Not** dst-tx top-3. It is **flow-level top-k retrieval** evaluated on tx-labeled rows.

Note in summary: `top3_recall reflects transport ranking, not admissibility filter.`

## Implication for baseline comparison

| RC-UOT-Q metric | Baseline alignment |
|-----------------|-------------------|
| pair P/R/F1 | Connector raw: same tx-pair exact match vs `gt_tx_pairs.csv` |
| coverage | Requires mapping abstentions to source flows; Connector uses per-src-tx outputs |
| abstention_rate | Per labeled src tx (7296 denominator) |
| tx_level_cvr | Non-abstained tx-pairs only |
| top3_recall | Requires flow-level ranking matrix; Connector N/A without native top-k |
"""
    (AUDIT / "metric_unit_audit.md").write_text(text, encoding="utf-8")


def _write_baseline_policy(audit: dict[str, Any]) -> None:
    b = audit["bnb_dst_tx_to_dst_flow"]
    e = audit["eth_src_tx_to_src_flow"]
    p = audit["gt_tx_pair_to_flow_pair_projection"]

    text = f"""# Baseline comparison policy (frozen, Phase 0.2)

Generated: {datetime.now(timezone.utc).isoformat()}

Status: **frozen for Phase 1+** unless user revises.

## Shared evaluation substrate

| Role | File |
|------|------|
| tx-pair GT | `labels/gt_tx_pairs.csv` |
| GT dst txs (labels only) | `labels/gt_dst_txs.csv` |
| BNB candidate pool (shared) | `labels/candidate_bnb_universe_all_txs.csv` |
| BNB flow universe | `labels/candidate_bnb_universe_flows.csv` |
| tx→flow memberships | `labels/candidate_bnb_tx_flow_memberships.csv` |
| RC-UOT-Q reference | `rc_uot_q_frozen/rc_uot_q_reference_metrics.json` |

## A. Connector raw (primary)

- **Output:** `(src_tx, dst_tx)` top-1 from original `WithdrawLocator.search_withdraw()`.
- **Primary metrics:** `pair_precision`, `pair_recall`, `pair_f1` via tx-pair exact match against `gt_tx_pairs.csv`.
- **Do not** require unique flow projection for raw P/R/F1 — BNB `dst_tx → dst_flow` is one-to-many ({b['n_gt_dst_tx_with_multiple_dst_flows']}/{b['stats_gt_dst_txs']['n']} GT dst txs in multiple flows).
- **Do not** use `gt_dst_txs.csv` as search pool; use `candidate_bnb_universe_all_txs.csv`.

## B. Connector top1 admissible (diagnostic only)

- **Method name:** `connector_top1_admissible_filter` (never `joint_time_admissible_filter`).
- **Input:** Connector top-1 `(src_tx, dst_tx)`.
- **Rule:** same fixed-delay timestamp tables as RC-UOT-Q; keep if `bnb_ts[dst] - eth_ts[src] >= 0`; else abstain.
- **Metrics:** pair P/R/F1, tx_level_cvr, coverage, abstention_rate using same denominators as RC-UOT-Q audit.
- **Not** RC-UOT-Q joint parity (no flow-level admissibility on ranked target flows).

## C. Connector top3 / joint

- **Status: N/A** — original `search_withdraw()` returns top-1 only; `fulloutput=True` gives filter counts, not ranking.
- **Forbidden:** modify WithdrawLocator rules, forge ranking, or label as RC-UOT-Q `positive_delay_top3_rescue` / `joint_time_admissible_filter`.

## D. Flow-level projection (diagnostic only, not primary raw)

Membership facts:
- ETH GT src tx → src_flow: **unique** ({e['all_gt_src_tx_map_to_exactly_one_src_flow']}).
- BNB GT dst tx → dst_flow: **one-to-many** (median {b['stats_gt_dst_txs']['median']} flows per dst tx).
- tx-pair → flow-pair: up to {p['stats_possible_flow_pairs_per_tx_pair']['max']} possible flow-pairs per tx-pair; {p['n_unique_flow_pairs']} unique flow-pairs from 7296 tx-pairs.

### Strict projection (diagnostic)

Given baseline prediction `(src_tx, dst_tx)`:
- `src_flow` = unique LAO src flow containing `src_tx` (exactly one for GT src txs).
- `dst_flows` = **all** LAO dst flows containing `dst_tx`.
- **Strict TP** if ∃ `dst_flow` in that set such that `(src_flow, dst_flow)` matches a GT flow-pair projection of `(src_tx, dst_tx)`.
- If predicted `dst_tx` wrong, strict flow-pair is FP for each expanded `(src_flow, dst_flow)` candidate.

### Relaxed projection (diagnostic only — not main table)

- Pick **one** dst_flow arbitrarily (e.g. first lexicographic) → **deprecated for main metrics**; inflates/deflates flow-pair scores when dst_tx is one-to-many.
- Phase 0 `labels/gt_flow_pairs.csv` effectively used this pattern (BNB dict last-wins) → 3934 unique flow-pairs vs **7386** membership-aware.
- **Never** mix relaxed projection into headline comparison table.

### baseline_ranking_matrix (if needed later)

- Name: `baseline_ranking_matrix` or `baseline_score_matrix` only.
- Shape concept: [3258 × 5226] for flow-level top-k metrics.
- **Forbidden names:** pseudo-P, pseudo-transport, transport_matrix.
- **Forbidden metric on baseline:** `flow_mass_recall`.

## E. Candidate pool disclosure (required prose)

- Shared pool = `candidate_bnb_universe_all_txs.csv` (7296 unique dst txs in 5226 LAO dst flows).
- **Closed-set strict subgraph:** tx-hash candidate set equals GT dst set; **not** open-world full-chain discovery.
- Each src must still identify the correct counterparty among all 7296 dst txs (global closed pool).
- 14624 rows in `candidate_bnb_tx_flow_memberships.csv` = tx→flow memberships, not extra unique txs.

## ABCTracer

- **Gate-1: BLOCKED** (no official checkpoint). No Phase 1 run until checkpoint supplied or user approves training.

## Phase 1 permissions (from this policy)

| Item | Allowed |
|------|---------|
| Connector raw tx-pair exact-match | **YES** |
| Connector top1 admissible diagnostic | **YES** |
| Connector top3 / RC-UOT-Q joint parity | **NO** |
| ABCTracer | **NO** (BLOCKED) |
| Flow-level projection in main table | **NO** (diagnostic only) |
"""
    (AUDIT / "baseline_comparison_policy.md").write_text(text, encoding="utf-8")


def _phase02_conclusion(audit: dict[str, Any]) -> str:
    b = audit["bnb_dst_tx_to_dst_flow"]
    if b["n_gt_dst_tx_with_multiple_dst_flows"] > 0:
        return "WARN"
    return "PASS"


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)

    audit = _audit_membership()
    (AUDIT / "tx_flow_membership_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_membership_md(audit)
    _write_metric_unit_audit()
    _write_baseline_policy(audit)

    conclusion = _phase02_conclusion(audit)

    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["phase0_2"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conclusion": conclusion,
        "membership_audit_summary": {
            "eth_gt_src_unique_src_flow": audit["eth_src_tx_to_src_flow"]["all_gt_src_tx_map_to_exactly_one_src_flow"],
            "bnb_gt_dst_multi_flow_count": audit["bnb_dst_tx_to_dst_flow"]["n_gt_dst_tx_with_multiple_dst_flows"],
            "n_unique_flow_pairs_membership_aware": audit["gt_tx_pair_to_flow_pair_projection"]["n_unique_flow_pairs"],
            "n_unique_flow_pairs_phase0_gt_flow_pairs_last_wins": 3934,
            "gt_flow_pairs_csv_not_membership_aware": True,
            "tx_flow_ambiguity_affects_tx_pair_metrics": False,
            "tx_flow_ambiguity_affects_flow_projection_diagnostics": audit["bnb_dst_tx_to_dst_flow"]["n_gt_dst_tx_with_multiple_dst_flows"] > 0,
        },
        "phase1_permissions": {
            "connector_raw_tx_pair_exact_match": True,
            "connector_top1_admissible_diagnostic": True,
            "connector_top3": False,
            "connector_joint_rc_uot_q_parity": False,
            "abctracer": False,
            "abctracer_blocked": True,
        },
        "artifacts": {
            "tx_flow_membership_audit": "projection_audit/tx_flow_membership_audit.json",
            "metric_unit_audit": "projection_audit/metric_unit_audit.md",
            "baseline_comparison_policy": "projection_audit/baseline_comparison_policy.md",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight = OUT / "preflight_report.md"
    block = f"""
---

## Phase 0.2 — projection-policy audit

Generated: {manifest['phase0_2']['generated_at_utc']}

**Conclusion: {conclusion}**

### Membership summary

| Side | GT txs with multiple flows | All txs multi-flow |
|------|---------------------------:|-------------------:|
| ETH src | {audit['eth_src_tx_to_src_flow']['n_gt_src_tx_with_multiple_src_flows']} | {audit['eth_src_tx_to_src_flow']['n_tx_with_multiple_src_flows']} |
| BNB dst | {audit['bnb_dst_tx_to_dst_flow']['n_gt_dst_tx_with_multiple_dst_flows']} | {audit['bnb_dst_tx_to_dst_flow']['n_tx_with_multiple_dst_flows']} |

- 7296 tx-pairs → {audit['gt_tx_pair_to_flow_pair_projection']['n_unique_flow_pairs']} unique flow-pairs
- tx-flow ambiguity affects **tx-pair P/R/F1**: **NO** (tx-pair exact match needs no flow projection)
- tx-flow ambiguity affects **flow projection diagnostics**: **YES** (BNB dst one-to-many)

### Phase 1 permissions

| Item | Status |
|------|--------|
| Connector raw tx-pair exact-match | **ALLOWED** |
| Connector top1 admissible diagnostic | **ALLOWED** |
| Connector top3 / joint RC-UOT-Q parity | **FORBIDDEN** |
| ABCTracer | **BLOCKED** |

Phase 1 **not entered**.
"""
    preflight.write_text(preflight.read_text(encoding="utf-8").rstrip() + block + "\n", encoding="utf-8")

    print(f"Phase 0.2 complete. Conclusion: {conclusion}")
    print(f"BNB GT dst multi-flow: {audit['bnb_dst_tx_to_dst_flow']['n_gt_dst_tx_with_multiple_dst_flows']}")
    print(f"unique flow-pairs: {audit['gt_tx_pair_to_flow_pair_projection']['n_unique_flow_pairs']}")


if __name__ == "__main__":
    main()
