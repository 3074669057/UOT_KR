#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 0: frozen evaluation substrate + Gate-1 feasibility audit only.

All outputs under cross/out/baseline_compare/. Does not run baselines or modify frozen paper artifacts.
"""
from __future__ import annotations

import hashlib
import json
import sys
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
LABELS = OUT / "labels"
GATE = OUT / "gate_reports"
FROZEN = OUT / "rc_uot_q_frozen"

LAO_ETH = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_eth.csv"
LAO_BNB = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_bnb.csv"
GT_TX_SRC = REPO / "label" / "celer_label.csv"

MAIN_TABLE = REPO / "out" / "final_paper_tables" / "main_table_rc_uot_q_fixed_delay.json"
ADM_SUMMARY = REPO / "out" / "admissible_decoding" / "admissible_decoding_summary.json"
ADM_MANIFEST = REPO / "out" / "admissible_decoding" / "admissible_decoding_manifest.json"

CONNECTOR_ROOT = REPO.parent / "Connector" / "Connector-main"
ABCT_ROOT = REPO.parent / "ABCTracer"

HEADLINE_METHODS = (
    "raw_argmax_fixed_delay",
    "positive_delay_top3_rescue",
    "joint_time_admissible_filter",
)

METRIC_KEYS = (
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "top3_recall",
    "tx_level_cvr",
    "flow_pair_cvr",
    "coverage",
    "abstention_rate",
    "n_true_positive",
    "n_false_positive",
    "n_abstained",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tx_to_flow_index(source_flows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            out[norm_addr(str(txh))] = i
    return out


def _build_tx_to_flow_map_rows(seg_path: Path, chain: str) -> list[dict[str, str]]:
    df = pd.read_csv(seg_path, dtype=str, keep_default_na=False)
    rows: list[dict[str, str]] = []
    for _, r in df.iterrows():
        fid = str(r["flow_id"])
        for tx in str(r.get("tx_hashes", "")).split("|"):
            tx = tx.strip()
            if not tx:
                continue
            rows.append(
                {
                    "chain": chain,
                    "tx_hash": norm_addr(tx),
                    "flow_id": fid,
                    "mapping_source": "lao_flow_segments_lookup",
                }
            )
    return rows


def _export_labels() -> dict[str, Any]:
    eth_rows = _build_tx_to_flow_map_rows(LAO_ETH, "ETH")
    bnb_rows = _build_tx_to_flow_map_rows(LAO_BNB, "BNB")
    tx_map = pd.DataFrame(eth_rows + bnb_rows)
    tx_map.to_csv(LABELS / "tx_to_flow_map_lao.csv", index=False)

    eth_flows = load_flow_segments(LAO_ETH)
    bnb_flows = load_flow_segments(LAO_BNB)
    tx_to_i = _tx_to_flow_index(eth_flows)

    # Consistency: every tx in map must match _tx_to_flow_index flow_id via segment index
    eth_map = tx_map[tx_map["chain"] == "ETH"]
    flow_id_by_index = {i: str(eth_flows[i]["flow_id"]) for i in range(len(eth_flows))}
    mismatches = []
    for _, row in eth_map.iterrows():
        tx = row["tx_hash"]
        idx = tx_to_i.get(tx, -1)
        if idx < 0 or flow_id_by_index.get(idx) != row["flow_id"]:
            mismatches.append(tx)
    if mismatches:
        raise RuntimeError(
            f"tx_to_flow_map_lao inconsistent with _tx_to_flow_index: {len(mismatches)} mismatches"
        )

    eth_seg = pd.read_csv(LAO_ETH, dtype=str, keep_default_na=False)
    bnb_seg = pd.read_csv(LAO_BNB, dtype=str, keep_default_na=False)
    eth_universe = eth_map.drop_duplicates("tx_hash")[["tx_hash", "flow_id"]].rename(
        columns={"flow_id": "src_flow_id"}
    )
    bnb_universe = tx_map[tx_map["chain"] == "BNB"].drop_duplicates("tx_hash")[
        ["tx_hash", "flow_id"]
    ].rename(columns={"flow_id": "dst_flow_id"})
    eth_universe.to_csv(LABELS / "universe_eth_txs.csv", index=False)
    bnb_universe.to_csv(LABELS / "universe_bnb_txs.csv", index=False)

    gt_tx = pd.read_csv(GT_TX_SRC, dtype=str, keep_default_na=False)
    gt_tx = gt_tx.rename(columns={"srcTxhash": "src_tx_hash", "dstTxhash": "dst_tx_hash"})
    gt_tx["src_tx_hash"] = gt_tx["src_tx_hash"].map(norm_addr)
    gt_tx["dst_tx_hash"] = gt_tx["dst_tx_hash"].map(norm_addr)
    gt_tx.to_csv(LABELS / "gt_tx_pairs.csv", index=False)

    eth_lookup = dict(zip(eth_map["tx_hash"], eth_map["flow_id"]))
    bnb_lookup = dict(zip(bnb_universe["tx_hash"], bnb_universe["dst_flow_id"]))

    src_missing = [s for s in gt_tx["src_tx_hash"] if s not in eth_lookup]
    dst_missing = [d for d in gt_tx["dst_tx_hash"] if d not in bnb_lookup]

    flow_rows = []
    for _, r in gt_tx.iterrows():
        flow_rows.append(
            {
                "src_tx_hash": r["src_tx_hash"],
                "dst_tx_hash": r["dst_tx_hash"],
                "src_flow_id": eth_lookup[r["src_tx_hash"]],
                "dst_flow_id": bnb_lookup[r["dst_tx_hash"]],
            }
        )
    gt_flow = pd.DataFrame(flow_rows)
    gt_flow.to_csv(LABELS / "gt_flow_pairs.csv", index=False)

    unique_flow_pairs = gt_flow.drop_duplicates(["src_flow_id", "dst_flow_id"])
    collapse = gt_flow.groupby(["src_flow_id", "dst_flow_id"]).size().reset_index(name="tx_pair_count")
    multi = collapse[collapse["tx_pair_count"] > 1]

    return {
        "n_src_flows": len(eth_seg),
        "n_dst_flows": len(bnb_seg),
        "n_tx_pair_gt": len(gt_tx),
        "n_flow_pair_gt_rows": len(gt_flow),
        "n_unique_flow_pairs": len(unique_flow_pairs),
        "n_collapsed_flow_pairs": len(multi),
        "max_tx_per_flow_pair": int(multi["tx_pair_count"].max()) if len(multi) else 1,
        "gt_src_mapping_rate": float((len(gt_tx) - len(src_missing)) / max(len(gt_tx), 1)),
        "gt_dst_mapping_rate": float((len(gt_tx) - len(dst_missing)) / max(len(gt_tx), 1)),
        "n_src_missing": len(src_missing),
        "n_dst_missing": len(dst_missing),
        "n_eth_universe_txs": len(eth_universe),
        "n_bnb_universe_txs": len(bnb_universe),
        "tx_map_eth_rows": len(eth_map),
        "tx_map_bnb_rows": len(bnb_rows),
        "tx_to_flow_index_consistent": True,
        "flow_pair_collapse_rows": multi.to_dict(orient="records") if len(multi) else [],
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def _export_rc_uot_q_reference() -> dict[str, Any]:
    main = _load_json(MAIN_TABLE)
    summary = _load_json(ADM_SUMMARY)
    manifest = _load_json(ADM_MANIFEST)

    main_by_method = {r["method"]: r for r in main["rows"]}
    strat = summary.get("strategies") or {}

    ref: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "main_table": str(MAIN_TABLE.relative_to(REPO)),
            "admissible_summary": str(ADM_SUMMARY.relative_to(REPO)),
            "admissible_manifest": str(ADM_MANIFEST.relative_to(REPO)),
        },
        "manifest_excerpt": {
            "paper_ready": manifest.get("paper_ready"),
            "n_source_flows": manifest.get("n_source_flows"),
            "n_ground_truth_pairs": manifest.get("n_ground_truth_pairs"),
            "coverage_definition": manifest.get("coverage_definition"),
            "abstention_rate_definition": manifest.get("abstention_rate_definition"),
        },
        "methods": {},
        "validation": {},
    }

    coverage_def = summary.get("coverage_definition") or manifest.get("coverage_definition")
    abst_def = summary.get("abstention_rate_definition") or manifest.get("abstention_rate_definition")
    n_src_flows = int(summary.get("n_source_flows") or manifest.get("n_source_flows") or 3258)
    n_gt = int(summary.get("n_ground_truth_pairs") or manifest.get("n_ground_truth_pairs") or 7296)

    ref["metric_definitions"] = {
        "coverage": coverage_def,
        "abstention_rate": abst_def,
        "coverage_abstention_not_complementary": (
            "coverage and abstention_rate use different denominators "
            f"(n_source_flows={n_src_flows} vs n_ground_truth_pairs={n_gt}); "
            "they must NOT be expected to sum to 1."
        ),
    }

    all_ok = True
    checks: list[dict[str, Any]] = []

    for method in HEADLINE_METHODS:
        m_row = main_by_method.get(method)
        s_row = strat.get(method)
        if not m_row or not s_row:
            all_ok = False
            checks.append({"method": method, "status": "FAIL", "reason": "missing in source JSON"})
            continue

        entry = {"main_table": {}, "admissible_summary": {}}
        method_ok = True
        for k in METRIC_KEYS:
            mv = m_row.get(k)
            sv = s_row.get(k)
            entry["main_table"][k] = mv
            entry["admissible_summary"][k] = sv
            if k in ("n_true_positive", "n_false_positive", "n_abstained"):
                if int(mv) != int(sv):
                    method_ok = False
                    checks.append(
                        {"method": method, "field": k, "main": mv, "summary": sv, "status": "MISMATCH"}
                    )
            elif isinstance(mv, (int, float)) and isinstance(sv, (int, float)):
                if not _close(mv, sv):
                    method_ok = False
                    checks.append(
                        {"method": method, "field": k, "main": mv, "summary": sv, "status": "MISMATCH"}
                    )

        # Internal self-consistency for coverage / abstention
        cov = float(s_row["coverage"])
        abst = float(s_row["abstention_rate"])
        n_abst = int(s_row["n_abstained"])
        n_pred = int(s_row.get("n_predicted_pairs") or 0)

        abst_recomputed = n_abst / n_gt
        if not _close(abst, abst_recomputed, 1e-9):
            method_ok = False
            checks.append(
                {
                    "method": method,
                    "check": "abstention_rate_self_consistency",
                    "reported": abst,
                    "recomputed_n_abstained_over_n_gt": abst_recomputed,
                    "status": "FAIL",
                }
            )

        if n_pred != n_gt - n_abst:
            method_ok = False
            checks.append(
                {
                    "method": method,
                    "check": "n_predicted_pairs",
                    "reported": n_pred,
                    "expected_n_gt_minus_n_abstained": n_gt - n_abst,
                    "status": "FAIL",
                }
            )

        # coverage + abstention != 1 is OK; document if someone might misread
        if _close(cov + abst, 1.0, 1e-6):
            checks.append(
                {
                    "method": method,
                    "check": "coverage_plus_abstention",
                    "note": "accidentally sums to ~1.0 but denominators differ; not required",
                    "status": "INFO",
                }
            )
        else:
            checks.append(
                {
                    "method": method,
                    "check": "coverage_plus_abstention",
                    "coverage": cov,
                    "abstention_rate": abst,
                    "sum": cov + abst,
                    "status": "EXPECTED_DIFFERENT_DENOMINATORS",
                }
            )

        ref["methods"][method] = entry
        checks.append({"method": method, "main_vs_summary": "PASS" if method_ok else "FAIL"})
        all_ok = all_ok and method_ok

    ref["validation"] = {
        "all_headline_methods_match": all_ok,
        "checks": checks,
    }

    (FROZEN / "rc_uot_q_reference_metrics.json").write_text(
        json.dumps(ref, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return ref


def _connector_gate1() -> dict[str, Any]:
    dst_chain = CONNECTOR_ROOT / "core" / "dst_chain.py"
    model_pkl = CONNECTOR_ROOT / "data" / "Model" / "model.pkl"
    word2vec = CONNECTOR_ROOT / "data" / "Model" / "word2vec_model.bin"
    norm_map = CONNECTOR_ROOT / "data" / "Model" / "normalization_map.csv"

    src_text = dst_chain.read_text(encoding="utf-8") if dst_chain.is_file() else ""
    fulloutput_exposes_ranking = False
    top_k_capable = False
    notes = []

    if "fulloutput=True" in src_text or "fulloutput=False" in src_text:
        notes.append(
            "fulloutput=True returns per-src filter-stage row counts (debug dict), "
            "NOT an ordered multi-candidate ranking list."
        )
    if "idxmin" in src_text and "time_diff" in src_text:
        notes.append(
            "_match_timestamp selects exactly one dst row per src via groupby(key).time_diff.idxmin()."
        )
        top_k_capable = False

    lines = [
        "# Connector Gate-1 feasibility report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Disclosure",
        "",
        "- **Mode for fair comparison:** Connector original matcher core (`WithdrawLocator` in "
        "`Connector-main/core/dst_chain.py`) on **frozen Celer CSV universe** with candidate pool "
        "restricted to LAO BNB txs.",
        "- **Not in scope:** online `BridgeSpider` pipeline from `experiment/param/main.py` "
        "(dynamic per-src dst fetch via RPC; candidate pool varies by time window).",
        "",
        "## Source paths",
        "",
        f"- Connector root: `{CONNECTOR_ROOT}`",
        f"- Matcher core: `{dst_chain}`",
        f"- Experiment entry (online, not used): `{CONNECTOR_ROOT / 'experiment' / 'param' / 'main.py'}`",
        "",
        "## DepositLocator / model artifacts",
        "",
        f"- `model.pkl` present: **{model_pkl.is_file()}** (`{model_pkl}`)",
        f"- `word2vec_model.bin` present: **{word2vec.is_file()}** (`{word2vec}`)",
        f"- `normalization_map.csv` present: **{norm_map.is_file()}**",
        "",
        "**WithdrawLocator-only path does not call `DepositLocator.filter_deposit()`.** "
        "The frozen-universe evaluation uses pre-identified labeled src txs (7296 GT src) "
        "converted to Connector `src_txs` schema; deposit classification is bypassed.",
        "",
        "Missing `model.pkl` / `word2vec_model.bin` **blocks full Connector pipeline** "
        "(src_chain deposit filtering) but **does not block WithdrawLocator matcher core** "
        "for labeled src txs.",
        "",
        "## Top-k / ranking exposure (no logic changes)",
        "",
        f"- `search_withdraw(fulloutput=True)` exposes filter-stage counts only: **not top-k ranking**.",
        f"- Final output is **top-1 dst per src** (minimum time gap after rule filters).",
        f"- **top_k_capable for RC-UOT-Q top3/joint decode parity:** `{top_k_capable}`",
        "",
        "### Allowed downstream decoding labels (if top-1 only)",
        "",
        "- `raw`: **allowed**",
        "- `positive_delay_top3_rescue`: **N/A** (no native multi-flow ranking)",
        "- `joint_time_admissible_filter`: **N/A** as RC-UOT-Q parity; optional `top1_admissible_filter` only",
        "",
        "## Gate-1 status",
        "",
    ]

    blocked = not dst_chain.is_file()
    status = "BLOCKED" if blocked else "WARN" if not top_k_capable else "PASS"
    lines.append(f"**Status: {status}**")
    lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")

    report_path = GATE / "connector_gate1_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": status,
        "connector_root": str(CONNECTOR_ROOT),
        "withdraw_locator_path": str(dst_chain),
        "withdraw_locator_present": dst_chain.is_file(),
        "model_pkl_present": model_pkl.is_file(),
        "word2vec_present": word2vec.is_file(),
        "deposit_locator_required_for_withdraw_only": False,
        "online_bridge_spider_used": False,
        "fulloutput_exposes_ranking": fulloutput_exposes_ranking,
        "top_k_capable": top_k_capable,
        "allowed_decodings": {
            "raw": True,
            "positive_delay_top3_rescue": False,
            "joint_time_admissible_filter_rc_uot_q_parity": False,
            "top1_admissible_filter_optional": True,
        },
    }


def _abctracer_gate1() -> dict[str, Any]:
    wgt_candidates = list(ABCT_ROOT.rglob("wgt.pth"))
    wgt_candidates = [p for p in wgt_candidates if ".venv" not in p.parts]
    ckpt_candidates = list(ABCT_ROOT.rglob("checkpoint.pth"))
    ckpt_candidates = [p for p in ckpt_candidates if ".venv" not in p.parts]

    checkpoint_found = len(wgt_candidates) > 0 or len(ckpt_candidates) > 0
    status = "BLOCKED" if not checkpoint_found else "PASS"

    lines = [
        "# ABCTracer Gate-1 feasibility report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Disclosure",
        "",
        "- Fair comparison requires **original ABCTracer IR/TIR inference** with official checkpoint.",
        "- Phase 7.5 `ABCTracer-style` reimplementation is **explicitly disallowed** as substitute.",
        "- Training a new model without user approval is **not performed** in Phase 0.",
        "",
        "## Source paths",
        "",
        f"- ABCTracer root: `{ABCT_ROOT}`",
        f"- IR model: `{ABCT_ROOT / 'model' / 'ir.py'}`",
        f"- IR entry: `{ABCT_ROOT / 'exp' / 'ir.py'}`",
        f"- Dataset builder: `{ABCT_ROOT / 'build_cct_dataset.py'}`",
        "",
        "## Checkpoint search (excluding .venv)",
        "",
        f"- `wgt.pth` found: **{len(wgt_candidates)}**",
    ]
    for p in wgt_candidates[:10]:
        lines.append(f"  - `{p}`")
    lines.extend(
        [
            f"- `checkpoint.pth` found: **{len(ckpt_candidates)}**",
        ]
    )
    for p in ckpt_candidates[:10]:
        lines.append(f"  - `{p}`")

    lines.extend(
        [
            "",
            "## If training were requested (NOT executed)",
            "",
            "Would require user confirmation before:",
            "- Building CCT IR tasks from LAO universe via I/O adapter (not implemented in Phase 0).",
            "- Running `ABCTracer/exp/ir.py` with official defaults (`utils/args.py`).",
            "- Documenting train/valid/test split, random seed, GPU time, and divergence from "
            "official checkpoint inference.",
            "",
            "## Gate-1 status",
            "",
            f"**Status: {status}**",
            "",
            "No official checkpoint → **ABCTracer baseline comparison blocked** until user supplies "
            "`wgt.pth` or explicitly approves retraining.",
            "",
        ]
    )

    (GATE / "abctracer_gate1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": status,
        "abctracer_root": str(ABCT_ROOT),
        "wgt_pth_count": len(wgt_candidates),
        "checkpoint_pth_count": len(ckpt_candidates),
        "wgt_pth_paths": [str(p) for p in wgt_candidates],
        "checkpoint_pth_paths": [str(p) for p in ckpt_candidates],
        "phase7_5_substitute_allowed": False,
        "training_executed": False,
    }


def _substrate_hashes() -> dict[str, str]:
    files = [
        LAO_ETH,
        LAO_BNB,
        GT_TX_SRC,
        MAIN_TABLE,
        ADM_SUMMARY,
        ADM_MANIFEST,
        LABELS / "tx_to_flow_map_lao.csv",
        LABELS / "universe_eth_txs.csv",
        LABELS / "universe_bnb_txs.csv",
        LABELS / "gt_tx_pairs.csv",
        LABELS / "gt_flow_pairs.csv",
        FROZEN / "rc_uot_q_reference_metrics.json",
    ]
    return {str(p.relative_to(REPO)): _sha256(p) for p in files if p.is_file()}


def _overall_status(label_info: dict[str, Any], rc_ref: dict[str, Any], conn: dict[str, Any], abct: dict[str, Any]) -> str:
    if label_info["n_src_missing"] or label_info["n_dst_missing"]:
        return "BLOCKED"
    if not rc_ref["validation"]["all_headline_methods_match"]:
        return "BLOCKED"
    if abct["status"] == "BLOCKED":
        return "WARN"  # ABCT blocked but substrate OK — Phase 0 can pass with WARN
    if conn["status"] == "BLOCKED":
        return "BLOCKED"
    if conn["status"] == "WARN":
        return "WARN"
    return "PASS"


def main() -> None:
    for d in (OUT, LABELS, GATE, FROZEN):
        d.mkdir(parents=True, exist_ok=True)

    label_info = _export_labels()
    rc_ref = _export_rc_uot_q_reference()
    conn = _connector_gate1()
    abct = _abctracer_gate1()
    hashes = _substrate_hashes()
    overall = _overall_status(label_info, rc_ref, conn, abct)

    manifest = {
        "phase": 0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(OUT.relative_to(REPO)),
        "counts": {
            "n_src_flows": label_info["n_src_flows"],
            "n_dst_flows": label_info["n_dst_flows"],
            "n_tx_pair_gt": label_info["n_tx_pair_gt"],
            "n_flow_pair_gt_rows": label_info["n_flow_pair_gt_rows"],
            "n_unique_flow_pairs": label_info["n_unique_flow_pairs"],
            "n_collapsed_flow_pairs": label_info["n_collapsed_flow_pairs"],
            "n_eth_universe_txs": label_info["n_eth_universe_txs"],
            "n_bnb_universe_txs": label_info["n_bnb_universe_txs"],
            "gt_src_tx_mapping_rate": label_info["gt_src_mapping_rate"],
            "gt_dst_tx_mapping_rate": label_info["gt_dst_mapping_rate"],
        },
        "flow_mapping": {
            "canonical_source": [
                str(LAO_ETH.relative_to(REPO)),
                str(LAO_BNB.relative_to(REPO)),
            ],
            "exported_map": str((LABELS / "tx_to_flow_map_lao.csv").relative_to(REPO)),
            "paper_full_pipeline_tx_to_flow_map_used": False,
            "tx_to_flow_index_consistent": label_info["tx_to_flow_index_consistent"],
        },
        "gt": {
            "tx_pair_source": str(GT_TX_SRC.relative_to(REPO)),
            "flow_pair_projection": str((LABELS / "gt_flow_pairs.csv").relative_to(REPO)),
            "flow_pair_collapse_reported": label_info["n_collapsed_flow_pairs"] > 0,
            "evaluation_uses_tx_pair_gt_for_pair_metrics": True,
            "flow_pair_gt_role": "flow-level audit / collapse reporting; pair P/R/F1 in admissible decoding uses tx-pair GT",
        },
        "rc_uot_q_frozen": {
            "reference_metrics": str((FROZEN / "rc_uot_q_reference_metrics.json").relative_to(REPO)),
            "validation_pass": rc_ref["validation"]["all_headline_methods_match"],
            "frozen_dirs_read_only": True,
        },
        "gate1": {
            "connector": conn,
            "abctracer": abct,
        },
        "substrate_sha256": hashes,
        "phase0_conclusion": overall,
        "blockers_for_phase1": [],
    }

    blockers = []
    if not rc_ref["validation"]["all_headline_methods_match"]:
        blockers.append("RC-UOT-Q reference metrics validation failed")
    if label_info["n_src_missing"] or label_info["n_dst_missing"]:
        blockers.append("GT tx not 100% mapped to LAO flows")
    if abct["status"] == "BLOCKED":
        blockers.append("ABCTracer: no official checkpoint (wgt.pth)")
    if not conn["withdraw_locator_present"]:
        blockers.append("Connector WithdrawLocator source missing")
    if not conn["top_k_capable"]:
        blockers.append(
            "Connector: top-k ranking not exposed by original search_withdraw(); "
            "top3/joint RC-UOT-Q parity N/A — raw and optional top1_admissible only"
        )
    manifest["blockers_for_phase1"] = blockers

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight = [
        "# Baseline compare — Phase 0 preflight report",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        f"## Conclusion: **{overall}**",
        "",
        "## Universe counts",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| ETH src flows | {label_info['n_src_flows']} |",
        f"| BNB dst flows | {label_info['n_dst_flows']} |",
        f"| tx-pair GT | {label_info['n_tx_pair_gt']} |",
        f"| flow-pair GT rows (tx-projected) | {label_info['n_flow_pair_gt_rows']} |",
        f"| unique (src_flow, dst_flow) pairs | {label_info['n_unique_flow_pairs']} |",
        f"| collapsed flow-pairs (>1 tx) | {label_info['n_collapsed_flow_pairs']} |",
        f"| ETH universe txs | {label_info['n_eth_universe_txs']} |",
        f"| BNB universe txs | {label_info['n_bnb_universe_txs']} |",
        f"| GT src mapping rate | {label_info['gt_src_mapping_rate']:.6f} |",
        f"| GT dst mapping rate | {label_info['gt_dst_mapping_rate']:.6f} |",
        "",
        "## Flow mapping",
        "",
        "- Canonical: LAO `uot_flow_segments_{eth,bnb}.csv` → `labels/tx_to_flow_map_lao.csv`",
        "- Verified consistent with `_tx_to_flow_index()` used in admissible decoding.",
        "- **Not used:** `paper_full_pipeline_run/labels/tx_to_flow_map.csv` (5735-flow universe; ID mismatch).",
        "",
        "## Flow-pair collapse (not deduplicated)",
        "",
    ]
    if label_info["n_collapsed_flow_pairs"]:
        preflight.append(
            f"{label_info['n_collapsed_flow_pairs']} unique flow-pairs map from multiple tx-pairs "
            f"(max {label_info['max_tx_per_flow_pair']} tx-pairs per flow-pair). "
            "Full list in `manifest.json` → counts / see `gt_flow_pairs.csv`."
        )
    else:
        preflight.append("No flow-pair collapse (7296 tx-pairs → 7296 unique flow-pairs).")

    preflight.extend(
        [
            "",
            "## RC-UOT-Q reference metrics",
            "",
            f"- Loaded from frozen JSON (not hand-copied).",
            f"- Headline validation: **{'PASS' if rc_ref['validation']['all_headline_methods_match'] else 'FAIL'}**",
            "",
            "### coverage vs abstention (joint example from source JSON)",
            "",
            "- `coverage` denominator: **n_source_flows = 3258** (source flows with ≥1 non-abstained tx).",
            "- `abstention_rate` denominator: **n_ground_truth_pairs = 7296** (labeled tx-pairs abstained).",
            "- These metrics are **not complementary** and **must not** be expected to sum to 1.",
            "",
            "| method | coverage | abstention_rate | sum | n_abstained |",
            "|--------|----------|-----------------|-----|-------------|",
        ]
    )
    for method in HEADLINE_METHODS:
        s = rc_ref["methods"][method]["admissible_summary"]
        cov = s["coverage"]
        abst = s["abstention_rate"]
        preflight.append(
            f"| {method} | {cov} | {abst} | {float(cov)+float(abst):.6f} | {s['n_abstained']} |"
        )

    preflight.extend(
        [
            "",
            "## Gate-1",
            "",
            f"- Connector: **{conn['status']}** (WithdrawLocator on frozen CSV; top_k_capable={conn['top_k_capable']})",
            f"- ABCTracer: **{abct['status']}** (checkpoints found: wgt={abct['wgt_pth_count']})",
            "",
            "## Phase 1 blockers / warnings",
            "",
        ]
    )
    for b in blockers:
        preflight.append(f"- {b}")
    if not blockers:
        preflight.append("- None")

    preflight.append("")
    preflight.append("## Substrate SHA256")
    preflight.append("")
    for k, v in sorted(hashes.items()):
        preflight.append(f"- `{k}`: `{v}`")

    (OUT / "preflight_report.md").write_text("\n".join(preflight) + "\n", encoding="utf-8")

    print(f"Phase 0 complete. Conclusion: {overall}")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"RC-UOT-Q validation: {rc_ref['validation']['all_headline_methods_match']}")
    print(f"Connector Gate-1: {conn['status']}, top_k={conn['top_k_capable']}")
    print(f"ABCTracer Gate-1: {abct['status']}")


if __name__ == "__main__":
    main()
