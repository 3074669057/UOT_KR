"""Rebuild UOT cost/transport with fixed delay policy from cached flow segments."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
from cross.application.experiments.production_same_source_metrics import (
    compute_production_same_source_metrics,
    delay_policy_manifest,
    metrics_to_markdown,
    validate_production_acceptance,
)
from cross.application.experiments.uot_cache_utils import (
    build_cost_matrix_from_components,
    load_cost_component_cache,
    load_flow_segments,
    solve_from_cache,
)
from cross.config.paths import CROSS_ROOT
from cross.domain.uot.delay_policy import (
    DEFAULT_TIME_DELAY_POLICY,
    DELAY_POLICIES,
    DELAY_POLICY_DOCS,
    build_delay_sec_matrix,
)
from cross.domain.uot.tx_decode_policy import DEFAULT_TX_DECODE_POLICY, TX_DECODE_POLICIES
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs
from cross.infrastructure.run_init import setup_logging

logger = logging.getLogger(__name__)


def _write_uot_cache_layout(
    out_dir: Path,
    base_run: Path,
    *,
    cache: dict[str, Any],
    c_mat: np.ndarray,
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    time_delay_policy: str,
) -> None:
    """Write uot/ NPZ + flow segments so sensitivity sweep can use this as base_run."""
    uot_dir = out_dir / "uot"
    uot_dir.mkdir(parents=True, exist_ok=True)
    base_uot = base_run / "uot"

    for name in ("uot_flow_segments_eth.csv", "uot_flow_segments_bnb.csv"):
        src = base_uot / name
        if src.is_file():
            shutil.copy2(src, uot_dir / name)

    components = dict(cache["components"])
    delay_sec = build_delay_sec_matrix(eth_flows, bnb_flows, policy=time_delay_policy)
    components["delay_sec"] = delay_sec
    components["time_delay_policy"] = np.array(time_delay_policy)
    if "C" in components:
        components["C"] = c_mat
    np.savez_compressed(uot_dir / "uot_cost_matrix.npz", **components)

    np.savez_compressed(
        uot_dir / "uot_transport_matrix.npz",
        P=p,
        source_mass_risk_weighted=cache["source_mass"],
        target_mass_evidence_weighted=cache["target_mass"],
    )

    diag: dict[str, Any] = {}
    diag_src = base_uot / "uot_diagnostics.json"
    if diag_src.is_file():
        diag = json.loads(diag_src.read_text(encoding="utf-8"))
    diag.update(
        {
            "time_delay_policy": time_delay_policy,
            "max_delay_sec": float(cache["max_delay_sec"]),
            "causal_violation_penalty": float(cache["baseline_causal_penalty"]),
            "reg": float(cache["reg"]),
            "reg_m": float(cache["reg_m"]),
            "decode_threshold": float(cache["decode_threshold"]),
            "P_shape": list(p.shape),
            "rebuilt_from_base_run": str(base_run.resolve()),
        }
    )
    (uot_dir / "uot_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")


def _anchor_leakage_sanity(base_run: Path) -> dict[str, Any]:
    report_path = base_run / "anchor_mask_report.json"
    if not report_path.is_file():
        return {
            "passed": False,
            "reason": "anchor_mask_report.json missing from base run",
            "note": "Delay policy change does not alter anchor masking; reuse base-run leakage audit.",
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    leakage_passed = bool(report.get("leakage_scan_passed", report.get("leakage_scan", {}).get("leakage_scan_passed")))
    return {
        "passed": leakage_passed,
        "anchor_mask_mode": report.get("anchor_mask_mode"),
        "leakage_scan_passed": leakage_passed,
        "masked_field_count": report.get("masked_field_count"),
        "note": (
            "Anchor masking unchanged from base run; fixed delay policy applies only to "
            "time cost and causal penalty, not leave-anchor-out feature masking."
        ),
        "source_report": str(report_path.resolve()),
    }


def _write_bad_cases(out_dir: Path, bad_cases: list[dict[str, Any]]) -> None:
    neg = sorted(bad_cases, key=lambda r: min(r.get("delay_tx_sec", 0), r.get("delay_flow_representative_sec", 0)))[:50]
    if not neg:
        return
    df = pd.DataFrame(neg)
    df.to_csv(out_dir / "negative_delay_decoded_cases.csv", index=False)
    lines = ["# Negative-delay decoded cases (top 50)", "", "| srcTxHash | dstTxHash | delay_tx_sec | delay_flow_repr |", "|-----------|-----------|-------------:|----------------:|"]
    for r in neg:
        lines.append(
            f"| {r.get('srcTxHash','')} | {r.get('dstTxHash','')} | {r.get('delay_tx_sec','')} | {r.get('delay_flow_representative_sec','')} |"
        )
    (out_dir / "negative_delay_decoded_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_delay_fixed_production(
    *,
    base_run: Path,
    out_dir: Path,
    eth_path: Path,
    bnb_path: Path,
    label_path: Path,
    time_delay_policy: str = DEFAULT_TIME_DELAY_POLICY,
    tx_decode_policy: str = DEFAULT_TX_DECODE_POLICY,
    reuse_transport_from: Path | None = None,
) -> dict[str, Any]:
    if time_delay_policy not in DELAY_POLICIES:
        raise ValueError(f"Unknown time_delay_policy {time_delay_policy!r}")
    if tx_decode_policy not in TX_DECODE_POLICIES:
        raise ValueError(f"Unknown tx_decode_policy {tx_decode_policy!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_run = Path(base_run)

    cache = load_cost_component_cache(base_run)
    flow_root = base_run / "uot"
    eth_flows = load_flow_segments(flow_root / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(flow_root / "uot_flow_segments_bnb.csv")
    label_df = pd.read_csv(label_path)
    eth_df = pd.read_csv(eth_path, low_memory=False)
    bnb_df = pd.read_csv(bnb_path, low_memory=False)
    flow_txs = {norm_addr(str(h)) for f in eth_flows for h in (f.get("tx_hashes") or [])}
    src_all = eth_df_to_src_txs(eth_df)
    if flow_txs:
        src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)
    eth_ts, bnb_ts, ts_missing = _tx_timestamp_lookup(eth_df, bnb_df)

    c_mat: np.ndarray
    if reuse_transport_from is not None:
        reuse_transport_from = Path(reuse_transport_from)
        p_loaded: np.ndarray | None = None
        c_loaded: np.ndarray | None = None
        for rel in ("matching_transport_matrix.npz", "uot/uot_transport_matrix.npz"):
            tp = reuse_transport_from / rel
            if tp.is_file():
                data = np.load(tp)
                p_loaded = np.asarray(data["P"], dtype=float)
                if "C" in data.files:
                    c_loaded = np.asarray(data["C"], dtype=float)
                break
        if p_loaded is None:
            raise FileNotFoundError(f"No transport matrix in {reuse_transport_from}")
        p = p_loaded
        if c_loaded is not None:
            c_mat = c_loaded
        else:
            c_mat = build_cost_matrix_from_components(
                cache["components"],
                time_weight=1.0,
                causal_weight=1.0,
                baseline_causal_penalty=float(cache["baseline_causal_penalty"]),
                max_delay_sec=float(cache["max_delay_sec"]),
                delay_policy=time_delay_policy,
                source_flows=eth_flows,
                target_flows=bnb_flows,
            )
    else:
        c_mat = build_cost_matrix_from_components(
            cache["components"],
            time_weight=1.0,
            causal_weight=1.0,
            baseline_causal_penalty=float(cache["baseline_causal_penalty"]),
            max_delay_sec=float(cache["max_delay_sec"]),
            delay_policy=time_delay_policy,
            source_flows=eth_flows,
            target_flows=bnb_flows,
        )
        p = solve_from_cache(cache, c_mat)

    metrics = compute_production_same_source_metrics(
        p,
        eth_flows=eth_flows,
        bnb_flows=bnb_flows,
        label_df=label_df,
        src_all=src_all,
        dst_norm=dst_norm,
        source_mass=cache["source_mass"],
        target_mass=cache["target_mass"],
        decode_threshold=float(cache["decode_threshold"]),
        eth_ts=eth_ts,
        bnb_ts=bnb_ts,
        time_delay_policy=time_delay_policy,
        tx_decode_policy=tx_decode_policy,
    )
    bad_cases = metrics.pop("_bad_cases", [])
    pairs_df = metrics.pop("_pairs_df", pd.DataFrame())
    decoded = metrics.pop("_decoded", [])

    acceptance = validate_production_acceptance(metrics)
    anchor_leakage = _anchor_leakage_sanity(base_run)

    manifest = delay_policy_manifest(
        time_delay_policy=time_delay_policy,
        base_run=str(base_run.resolve()),
        metrics=metrics,
        acceptance=acceptance,
        anchor_leakage=anchor_leakage,
        production_sweep_metric_alignment=None,
    )
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["eth_path"] = str(eth_path.resolve())
    manifest["bnb_path"] = str(bnb_path.resolve())
    manifest["label_path"] = str(label_path.resolve())
    manifest["delay_policy_docs"] = DELAY_POLICY_DOCS
    manifest["tx_timestamp_lookup_missing"] = ts_missing
    manifest["solver"] = {
        "reg": cache["reg"],
        "reg_m": cache["reg_m"],
        "decode_threshold": cache["decode_threshold"],
        "max_delay_sec": cache["max_delay_sec"],
        "baseline_causal_penalty": cache["baseline_causal_penalty"],
        "P_shape": list(p.shape),
    }

    manifest["tx_decode_policy"] = tx_decode_policy
    manifest["reuse_transport_from"] = str(reuse_transport_from.resolve()) if reuse_transport_from else None

    public_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    metrics_name = (
        "production_delay_fixed_txdecode_metrics"
        if tx_decode_policy != DEFAULT_TX_DECODE_POLICY
        else "production_delay_fixed_metrics"
    )
    (out_dir / f"{metrics_name}.json").write_text(
        json.dumps(public_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"{metrics_name}.md").write_text(
        metrics_to_markdown(public_metrics, title=f"Production metrics ({metrics_name})"),
        encoding="utf-8",
    )
    if tx_decode_policy == DEFAULT_TX_DECODE_POLICY:
        (out_dir / "production_delay_fixed_metrics.json").write_text(
            json.dumps(public_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    delay_dist = {
        "time_delay_policy": time_delay_policy,
        "delay_tx_distribution": {
            "median_sec": metrics.get("delay_tx_median_sec"),
            "p05_sec": metrics.get("delay_tx_p05_sec"),
            "p95_sec": metrics.get("delay_tx_p95_sec"),
            "negative_ratio": metrics.get("negative_delay_ratio_tx_level"),
        },
        "delay_flow_representative_distribution": {
            "median_sec": metrics.get("delay_flow_representative_median_sec"),
            "p05_sec": metrics.get("delay_flow_representative_p05_sec"),
            "p95_sec": metrics.get("delay_flow_representative_p95_sec"),
            "negative_ratio": metrics.get("negative_delay_ratio_flow_representative"),
        },
        "tx_timestamp_available_ratio": metrics.get("tx_timestamp_available_ratio"),
        "flow_representative_fallback_ratio": metrics.get("flow_representative_fallback_ratio"),
    }
    (out_dir / "production_delay_distribution.json").write_text(json.dumps(delay_dist, indent=2), encoding="utf-8")
    (out_dir / "production_delay_distribution.md").write_text(
        metrics_to_markdown(delay_dist, title="Production delay distribution"),
        encoding="utf-8",
    )

    cvr_doc = {
        "causality_violation_rate_tx_level": metrics.get("causality_violation_rate_tx_level"),
        "causality_violation_rate_flow_level_representative": metrics.get(
            "causality_violation_rate_flow_level_representative"
        ),
        "same_source_plan": True,
        "note": "CVR computed on decoded tx pairs from this transport plan.",
    }
    (out_dir / "production_plan_cvr.json").write_text(json.dumps(cvr_doc, indent=2), encoding="utf-8")
    (out_dir / "production_plan_cvr.md").write_text(
        metrics_to_markdown(cvr_doc, title="Production plan CVR (same source)"),
        encoding="utf-8",
    )

    (out_dir / "delay_policy_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if tx_decode_policy != DEFAULT_TX_DECODE_POLICY:
        tx_manifest = {
            "tx_decode_policy": tx_decode_policy,
            "time_delay_policy": time_delay_policy,
            "reuse_transport_from": manifest.get("reuse_transport_from"),
            "acceptance": acceptance,
            "metrics": public_metrics,
            "generated_at_utc": manifest["generated_at_utc"],
        }
        (out_dir / "tx_decode_policy_manifest.json").write_text(json.dumps(tx_manifest, indent=2), encoding="utf-8")
        (out_dir / "anchor_leakage_sanity_after_txdecode_fix.json").write_text(
            json.dumps({**anchor_leakage, "passed": anchor_leakage.get("passed")}, indent=2),
            encoding="utf-8",
        )
    (out_dir / "anchor_leakage_sanity_after_delay_fix.json").write_text(
        json.dumps({**anchor_leakage, "passed": anchor_leakage.get("passed")}, indent=2),
        encoding="utf-8",
    )

    flow_metrics_out = {k: public_metrics.get(k) for k in ("pair_f1", "pair_recall", "flow_mass_recall", "unmatched_mass_ratio")}
    (out_dir / "matching_flow_metrics.json").write_text(json.dumps(flow_metrics_out, indent=2), encoding="utf-8")
    (out_dir / "matching_flow_correspondence.json").write_text(
        json.dumps({"decoded_correspondences": decoded, "n_pairs": len(pairs_df)}, indent=2),
        encoding="utf-8",
    )

    np.savez_compressed(
        out_dir / "matching_transport_matrix.npz",
        P=p,
        C=c_mat,
        source_mass_risk_weighted=cache["source_mass"],
        target_mass_evidence_weighted=cache["target_mass"],
        source_flow_ids=np.array([f.get("flow_id") for f in eth_flows], dtype=object),
        target_flow_ids=np.array([f.get("flow_id") for f in bnb_flows], dtype=object),
        time_delay_policy=np.array(time_delay_policy),
    )

    _write_uot_cache_layout(
        out_dir,
        base_run,
        cache=cache,
        c_mat=c_mat,
        p=p,
        eth_flows=eth_flows,
        bnb_flows=bnb_flows,
        time_delay_policy=time_delay_policy,
    )

    if bad_cases and (metrics.get("causality_violation_rate_tx_level") or 0) > 0.05:
        _write_bad_cases(out_dir, bad_cases)

    logger.info("Wrote delay-fixed production run to %s (pair_f1=%s)", out_dir, public_metrics.get("pair_f1"))
    return {
        "out_dir": str(out_dir.resolve()),
        "metrics": public_metrics,
        "acceptance": acceptance,
        "manifest": manifest,
        "anchor_leakage": anchor_leakage,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild UOT with fixed delay policy from cached flows.")
    p.add_argument("--base-run", type=Path, default=Path("out/leave_anchor_out_real"))
    p.add_argument("--out", type=Path, default=Path("out/uot_delay_fixed_production"))
    p.add_argument("--eth", type=Path, required=True)
    p.add_argument("--bnb", type=Path, required=True)
    p.add_argument("--label", type=Path, required=True)
    p.add_argument(
        "--time-delay-policy",
        default=DEFAULT_TIME_DELAY_POLICY,
        choices=tuple(sorted(DELAY_POLICIES)),
    )
    p.add_argument(
        "--tx-decode-policy",
        default=DEFAULT_TX_DECODE_POLICY,
        choices=tuple(sorted(TX_DECODE_POLICIES)),
    )
    p.add_argument(
        "--reuse-transport-from",
        type=Path,
        default=None,
        help="Reuse transport plan P from another run (tx-decode-only refresh)",
    )
    args = p.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = (CROSS_ROOT / out).resolve()
    base = Path(args.base_run)
    if not base.is_absolute():
        base = (CROSS_ROOT / base).resolve()

    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out, {"log_to_file": True, "log_file_name": "run.log", "console_log_level": "INFO", "file_log_level": "INFO"})

    result = run_delay_fixed_production(
        base_run=base,
        out_dir=out,
        eth_path=Path(args.eth),
        bnb_path=Path(args.bnb),
        label_path=Path(args.label),
        time_delay_policy=args.time_delay_policy,
        tx_decode_policy=args.tx_decode_policy,
        reuse_transport_from=Path(args.reuse_transport_from) if args.reuse_transport_from else None,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
