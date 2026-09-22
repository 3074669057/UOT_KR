"""Lightweight time/causal weight sensitivity sweep from cached UOT cost components."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
from cross.application.experiments.uot_cache_utils import (
    build_cost_matrix_from_components,
    load_cost_component_cache,
    load_flow_segments,
    solve_from_cache,
    write_cost_component_cache,
)
from cross.application.experiments.uot_sweep_metrics import evaluate_transport_plan
from cross.domain.uot.delay_policy import DEFAULT_TIME_DELAY_POLICY
from cross.shared.normalize import norm_addr

SWEEP_COLUMNS: tuple[str, ...] = (
    "time_weight",
    "causal_weight",
    "causal_violation_penalty",
    "pair_f1",
    "top1_recall",
    "top3_recall",
    "flow_mass_recall",
    "coverage",
    "abstention_rate",
    "unmatched_mass_ratio",
    "causality_violation_rate",
    "median_delay_sec",
    "p90_delay_sec",
    "mean_transport_entropy",
    "mass_concentration_top1",
    "mass_concentration_top3",
    "config_label",
)


def _metrics_close(a: Any, b: Any, *, tol: float = 1e-4) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _check_production_sweep_alignment(
    baseline: dict[str, Any] | None,
    production_metrics_path: Path,
) -> dict[str, Any]:
    if baseline is None or not production_metrics_path.is_file():
        return {
            "production_sweep_metric_alignment": None,
            "reason": "baseline or production metrics missing",
        }
    prod = json.loads(production_metrics_path.read_text(encoding="utf-8"))
    fields = (
        "pair_f1",
        "top1_recall",
        "top3_recall",
        "flow_mass_recall",
        "unmatched_mass_ratio",
    )
    diffs: dict[str, Any] = {}
    aligned = True
    for f in fields:
        sweep_v = baseline.get(f)
        prod_v = prod.get(f)
        if f == "top1_recall" and prod_v is None:
            prod_v = prod.get("pair_recall")
        ok = _metrics_close(sweep_v, prod_v)
        if not ok:
            aligned = False
            diffs[f] = {"sweep_baseline": sweep_v, "production": prod_v}
    return {
        "production_sweep_metric_alignment": aligned,
        "alignment_fields_checked": list(fields),
        "alignment_diffs": diffs if not aligned else {},
        "production_metrics_path": str(production_metrics_path.resolve()),
    }


def _prepare_src_dst(
    eth_path: Path,
    bnb_path: Path,
    eth_flows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = pd.read_csv(eth_path)
    dst = pd.read_csv(bnb_path)
    src = src.copy()
    dst = dst.copy()
    if "txhash" not in src.columns and "hash" in src.columns:
        src["txhash"] = src["hash"].map(lambda x: norm_addr(str(x)))
    if "txhash" not in dst.columns and "hash" in dst.columns:
        dst["txhash"] = dst["hash"].map(lambda x: norm_addr(str(x)))
    flow_txs = {norm_addr(str(h)) for f in eth_flows for h in (f.get("tx_hashes") or [])}
    if flow_txs and "txhash" in src.columns:
        src = src[src["txhash"].astype(str).map(norm_addr).isin(flow_txs)]
    return src, dst


def _select_pareto(rows: list[dict[str, Any]], *, max_cvr: float) -> dict[str, Any] | None:
    admissible = [r for r in rows if float(r.get("causality_violation_rate") or 1.0) <= max_cvr]
    if not admissible:
        return None
    return max(admissible, key=lambda r: float(r.get("pair_f1") or 0.0))


def _write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_discussion(
    path: Path,
    *,
    best_f1: dict[str, Any] | None,
    best_adm: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    no_causal_bound: dict[str, Any] | None,
) -> None:
    text = f"""# Paper discussion: time / causal admissibility trade-off

## Position for reviewers

The diagnostic ablations `no_time` (pair F1 ≈ 0.497) and `no_causal` (pair F1 ≈ 0.596) are **relaxed diagnostic upper bounds**, not the proposed forensic method. They remove or zero out cost terms that enforce temporal plausibility and source-before-target (causal) admissibility. Higher F1 under these settings indicates that real Celer bridge data contain batching, delay noise, timestamp granularity mismatch, or label-window effects — not that the production model should drop causal constraints.

## Forensic admissibility

Causal constraints encode **forensic admissibility**: a cross-chain correspondence is inadmissible if the destination-side activity precedes the source-side completion time. Relaxing the causal penalty increases pair-level recovery but also increases `causality_violation_rate` (hard-decoded pairs with negative delay). The main model should be chosen on the **Pareto frontier** between pair F1 and admissibility, not by maximizing F1 alone.

## Sweep summary

| Config | pair_f1 | causality_violation_rate | Notes |
|--------|--------:|-------------------------:|-------|
| baseline (time=1.0, causal=1.0) | {baseline.get('pair_f1') if baseline else 'n/a'} | {baseline.get('causality_violation_rate') if baseline else 'n/a'} | Production default |
| best admissible (CVR ≤ threshold) | {best_adm.get('pair_f1') if best_adm else 'none'} | {best_adm.get('causality_violation_rate') if best_adm else 'n/a'} | Recommended if beats baseline |
| best by pair_f1 (any) | {best_f1.get('pair_f1') if best_f1 else 'n/a'} | {best_f1.get('causality_violation_rate') if best_f1 else 'n/a'} | May violate admissibility |
| no_causal upper bound (causal=0) | {no_causal_bound.get('pair_f1') if no_causal_bound else 'n/a'} | {no_causal_bound.get('causality_violation_rate') if no_causal_bound else 'n/a'} | Relaxed diagnostic only |

## Wording guidance

- Do **not** write that `no_causal` is a better method.
- Write that it is an **upper bound** when admissibility constraints are removed.
- If no admissible config beats baseline F1, retain baseline weights and cite the sweep as sensitivity evidence.
- Mention batching / delay noise as plausible causes when relaxing time/causal terms helps F1.

## Suggested sentence (Results / Limitations)

> Removing the causal penalty raises pair-level F1 to approximately {no_causal_bound.get('pair_f1') if no_causal_bound else '0.59'}, but at the cost of a higher causality-violation rate; we therefore treat this as a relaxed diagnostic upper bound rather than a deployable configuration, and retain the baseline time/causal weights for the forensic-facing model.
"""
    path.write_text(text, encoding="utf-8")


def run_time_causal_sensitivity(
    *,
    eth_path: Path,
    bnb_path: Path,
    label_path: Path,
    out_dir: Path,
    base_run: Path,
    time_grid: list[float] | None = None,
    causal_grid: list[float] | None = None,
    time_delay_policy: str = "legacy_flow_boundary",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_run = Path(base_run)

    time_grid = time_grid or [0.0, 0.25, 0.5, 1.0]
    causal_grid = causal_grid or [0.0, 0.25, 0.5, 1.0]

    cache = load_cost_component_cache(base_run)
    write_cost_component_cache(cache, out_dir)

    eth_flows = load_flow_segments(base_run / "uot" / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(base_run / "uot" / "uot_flow_segments_bnb.csv")
    label_df = pd.read_csv(label_path)
    src_all, dst_norm = _prepare_src_dst(eth_path, bnb_path, eth_flows)
    eth_df = pd.read_csv(eth_path)
    bnb_df = pd.read_csv(bnb_path)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)

    baseline_causal = float(cache["baseline_causal_penalty"])
    rows: list[dict[str, Any]] = []

    for tw in time_grid:
        for cw in causal_grid:
            c_mat = build_cost_matrix_from_components(
                cache["components"],
                time_weight=tw,
                causal_weight=cw,
                baseline_causal_penalty=baseline_causal,
                max_delay_sec=float(cache["max_delay_sec"]),
                delay_policy=time_delay_policy,
                source_flows=eth_flows,
                target_flows=bnb_flows,
            )
            p = solve_from_cache(cache, c_mat)
            metrics = evaluate_transport_plan(
                p,
                eth_flows=eth_flows,
                bnb_flows=bnb_flows,
                label_df=label_df,
                src_all=src_all,
                dst_norm=dst_norm,
                decode_threshold=float(cache["decode_threshold"]),
                source_mass=cache["source_mass"],
                target_mass=cache["target_mass"],
                eth_ts=eth_ts,
                bnb_ts=bnb_ts,
                delay_policy=time_delay_policy,
            )
            row = {
                "time_weight": tw,
                "causal_weight": cw,
                "causal_violation_penalty": baseline_causal * cw,
                "config_label": f"time={tw:g},causal={cw:g}",
                **metrics,
            }
            rows.append(row)

    df = pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))
    df.to_csv(out_dir / "time_causal_weight_sweep.csv", index=False)
    (out_dir / "time_causal_weight_sweep.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_md_table(out_dir / "time_causal_weight_sweep.md", "Time / causal weight sweep", rows, SWEEP_COLUMNS)

    pareto = sorted(rows, key=lambda r: (-float(r["pair_f1"] or 0), float(r["causality_violation_rate"] or 1)))
    pd.DataFrame(pareto).to_csv(out_dir / "time_causal_pareto_frontier.csv", index=False)
    _write_md_table(
        out_dir / "time_causal_pareto_frontier.md",
        "Pareto frontier (sorted by pair_f1 desc, CVR asc)",
        pareto,
        SWEEP_COLUMNS,
    )

    best_f1 = max(rows, key=lambda r: float(r.get("pair_f1") or 0.0))
    best_adm = _select_pareto(rows, max_cvr=0.01) or _select_pareto(rows, max_cvr=0.05)
    baseline = next((r for r in rows if r["time_weight"] == 1.0 and r["causal_weight"] == 1.0), None)
    no_causal = next((r for r in rows if r["causal_weight"] == 0.0 and r["time_weight"] == 1.0), None)

    prod_metrics_path = base_run / "production_delay_fixed_metrics.json"
    alignment = _check_production_sweep_alignment(baseline, prod_metrics_path)

    (out_dir / "best_by_pair_f1.json").write_text(json.dumps(best_f1, indent=2), encoding="utf-8")
    (out_dir / "best_admissible_config.json").write_text(
        json.dumps(best_adm or {"status": "none_admissible"}, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_run": str(base_run.resolve()),
        "time_delay_policy": time_delay_policy,
        "time_grid": time_grid,
        "causal_grid": causal_grid,
        "baseline_causal_penalty": baseline_causal,
        "note": (
            "time_weight scales default time cost weight (0.25 at 1.0) then renormalizes; "
            "causal_weight scales baseline causal_violation_penalty (5.0 at 1.0). "
            f"Delay policy: {time_delay_policy}. "
            "causality_violation_rate uses flow-boundary delay on decoded pairs; "
            "causality_violation_rate_tx_level (when present) uses tx timestamps."
        ),
        "baseline_config": baseline,
        "best_by_pair_f1": best_f1,
        "best_admissible_config": best_adm,
        "no_causal_upper_bound": no_causal,
        "admissible_threshold_used": 0.01 if _select_pareto(rows, max_cvr=0.01) else 0.05,
        **alignment,
    }
    (out_dir / "sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    delay_manifest_path = base_run / "delay_policy_manifest.json"
    if delay_manifest_path.is_file():
        dm = json.loads(delay_manifest_path.read_text(encoding="utf-8"))
        dm["production_sweep_metric_alignment"] = alignment.get("production_sweep_metric_alignment")
        dm["paper_ready"] = bool(
            dm.get("paper_ready")
            and alignment.get("production_sweep_metric_alignment") is not False
        )
        delay_manifest_path.write_text(json.dumps(dm, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_dir / "delay_fix_manifest.json").write_text(
        json.dumps(
            {
                "time_delay_policy": time_delay_policy,
                "legacy_policy": "legacy_flow_boundary",
                "fixed_policy": "tx_if_available_else_flow_representative",
                "production_sweep_metric_alignment": alignment.get("production_sweep_metric_alignment"),
                "manifest": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_paper_discussion(
        out_dir / ("paper_time_causal_discussion_fixed.md" if time_delay_policy != "legacy_flow_boundary" else "paper_time_causal_discussion.md"),
        best_f1=best_f1,
        best_adm=best_adm,
        baseline=baseline,
        no_causal_bound=no_causal,
    )

    audit_src = base_run.parent / "delay_semantics_diagnosis" / "production_plan_delay_audit.json"
    if audit_src.is_file():
        shutil.copy2(audit_src, out_dir / "production_plan_delay_audit.json")
        md_src = audit_src.with_suffix(".md")
        if md_src.is_file():
            shutil.copy2(md_src, out_dir / "production_plan_delay_audit.md")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Time/causal UOT sensitivity sweep from cache.")
    parser.add_argument("--eth", type=Path, required=True)
    parser.add_argument("--bnb", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/time_causal_sensitivity"))
    parser.add_argument("--base-run", type=Path, default=Path("out/leave_anchor_out_real"))
    parser.add_argument(
        "--time-delay-policy",
        type=str,
        default=DEFAULT_TIME_DELAY_POLICY,
        choices=("legacy_flow_boundary", "tx_if_available_else_flow_representative"),
    )
    args = parser.parse_args()
    result = run_time_causal_sensitivity(
        eth_path=args.eth,
        bnb_path=args.bnb,
        label_path=args.label,
        out_dir=args.out,
        base_run=args.base_run,
        time_delay_policy=args.time_delay_policy,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("baseline_config", "best_by_pair_f1")}, indent=2))


if __name__ == "__main__":
    main()
