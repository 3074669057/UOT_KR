"""Run UOT flow matching on local ETH + BNB CSVs (paper experiments / ablations).

Example::

    python -m cross.application.experiments.run_uot_matching \\
        --eth in/Celer_ETH_cun.csv --bnb path/to/bnb.csv --out out/uot_exp \\
        --label label/celer_label.csv --method uot --uot-backend numpy

Leave-the-anchor-out ablation::

    python -m cross.application.experiments.run_uot_matching \\
        --eth in/Celer_ETH_cun.csv --bnb path/to/bnb.csv \\
        --label label/celer_label.csv --out out/leave_anchor_out \\
        --method uot --uot-backend numpy --leave-anchor-out --anchor-mask-strict

Requires ``bnb`` to be a BNB export compatible with ``bnb_df_to_dst_txs`` (hash, from, to, contractAddress, timeStamp, value).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from cross.application.experiments.leave_anchor_out_ablation import run_leave_anchor_out_ablation
from cross.domain.uot.delay_policy import DEFAULT_TIME_DELAY_POLICY, DELAY_POLICIES
from cross.config.output_layout import output_file
from cross.config.paths import CROSS_ROOT
from cross.domain.path_b.service import execute_path_b, persist_path_b_outputs
from cross.infrastructure.run_init import setup_logging

logger = logging.getLogger(__name__)


def _parse_weights(s: str | None) -> dict[str, float] | None:
    if not s or not str(s).strip():
        return None
    return json.loads(s)


def _path_b_kwargs(args: argparse.Namespace, weights: dict[str, float] | None, bnb_df: pd.DataFrame) -> dict:
    kw = dict(
        eth_path=args.eth,
        label_path=args.label,
        chunk_size=args.chunk_size,
        bnb_df_override=bnb_df,
        unlabeled=args.label is None,
        matching_method=args.method,
        uot_reg=args.uot_reg,
        uot_reg_m=args.uot_reg_m,
        uot_decode_threshold=args.uot_decode_threshold,
        uot_cost_weights=weights,
        uot_flow_mode=args.uot_flow_mode,
        uot_backend=args.uot_backend,
        uot_segment_time_bucket_sec=int(args.uot_segment_time_bucket_sec),
        uot_time_delay_policy=str(
            getattr(args, "time_delay_policy", None)
            or getattr(args, "uot_time_delay_policy", None)
            or DEFAULT_TIME_DELAY_POLICY
        ),
    )
    if getattr(args, "anchor_mask_mode", None):
        kw["anchor_mask_mode"] = args.anchor_mask_mode
        kw["leave_anchor_out"] = args.anchor_mask_mode != "none"
    return kw


def main() -> int:
    p = argparse.ArgumentParser(description="UOT cross-chain matching experiment runner")
    p.add_argument("--eth", type=Path, required=True)
    p.add_argument("--bnb", type=Path, required=True, help="BNB transfers CSV (hash/from/to/contractAddress/timeStamp/value)")
    p.add_argument("--label", type=Path, default=None, help="Optional label CSV for metrics")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", choices=("uot", "hungarian", "greedy"), default="uot")
    p.add_argument("--uot-reg", type=float, default=0.05)
    p.add_argument("--uot-reg-m", type=float, default=0.5)
    p.add_argument("--uot-decode-threshold", type=float, default=0.01)
    p.add_argument("--uot-flow-mode", choices=("segment", "tx"), default="segment")
    p.add_argument("--uot-backend", choices=("pot", "numpy"), default="pot")
    p.add_argument("--uot-cost-weights", type=str, default=None)
    p.add_argument("--uot-segment-time-bucket-sec", type=int, default=600)
    p.add_argument(
        "--time-delay-policy",
        dest="time_delay_policy",
        choices=tuple(sorted(DELAY_POLICIES)),
        default=DEFAULT_TIME_DELAY_POLICY,
        help="Delay semantics for UOT time cost and causal penalty",
    )
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument(
        "--leave-anchor-out",
        action="store_true",
        help="Run full leakage-resistant anchor ablation suite",
    )
    p.add_argument(
        "--anchor-mask-mode",
        choices=("none", "leave_key_out", "leave_anchor_out_strict"),
        default=None,
        help="Optional single-run masking tier (without full suite)",
    )
    p.add_argument(
        "--anchor-mask-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strict anchor masking: raise on leakage scan failure (default: true)",
    )
    p.add_argument(
        "--anchor-mask-report",
        type=Path,
        default=None,
        help="Optional path for anchor_mask_report.json (default: <out>/anchor_mask_report.json)",
    )
    args = p.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = (CROSS_ROOT / out).resolve()
    else:
        out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(
        out,
        {
            "log_to_file": True,
            "log_file_name": "run.log",
            "console_log_level": "INFO",
            "file_log_level": "INFO",
        },
    )
    args.out = out

    bnb_df = pd.read_csv(args.bnb, low_memory=False)
    weights = _parse_weights(args.uot_cost_weights)
    pb_kw = _path_b_kwargs(args, weights, bnb_df)

    if args.leave_anchor_out:
        report_path = args.anchor_mask_report
        if report_path is not None and not report_path.is_absolute():
            report_path = (CROSS_ROOT / report_path).resolve()
        elif report_path is None:
            report_path = output_file(out, "anchor_mask_report.json")
        run_leave_anchor_out_ablation(
            out_dir=out,
            path_b_kwargs=pb_kw,
            eth_path=args.eth,
            bnb_df=bnb_df,
            label_path=args.label,
            anchor_mask_strict=bool(args.anchor_mask_strict),
            anchor_mask_report=report_path,
            bnb_path=Path(args.bnb),
            command_argv=sys.argv,
        )
        logger.info("Leave-anchor-out ablation complete under %s", out)
        return 0

    pb_df, cmp_dict = execute_path_b(**pb_kw)

    persist_path_b_outputs(
        args.out,
        pb_df,
        cmp_dict,
        eth_path=args.eth,
        bnb_df=bnb_df,
        validate_evidence_schema=False,
    )

    fm = (cmp_dict.get("uot") or {}).get("flow_metrics") or {}
    row = {
        "method": args.method,
        "uot_reg": args.uot_reg,
        "uot_reg_m": args.uot_reg_m,
        "pair_f1": fm.get("pair_f1"),
        "accuracy": cmp_dict.get("accuracy"),
        "flow_mass_recall": fm.get("flow_mass_recall"),
    }
    with open(output_file(args.out, "uot_experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    logger.info("Wrote outputs under %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
