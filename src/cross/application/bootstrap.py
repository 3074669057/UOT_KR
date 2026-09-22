"""Single entry path for CLI / ``python run.py`` (config, output dir, logging, then pipeline)."""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

from cross.config.paths import CROSS_ROOT
from cross.infrastructure.config.service import load_and_validate_config
from cross.infrastructure.run_init import prepare_output_dir, setup_logging
from cross.interfaces.cli import build_parser

from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _resolve_output_path(
    *,
    args: Namespace,
    cfg: dict,
    project_root: Path,
) -> tuple[Path, str, bool]:
    out_rel = str(cfg.get("output_dir") or "out_smoke_uot_run").strip() or "out_smoke_uot_run"
    clear_output = bool(cfg.get("clear_output_on_start", True))

    if getattr(args, "out", None) is not None:
        out_raw = Path(args.out)
    else:
        out_raw = Path(out_rel)

    if not out_raw.is_absolute():
        out_path = (project_root / out_raw).resolve()
    else:
        out_path = out_raw.resolve()

    return out_path, out_rel, clear_output


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        cfg = load_and_validate_config(args.config, args.local_config)

        out_path, out_rel, clear_output = _resolve_output_path(args=args, cfg=cfg, project_root=CROSS_ROOT)
        incremental = bool(
            getattr(args, "build_tx_anchors", False)
            or getattr(args, "build_flow_labels", False)
            or getattr(args, "eval_flow_level", False)
            or getattr(args, "eval_semi_synthetic", False)
            or getattr(args, "paper_experiment_closure", False)
            or getattr(args, "freeze_label_layer_v1", False)
            or getattr(args, "run_synthetic_multi_seed", False)
            or getattr(args, "flow_uot_resume", False)
            or getattr(args, "candidate_recall_diagnostics", False)
            or getattr(args, "candidate_recall_diagnostics_fast", False)
            or getattr(args, "decode_threshold_sweep", False)
            or getattr(args, "paper_finalization", False)
            or getattr(args, "synthetic_failure_debug", False)
            or getattr(args, "candidate_pool_sweep_only", None) is not None
            or (
                str(getattr(args, "matching_method", "")).lower() == "uot"
                and str(getattr(args, "uot_input_level", "")).lower() == "flow"
                and getattr(args, "src_flows", None) is not None
            )
        )
        prepare_output_dir(
            out_path,
            clear=clear_output and not incremental,
            project_root=CROSS_ROOT,
            config_output_dir=out_rel,
        )
        log_file = setup_logging(out_path, cfg)

        args.out = out_path

        logger.info("Starting pipeline run")
        logger.info("Output directory prepared: %s", out_path)
        logger.info("Run log file: %s", log_file)
        logger.info("clear_output_on_start=%s", clear_output)

        return run_pipeline(args, cfg=cfg)
    except Exception:
        logger.exception("Pipeline failed")
        return 1
