"""Precision-coverage curve computation for M1 ablation.

Compares UOT mass-driven abstention against score-threshold abstention
for all solvers. Outputs AUC and curve data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cross.domain.uot.m1_ablation.fixed_decoder import (
    decode_with_fixed_rc_uot_q,
    DecodeConfig,
)
from cross.domain.uot.m1_ablation.evaluation import compute_metrics


@dataclass
class PrecisionCoveragePoint:
    """One point on the precision-coverage curve."""
    coverage: float
    precision: float
    recall: float
    f1: float
    n_predicted: int
    n_abstained: int
    threshold: float | None


@dataclass
class PrecisionCoverageResult:
    """Precision-coverage curve for one solver."""
    solver: str
    points: list[PrecisionCoveragePoint]
    auc: float
    best_precision_at_target_coverage: float
    target_coverage: float
    best_threshold: float | None


def compute_precision_coverage_curve(
    T: np.ndarray,
    C: np.ndarray,
    *,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    src_all: Any,
    dst_norm: Any,
    truth: dict[str, str],
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    solver_name: str = "",
    decode_config: DecodeConfig | None = None,
    n_source_flows: int | None = None,
    threshold_grid: list[float] | None = None,
    target_coverage: float | None = None,
) -> PrecisionCoverageResult:
    """Compute precision-coverage curve for a transport plan.

    Sweeps score thresholds and records (precision, coverage) at each point.

    Args:
        T: Transport/assignment matrix.
        C: Cost matrix.
        source_flows, target_flows: Flow segments.
        src_all, dst_norm: Transaction DataFrames.
        truth: Ground truth.
        eth_ts, bnb_ts: Timestamp dicts.
        solver_name: Solver label.
        decode_config: Decoder config.
        n_source_flows: Total source flows.
        threshold_grid: Score thresholds to sweep.
        target_coverage: Target coverage for computing best precision.

    Returns:
        PrecisionCoverageResult with curve points and AUC.
    """
    cfg = decode_config or DecodeConfig()
    T = np.asarray(T, dtype=float)
    n_sf = n_source_flows or T.shape[0]

    # Build threshold grid from mass distribution if not provided
    if threshold_grid is None:
        masses = T[T > 0] if T.size > 0 else np.array([0.0])
        if masses.size == 0:
            masses = np.array([0.0])
        pcts = np.linspace(0, 100, 7)
        grid = list(np.percentile(masses, pcts))
        grid = sorted(set(round(float(x), 10) for x in grid))
        grid = [x for x in grid if x >= 0]
        if not grid:
            grid = [0.0]
        if grid[0] != 0.0:
            grid.insert(0, 0.0)
    else:
        grid = sorted(set(float(t) for t in threshold_grid))

    # Also include no-threshold point
    grid_to_eval = [None] + [t for t in grid if t > 0]

    points: list[PrecisionCoveragePoint] = []
    for thr in grid_to_eval:
        decode_result = decode_with_fixed_rc_uot_q(
            T=T, C=C,
            source_flows=source_flows,
            target_flows=target_flows,
            src_all=src_all,
            dst_norm=dst_norm,
            truth=truth,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            config=cfg,
            score_threshold=thr,
        )
        metrics = compute_metrics(
            mapping=decode_result.mapping,
            truth=truth,
            n_source_flows=n_sf,
            n_abstained=decode_result.n_abstained,
            n_predicted=decode_result.n_predicted,
            solver_name=solver_name,
            score_threshold=thr,
        )
        if metrics.n_predicted > 0:
            points.append(PrecisionCoveragePoint(
                coverage=metrics.coverage,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                n_predicted=metrics.n_predicted,
                n_abstained=metrics.n_abstained,
                threshold=thr,
            ))

    # Sort by coverage ascending, deduplicate coverage values (keep max precision per coverage)
    points.sort(key=lambda p: (p.coverage, -p.precision))
    deduped: list[PrecisionCoveragePoint] = []
    for p in points:
        if deduped and abs(deduped[-1].coverage - p.coverage) < 1e-9:
            deduped[-1] = p  # keep the one with higher precision
        else:
            deduped.append(p)

    # Compute AUC using trapezoidal rule (x=coverage, y=precision)
    # Extend to x=0 and x=max_coverage for fair comparison
    if deduped:
        x_vals = [float(p.coverage) for p in deduped]
        y_vals = [float(p.precision) for p in deduped]
        min_x, max_x = min(x_vals), max(x_vals)

        # Pad with precision=0 at coverage=0 if needed
        if min_x > 1e-9:
            x_vals.insert(0, 0.0)
            y_vals.insert(0, 0.0)

        # Normalize AUC by max coverage for fair comparison across solvers
        auc = 0.0
        for k in range(len(x_vals) - 1):
            x0, x1 = x_vals[k], x_vals[k + 1]
            y0, y1 = y_vals[k], y_vals[k + 1]
            if x1 > x0 + 1e-12:
                auc += (y0 + y1) * (x1 - x0) / 2.0
    else:
        auc = 0.0

    # Best precision at target coverage
    best_prec = 0.0
    best_thr = None
    if target_coverage is not None:
        for p in points:
            if p.coverage >= target_coverage:
                if p.precision > best_prec:
                    best_prec = p.precision
                    best_thr = p.threshold

    return PrecisionCoverageResult(
        solver=solver_name,
        points=points,
        auc=round(float(auc), 6),
        best_precision_at_target_coverage=round(best_prec, 6),
        target_coverage=target_coverage or 0.0,
        best_threshold=best_thr,
    )


def save_precision_coverage_curve(
    result: PrecisionCoverageResult,
    output_dir: Path,
    *,
    save_png: bool = True,
) -> dict[str, Path]:
    """Save precision-coverage curve data as CSV and optional PNG.

    Returns:
        {ext: path} for saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # CSV of curve points
    csv_path = output_dir / f"precision_coverage_{result.solver}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("solver,coverage,precision,recall,f1,n_predicted,n_abstained,threshold\n")
        for p in result.points:
            f.write(f"{result.solver},{p.coverage:.6f},{p.precision:.6f},{p.recall:.6f},"
                    f"{p.f1:.6f},{p.n_predicted},{p.n_abstained},{p.threshold}\n")
    paths["csv"] = csv_path

    # Try to render PNG using matplotlib if available
    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            covs = [p.coverage for p in result.points]
            precs = [p.precision for p in result.points]
            ax.plot(covs, precs, "b-", linewidth=1.5, label=f"{result.solver} (AUC={result.auc:.4f})")
            if hasattr(result, "target_coverage") and result.target_coverage:
                ax.axvline(x=result.target_coverage, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Coverage")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Coverage Curve")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            png_path = output_dir / f"precision_coverage_{result.solver}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths["png"] = png_path

            pdf_path = output_dir / f"precision_coverage_{result.solver}.pdf"
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(covs, precs, "b-", linewidth=1.5, label=f"{result.solver} (AUC={result.auc:.4f})")
            if hasattr(result, "target_coverage") and result.target_coverage:
                ax.axvline(x=result.target_coverage, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Coverage")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Coverage Curve")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            paths["pdf"] = pdf_path
        except Exception:
            pass

    # Combined CSV for all curves
    combined_csv = output_dir / "precision_coverage.csv"
    header_written = combined_csv.exists()
    with open(combined_csv, "a" if header_written else "w", encoding="utf-8") as f:
        if not header_written:
            f.write("solver,coverage,precision,recall,f1,n_predicted,n_abstained,threshold\n")
        for p in result.points:
            f.write(f"{result.solver},{p.coverage:.6f},{p.precision:.6f},{p.recall:.6f},"
                    f"{p.f1:.6f},{p.n_predicted},{p.n_abstained},{p.threshold}\n")
    paths["combined_csv"] = combined_csv

    return paths
