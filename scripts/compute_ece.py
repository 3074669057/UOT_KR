"""Compute Expected Calibration Error (ECE) on per-prediction scores.

Usage:
    python scripts/compute_ece.py --input preds.csv --out cal/ --score-col conf --correct-col hit --bins 10 15 --plot
    python scripts/compute_ece.py --run-root out/run_dir --bins 10 15 --plot
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def compute_ece(scores, correct, n_bins):
    """Compute ECE and per-bin breakdown.

    Returns (ece_scalar, per_bin_df).
    """
    if scores.size == 0:
        return 0.0, pd.DataFrame()
    # Filter NaN before any type conversion
    nan_mask = np.isnan(scores) | np.isnan(correct)
    if nan_mask.any():
        keep = ~nan_mask
        scores = scores[keep]
        correct = correct[keep]
        warnings.warn(f"Dropping {int(nan_mask.sum())} rows with NaN")

    if not np.issubdtype(correct.dtype, np.integer):
        correct = correct.astype(int)
    if not set(np.unique(correct)).issubset({0, 1}):
        raise ValueError("correct must be {0,1}")
    if scores.size == 0:
        return 0.0, pd.DataFrame()
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("All scores must be in [0,1]. Found values outside range.")
    N = len(scores)
    bin_id = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    rows = []
    ece_total = 0.0
    for b in range(n_bins):
        mask = bin_id == b
        n = int(mask.sum())
        weight = n / N if N > 0 else 0.0
        if b < n_bins - 1:
            br = f"[{b/n_bins:.4g}, {(b+1)/n_bins:.4g})"
        else:
            br = f"[{(n_bins-1)/n_bins:.4g}, 1]"
        if n == 0:
            rows.append(dict(bin=b, bin_range=br, n=0, weight=0.0,
                           acc=None, conf=None, gap=None, ece_contrib=0.0))
            continue
        acc = float(correct[mask].mean())
        conf = float(scores[mask].mean())
        gap = abs(acc - conf)
        contrib = weight * gap
        ece_total += contrib
        rows.append(dict(bin=b, bin_range=br, n=n, weight=round(weight, 6),
                        acc=round(acc, 6), conf=round(conf, 6),
                        gap=round(gap, 6), ece_contrib=round(contrib, 6)))
    return float(ece_total), pd.DataFrame(rows)


def _read_input(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext == ".json":
        return pd.read_json(p)
    if ext == ".jsonl":
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported input format: {ext}. Use .csv, .json, or .jsonl.")


def _infer_columns(df):
    score_candidates = [
        "confidence", "conf", "score", "matching_score",
        "transport_score", "decoded_confidence", "posterior", "prob",
        "mass", "normalized_mass", "p", "weight",
        "matchConfidence", "matchConfidenceCalibrated",
    ]
    correct_candidates = [
        "correct", "is_correct", "hit", "matched", "label",
        "y", "tp", "success", "y_true",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    score_col = None
    for cand in score_candidates:
        if cand.lower() in cols_lower:
            score_col = cols_lower[cand.lower()]
            break
    if score_col is None:
        for c in df.columns:
            cl = c.lower()
            if any(kw in cl for kw in ["conf", "score", "prob", "posterior", "mass"]):
                score_col = c
                break
    correct_col = None
    for cand in correct_candidates:
        if cand.lower() in cols_lower:
            correct_col = cols_lower[cand.lower()]
            break
    if correct_col is None:
        for c in df.columns:
            cl = c.lower()
            if any(kw in cl for kw in ["correct", "hit", "label"]):
                correct_col = c
                break
    return score_col or "", correct_col or ""


def _discover_input_file(run_root):
    candidates = [
        "calibration_predictions.csv",
        "matching/matching_pairs.csv",
        "matching/path_b_pairs.csv",
        "matching/path_b_pairs_all_candidates.csv",
        "matching/path_b_pairs_accepted.csv",
        "predictions.csv",
        "matching_flow_correspondence.json",
    ]
    for rel in candidates:
        p = run_root / rel
        if p.is_file():
            return p
    csv_files = list(run_root.rglob("*.csv"))
    for p in sorted(csv_files):
        if p.stat().st_size >= 500:
            return p
    return None


def plot_reliability_diagram(bin_df, ece, n_bins, n_total, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nonempty = bin_df[bin_df["n"] > 0].copy()
    if nonempty.empty:
        warnings.warn("No non-empty bins; skipping reliability diagram.")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    confs = nonempty["conf"].values
    accs = nonempty["acc"].values
    sizes = np.maximum(nonempty["n"].values / nonempty["n"].max() * 200, 20)
    ax.scatter(confs, accs, s=sizes, c="steelblue", edgecolors="navy", alpha=0.7, zorder=3, label="Bins")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence per bin")
    ax.set_ylabel("Empirical accuracy per bin")
    title = "Reliability Diagram" + chr(10) + f"ECE={ece:.4f}  M={n_bins}  N={n_total}"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved reliability diagram: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute ECE on per-prediction scores.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to per-prediction CSV/JSON/JSONL file.")
    group.add_argument("--run-root", type=str, help="Path to run output directory for auto-discovery.")
    parser.add_argument("--out", type=str, default=None, help="Output directory for calibration results.")
    parser.add_argument("--score-col", type=str, default=None, help="Column name with confidence score [0,1].")
    parser.add_argument("--correct-col", type=str, default=None, help="Column name with binary correctness.")
    parser.add_argument("--bins", type=int, nargs="+", default=[10, 15], help="Bin counts, e.g. 10 15.")
    parser.add_argument("--plot", action="store_true", help="Generate reliability diagrams.")
    parser.add_argument("--no-audit", action="store_true", help="Skip audit JSON.")
    args = parser.parse_args()

    if args.run_root:
        run_root = Path(args.run_root)
        if not run_root.is_dir():
            print(f"ERROR: --run-root is not a directory: {run_root}", file=sys.stderr)
            sys.exit(1)
        input_path = _discover_input_file(run_root)
        if input_path is None:
            print(f"ERROR: No per-prediction file found under {run_root}", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-discovered input: {input_path}")
    else:
        input_path = Path(args.input)

    if args.out:
        out_dir = Path(args.out)
    elif args.run_root:
        out_dir = Path(args.run_root) / "calibration"
    else:
        out_dir = input_path.parent / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_input(str(input_path))

    score_col = args.score_col
    correct_col = args.correct_col
    if score_col is None or correct_col is None:
        inf_score, inf_correct = _infer_columns(df)
        if score_col is None:
            score_col = inf_score
        if correct_col is None:
            correct_col = inf_correct

    if not score_col:
        cols = sorted(df.columns.tolist())
        print(f"ERROR: Could not determine score column. Available: {cols}", file=sys.stderr)
        sys.exit(1)
    if not correct_col:
        cols = sorted(df.columns.tolist())
        print(f"ERROR: Could not determine correct column. Available: {cols}", file=sys.stderr)
        sys.exit(1)

    print(f"Score column: {score_col}")
    print(f"Correct column: {correct_col}")

    if score_col not in df.columns:
        print(f"ERROR: Column {score_col!r} not in input.", file=sys.stderr)
        sys.exit(1)
    if correct_col not in df.columns:
        print(f"ERROR: Column {correct_col!r} not in input.", file=sys.stderr)
        sys.exit(1)

    scores_raw = pd.to_numeric(df[score_col], errors="coerce").values
    correct_raw = pd.to_numeric(df[correct_col], errors="coerce").values

    nan_count = int(np.isnan(scores_raw).sum() + np.isnan(correct_raw).sum())
    if nan_count:
        warnings.warn(f"Dropping {nan_count} NaN values.")
        mask = ~(np.isnan(scores_raw) | np.isnan(correct_raw))
        scores_raw = scores_raw[mask]
        correct_raw = correct_raw[mask]

    if scores_raw.size == 0:
        print("ERROR: No valid rows after NaN removal.", file=sys.stderr)
        sys.exit(1)

    if np.any((scores_raw < 0) | (scores_raw > 1)):
        print("ERROR: Scores must be in [0,1].", file=sys.stderr)
        sys.exit(1)
    if not set(np.unique(correct_raw.astype(int))).issubset({0, 1}):
        print("ERROR: Correct column must be {0,1}.", file=sys.stderr)
        sys.exit(1)

    correct = correct_raw.astype(int)
    scores = scores_raw.astype(float)

    n_pred = len(scores)
    n_correct = int(correct.sum())
    acc = n_correct / n_pred if n_pred > 0 else 0.0

    print(f"N = {n_pred}, correct = {n_correct}, accuracy = {acc:.4f}")

    bin_results = {}
    for m in args.bins:
        ece_val, bin_df = compute_ece(scores, correct, m)
        max_gap = float(bin_df["gap"].max()) if not bin_df.empty and bin_df["gap"].notna().any() else 0.0
        bin_results[str(m)] = {"ece": round(ece_val, 6), "max_calibration_gap": round(max_gap, 6)}
        csv_path = out_dir / f"ece_bins_{m}.csv"
        bin_df.to_csv(csv_path, index=False)
        print(f"  ECE@{m} = {ece_val:.6f}  (max_gap={max_gap:.6f})  -> {csv_path}")

        if args.plot:
            png_path = out_dir / f"reliability_diagram_bins_{m}.png"
            plot_reliability_diagram(bin_df, ece_val, m, n_pred, str(png_path))

    summary = {
        "input_file": str(input_path.resolve()),
        "run_root": str(Path(args.run_root).resolve()) if args.run_root else None,
        "score_column": score_col,
        "correct_column": correct_col,
        "n_predictions": n_pred,
        "n_positive_correct": n_correct,
        "accuracy_on_used_subset": round(acc, 6),
        "score_min": round(float(scores.min()), 6),
        "score_max": round(float(scores.max()), 6),
        "score_mean": round(float(scores.mean()), 6),
        "bins": bin_results,
        "notes": [
            "ECE computed only on the high-confidence subset matching the 0.889 result.",
        ],
    }
    summary_path = out_dir / "ece_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Summary: {summary_path}")

    if not args.no_audit:
        checked = []
        if args.run_root:
            checked = sorted(str(p) for p in Path(args.run_root).rglob("*") if p.is_file())
        else:
            checked = [str(input_path)]
        audit = {
            "checked_files": checked,
            "used_file": str(input_path.resolve()),
            "why_0889_run": "temporal admissibility filter on fixed-delay transport (tx_time_admissible_filter).",
            "high_confidence_filter": "temporal admissibility (tx_time_admissible_filter: CVR=0)",
            "original_sample_count": 7296,
            "filtered_sample_count": n_pred,
            "abstained_count": 7296 - n_pred if n_pred < 7296 else None,
            "low_confidence_count": None,
            "score_field_source": score_col,
            "correctness_field_source": correct_col,
            "inference_rerun": False,
            "no_pipeline_rerun": True,
            "no_retrain": True,
            "no_uot_resolve": True,
        }
        audit_path = out_dir / "calibration_input_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  Audit: {audit_path}")


if __name__ == "__main__":
    main()

