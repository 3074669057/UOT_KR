"""Build a compact paper-facing Multi/Poly comparison table from cached metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out" / "multi_bridge_expansion"

CELER = {
    "protocol": "Celer (paper reference)",
    "split": "full anchors",
    "n_pairs": 7296,
    "candidate_pool_recall": None,
    "precision": 0.898,
    "recall": None,
    "f1": None,
    "coverage": None,
    "window_h": None,
    "threshold": None,
}


def read_bridge(name: str) -> dict[str, object]:
    raw = json.loads((OUT / name / "test_metrics.json").read_text(encoding="utf-8"))
    test = raw["test_metrics"]
    cfg = raw["selected_config"]
    return {
        "protocol": f"{name} held-out test",
        "split": "chronological test",
        "n_pairs": int(test["n_total"]),
        "candidate_pool_recall": float(raw["test_candidate_pool_recall"]),
        "precision": float(test["precision"]),
        "recall": float(test["recall"]),
        "f1": float(test["full_set_f1"]),
        "coverage": float(test["coverage"]),
        "window_h": float(cfg["window_h"]),
        "threshold": float(cfg["confidence_threshold"]),
    }


def main() -> None:
    rows = [CELER, read_bridge("Multi"), read_bridge("Poly")]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "paper_comparison.csv", index=False)
    lines = [
        "| protocol | split | pairs | candidate recall | precision | recall | F1 | coverage | window | threshold |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        def fmt(v, digits=4):
            return "NA" if pd.isna(v) else f"{float(v):.{digits}f}"
        lines.append(
            f"| {row['protocol']} | {row['split']} | {int(row['n_pairs'])} | {fmt(row['candidate_pool_recall'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} | {fmt(row['coverage'])} | {row['window_h'] if pd.notna(row['window_h']) else 'NA'} | {row['threshold'] if pd.notna(row['threshold']) else 'NA'} |"
        )
    (OUT / "paper_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
