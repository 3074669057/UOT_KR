"""ONE-SHOT locked test on seeds 42-46.

Requires calibration/locked_decoder.json (aborts otherwise). After this script produces
locked_test/aggregated.csv, the decoder is considered TESTED: no further tuning is allowed.

Method variants (same grids, same ground truth):
  RC-UOT-Q legacy (D0: pi >= 1e-9), RC-UOT-Q locked (shared decoder),
  Balanced-OT legacy (D0), Balanced-OT locked (same shared decoder),
  Threshold-MM frozen (previous study tau/cutoff, NOT re-tuned),
  Connector-style / ABCTracer-style (project's existing rules).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import (  # noqa: E402
    decode_abctracer, decode_connector, decode_threshold_mm,
)
from decoder_audit.da_common import (  # noqa: E402
    AUDIT, BRIDGES, TEST_SEEDS, THRESHOLD_MM_CUTOFF, bootstrap_ci, decode, evaluate_edges,
    load_test_bot, load_test_uot,
)


def main() -> int:
    lock_path = AUDIT / "calibration" / "locked_decoder.json"
    if not lock_path.is_file():
        raise SystemExit("locked_decoder.json missing — the decoder has not been locked; aborting test.")
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_cfg = {"name": locked["decoder_name"], "family": locked["decoder_family"],
                "params": locked["params"]}
    legacy_cfg = {"name": "D0_legacy", "family": "D0", "params": {"thr": 1e-9}}
    out = AUDIT / "locked_test"
    out.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    for bridge in BRIDGES:
        for seed in TEST_SEEDS:
            inst_uot = load_test_uot(bridge, seed)
            inst_bot = load_test_bot(bridge, seed)
            variants = [
                ("RC-UOT-Q legacy", inst_uot, legacy_cfg, None),
                ("RC-UOT-Q locked", inst_uot, lock_cfg, None),
                ("Balanced-OT legacy", inst_bot, legacy_cfg, None),
                ("Balanced-OT locked", inst_bot, lock_cfg, None),
                ("Threshold-MM frozen", inst_uot, None, decode_threshold_mm(inst_uot, THRESHOLD_MM_CUTOFF)),
                ("Connector-style", inst_uot, None, decode_connector(inst_uot)),
                ("ABCTracer-style", inst_uot, None, decode_abctracer(inst_uot)),
            ]
            for name, inst, cfg, pre_edges in variants:
                edges = pre_edges if pre_edges is not None else decode(inst, inst["P"], cfg)
                df, summ = evaluate_edges(inst, edges)
                # abstention: sources with positive mass but zero decoded edges
                edge_set = {(s, d) for s, d in edges}
                srcs_with_edges = {s for s, _ in edge_set}
                n_abstain = sum(1 for s in inst["sids"] if s not in srcs_with_edges)
                df["bridge"] = bridge
                df["seed"] = seed
                df["method"] = name
                all_frames.append(df)
                per_seed_rows.append({
                    "bridge": bridge, "seed": seed, "method": name,
                    "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                    "overall_exact": summ["overall_exact"],
                    "edge_precision": summ["edge_precision"], "edge_recall": summ["edge_recall"],
                    "edge_f1": summ["edge_f1"],
                    "split_edge_f1": summ["split_edge_f1"], "merge_edge_f1": summ["merge_edge_f1"],
                    "degree_acc": float(np.mean([summ["deg_acc_split"], summ["deg_acc_merge"]])),
                    "fp_per_template": summ["edge_fp_total"] / 48,
                    "pred_edges_per_template": summ["n_pred_edges"],
                    "coverage": summ["coverage"],
                    "abstained_sources": n_abstain,
                })
            print(f"[locked-test] {bridge} seed {seed} done", flush=True)
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    all_f = pd.concat(all_frames, ignore_index=True)
    all_f.to_csv(out / "per_template.csv", index=False)

    # aggregated: mean over seeds, then mean/std/CI over the 5 seeds per bridge x method
    agg_rows: list[dict[str, Any]] = []
    for (br, method), g in per_seed.groupby(["bridge", "method"]):
        row = {"bridge": br, "method": method, "n_seeds": int(len(g))}
        for col in ("split_exact", "merge_exact", "overall_exact", "edge_precision",
                    "edge_recall", "edge_f1", "split_edge_f1", "merge_edge_f1", "degree_acc",
                    "fp_per_template", "pred_edges_per_template", "coverage",
                    "abstained_sources"):
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            row[f"{col}_mean"] = float(v.mean())
            row[f"{col}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            lo, hi = bootstrap_ci(v, seed=42)
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out / "aggregated.csv", index=False)

    # decoder comparison markdown
    lines = ["# Locked-test decoder comparison (seeds 42-46, mean +/- std over 5 seeds)", "",
             f"Locked decoder: `{locked['decoder_name']}` "
             f"(family {locked['decoder_family']}, params {locked['params']}); "
             "selected on calibration seeds 101-103 before this test was run; "
             "Threshold-MM frozen (tau 0.05, cutoff 0.477623).",
             "",
             "| Bridge | Method | Split exact | Merge exact | Edge P | Edge R | Edge F1 | "
             "Split F1 | Merge F1 | FP/tpl | Pred/tpl | Coverage | Abstained src |"]
    for (br, method), g in per_seed.groupby(["bridge", "method"]):
        m = g.mean(numeric_only=True)
        s = g.std(numeric_only=True)
        lines.append(f"| {br} | {method} | {m['split_exact']:.3f} | {m['merge_exact']:.3f} | "
                     f"{m['edge_precision']:.4f} +/- {s['edge_precision']:.4f} | "
                     f"{m['edge_recall']:.4f} | {m['edge_f1']:.4f} +/- {s['edge_f1']:.4f} | "
                     f"{m['split_edge_f1']:.4f} | {m['merge_edge_f1']:.4f} | "
                     f"{m['fp_per_template']:.1f} | {m['pred_edges_per_template']:.1f} | "
                     f"{m['coverage']:.3f} | {m['abstained_sources']:.1f} |")
    (out / "decoder_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
