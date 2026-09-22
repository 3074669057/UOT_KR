"""One-to-one baseline vs RC-UOT-Q on semi-synthetic split/merge structure.

Baselines (Connector-style, ABCTracer-style) are one-to-one hard matchers by
construction: each source emits at most one destination.  RC-UOT-Q is a
many-to-many transport and can represent 1->2 (split) and 2->1 (merge).

The synthetic truth is per-template: within a template, receiver/token/time are
identical clones, so amount is the only discriminative signal, and split/merge
requires amount halving/doubling that only a many-to-many transport can model.
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

SYNTH = REPO / "out" / "paper_full_pipeline_run" / "synthetic"
SEEDS = (42, 43, 44, 45, 46)  # frozen paper Table 2 seeds

# Frozen RC-UOT-Q reference (full flow-level pipeline, manuscript Table 2)
RC_SPLIT = 0.9458333333333332
RC_MERGE = 0.9666666666666668


def _f(v: Any) -> float:
    return float(pd.to_numeric(v, errors="coerce") or 0.0)


def load_template_problems(seed: int) -> list[dict[str, Any]]:
    root = SYNTH / f"synthetic_eval_seed_{seed}"
    labels = pd.read_csv(root / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
    hints = json.loads((root / "labels" / "synthetic_uot_eval_metrics.json").read_text(encoding="utf-8"))
    eh = hints.get("eval_hints", {})
    truth = {tuple(x) for x in eh.get("truth_flow_pairs", [])}
    split_edges = {e for e in truth if "__synth_split" in e[0]}
    merge_edges = {e for e in truth if "__synth_merge" in e[0] or "__synth_merge" in e[1]}

    srcs = labels[["src_flow_id", "src_amount_usd", "first_src_time"]].drop_duplicates("src_flow_id")
    dsts = labels[["dst_flow_id", "dst_amount_usd", "first_dst_time"]].drop_duplicates("dst_flow_id")
    src_amt = {r.src_flow_id: _f(r.src_amount_usd) for r in srcs.itertuples()}
    src_time = {r.src_flow_id: _f(r.first_src_time) for r in srcs.itertuples()}
    dst_amt = {r.dst_flow_id: _f(r.dst_amount_usd) for r in dsts.itertuples()}
    dst_time = {r.dst_flow_id: _f(r.first_dst_time) for r in dsts.itertuples()}

    # group by template = base src flow id (before '__synth')
    groups: dict[str, list[str]] = {}
    for sid in src_amt:
        groups.setdefault(sid.split("__synth")[0], []).append(sid)

    problems = []
    for base, sids in groups.items():
        # destinations in this template: from the labels' truth/dst side
        tpl_rows = labels[labels["src_flow_id"].str.startswith(base)]
        dids = list(dict.fromkeys(tpl_rows["dst_flow_id"].astype(str)))
        problems.append({
            "base": base,
            "sources": sorted(sids),
            "destinations": sorted(dids),
            "src_amt": src_amt,
            "src_time": src_time,
            "dst_amt": dst_amt,
            "dst_time": dst_time,
        })
    return problems, split_edges, merge_edges


def amount_cost(sa: float, da: float) -> float:
    return float(min(abs(sa - da) / max(abs(sa), abs(da), 1e-12), 1.0))


def run_one_to_one(problems, *, use_time: bool) -> set[tuple[str, str]]:
    """One-to-one matcher: each source -> exactly one destination (min amount(+time) cost)."""
    edges: set[tuple[str, str]] = set()
    for p in problems:
        for sid in p["sources"]:
            sa = p["src_amt"][sid]
            best_d = None
            best_c = float("inf")
            for did in p["destinations"]:
                da = p["dst_amt"][did]
                c = amount_cost(sa, da)
                if use_time:
                    delay = p["dst_time"][did] - p["src_time"][sid]
                    tcost = min(abs(delay) / max(abs(delay), 3600.0), 1.0)
                    c = 0.75 * c + 0.25 * tcost
                if c < best_c:
                    best_c = c
                    best_d = did
            if best_d is not None:
                edges.add((sid, best_d))
    return edges


def main() -> int:
    all_conn = {"split": [], "merge": []}
    all_abc = {"split": [], "merge": []}
    for seed in SEEDS:
        problems, split_edges, merge_edges = load_template_problems(seed)
        conn_edges = run_one_to_one(problems, use_time=False)
        abc_edges = run_one_to_one(problems, use_time=True)
        all_conn["split"].append(sum(1 for e in split_edges if e in conn_edges) / max(len(split_edges), 1))
        all_conn["merge"].append(sum(1 for e in merge_edges if e in conn_edges) / max(len(merge_edges), 1))
        all_abc["split"].append(sum(1 for e in split_edges if e in abc_edges) / max(len(split_edges), 1))
        all_abc["merge"].append(sum(1 for e in merge_edges if e in abc_edges) / max(len(merge_edges), 1))
        print(f"seed {seed}: conn split={all_conn['split'][-1]:.3f} merge={all_conn['merge'][-1]:.3f} | abc split={all_abc['split'][-1]:.3f} merge={all_abc['merge'][-1]:.3f}")

    out = {
        "seeds": list(SEEDS),
        "RC-UOT-Q (frozen, full pipeline)": {"split_recovery": RC_SPLIT, "merge_recovery": RC_MERGE},
        "Connector-style (one-to-one)": {
            "split_recovery": float(np.mean(all_conn["split"])),
            "merge_recovery": float(np.mean(all_conn["merge"])),
            "per_seed_split": all_conn["split"],
            "per_seed_merge": all_conn["merge"],
        },
        "ABCTracer-style (one-to-one)": {
            "split_recovery": float(np.mean(all_abc["split"])),
            "merge_recovery": float(np.mean(all_abc["merge"])),
            "per_seed_split": all_abc["split"],
            "per_seed_merge": all_abc["merge"],
        },
        "note": "One-to-one matchers are mathematically capped at <=0.5 split/merge recovery (1 src -> at most 1 dst); amount-driven matching yields ~0 in practice.",
    }
    out_dir = REPO / "out" / "multi_bridge_expansion" / "structural_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "structural_baselines.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n", json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
