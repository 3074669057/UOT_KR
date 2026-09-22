"""Three-bridge structural recovery: one-to-one baselines vs (paradigm) RC-UOT-Q.

Synthetic split/merge is seeded from each bridge's real 1-to-1 anchors (dev split).
For every bridge, the one-to-one baselines (Connector-style, ABCTracer-style) are
mathematically unable to emit a 1->2 split or 2->1 merge, so their split/merge
recovery is 0 on all bridges.  RC-UOT-Q's many-to-many transport is the only
method capable of representing split/merge (Celer full-pipeline reference 0.946 /
0.967).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = REPO / "out" / "multi_bridge_expansion" / "structural_recovery_three_bridges"
BRIDGES = ("Celer", "Multi", "Poly")
N_TEMPLATES = 48
SEEDS = (42, 43, 44, 45, 46)

RC_SPLIT = 0.9458333333333332  # Celer full-pipeline reference (manuscript Table 2)
RC_MERGE = 0.9666666666666668


def _f(v: Any) -> float:
    return float(pd.to_numeric(v, errors="coerce") or 0.0)


def load_dev_anchors(bridge: str) -> pd.DataFrame:
    root = REPO / "out" / "multi_bridge_expansion" / bridge
    src = pd.read_csv(root / "source_features.csv", dtype=str, keep_default_na=False)
    dev = src[src["split"] == "development"].copy()
    dev["amt"] = dev["source_amount_raw"].map(_f)
    dev["ts"] = dev["source_timestamp"].map(_f)
    dev = dev[dev["amt"] > 0]
    return dev[["source_tx_hash", "amt", "ts"]].reset_index(drop=True)


def generate_synthetic(anchors: pd.DataFrame, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    idxs = list(range(len(anchors)))
    rng.shuffle(idxs)
    take = idxs[:N_TEMPLATES]
    src_rows: list[dict[str, Any]] = []
    dst_rows: list[dict[str, Any]] = []
    split_edges: set[tuple[str, str]] = set()
    merge_edges: set[tuple[str, str]] = set()
    for k, i in enumerate(take):
        a = anchors.loc[i, "amt"]
        t = anchors.loc[i, "ts"]
        base = f"tpl_{k}"
        # sources
        srcs = {
            "split_src": a,
            "merge_src1": a * 0.5,
            "merge_src2": a * 0.5,
            "unmatched_src": a,
            "noise_src0": a,
            "noise_src1": a,
        }
        dsts = {
            "split_a": a * 0.5,
            "split_b": a * 0.5,
            "merge_dst": a,
            "hidden_dst": a * 0.0,
            "noise_dst0": a,
            "noise_dst1": a,
        }
        for sid, amt in srcs.items():
            src_rows.append({"flow_id": f"{base}__{sid}", "amount": amt, "time": t})
        for did, amt in dsts.items():
            dst_rows.append({"flow_id": f"{base}__{did}", "amount": amt, "time": t + 277})
        split_edges.add((f"{base}__split_src", f"{base}__split_a"))
        split_edges.add((f"{base}__split_src", f"{base}__split_b"))
        merge_edges.add((f"{base}__merge_src1", f"{base}__merge_dst"))
        merge_edges.add((f"{base}__merge_src2", f"{base}__merge_dst"))
    return {
        "sources": pd.DataFrame(src_rows),
        "destinations": pd.DataFrame(dst_rows),
        "split_edges": split_edges,
        "merge_edges": merge_edges,
    }


def amount_cost(sa: float, da: float) -> float:
    return float(min(abs(sa - da) / max(abs(sa), abs(da), 1e-12), 1.0))


def run_one_to_one(syn: dict[str, Any], use_time: bool) -> set[tuple[str, str]]:
    srcs = syn["sources"]
    dsts = syn["destinations"]
    # group by template (base)
    edges: set[tuple[str, str]] = set()
    for base, grp in srcs.groupby(srcs["flow_id"].str.split("__").str[0]):
        dids = dsts[dsts["flow_id"].str.startswith(base)]
        for _, srow in grp.iterrows():
            sa = srow["amount"]
            best = None
            best_c = float("inf")
            for _, drow in dids.iterrows():
                da = drow["amount"]
                c = amount_cost(sa, da)
                if use_time:
                    delay = drow["time"] - srow["time"]
                    c = 0.75 * c + 0.25 * min(abs(delay) / 3600.0, 1.0)
                if c < best_c:
                    best_c = c
                    best = drow["flow_id"]
            if best is not None:
                edges.add((srow["flow_id"], best))
    return edges


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default: out/multi_bridge_expansion/structural_recovery_three_bridges)")
    cli = ap.parse_args()

    global OUT
    if cli.out:
        OUT = Path(cli.out)
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "RC-UOT-Q": {"scope": "Celer full flow-level pipeline (frozen, manuscript Table 2)",
                      "split_recovery": RC_SPLIT, "merge_recovery": RC_MERGE},
        "baselines_one_to_one": {},
    }
    per_seed_rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        anchors = load_dev_anchors(bridge)
        conn_s, conn_m, abc_s, abc_m = [], [], [], []
        for seed in SEEDS:
            syn = generate_synthetic(anchors, seed)
            ce = run_one_to_one(syn, use_time=False)
            ae = run_one_to_one(syn, use_time=True)
            cs = sum(1 for e in syn["split_edges"] if e in ce) / max(len(syn["split_edges"]), 1)
            cm = sum(1 for e in syn["merge_edges"] if e in ce) / max(len(syn["merge_edges"]), 1)
            as_ = sum(1 for e in syn["split_edges"] if e in ae) / max(len(syn["split_edges"]), 1)
            am = sum(1 for e in syn["merge_edges"] if e in ae) / max(len(syn["merge_edges"]), 1)
            conn_s.append(cs)
            conn_m.append(cm)
            abc_s.append(as_)
            abc_m.append(am)
            per_seed_rows.append({"bridge": bridge, "method": "Connector-style", "seed": seed,
                                  "split_recovery": cs, "merge_recovery": cm, "n_templates": N_TEMPLATES})
            per_seed_rows.append({"bridge": bridge, "method": "ABCTracer-style", "seed": seed,
                                  "split_recovery": as_, "merge_recovery": am, "n_templates": N_TEMPLATES})
        result["baselines_one_to_one"][bridge] = {
            "Connector-style": {"split_recovery": float(np.mean(conn_s)), "merge_recovery": float(np.mean(conn_m))},
            "ABCTracer-style": {"split_recovery": float(np.mean(abc_s)), "merge_recovery": float(np.mean(abc_m))},
            "n_dev_anchors_available": int(len(anchors)),
        }
        print(f"{bridge}: anchors={len(anchors)} conn split={np.mean(conn_s):.3f} merge={np.mean(conn_m):.3f} | abc split={np.mean(abc_s):.3f} merge={np.mean(abc_m):.3f}")

    (OUT / "structural_three_bridges.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(per_seed_rows).to_csv(OUT / "baseline_structural_per_seed.csv", index=False)
    print("\n", json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
