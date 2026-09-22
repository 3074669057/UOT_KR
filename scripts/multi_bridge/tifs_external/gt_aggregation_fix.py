"""Corrected aggregated GT edge layer builder (GT-CORRECTION, data-only).

GT_LINKAGE/AGGREGATION CORRECTION #2 (see DATASET_ACCOUNTING_RECONCILIATION.md):
the Stage B label files contained one row PER ANCHOR; multiple anchors of the
same flow pair produced duplicate (src_flow, dst_flow) rows. The frozen
historical schema aggregates per flow pair with a support list. This script
rebuilds the GT EDGE layer (one row per unique flow pair, support list attached)
from the frozen anchor layer, WITHOUT touching anchors.json and WITHOUT any
method involvement. After this correction GT semantics are frozen.

Outputs: flow_edges_canonical.json per block + cumulative primary population
list + recomputed adequacy statistics.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"
BLOCKS = ["b01", "b02", "b03"]


def main() -> int:
    all_edges: dict[tuple, dict] = {}
    per_block_edges: dict[str, list[dict]] = {}
    for b in BLOCKS:
        labels = json.loads((ROOT / "blocks" / b / "flow_labels_canonical.json")
                            .read_text(encoding="utf-8"))
        agg: dict[tuple, list[dict]] = defaultdict(list)
        for lb in labels:
            agg[(lb["src_flow_id"], lb["dst_flow_id"])].append(lb)
        rows = []
        for (s, t), grp in sorted(agg.items()):
            first = grp[0]
            rows.append({
                "src_flow_id": s, "dst_flow_id": t,
                "support_tx_pair_count": len(grp),
                "support_anchors": sorted({g["transferId"] for g in grp}),
                "sender": first["sender"], "receiver": first["receiver"],
                "ctx_src": first["ctx_src"], "ctx_dst": first["ctx_dst"],
                "ts_min": min(g["ts"] for g in grp),
                "ts_max": max(g["ts"] for g in grp),
                "tier": "A",
                "block": b,
            })
        per_block_edges[b] = rows
        for r in rows:
            all_edges[(r["src_flow_id"], r["dst_flow_id"])] = r

    out_deg = Counter(s for (s, t) in all_edges)
    in_deg = Counter(t for (s, t) in all_edges)
    for (s, t), r in all_edges.items():
        sd, dd = out_deg[s], in_deg[t]
        r["n_src"] = dd
        r["n_dst"] = sd
        # canonical row-level tag: n_src=dd (sources of this destination),
        # n_dst=sd (destinations of this source)
        r["canonical_pattern"] = ("1to1" if sd == 1 and dd == 1 else
                                  "1toN_fanout" if sd >= 2 and dd == 1 else
                                  "Nto1_merge" if sd == 1 and dd >= 2 else "NtoM")

    for b in BLOCKS:
        out_f = ROOT / "blocks" / b / "flow_edges_canonical.json"
        out_f.write_text(json.dumps(per_block_edges[b], indent=1), encoding="utf-8")

    fan = {s: d for s, d in out_deg.items() if d >= 2}
    merge = {t: d for t, d in in_deg.items() if d >= 2}
    sender_of: dict[str, str] = {}
    block_of: dict[str, str] = {}
    for (s, t), r in all_edges.items():
        sender_of.setdefault(s, r["sender"])
        block_of.setdefault(s, r["block"])
    deg = Counter(fan.values())
    addr_counts = Counter(sender_of[s] for s in fan)
    ts_all = [r["ts_min"] for r in all_edges.values()]
    spread = max(ts_all) - min(ts_all)
    n_fan = len(fan)
    max_share = max(addr_counts.values()) / n_fan if n_fan else None
    G = len(addr_counts)
    gates = {
        "G_ge_8": G >= 8,
        "max_share_lt_0.5": (max_share is not None and max_share < 0.5),
        "n_fanout_ge_30": n_fan >= 30,
        "spread_ge_2_months": spread >= 2 * 2592000,
    }
    edge_topo = Counter(all_edges[k]["canonical_pattern"] for k in all_edges)
    report = {
        "correction": "per-anchor rows -> per-flow-pair edges with support lists",
        "n_anchor_rows": sum(len(json.loads((ROOT / 'blocks' / b /
                          'flow_labels_canonical.json').read_text(encoding='utf-8')))
                          for b in BLOCKS),
        "n_unique_edges": len(all_edges),
        "edge_topology": dict(edge_topo),
        "n_src_flows": len(out_deg), "n_dst_flows": len(in_deg),
        "n_fanout_components": n_fan,
        "n_fanout_edges": sum(fan.values()),
        "degree_distribution": {str(k): v for k, v in sorted(deg.items())},
        "identity1_sum_components": sum(deg.values()),
        "identity2_sum_deg_weighted": sum(d * n for d, n in deg.items()),
        "n_merge_components": len(merge),
        "n_merge_edges": sum(merge.values()),
        "merge_invariant_2x": sum(merge.values()) >= 2 * len(merge),
        "cluster_addrs_G": G,
        "max_cluster_share": round(max_share, 4) if max_share is not None else None,
        "temporal_spread_days": round(spread / 86400, 1),
        "adequacy_gates": gates,
        "DATA_ADEQUACY": "PASS" if all(gates.values()) else "FAIL",
    }
    (ROOT / "corrected_adequacy.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    # primary population manifest list (frozen)
    primary_rows = []
    for s, d in sorted(fan.items()):
        primary_rows.append({
            "component_id": s,
            "cluster_key": "addr_" + sender_of[s][2:10],
            "cluster_full": sender_of[s],
            "degree": d,
            "block": block_of[s],
            "tier": "A",
            "dsts": sorted(t for (x, t) in all_edges if x == s),
        })
    (ROOT / "primary_population_list.json").write_text(
        json.dumps(primary_rows, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("GT_AGGREGATION_CORRECTION:", "APPLIED")
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
