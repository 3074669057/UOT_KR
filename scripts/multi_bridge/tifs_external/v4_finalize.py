"""v4 finalization (data-only): cumulative statistics, primary source-unit
list + SHA256, topology, tiers, disjointness summary, statistical branch.
Runs AFTER the progressive accrual has stopped. NO solver, NO prediction.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v4"
PKG4 = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_validation_preregistration_v4"
V4_START = 1716076800


def main() -> int:
    cum = json.loads((ROOT / "cumulative_adequacy.json").read_text(encoding="utf-8"))
    stop = cum.get("stop_block")
    edges = json.loads((ROOT / "flow_edges_canonical_cumulative.json").read_text(encoding="utf-8"))
    anchors = []
    for blk in range(1, (stop or 12) + 1):
        f = ROOT / "blocks" / f"b{blk:02d}" / "anchors.json"
        if f.is_file():
            anchors += json.loads(f.read_text(encoding="utf-8"))

    unique_keys = [(e["src_flow_id"], e["dst_flow_id"]) for e in edges]
    dup = [k for k, n in Counter(unique_keys).items() if n > 1]
    out_deg = Counter(e["src_flow_id"] for e in edges)
    in_deg = Counter(e["dst_flow_id"] for e in edges)
    fan = {s: d for s, d in out_deg.items() if d >= 2}
    merge = {t: d for t, d in in_deg.items() if d >= 2}
    sender_of = {}
    block_of = {}
    for e in edges:
        sender_of.setdefault(e["src_flow_id"], e["sender"])
        block_of.setdefault(e["src_flow_id"], e["block"])
    addr_counts = Counter(sender_of[s] for s in fan)
    ts_all = [e["ts_min"] for e in edges]
    G = len(addr_counts)
    n_fan = len(fan)
    max_share = max(addr_counts.values()) / n_fan if n_fan else None
    spread = max(ts_all) - min(ts_all) if ts_all else 0
    deg_dist = Counter(fan.values())
    edge_topo = Counter(e["canonical_pattern"] for e in edges)
    tiers_edges = Counter(e["tier"] for e in edges)

    primary_rows = []
    for s, d in sorted(fan.items()):
        primary_rows.append({"source_unit_id": s, "source_flow_id": s,
                             "gt_dst_set": sorted(t for (x, t) in unique_keys if x == s),
                             "degree": d,
                             "cluster_id": "addr_" + sender_of[s][2:10],
                             "time_block": block_of[s], "tier": "A"})
    pl_payload = json.dumps(primary_rows, sort_keys=True, separators=(",", ":"))
    pl_sha = hashlib.sha256(pl_payload.encode()).hexdigest()

    # connected components (descriptive)
    adj = defaultdict(set)
    rev = defaultdict(set)
    srcs = {e["src_flow_id"] for e in edges}
    dsts = {e["dst_flow_id"] for e in edges}
    for e in edges:
        adj[e["src_flow_id"]].add(e["dst_flow_id"])
        rev[e["dst_flow_id"]].add(e["src_flow_id"])
    visited = set()
    conn = []
    for node in sorted(srcs | dsts):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        cs, cd = set(), set()
        while stack:
            u = stack.pop()
            (cs if u in srcs else cd).add(u)
            for v in adj.get(u, set()):
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
            for v in rev.get(u, set()):
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        ns, nd = len(cs), len(cd)
        cls = ("1to1" if ns == 1 and nd == 1 else "1toN" if ns == 1 and nd >= 2
               else "Nto1" if ns >= 2 and nd == 1 else "NtoM")
        conn.append((ns, nd, cls))

    branch = ("NOT_EVALUABLE" if G < 8 else
              "EXHAUSTIVE" if G <= 20 else "MONTE_CARLO")
    report = {
        "v4_start": V4_START,
        "stop_block": stop,
        "untouched_after_stop": cum.get("UNTOUCHED_AFTER_V4_STOP"),
        "n_anchors_total": len(anchors),
        "n_unique_edges": len(edges),
        "duplicate_unique_edge_count": len(dup),
        "edge_topology": dict(edge_topo),
        "tier_edge_counts": dict(tiers_edges),
        "n_src_flows": len(out_deg), "n_dst_flows": len(in_deg),
        "n_source_level_fanout_units": n_fan,
        "n_merge_units": len(merge),
        "n_connected_components": len(conn),
        "conn_counts_by_class": dict(Counter(c[2] for c in conn)),
        "fanout_degree_distribution": {str(k): v for k, v in sorted(deg_dist.items())},
        "degree_le_5": sum(1 for d in fan.values() if d <= 5),
        "degree_gt_5": sum(1 for d in fan.values() if d > 5),
        "cluster_count_G": G,
        "cluster_size_vector_sorted": sorted(addr_counts.values(), reverse=True),
        "max_cluster_share": round(max_share, 6) if max_share is not None else None,
        "temporal_spread_days": round(spread / 86400, 2),
        "gates": {"G_ge_8": G >= 8,
                  "max_share_lt_0.5": (max_share is not None and max_share < 0.5),
                  "n_fanout_ge_30": n_fan >= 30,
                  "spread_ge_2_months": spread >= 2 * 2592000},
        "DATA_ADEQUACY": "PASS" if (G >= 8 and max_share is not None
                                    and max_share < 0.5 and n_fan >= 30
                                    and spread >= 2 * 2592000) else "FAIL",
        "statistical_branch": branch,
        "PRIMARY_V4_SOURCE_UNIT_SHA256": pl_sha,
    }
    (ROOT / "v4_final_report.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    (ROOT / "primary_source_unit_list_v4.json").write_text(
        json.dumps(primary_rows, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
