"""Bipartite connected-component descriptive table. DATA-ONLY.

Constructs connected components of the frozen GT bipartite graph (source flow
nodes, destination flow nodes, unique GT edges) and reports counts by
(n_src, n_dst, topology class). Descriptive only; never enters primary
inference. No method, no prediction, no performance.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"


def main() -> int:
    edges = []
    for b in ("b01", "b02", "b03"):
        edges += json.loads((ROOT / "blocks" / b / "flow_edges_canonical.json")
                            .read_text(encoding="utf-8"))
    adj: dict[str, set[str]] = defaultdict(set)
    rev: dict[str, set[str]] = defaultdict(set)
    srcs: set[str] = set()
    dsts: set[str] = set()
    for e in edges:
        s, t = e["src_flow_id"], e["dst_flow_id"]
        adj[s].add(t)
        rev[t].add(s)
        srcs.add(s)
        dsts.add(t)
    visited: set[str] = set()
    conn: list[tuple[int, int, str]] = []
    for node in sorted(srcs | dsts):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        cs: set[str] = set()
        cd: set[str] = set()
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
        cls = ("1to1" if ns == 1 and nd == 1 else
               "1toN" if ns == 1 and nd >= 2 else
               "Nto1" if ns >= 2 and nd == 1 else "NtoM")
        conn.append((ns, nd, cls))
    counts = Counter(conn)
    table = {"n_connected_components": len(conn),
             "counts_by_class": {f"{k[0]}x{k[1]}_{k[2]}": v for k, v in counts.items()},
             "counts_by_nsrc_ndst": {f"{k[0]}x{k[1]}": v for k, v in
                                     sorted(Counter((ns, nd) for ns, nd, _ in conn).items())},
             "largest": sorted(conn, reverse=True)[:5],
             "note": ("descriptive topology view only; the PRIMARY evaluation "
                      "unit is the source-level fan-out unit (1,770), NOT the "
                      "connected component")}
    (ROOT / "connected_components_table.json").write_text(
        json.dumps(table, indent=2, default=str), encoding="utf-8")
    print(json.dumps(table, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
