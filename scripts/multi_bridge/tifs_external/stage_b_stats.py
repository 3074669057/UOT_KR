"""Stage B/C cumulative statistics and adequacy evaluation. DATA-ONLY.

Recomputes canonical statistics from the collected block label JSONs using the
FROZEN component definition: a fan-out component is a source flow with >= 2 GT
destination flows (out-degree >= 2); merge = destination flow with >= 2 GT source
flows. Row-level pattern tags stay row-level descriptive. No method, no
prediction, no performance.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"
WINDOW_START = 1684454400

ADEQUACY = {"G_min": 8, "max_share_max": 0.5, "n_fanout_min": 30,
            "spread_sec_min": 2 * 2592000}


def main() -> int:
    blocks = sorted(p for p in (ROOT / "blocks").iterdir() if p.is_dir())
    all_labels: list[dict] = []
    for b in blocks:
        f = b / "flow_labels_canonical.json"
        if f.is_file():
            all_labels += json.loads(f.read_text(encoding="utf-8"))
    if not all_labels:
        print("no labels collected")
        return 1

    src_deg = Counter(lb["src_flow_id"] for lb in all_labels)
    dst_deg = Counter(lb["dst_flow_id"] for lb in all_labels)
    sender_of: dict[str, str] = {}
    for lb in all_labels:
        sender_of.setdefault(lb["src_flow_id"], lb["sender"])

    fan = {s: d for s, d in src_deg.items() if d >= 2}
    merge = {t: d for t, d in dst_deg.items() if d >= 2}
    deg_dist = Counter(fan.values())
    addr_counts = Counter(sender_of[s] for s in fan)
    ts_all = [lb["ts"] for lb in all_labels]
    spread = (max(ts_all) - min(ts_all)) if ts_all else 0
    n_fan = len(fan)
    max_share = (max(addr_counts.values()) / n_fan) if n_fan else None
    G = len(addr_counts)

    gates = {
        "G_ge_8": G >= ADEQUACY["G_min"],
        "max_share_lt_0.5": (max_share is not None and max_share < ADEQUACY["max_share_max"]),
        "n_fanout_ge_30": n_fan >= ADEQUACY["n_fanout_min"],
        "spread_ge_2_months": spread >= ADEQUACY["spread_sec_min"],
    }
    adequate = all(gates.values())

    stats = {
        "blocks_collected": [b.name for b in blocks],
        "n_label_rows": len(all_labels),
        "n_anchor_edges": len(all_labels),
        "topology_rows": dict(Counter(lb["canonical_pattern"] for lb in all_labels)),
        "tier_counts": dict(Counter(lb["tier"] for lb in all_labels)),
        "n_src_flows": len(src_deg), "n_dst_flows": len(dst_deg),
        "n_fanout_components": n_fan,
        "n_fanout_degree_le_5": sum(1 for d in fan.values() if d <= 5),
        "n_fanout_degree_gt_5": sum(1 for d in fan.values() if d > 5),
        "fanout_degree_distribution": {str(k): v for k, v in sorted(deg_dist.items())},
        "n_merge_components": len(merge),
        "cluster_addrs_G": G,
        "max_cluster_share": round(max_share, 4) if max_share is not None else None,
        "addr_counts": dict(addr_counts),
        "temporal_spread_sec": spread,
        "temporal_spread_days": round(spread / 86400, 1),
        "adequacy_gates": gates,
        "DATA_ADEQUACY": "PASS" if adequate else "FAIL",
    }
    (ROOT / "cumulative_adequacy.json").write_text(json.dumps(stats, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
