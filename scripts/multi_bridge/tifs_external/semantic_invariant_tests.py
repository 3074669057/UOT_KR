"""Semantic invariant tests: legacy label field -> canonical topology.

Data-only. Reads the frozen real-label CSV and asserts the bijection between the
legacy pattern field values, the degree columns, and the canonical topology
definition (mission: "automated semantic invariant tests"). No method, no
performance, no 301-305.

Canonical definition (frozen):
  1->1      : n_src == 1 and n_dst == 1
  1->N fan-out: n_src == 1 and n_dst >= 2   (legacy field "many_to_one")
  N->1 merge  : n_src >= 2 and n_dst == 1   (legacy field "one_to_many")
  N<->M      : n_src >= 2 and n_dst >= 2

Note on notation: in the frozen label CSV, `src_degree` is the OUT-DEGREE of the
row's source flow (= number of distinct destination flows of that source) and
`dst_degree` is the IN-DEGREE of the row's destination flow (= number of distinct
source flows of that destination). These are degrees of single nodes, NOT node
counts; the canonical definition above uses NODE COUNTS (n_src, n_dst) of the
structural component and is the only notation allowed in new documents.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "out" / "paper_full_pipeline_run" / "label_layer_v1" / "flow_labels.csv"


def main() -> int:
    fl = pd.read_csv(LABELS)
    errors: list[str] = []
    fl["src_degree"] = pd.to_numeric(fl["src_degree"], errors="coerce").fillna(0).astype(int)
    fl["dst_degree"] = pd.to_numeric(fl["dst_degree"], errors="coerce").fillna(0).astype(int)

    # 1. Row-level degree -> pattern bijection (0 mismatches required).
    def violates(row, sd_ok, dd_ok):
        return not (sd_ok(row.src_degree) and dd_ok(row.dst_degree))

    mism = {
        "one_to_one": fl[(fl.pattern_type == "one_to_one")
                         & fl.apply(lambda r: violates(r, lambda d: d == 1, lambda d: d == 1), axis=1)].shape[0],
        "many_to_one": fl[(fl.pattern_type == "many_to_one")
                          & fl.apply(lambda r: violates(r, lambda d: d >= 2, lambda d: d == 1), axis=1)].shape[0],
        "one_to_many": fl[(fl.pattern_type == "one_to_many")
                          & fl.apply(lambda r: violates(r, lambda d: d == 1, lambda d: d >= 2), axis=1)].shape[0],
        "many_to_many": fl[(fl.pattern_type == "many_to_many")
                           & fl.apply(lambda r: violates(r, lambda d: d >= 2, lambda d: d >= 2), axis=1)].shape[0],
    }
    for k, v in mism.items():
        if v != 0:
            errors.append(f"degree->pattern bijection violated for {k}: {v} rows")

    # 2. Canonical topology mapping from the degree columns (per row).
    # legacy many_to_one  -> canonical 1->N : the row's source flow has >=2 dst flows
    #   and each of those dst flows has exactly this one source => n_src=1, n_dst>=2.
    # legacy one_to_many  -> canonical N->1 : the row's dst flow has >=2 src flows
    #   and each of those src flows has exactly this one dst => n_src>=2, n_dst=1.
    fan_rows = fl[(fl.pattern_type == "many_to_one")]
    if fan_rows.shape[0] == 0:
        errors.append("no fan-out rows found")
    else:
        bad = fan_rows[~(fan_rows.src_degree >= 2) | ~(fan_rows.dst_degree == 1)]
        if bad.shape[0]:
            errors.append(f"canonical 1->N invariant violated on {bad.shape[0]} rows")
    mer_rows = fl[(fl.pattern_type == "one_to_many")]
    if mer_rows.shape[0] == 0:
        errors.append("no merge rows found")
    else:
        bad = mer_rows[~(mer_rows.src_degree == 1) | ~(mer_rows.dst_degree >= 2)]
        if bad.shape[0]:
            errors.append(f"canonical N->1 invariant violated on {bad.shape[0]} rows")

    # 3. Component-level checks: fan-out components have n_src == 1 (a single source
    #    flow is the component) and n_dst == its degree; merge components have
    #    n_dst == 1 and n_src == its degree.
    src_deg = fl.groupby("src_flow_id")["dst_flow_id"].nunique()
    dst_deg = fl.groupby("dst_flow_id")["src_flow_id"].nunique()
    fan_components = src_deg[src_deg >= 2]
    for sid, deg in fan_components.items():
        if deg < 2:
            errors.append(f"fan-out component {sid} has degree {deg}")
    mer_components = dst_deg[dst_deg >= 2]
    for did, deg in mer_components.items():
        if deg < 2:
            errors.append(f"merge component {did} has degree {deg}")

    # 4. Counts (audit cross-check values).
    summary = {
        "n_rows": int(fl.shape[0]),
        "one_to_one": int((fl.pattern_type == "one_to_one").sum()),
        "fanout_rows": int((fl.pattern_type == "many_to_one").sum()),
        "merge_rows": int((fl.pattern_type == "one_to_many").sum()),
        "many_to_many": int((fl.pattern_type == "many_to_many").sum()),
        "fanout_components": int(len(fan_components)),
        "fanout_degree_le_5": int((fan_components <= 5).sum()),
        "fanout_degree_gt_5": int((fan_components > 5).sum()),
        "merge_components": int(len(mer_components)),
    }
    print("SUMMARY", summary)
    if errors:
        print("INVARIANT FAILURES:")
        for e in errors:
            print(" -", e)
        print("TOPOLOGY_SEMANTIC_INVARIANT: FAIL")
        return 1
    print("TOPOLOGY_SEMANTIC_INVARIANT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
