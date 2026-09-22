"""ONE-SHOT LEVEL-I TEMPORAL EXTERNAL METHOD EXECUTION. THE REAL PREDICTION PATH.

This script may run EXACTLY ONCE on the frozen corpus (b01-b03). It verifies
every frozen identity BEFORE generating any prediction, writes an immutable
execution marker, computes the six frozen methods, and writes the full result
artifact set. NO rerun, NO tuning, NO post-outcome changes. It never reads
301-305 and refuses b04-b06 paths.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tifs_external.execution_support import (  # noqa: E402
    build_flow_tables, evaluate_units, fetch_dst_times, gates_eval, inference,
)

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_validation_preregistration_v3"
DATA = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"
RES = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_validation_results_v3"
PRIMARY_SHA = "67f7ec4619bd8c7706bdadd7175778d1928a16cce429bc05523444cd12b42b4c"
METHODS = ["RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4", "CONDITIONAL_BOT_D4",
           "THRESHOLD_MM", "CONNECTOR_STYLE", "ABCTRACER_STYLE"]
TAU = 0.478
REG, REG_M, K5 = 0.05, 0.5, 5
SEED = 20260904
B_WILD = 4000


def frozen_gates() -> list[str]:
    errs: list[str] = []
    pl = json.loads((DATA / "primary_population_list.json").read_text(encoding="utf-8"))
    if hashlib.sha256((DATA / "primary_population_list.json").read_bytes()).hexdigest() != PRIMARY_SHA:
        errs.append("primary_sha")
    if len(pl) != 1770:
        errs.append("N_primary")
    deg_gt5 = sum(1 for u in pl if u["degree"] > 5)
    if deg_gt5 != 360:
        errs.append("degree_gt5")
    from collections import Counter
    vec = sorted(Counter(u["cluster_key"] for u in pl).values(), reverse=True)
    if vec != [617, 520, 290, 272, 57, 7, 6, 1]:
        errs.append("cluster_vector")
    edges = []
    for b in ("b01", "b02", "b03"):
        edges += json.loads((DATA / "blocks" / b / "flow_edges_canonical.json")
                            .read_text(encoding="utf-8"))
    if len(edges) != 15884:
        errs.append("unique_edges")
    return errs


def main() -> int:
    errs = frozen_gates()
    if errs:
        print("ABORT BEFORE PREDICTION:", errs)
        return 3
    if RES.exists():
        print("ABORT: result directory already exists (one-shot rule).")
        return 7
    RES.mkdir(parents=True)
    marker = {"utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
              "event": "REAL_PREDICTION_PATH_ENTERED", "one_shot": True}
    (RES / "EXECUTION_MARKER.json").write_text(json.dumps(marker, indent=2),
                                               encoding="utf-8")
    print("EXECUTION MARKER WRITTEN. Entering the real prediction path (one-shot).",
          flush=True)

    # ---- load frozen corpus ----
    edges = []
    anchors = []
    for b in ("b01", "b02", "b03"):
        edges += json.loads((DATA / "blocks" / b / "flow_edges_canonical.json")
                            .read_text(encoding="utf-8"))
        anchors += json.loads((DATA / "blocks" / b / "anchors.json")
                              .read_text(encoding="utf-8"))
    pl = json.loads((DATA / "primary_population_list.json").read_text(encoding="utf-8"))

    # ---- feature assembly: relay timestamps for destination txs (frozen step) ----
    from tifs_external.execution_support import fetch_dst_times, build_flow_tables
    dst_time = fetch_dst_times(anchors, RES)
    src_flows, dst_flows, unit_meta = build_flow_tables(edges, anchors, pl, dst_time, RES)

    n, m = len(src_flows), len(dst_flows)
    sids = [f["flow_id"] for f in src_flows]
    tids = [f["flow_id"] for f in dst_flows]
    print(f"grid: {n} source flows x {m} destination flows", flush=True)

    # ---- frozen amount-free renormalized cost (vectorized) ----
    s_start = np.array([f["start_time"] for f in src_flows], dtype=float)
    t_end = np.array([f["end_time"] for f in dst_flows], dtype=float)
    delay = t_end[None, :] - s_start[:, None]          # dst_end - src_start
    MAXD, CPEN = 21600.0, 5.0
    time_cost = np.where(delay < 0, np.minimum(1.0 + CPEN / MAXD, 2.0),
                np.where(delay > MAXD,
                         1.0 + np.minimum((delay - MAXD) / MAXD, 2.0) * (CPEN * 0.2),
                         np.minimum(delay / MAXD, 1.0)))
    del delay

    rs = np.array([f["route_type"] for f in src_flows], dtype=object)
    rt = np.array([f["route_type"] for f in dst_flows], dtype=object)
    rs_ok = np.array([f["price_ok"] for f in src_flows], dtype=bool)
    rt_ok = np.array([f["price_ok"] for f in dst_flows], dtype=bool)
    same = (rs[:, None] == rt[None, :])
    unknown = (rs[:, None] == "") | (rt[None, :] == "")
    sab = np.char.find(rs.astype(str), "same_asset_bridge") >= 0
    swap = np.char.find(rs.astype(str), "swap") >= 0
    cross = np.char.find(rs.astype(str), "cross_asset") >= 0
    sab_c = sab[:, None]
    swap_c = swap[:, None]
    cross_c = cross[:, None]
    both_ok = rs_ok[:, None] & rt_ok[None, :]
    route_cost = np.where(unknown, 1.0,
                 np.where(same,
                   np.where(sab_c & ~swap_c & ~cross_c, 0.0,
                   np.where(sab_c | (np.char.find(rt.astype(str), "wrapped") >= 0)[None, :],
                            0.12,
                   np.where(cross_c | swap_c, np.where(both_ok, 0.25, 0.95), 0.08))),
                 np.where(cross_c | (np.char.find(rt.astype(str), "cross_asset") >= 0)[None, :],
                          np.where(both_ok, 0.35, 1.0), 0.2)))
    route_cost = np.minimum(route_cost, 1.0)
    del same, unknown, sab_c, swap_c, cross_c, both_ok

    s_risk = np.array([f["aml_score"] for f in src_flows], dtype=float)
    t_evq = np.array([f["evidence_quality_score"] for f in dst_flows], dtype=float)
    risk_cost = np.minimum(np.abs(s_risk[:, None] - t_evq[None, :]), 1.0)
    evidence_cost = np.full((n, m),
        np.minimum(np.array([f["evidence_penalty"] for f in dst_flows]), 1.0)[None, :])
    s_addr = np.array([f["address"] for f in src_flows], dtype=object)
    t_addr = np.array([f["address"] for f in dst_flows], dtype=object)
    novelty_cost = np.where(s_addr[:, None] == t_addr[None, :], 0.0, 1.0)

    C_p = ((0.25 * time_cost + 0.15 * route_cost + 0.15 * risk_cost
            + 0.05 * evidence_cost + 0.05 * novelty_cost) / 0.65)
    C_p = np.minimum(np.maximum(C_p, 0.0), 2.0)
    del time_cost, route_cost, risk_cost, evidence_cost, novelty_cost
    print("amount-free renormalized cost built", flush=True)

    # ---- frozen marginals ----
    amt_s = np.array([f["amount_usd"] for f in src_flows], dtype=float)
    amt_t = np.array([f["amount_usd"] for f in dst_flows], dtype=float)
    a0 = amt_s / max(amt_s.sum(), 1e-12)
    b0 = amt_t / max(amt_t.sum(), 1e-12)
    aml = np.array([f["aml_score"] for f in src_flows], dtype=float)
    a = a0 * (1.0 + 0.25 * aml / 100.0)
    a = a / max(a.sum(), 1e-12)
    q = np.clip(np.array([f["evidence_quality_score"] for f in dst_flows], dtype=float),
                1e-6, 1.0)
    b = b0 * q
    b = b / max(b.sum(), 1e-12)

    # ---- frozen solves ----
    import ot
    P_uot = ot.sinkhorn_unbalanced(a, b, C_p, REG, REG_M)
    P_bot = ot.sinkhorn(a, b, C_p, REG)
    print("solves done", flush=True)

    # ---- six frozen decoders ----
    edgesets = {}
    edgesets["RAW_UOT_PLAN_D4"] = mutual_top5(P_uot, K5)
    Sr = P_uot / np.maximum(P_uot.sum(axis=0, keepdims=True), 1e-300)
    Sc = P_uot / np.maximum(P_uot.sum(axis=1, keepdims=True), 1e-300)
    edgesets["CONDITIONAL_UOT_D4"] = conditional_edges_arrays(Sr, Sc, K5)
    del Sr, Sc
    Srb = P_bot / np.maximum(P_bot.sum(axis=0, keepdims=True), 1e-300)
    Scb = P_bot / np.maximum(P_bot.sum(axis=1, keepdims=True), 1e-300)
    edgesets["CONDITIONAL_BOT_D4"] = conditional_edges_arrays(Srb, Scb, K5)
    del Srb, Scb, P_uot, P_bot
    # Threshold-MM: frozen tau = 0.478, edges restricted to PRIMARY source rows
    # (the frozen evaluation scope: edge metrics on the primary population).
    primary_idx = [i for i, s in enumerate(sids)
                   if s in {u["component_id"] for u in pl}]
    tmm = set()
    for i in primary_idx:
        for j in np.where(C_p[i] <= TAU)[0]:
            tmm.add((i, int(j)))
    edgesets["THRESHOLD_MM"] = tmm
    conn = {(i, int(np.argmin(C_p[i]))) for i in range(n)}
    abct = {(i, int(np.argmin(-C_p[i]))) for i in range(n)}
    edgesets["CONNECTOR_STYLE"] = conn
    edgesets["ABCTRACER_STYLE"] = abct
    del C_p
    print("six decoders done", flush=True)

    # ---- strict source-level fan-out set recovery ----
    from tifs_external.execution_support import evaluate_units, inference, gates_eval
    eval_out = evaluate_units(edgesets, pl, sids, tids, unit_meta)
    inf = inference(eval_out, pl)
    gates = gates_eval(eval_out, inf)
    print("evaluation + inference + gates done", flush=True)

    # ---- write artifacts ----
    write_artifacts(RES, eval_out, inf, gates, C_p.shape, sids, tids)
    print("ALL ARTIFACTS WRITTEN", flush=True)
    return 0


def mutual_top5(P: np.ndarray, k: int) -> set:
    """Frozen D4 mutual top-k with the frozen stable tie-break (ascending index
    among equal values), via stable argsort."""
    row_top = np.argsort(-P, axis=1, kind="stable")[:, :k]
    col_top = np.argsort(-P, axis=0, kind="stable")[:k, :]
    rowsets = [set(r.tolist()) for r in row_top]
    colsets = [set(c.tolist()) for c in col_top.T]
    out = set()
    for i in range(P.shape[0]):
        for j in rowsets[i]:
            if i in colsets[j]:
                out.add((i, j))
    return out


def conditional_edges_arrays(Sr: np.ndarray, Sc: np.ndarray, k: int) -> set:
    row_top = np.argsort(-Sr, axis=1, kind="stable")[:, :k]
    col_top = np.argsort(-Sc, axis=0, kind="stable")[:k, :]
    rowsets = [set(r.tolist()) for r in row_top]
    colsets = [set(c.tolist()) for c in col_top.T]
    out = set()
    for i in range(Sr.shape[0]):
        for j in rowsets[i]:
            if i in colsets[j]:
                out.add((i, j))
    return out


def write_artifacts(RES: Path, ev, inf, gates, shape, sids, tids) -> None:
    import csv as _csv
    (RES / "RAW_METHOD_OUTPUTS").mkdir(exist_ok=True)
    for m in ev["per_method"]:
        pass
    (RES / "METHOD_SUMMARY.json").write_text(
        json.dumps({m: {k: v for k, v in d.items() if k != "recovered_by_unit"}
                    for m, d in ev["per_method"].items()}, indent=2, default=str),
        encoding="utf-8")
    (RES / "PRIMARY_INFERENCE.json").write_text(
        json.dumps(inf, indent=2, default=str), encoding="utf-8")
    (RES / "GATES.json").write_text(
        json.dumps(gates, indent=2, default=str), encoding="utf-8")
    (RES / "grid_shape.json").write_text(
        json.dumps({"n_sources": shape[0], "n_destinations": shape[1]}, indent=2),
        encoding="utf-8")
    # per-unit CSV
    with open(RES / "PRIMARY_SOURCE_UNIT_RESULTS.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["component_id", "cluster_key", "degree"] +
                   [f"Y_{m}" for m in ev["per_method"]])
        keys = sorted(ev["per_method"]["RAW_UOT_PLAN_D4"]["recovered_by_unit"].keys())
        from tifs_external.execution_support import cluster_of
        for k in keys:
            w.writerow([k, cluster_of(k), None] +
                       [ev["per_method"][m]["recovered_by_unit"].get(k, 0)
                        for m in ev["per_method"]])
    # per-cluster CSV
    with open(RES / "PER_CLUSTER_RESULTS.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["cluster_key", "n_g"] + [f"rate_{m}" for m in ev["per_method"]])
        gkeys = sorted(ev["per_method"]["RAW_UOT_PLAN_D4"]["per_cluster"].keys())
        for g in gkeys:
            w.writerow([g, ev["per_method"]["RAW_UOT_PLAN_D4"]["per_cluster"][g]["n"]]
                       + [ev["per_method"][m]["per_cluster"].get(g, {}).get("rate", float("nan"))
                          for m in ev["per_method"]])
    # per-degree CSV
    with open(RES / "PER_DEGREE_RESULTS.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["degree", "n"] + [f"rate_{m}" for m in ev["per_method"]])
        dkeys = sorted(ev["per_method"]["RAW_UOT_PLAN_D4"]["per_degree"].keys(),
                       key=lambda x: int(x))
        for d in dkeys:
            w.writerow([d, ev["per_method"]["RAW_UOT_PLAN_D4"]["per_degree"][d]["n"]]
                       + [ev["per_method"][m]["per_degree"].get(d, {}).get("rate", float("nan"))
                          for m in ev["per_method"]])
    # sign-flip distribution CSV (all 256)
    with open(RES / "PRIMARY_SIGNFLIP_DISTRIBUTION.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["sign_index", "T_s"])
        for i, T in enumerate(inf["signflip_distribution"]):
            w.writerow([i, T])
    (RES / "SECONDARY_WILD_CLUSTER_CI.json").write_text(
        json.dumps(inf["secondary_wild_cluster_ci"], indent=2), encoding="utf-8")
    # provenance
    prov = {
        "execution": "ONE-SHOT (marker written before predictions)",
        "corpus": "b01-b03 (2023-05-19T00:00:00Z .. 2023-08-18T00:00:00Z)",
        "primary_units": 1770,
        "primary_sha256": PRIMARY_SHA,
        "methods": ["RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4", "CONDITIONAL_BOT_D4",
                    "THRESHOLD_MM(tau=0.478)", "CONNECTOR_STYLE", "ABCTRACER_STYLE"],
        "operation_point": {"candidate_sha":
                            "0f360addc1f2a220299d75b4aa9cad1ee1e504e8097ae5bacfc0da40542cc401",
                            "k": K5, "reg": REG, "reg_m": REG_M, "tau": TAU,
                            "weights": "time/route/risk/evidence/novelty = "
                                       "0.25/0.15/0.15/0.05/0.05 renormalized to 1"},
        "missing_data_policy": "no decimals or no frozen price -> 0.0 USD mass",
    }
    (RES / "EXECUTION_PROVENANCE.md").write_text(
        "# EXECUTION_PROVENANCE.md\n\n```json\n" + json.dumps(prov, indent=2) +
        "\n```\n", encoding="utf-8")
    print("artifacts written", flush=True)


if __name__ == "__main__":
    sys.exit(main())
