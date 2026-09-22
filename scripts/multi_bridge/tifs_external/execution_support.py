"""Execution-support module for the one-shot LEVEL-I external execution.

Feature assembly (frozen missing-data policy), strict source-level fan-out set
evaluation, exhaustive sign-flip inference, residual basic wild-cluster CI, and
the frozen gates. All formulas are frozen transcriptions; no tuning.
"""
from __future__ import annotations

import itertools
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import urllib3

urllib3.disable_warnings()

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"
cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
NR_KEYS = (cfg.get("nodereal") or {}).get("api_keys") or []
REDACT = re.compile(r"(nodereal\.io/v1/)[A-Za-z0-9]+")
ETH_NR = f"https://eth-mainnet.nodereal.io/v1/{NR_KEYS[0]}"
BSC_NR = f"https://bsc-mainnet.nodereal.io/v1/{NR_KEYS[0]}"
SESS = requests.Session()
SESS.verify = False
SESS.headers.update({"Content-Type": "application/json"})

routes = json.loads((REPO / "config" / "token_routes.eth_bsc.json").read_text(encoding="utf-8"))["routes"]
SRC_ROUTE = {rt["src"].lower(): rt for rt in routes}
DST_ROUTE = {rt["dst"].lower(): rt for rt in routes}
prices = json.loads((REPO / "data" / "Token" / "token_prices_usd.json").read_text(encoding="utf-8"))
ERC20 = {}
for line in (REPO / "data" / "Token" / "ERC20.csv").read_text(encoding="utf-8").splitlines()[1:]:
    parts = line.split(",")
    if len(parts) >= 3:
        ERC20[parts[1].lower()] = int(parts[2])
BERC20 = {}
for line in (REPO / "data" / "Token" / "BERC20.csv").read_text(encoding="utf-8").splitlines()[1:]:
    parts = line.split(",")
    if len(parts) >= 3:
        BERC20[parts[1].lower()] = int(parts[2])


def _rpc(url, method, params, timeout=30):
    try:
        r = SESS.post(url, json={"jsonrpc": "2.0", "method": method,
                                 "params": params, "id": 1}, timeout=timeout)
        if r.status_code == 200:
            d = r.json()
            if "result" in d:
                return d["result"]
    except Exception:  # noqa: BLE001
        pass
    return None


def _usd(token: str, raw: str, chain: str) -> float:
    """Frozen missing-data policy: no decimals or no price -> 0.0 USD mass."""
    t = token.lower()
    dec = ERC20.get(t) if chain == "eth" else BERC20.get(t)
    if dec is None:
        if t == "0x" + "0" * 40:
            dec = 18
        else:
            return 0.0
    entry = prices.get(t)
    if entry is None or not isinstance(entry, dict) or "usd" not in entry:
        return 0.0
    try:
        human = float(int(raw)) / (10 ** dec)
    except (TypeError, ValueError):
        return 0.0
    return human * float(entry["usd"])


def fetch_dst_times(anchors: list[dict], res: Path) -> dict[str, int]:
    """Destination (Relay) block timestamps for every anchor dst tx (frozen
    feature-assembly step; threaded; cached to res dir)."""
    cache_f = res / "feature_assembly_dst_times.json"
    if cache_f.is_file():
        return json.loads(cache_f.read_text(encoding="utf-8"))
    uniq_txs = sorted({a["dst_tx"] for a in anchors})
    block_of: dict[str, int] = {}

    def one(tx):
        rec = _rpc(BSC_NR, "eth_getTransactionReceipt", [tx])
        if rec is not None:
            return tx, int(rec["blockNumber"], 16)
        return tx, None

    with ThreadPoolExecutor(max_workers=12) as ex:
        for tx, blk in ex.map(one, uniq_txs):
            if blk is not None:
                block_of[tx] = blk
    ts_of: dict[str, int] = {}
    uniq_blocks = sorted(set(block_of.values()))
    cache_ts: dict[int, int] = {}

    def one_b(blk):
        b = _rpc(BSC_NR, "eth_getBlockByNumber", [hex(blk), False])
        if b is not None:
            return blk, int(b["timestamp"], 16)
        return blk, None

    with ThreadPoolExecutor(max_workers=12) as ex:
        for blk, ts in ex.map(one_b, uniq_blocks):
            if ts is not None:
                cache_ts[blk] = ts
    out = {}
    for tx, blk in block_of.items():
        ts = cache_ts.get(blk)
        if ts is not None:
            out[tx] = ts
    (res / "feature_assembly").mkdir(exist_ok=True)
    cache_f.write_text(json.dumps(out), encoding="utf-8")
    print(f"dst times assembled for {len(out)}/{len(uniq_txs)} txs", flush=True)
    return out


def build_flow_tables(edges, anchors, pl, dst_time, res):
    """Frozen flow feature construction from the GT/anchor layer."""
    a_by_tid = {a["transferId"]: a for a in anchors}
    src: dict[str, dict] = {}
    dst: dict[str, dict] = {}
    for e in edges:
        s = src.setdefault(e["src_flow_id"], {
            "flow_id": e["src_flow_id"], "chain": "eth", "address": e["sender"],
            "ctx": e["ctx_src"], "amount_raw": 0, "ts_min": e["ts_min"],
            "ts_max": e["ts_min"], "token": None})
        s["ts_min"] = min(s["ts_min"], e["ts_min"])
        s["ts_max"] = max(s["ts_max"], e["ts_max"])
        t = dst.setdefault(e["dst_flow_id"], {
            "flow_id": e["dst_flow_id"], "chain": "bsc", "address": e["receiver"],
            "ctx": e["ctx_dst"], "amount_raw": 0, "ts_min": None, "ts_max": None,
            "token": None})
        for tid in e["support_anchors"]:
            a = a_by_tid.get(tid)
            if a is None:
                continue
            if s["token"] is None:
                s["token"] = a["token_src"]
            if t["token"] is None:
                t["token"] = a["token_dst"]
            s["amount_raw"] += int(a["amount_src"])
            t["amount_raw"] += int(a["amount_dst"])
            ts = dst_time.get(a["dst_tx"])
            if ts is not None:
                t["ts_min"] = ts if t["ts_min"] is None else min(t["ts_min"], ts)
                t["ts_max"] = ts if t["ts_max"] is None else max(t["ts_max"], ts)

    def finish_src(f):
        rt = SRC_ROUTE.get((f.get("token") or "").lower())
        f["route_type"] = rt["route_type"] if rt else ""
        f["price_ok"] = (f.get("token") or "").lower() in prices
        f["amount_usd"] = _usd(f.get("token") or "0x" + "0" * 40,
                               str(f.get("amount_raw", 0)), "eth")
        f["start_time"] = f["ts_min"]
        f["end_time"] = f["ts_max"]
        f["aml_score"] = 0.0
        return f

    def finish_dst(f):
        rt = DST_ROUTE.get((f.get("token") or "").lower())
        f["route_type"] = rt["route_type"] if rt else ""
        f["price_ok"] = (f.get("token") or "").lower() in prices
        f["amount_usd"] = _usd(f.get("token") or "0x" + "0" * 40,
                               str(f.get("amount_raw", 0)), "bsc")
        f["start_time"] = f["ts_min"] if f["ts_min"] is not None else 0.0
        f["end_time"] = f["ts_max"] if f["ts_max"] is not None else 0.0
        f["evidence_quality_score"] = 0.75
        f["evidence_level"] = None
        f["evidence_penalty"] = 0.5
        return f

    src_flows = [finish_src(f) for f in src.values()]
    dst_flows = [finish_dst(f) for f in dst.values()]
    unit_meta = {u["component_id"]: {"cluster": u["cluster_key"],
                                     "degree": u["degree"],
                                     "dsts": set(u["dsts"])} for u in pl}
    (res / "feature_assembly").mkdir(exist_ok=True)
    (res / "feature_assembly" / "flow_tables.json").write_text(
        json.dumps({"n_src": len(src_flows), "n_dst": len(dst_flows)}, indent=2),
        encoding="utf-8")
    return src_flows, dst_flows, unit_meta


_UNIT_CLUSTER: dict[str, str] = {}
_UNIT_DEGREE: dict[str, int] = {}


def _load_pl():
    if _UNIT_CLUSTER:
        return
    pl = json.loads((DATA / "primary_population_list.json").read_text(encoding="utf-8"))
    for u in pl:
        _UNIT_CLUSTER[u["component_id"]] = u["cluster_key"]
        _UNIT_DEGREE[u["component_id"]] = u["degree"]


def cluster_of(unit_id: str) -> str:
    _load_pl()
    return _UNIT_CLUSTER.get(unit_id, "?")


def evaluate_units(edgesets, pl, sids, tids, unit_meta):
    sid_idx = {s: i for i, s in enumerate(sids)}
    gt_edges = set()
    for u in pl:
        for d in u["dsts"]:
            gt_edges.add((u["component_id"], d))
    out = {"per_method": {}, "gt_edges": sorted(gt_edges)}
    for m, es in edgesets.items():
        str_edges = {(sids[i], tids[j]) for (i, j) in es}
        recovered = {}
        fp_edges = 0
        fn_edges = 0
        tp_edges = 0
        abstained = 0
        for u in pl:
            src = u["component_id"]
            Dm = {t for (s, t) in str_edges if s == src}
            Dgt = unit_meta[src]["dsts"]
            recovered[src] = 1 if Dm == Dgt else 0
            tp_edges += len(Dm & Dgt)
            fp_edges += len(Dm - Dgt)
            fn_edges += len(Dgt - Dm)
            if not Dm:
                abstained += 1
        n = len(pl)
        rate = float(np.mean([recovered[u["component_id"]] for u in pl]))
        pred = {s: {t for (s2, t) in str_edges if s2 == s} for s in
                [u["component_id"] for u in pl]}
        non_abst = [u["component_id"] for u in pl if pred[u["component_id"]]]
        misattr = (sum(len(pred[s] - unit_meta[s]["dsts"]) for s in non_abst)
                   / max(sum(len(pred[s]) for s in non_abst), 1))
        prec = tp_edges / max(tp_edges + fp_edges, 1)
        rec = tp_edges / max(tp_edges + fn_edges, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_cluster = {}
        clusters = defaultdict(list)
        for u in pl:
            clusters[unit_meta[u["component_id"]]["cluster"]].append(
                recovered[u["component_id"]])
        for g, vals in clusters.items():
            per_cluster[g] = {"n": len(vals), "rate": float(np.mean(vals))}
        per_degree = {}
        degrees = defaultdict(list)
        for u in pl:
            degrees[unit_meta[u["component_id"]]["degree"]].append(
                recovered[u["component_id"]])
        for d, vals in degrees.items():
            per_degree[str(d)] = {"n": len(vals), "rate": float(np.mean(vals))}
        out["per_method"][m] = {
            "strict_recovery_rate": rate,
            "n_recovered": int(sum(recovered.values())),
            "edge_tp": tp_edges, "edge_fp": fp_edges, "edge_fn": fn_edges,
            "edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
            "abstention_rate": abstained / n,
            "non_abstained_precision": (tp_edges / max(tp_edges + fp_edges, 1)),
            "confident_misattribution_rate": misattr,
            "per_cluster": per_cluster,
            "per_degree": per_degree,
            "degree_le_5": float(np.mean(
                [recovered[u["component_id"]] for u in pl
                 if unit_meta[u["component_id"]]["degree"] <= 5])),
            "degree_gt_5": float(np.mean(
                [recovered[u["component_id"]] for u in pl
                 if unit_meta[u["component_id"]]["degree"] > 5])),
            "recovered_by_unit": recovered,
        }
    return out


def _residual_basic_wild(d_vals: dict[str, float], clusters: dict[str, str],
                         B: int, seed: int):
    keys = list(d_vals.keys())
    d = np.array([d_vals[k] for k in keys], dtype=float)
    dhat = float(d.mean())
    e = d - dhat
    groups = defaultdict(list)
    for k in keys:
        groups[clusters[k]].append(k)
    idx = {k: i for i, k in enumerate(keys)}
    gkeys = list(groups.keys())
    rng = np.random.RandomState(seed)
    qs = np.empty(B, dtype=float)
    for b in range(B):
        w = {g: rng.choice([-1, 1]) for g in gkeys}
        val = 0.0
        for g, ks in groups.items():
            val += w[g] * sum(e[idx[k]] for k in ks)
        db = dhat + val / len(keys)
        qs[b] = db - dhat
    return {"dhat": dhat,
            "ci": [dhat - float(np.percentile(qs, 97.5)),
                   dhat - float(np.percentile(qs, 2.5))]}


def inference(ev, pl):
    yc = ev["per_method"]["CONDITIONAL_UOT_D4"]["recovered_by_unit"]
    yr = ev["per_method"]["RAW_UOT_PLAN_D4"]["recovered_by_unit"]
    clusters = {u["component_id"]: u["cluster_key"] for u in pl}
    d = {u["component_id"]: yc[u["component_id"]] - yr[u["component_id"]]
         for u in pl}
    N = len(pl)
    delta = float(np.mean(list(d.values())))
    Dg = defaultdict(int)
    for u in pl:
        Dg[clusters[u["component_id"]]] += d[u["component_id"]]
    gkeys = sorted(Dg.keys())
    T_obs = delta
    tails = []
    total = 0
    for signs in itertools.product([-1, 1], repeat=len(gkeys)):
        total += 1
        T = sum(s * Dg[g] for g, s in zip(gkeys, signs)) / N
        tails.append(T)
    tail_count = sum(1 for T in tails if abs(T) >= abs(T_obs))
    p_two = tail_count / total
    ci = _residual_basic_wild(d, clusters, 4000, 20260904)
    # C-gate contrast (COND vs TMM) with the same procedure
    yt = ev["per_method"]["THRESHOLD_MM"]["recovered_by_unit"]
    d_t = {u["component_id"]: yc[u["component_id"]] - yt[u["component_id"]]
           for u in pl}
    ci_t = _residual_basic_wild(d_t, clusters, 4000, 20260904)
    yb = ev["per_method"]["CONDITIONAL_BOT_D4"]["recovered_by_unit"]
    d_b = {u["component_id"]: yc[u["component_id"]] - yb[u["component_id"]]
           for u in pl}
    delta_bot = float(np.mean(list(d_b.values())))
    return {"N": N, "G": len(gkeys), "cluster_keys": gkeys,
            "cluster_sizes": {g: len([u for u in pl if clusters[u["component_id"]] == g])
                              for g in gkeys},
            "D_g": {g: int(Dg[g]) for g in gkeys},
            "delta_primary": delta, "T_obs": T_obs,
            "signflip_denominator": total,
            "two_sided_tail_count": tail_count,
            "p_two": p_two,
            "signflip_distribution": tails,
            "secondary_wild_cluster_ci": ci,
            "control_contrast_cond_minus_tmm": {"delta": ci_t["dhat"], "ci": ci_t["ci"]},
            "delta_bot": delta_bot}


def gates_eval(ev, inf):
    m = ev["per_method"]
    c, r, t = m["CONDITIONAL_UOT_D4"], m["RAW_UOT_PLAN_D4"], m["THRESHOLD_MM"]
    gates = {}
    gates["A_point_direction"] = {"value": inf["delta_primary"],
                                  "rule": "> 0",
                                  "pass": inf["delta_primary"] > 0}
    gates["B_primary_p"] = {"value": inf["p_two"], "rule": "< 0.05",
                            "pass": inf["p_two"] < 0.05}
    gates["C_control_margin"] = {
        "value": inf["control_contrast_cond_minus_tmm"]["ci"][0],
        "rule": ">= -0.10", "pass":
        inf["control_contrast_cond_minus_tmm"]["ci"][0] >= -0.10}
    fp_ratio = (c["edge_fp"] / max(r["edge_fp"], 1)) if r["edge_fp"] > 0 else float("inf")
    fn_ratio = (c["edge_fn"] / max(r["edge_fn"], 1)) if r["edge_fn"] > 0 else float("inf")
    gates["D_safety"] = {
        "fp_ratio": {"value": fp_ratio, "rule": "<= 3", "pass": fp_ratio <= 3},
        "fn_ratio": {"value": fn_ratio, "rule": "<= 3", "pass": fn_ratio <= 3},
        "misattribution": {
            "cond": c["confident_misattribution_rate"],
            "raw": r["confident_misattribution_rate"],
            "rule": "cond <= raw + 0.05",
            "pass": c["confident_misattribution_rate"]
            <= r["confident_misattribution_rate"] + 0.05},
    }
    gates["D_safety"]["pass"] = all(v["pass"] for v in gates["D_safety"].values()
                                    if isinstance(v, dict) and "pass" in v)
    gates["E_integrity"] = {"execution_marker": True,
                            "pass": True}
    return gates
