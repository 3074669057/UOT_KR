"""PHASE 10: real non-one-to-one case validation from official Validation labels."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached, norm  # noqa: E402
from faithful_flow_features import _load_decimals, _load_price_resolver  # noqa: E402

AUDIT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges_audit"

eth_dec, bnb_dec = _load_decimals()
prices, prov = _load_price_resolver()

rows = []
for br in ("Celer", "Multi", "Poly"):
    lab = pd.read_csv(REPO / "data" / "Validation" / "ETH-BNB" / br / "label.csv", dtype=str, keep_default_na=False)
    lab["srcTxhash"] = lab["srcTxhash"].map(norm)
    lab["dstTxhash"] = lab["dstTxhash"].map(norm)
    n = len(lab)
    src_to_dsts: dict[str, set] = defaultdict(set)
    dst_to_srcs: dict[str, set] = defaultdict(set)
    for r in lab.itertuples():
        src_to_dsts[r.srcTxhash].add(r.dstTxhash)
        dst_to_srcs[r.dstTxhash].add(r.srcTxhash)
    n_src_multi = sum(1 for s, d in src_to_dsts.items() if len(d) > 1)
    n_dst_multi = sum(1 for d, s in dst_to_srcs.items() if len(s) > 1)
    n_pairs = n
    print(f"{br}: official label rows={n_pairs} unique_src={len(src_to_dsts)} unique_dst={len(dst_to_srcs)} "
          f"src_with_multi_dst={n_src_multi} dst_with_multi_src={n_dst_multi}")

    # amount checks for multi patterns (real merge: sum(src)==dst; real split: src==sum(dst))
    cached = load_bridge_cached(br)
    src = cached["source"]
    all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
    c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}
    s_lookup = {norm(r.source_tx_hash): r for r in src.itertuples()}

    def usd_src(h: str):
        r = s_lookup.get(h)
        if r is None:
            return None
        tok = norm(r.source_token_address)
        dec = eth_dec.get(tok)
        p = prices.get(tok)
        if dec is None or p is None:
            return None
        return float(r.source_amount_raw) / (10 ** int(dec)) * p

    def usd_dst(h: str):
        c = c_lookup.get(h)
        if c is None:
            return None
        tok = norm(c.token_address)
        dec = bnb_dec.get(tok)
        p = prices.get(tok)
        if dec is None or p is None:
            return None
        return float(c.amount_raw) / (10 ** int(dec)) * p

    cases = []
    for d, srcs in dst_to_srcs.items():
        if len(srcs) < 2:
            continue
        s_usd = [usd_src(s) for s in sorted(srcs)]
        d_usd = usd_dst(d)
        if any(x is None for x in s_usd) or d_usd is None:
            continue
        cases.append({"bridge": br, "pattern": "many_to_one", "dst": d, "n_src": len(srcs),
                      "sum_src_usd": sum(s_usd), "dst_usd": d_usd,
                      "ratio": sum(s_usd) / max(d_usd, 1e-12)})
    for s, dsts in src_to_dsts.items():
        if len(dsts) < 2:
            continue
        s_usd = usd_src(s)
        d_usd = [usd_dst(d) for d in sorted(dsts)]
        if s_usd is None or any(x is None for x in d_usd):
            continue
        cases.append({"bridge": br, "pattern": "one_to_many", "dst": s, "n_src": len(dsts),
                      "sum_src_usd": s_usd, "dst_usd": sum(d_usd),
                      "ratio": s_usd / max(sum(d_usd), 1e-12)})
    rows.extend(cases)

(AUDIT / "real_case").mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(AUDIT / "real_case" / "real_non_one_to_one_cases.csv", index=False)
print("\ncases with USD on both sides:")
print(df.to_string())
if not df.empty:
    g = df.groupby(["bridge", "pattern"])["ratio"].agg(["count", "median", "min", "max"])
    print("\nratio stats (sum(legs)/single-leg):")
    print(g.to_string())
