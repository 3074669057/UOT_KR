"""Verify Poly src==dst amount pattern: all dev pairs, raw fields."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached, norm  # noqa: E402

cached = load_bridge_cached("Poly")
split = cached["split"]
dev = split[split["split"] == "development"]
all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}

eth_dec = {norm(r.address): int(r.decimal) for r in pd.read_csv(REPO / "data" / "Token" / "ERC20.csv", dtype={"address": str}).itertuples(index=False)}
bnb_dec = {norm(r.address): int(r.decimal) for r in pd.read_csv(REPO / "data" / "Token" / "BERC20.csv", dtype={"address": str}).itertuples(index=False)}

sample = json.loads((REPO / "data" / "Validation" / "ETH-BNB" / "Poly" / "sample.json").read_text(encoding="utf-8"))
s_by_tx = {norm(it.get("txhash")): it for it in sample}

n = 0
eq_raw = 0
eq_human = 0
neq = 0
ratios: list[float] = []
for r in dev.itertuples():
    c = c_lookup.get(norm(r.dest_tx_hash))
    if c is None:
        continue
    n += 1
    sraw = float(r.source_amount_raw)
    draw = float(c.amount_raw)
    if abs(sraw - draw) < 1e-9:
        eq_raw += 1
    else:
        neq += 1
    sd = eth_dec.get(norm(r.source_token_address))
    dd = bnb_dec.get(norm(c.token_address))
    if sd is not None and dd is not None and int(sd) > 0 and int(dd) > 0:
        sh = sraw / (10 ** int(sd))
        dh = draw / (10 ** int(dd))
        ratios.append(dh / sh if sh else float("nan"))
        if abs(sh - dh) < 1e-9:
            eq_human += 1
print(f"Poly: n dev pairs with dst candidate={n}; src_raw==dst_raw exactly: {eq_raw}; neq: {neq}")
print(f"   human-equal (both decimals known): {eq_human}/{len(ratios)}")
rs = pd.Series([x for x in ratios if x == x])
print(f"   human ratio n={len(rs)} min={rs.min()} max={rs.max()} mean={rs.mean()}")

# show 3 example neq pairs raw
shown = 0
for r in dev.itertuples():
    c = c_lookup.get(norm(r.dest_tx_hash))
    if c is None:
        continue
    sraw = float(r.source_amount_raw)
    draw = float(c.amount_raw)
    if abs(sraw - draw) > 1e-9:
        it = s_by_tx.get(norm(r.source_tx_hash))
        print("example neq:", r.source_tx_hash[:16], "src_raw=", sraw, "dst_raw=", draw,
              "src_tok=", r.source_token_address[:14], "dst_tok=", c.token_address[:14],
              "sample args.amount=", (it or {}).get("args", {}).get("amount"))
        shown += 1
        if shown >= 3:
            break
