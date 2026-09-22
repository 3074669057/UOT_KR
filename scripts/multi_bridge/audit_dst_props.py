"""Check variation of Celer dst candidate properties within priced dev pairs."""
from __future__ import annotations

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

for b in ("Celer", "Multi", "Poly"):
    cached = load_bridge_cached(b)
    split = cached["split"]
    dev = split[split["split"] == "development"]
    all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
    c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}
    props = {"topic0": Counter(), "contract": Counter(), "bridge_hit": Counter(), "recv_missing": 0, "amt_zero": 0}
    n = 0
    for r in dev.itertuples():
        c = c_lookup.get(norm(r.dest_tx_hash))
        if c is None:
            continue
        n += 1
        props["topic0"][norm(c.topic0)[:18]] += 1
        props["contract"][norm(c.contract_address)] += 1
        props["bridge_hit"][bool(c.bridge_contract_hit)] += 1
        if not str(c.receiver).strip():
            props["recv_missing"] += 1
        if float(pd.to_numeric(c.amount_raw, errors="coerce") or 0) <= 0:
            props["amt_zero"] += 1
    print(f"{b}: n={n} topic0={dict(list(props['topic0'].items())[:6])}")
    print(f"   contracts={dict(list(props['contract'].items())[:6])} bridge_hit={dict(props['bridge_hit'])} recv_missing={props['recv_missing']} amt_zero={props['amt_zero']}")
