"""Inspect BNB-side candidate rows for Multi truth dst flows."""
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

cached = load_bridge_cached("Multi")
split = cached["split"]
dev = split[split["split"] == "development"]
all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}

print("dev candidate topic0 values:", Counter(all_c["topic0"].astype(str).str.lower()))
print("dev candidate contract_address sample:", Counter(all_c["contract_address"].astype(str).str.lower()).most_common(10))

# For anyETH (0x0615...) flows, show dst rows
src_anyeth = "0x0615dbba33fe61a31c7ed131bda6655ed76748b1"
src_anyusdt = "0x22648c12acd87912ea1710357b1302c6a4154ebc"
for label, stok in (("anyETH", src_anyeth), ("anyUSDT", src_anyusdt)):
    sub = dev[dev["source_token_address"].astype(str).str.lower() == stok]
    dsts = [c_lookup.get(norm(r.dest_tx_hash)) for r in sub.itertuples()]
    dtoks = Counter(str(d.token_address).lower() for d in dsts if d is not None)
    dcon = Counter(str(d.contract_address).lower() for d in dsts if d is not None)
    dtop = Counter(str(d.topic0).lower() for d in dsts if d is not None)
    print(f"\n{label}: n={len(sub)} dst tokens={dtoks.most_common(5)}")
    print(f"   dst contract_address={dcon.most_common(5)}")
    print(f"   dst topic0={dtop.most_common(5)}")
    # show one full row
    d = dsts[0]
    if d is not None:
        print("   example dst row:", d)
