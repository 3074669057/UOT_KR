"""Check candidate multiplicity per dst tx hash (pre-dedup) - real evidence trail width."""
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
    cand = cached["dev_candidates"].copy()
    cand["h"] = cand["candidate_tx_hash"].map(norm)
    mult = cand.groupby("h").size()
    dev_dsts = set(dev["dest_tx_hash"].map(norm))
    dev_mult = mult[mult.index.isin(dev_dsts)]
    print(f"{b}: dev dst txs={len(dev_dsts)} multiplicity distribution: {Counter(dev_mult.tolist())}")
