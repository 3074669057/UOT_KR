"""Compare dev truth pairs vs official Validation label.csv for Multi (delay check)."""
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

labels = pd.read_csv(REPO / "data" / "Validation" / "ETH-BNB" / "Multi" / "label.csv", dtype=str, keep_default_na=False)
labels["srcTxhash"] = labels["srcTxhash"].map(norm)
labels["dstTxhash"] = labels["dstTxhash"].map(norm)
official = {(r.srcTxhash, r.dstTxhash) for r in labels.itertuples()}
src_to_official_dst: dict[str, str] = {}
for s, d in official:
    src_to_official_dst.setdefault(s, d)
print("official labels:", len(official))

agree = 0
disagree = 0
not_in_official = 0
neg_delay_rows = 0
official_delay_stats = []
for r in dev.itertuples():
    sh = norm(r.source_tx_hash)
    dh = norm(r.dest_tx_hash)
    od = src_to_official_dst.get(sh)
    c = c_lookup.get(dh)
    delay = float(c.candidate_timestamp) - float(r.source_timestamp) if c is not None else None
    if od is None:
        not_in_official += 1
        continue
    if od == dh:
        agree += 1
    else:
        disagree += 1
        if delay is not None and delay < 0:
            neg_delay_rows += 1
            # what is the official dst's timestamp?
            oc = c_lookup.get(od)
            od_ts = float(oc.candidate_timestamp) if oc is not None else None
            official_delay_stats.append((delay, (od_ts - float(r.source_timestamp)) if od_ts else None))
print(f"dev pairs in official labels: agree={agree} disagree={disagree} not_in_official={not_in_official}")
print(f"of disagree, negative-delay dev pairs: {neg_delay_rows}")
print("(dev delay, official dst delay) samples:", official_delay_stats[:10])
