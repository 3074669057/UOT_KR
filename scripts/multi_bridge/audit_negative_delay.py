"""Diagnose Multi negative-delay pairs and missing decimals."""
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

delays: list[float] = []
neg_by_token: Counter = Counter()
for r in dev.itertuples():
    c = c_lookup.get(norm(r.dest_tx_hash))
    if c is None:
        continue
    d = float(c.candidate_timestamp) - float(r.source_timestamp)
    delays.append(d)
    if d < 0:
        neg_by_token[norm(r.source_token_address)] += 1

s = pd.Series(delays)
print("all dev pairs delay:", s.describe().round(1).to_dict())
print("negative:", int((s < 0).sum()), "| < -600:", int((s < -600).sum()), "| < -60:", int((s < -60).sum()))
print("neg by src token:", dict(neg_by_token.most_common(8)))
print("delay histogram (bins of 60s, -1200..1200):")
print(Counter(int(x // 60) for x in s[(s > -1200) & (s < 1200)]).most_common(30))
