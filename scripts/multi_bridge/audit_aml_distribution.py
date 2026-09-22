"""Audit: Hou-rule AML score distribution per bridge using FirstPhrase ETH tx data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.domain.aml.rules import _annotate_src_aml_rules  # noqa: E402

for b in ("Celer", "Multi", "Poly"):
    sample = json.loads((REPO / "data" / "Validation" / "ETH-BNB" / b / "sample.json").read_text(encoding="utf-8"))
    src_all = pd.DataFrame(
        [
            {
                "txhash": str(it.get("txhash") or ""),
                "args.receiver": str((it.get("args") or {}).get("receiver") or ""),
                "args.sender": str((it.get("args") or {}).get("sender") or ""),
            }
            for it in sample
        ]
    )
    fp = pd.read_csv(REPO / "data" / "FirstPhrase" / f"{b}_ETH.csv", dtype=str, keep_default_na=False)
    eth_df = fp.rename(columns={"from": "from_", "to": "to_"})
    eth_df["from"] = eth_df["from_"]
    eth_df["to"] = eth_df["to_"]
    eth_df = eth_df[["from", "to", "value", "timeStamp"]].copy()

    scored, meta = _annotate_src_aml_rules(src_all, eth_df)
    s = pd.to_numeric(scored["aml_risk_score"], errors="coerce").fillna(0.0)
    print(f"{b}: n={len(s)} unique={s.nunique()} mean={s.mean():.3f} std={s.std():.3f} min={s.min()} max={s.max()}")
    print("   distribution:", s.round(1).value_counts().sort_index().to_dict())
    hits = scored["aml_rule_hits"].astype(str)
    print("   flows with >=1 rule hit:", int((hits.str.len() > 0).sum()))
