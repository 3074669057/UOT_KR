"""Compare amount distributions: frozen Celer pool vs new Celer pool; and template subgraph masses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_42"
MINE = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "smoke" / "Celer" / "seed_42"

for tag, csv in (("frozen_pool", REPO / "out" / "paper_full_pipeline_run" / "uot" / "uot_flow_segments_eth.csv"),
                 ("mine_pool", REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "feature_stats" / "Celer" / "flow_segments_eth.csv")):
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    amt = pd.to_numeric(df["usd_amount_sum"], errors="coerce").clip(lower=0)
    print(f"{tag}: n={len(df)} usd q=[{amt.quantile(0.01):.2f},{amt.quantile(0.25):.2f},{amt.quantile(0.5):.2f},{amt.quantile(0.75):.2f},{amt.quantile(0.99):.2f}] max={amt.max():.2f} mean={amt.mean():.2f}")

for tag, csv in (("frozen_templates", FROZEN / "flow_segments_eth_synth.csv"),
                 ("mine_templates", MINE / "flow_segments_eth_synth.csv")):
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    amt = pd.to_numeric(df["usd_amount_sum"], errors="coerce").clip(lower=0)
    # per-template amount = max over the 6 src clones
    df["tpl"] = df["flow_id"].str.split("__").str[0]
    tpl_amt = df.groupby("tpl")["usd_amount_sum"].apply(lambda s: pd.to_numeric(s, errors="coerce").max())
    print(f"{tag}: n_tpl={len(tpl_amt)} tpl_usd q=[{tpl_amt.quantile(0.1):.2f},{tpl_amt.quantile(0.5):.2f},{tpl_amt.quantile(0.9):.2f}] max={tpl_amt.max():.2f} min={tpl_amt.min():.2f}")
    print("   sample:", sorted(round(float(x), 2) for x in tpl_amt)[:8])
