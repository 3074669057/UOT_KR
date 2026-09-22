"""Compare template amount spreads and pool amount distributions across bridges."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_42"

for tag, csv in (("frozen", FROZEN / "flow_segments_eth_synth.csv"),):
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df["tpl"] = df["flow_id"].str.split("__").str[0]
    tpl_amt = df.groupby("tpl")["usd_amount_sum"].apply(lambda s: pd.to_numeric(s, errors="coerce").max())
    print(f"{tag}: templates n={len(tpl_amt)} amt q=[{tpl_amt.quantile(0.02):.3f},{tpl_amt.quantile(0.25):.3f},{tpl_amt.quantile(0.5):.3f},{tpl_amt.quantile(0.98):.3f}] max/min={tpl_amt.max()/max(tpl_amt.min(),1e-12):.1f}")

for br in ("Multi", "Poly"):
    pool = pd.read_csv(OUT / "feature_stats" / br / "flow_segments_eth.csv", dtype=str, keep_default_na=False)
    amt = pd.to_numeric(pool["usd_amount_sum"], errors="coerce").clip(lower=1e-9)
    print(f"{br} pool: n={len(amt)} q=[{amt.quantile(0.02):.3f},{amt.quantile(0.25):.3f},{amt.quantile(0.5):.3f},{amt.quantile(0.98):.3f}] max/min={amt.max()/max(amt.min(),1e-12):.1f} max={amt.max():.2f} min={amt.min():.4f}")
    csv = OUT / "smoke" / br / "seed_42" / "flow_segments_eth_synth.csv"
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df["tpl"] = df["flow_id"].str.split("__").str[0]
    tpl_amt = df.groupby("tpl")["usd_amount_sum"].apply(lambda s: pd.to_numeric(s, errors="coerce").max())
    print(f"   templates: n={len(tpl_amt)} amt q=[{tpl_amt.quantile(0.02):.3f},{tpl_amt.quantile(0.25):.3f},{tpl_amt.quantile(0.5):.3f},{tpl_amt.quantile(0.98):.3f}] max/min={tpl_amt.max()/max(tpl_amt.min(),1e-12):.1f} max={tpl_amt.max():.2f} min={tpl_amt.min():.4f}")
    # source masses from diag
    d = json.loads((OUT / "smoke" / br / "seed_42" / "uot" / "uot_diagnostics.json").read_text(encoding="utf-8"))
    a = np.asarray(d["solver"]["source_mass_original"])
    print(f"   src masses: max={a.max():.2e} min={a[a>0].min():.2e} ratio={a.max()/a[a>0].min():.1e}")
