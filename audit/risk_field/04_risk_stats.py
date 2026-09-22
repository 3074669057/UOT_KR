"""Step 2/3: empirical distribution of risk-related fields.

Computes, per file per risk column:
  n_rows, missing/NaN rate, min, max, mean, median, std,
  quantiles q05/q25/q50/q75/q95/q99, n_distinct, is_constant,
  value_counts when n_distinct <= 20.
Also value_counts for `aml_risk_level`.
Also groups by the `bridge` column when present.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\04_risk_stats.py
Output: stats_report.txt (same dir) + console
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"<REPO>"

RISK_COLS = [
    "aml_risk_score", "aml_score", "aml_risk_score_raw",
    "evidence_quality_score", "risk_weighted_mass",
    "aml_risk_level", "aml_score_mean", "aml_score_max",
    "evidence_quality_mean", "evidence_quality", "evidence_level",
]
Q = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

out = io.StringIO()


def _clean(x):
    if isinstance(x, str):
        return x.replace("\ufeff", "<BOM>")
    return x


def p(*a):
    a = tuple(_clean(x) for x in a)
    print(*a)
    print(*a, file=out)


def colstats(s: pd.Series) -> dict:
    n = len(s)
    num = pd.to_numeric(s, errors="coerce")
    nn = num.notna().sum()
    d = {
        "n_rows": n,
        "n_nonnull": int(nn),
        "n_missing": int(n - nn),
        "missing_rate": (n - nn) / n if n else float("nan"),
        "n_distinct": int(num.nunique(dropna=True)),
    }
    if nn == 0:
        d.update(is_constant=None)
        return d
    d.update(
        min=float(num.min()), max=float(num.max()), mean=float(num.mean()),
        median=float(num.median()), std=float(num.std(ddof=1)) if nn > 1 else float("nan"),
    )
    for q in Q:
        d[f"q{int(q*100):02d}"] = float(num.quantile(q))
    d["is_constant"] = bool(num.nunique(dropna=True) == 1)
    d["_series"] = num
    return d


FMT = (
    "    {col:<24} n={n_rows:<7} miss={n_missing:<6} miss%={mr:>6.2%} "
    "min={min:>10.4f} max={max:>10.4f} mean={mean:>10.4f} med={median:>10.4f} std={std:>10.4f}"
)
QFMT = "        q05={q05:>9.4f} q25={q25:>9.4f} q50={q50:>9.4f} q75={q75:>9.4f} q95={q95:>9.4f} q99={q99:>9.4f}  ndistinct={nd} const={const}"


def report_frame(label: str, df: pd.DataFrame, indent: str = ""):
    p(f"{indent}--- {label}   rows={len(df)}")
    found = [c for c in RISK_COLS if c in df.columns]
    if not found:
        p(f"{indent}    (no risk-like columns present; columns={list(df.columns)[:40]})")
        return
    for c in found:
        if c == "aml_risk_level":
            vc = df[c].value_counts(dropna=False)
            p(f'    {c:<24} n={len(df)} n_distinct={df[c].nunique(dropna=True)} const={df[c].nunique(dropna=True)==1}')
            for k, v in vc.items():
                p(f'        {str(k)!r:<28} {v}')
            continue
        if c in ("evidence_level",):
            vc = df[c].value_counts(dropna=False)
            p(f'    {c:<24} n={len(df)} n_distinct={df[c].nunique(dropna=True)}')
            for k, v in vc.head(20).items():
                p(f'        {str(k)!r:<28} {v}')
            continue
        st = colstats(df[c])
        if st["n_nonnull"] == 0:
            p(f'    {c:<24} n={st["n_rows"]} ALL MISSING/non-numeric (n_distinct=0)')
            continue
        p(FMT.format(col=c, n_rows=st["n_rows"], n_missing=st["n_missing"],
                     mr=st["missing_rate"], min=st["min"], max=st["max"],
                     mean=st["mean"], median=st["median"], std=st["std"]))
        p(QFMT.format(q05=st["q05"], q25=st["q25"], q50=st["q50"], q75=st["q75"],
                      q95=st["q95"], q99=st["q99"], nd=st["n_distinct"], const=st["is_constant"]))
        if st["n_distinct"] <= 20:
            vc = st["_series"].value_counts(dropna=False).sort_index()
            p(f'        value_counts (<=20 distinct):')
            for k, v in vc.items():
                p(f'            {k!r:<28} {v}')


def analyze(path: str, group_by_bridge=True):
    rel = os.path.relpath(path, ROOT)
    if not os.path.exists(path):
        p(f"### {rel}\n    FILE NOT FOUND\n")
        return
    size = os.path.getsize(path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        p(f"### {rel}  [{size} B]\n    READ ERROR: {e!r}\n")
        return
    p(f"### {rel}  [{size} B]")
    report_frame("whole file", df)
    if group_by_bridge and "bridge" in df.columns:
        vals = [v for v in df["bridge"].dropna().unique()]
        if 0 < len(vals) <= 12:
            for v in sorted(vals):
                report_frame(f"bridge == {v}", df[df["bridge"] == v], indent="  ")
    if group_by_bridge and "bridge" not in df.columns and "chain" in df.columns:
        p(f'    [note] no `bridge` column; chain values = {sorted(map(str, df["chain"].dropna().unique()))[:10]}')
    p("")


# ---------------------------------------------------------------- file list
THREE = os.path.join(ROOT, "out", "multi_bridge_expansion", "flow_structural_three_bridges")
PIPE = os.path.join(ROOT, "out", "paper_full_pipeline_run")
CH4 = os.path.join(ROOT, "out", "chapter4_data_package")

targets = []
for b in ("Celer", "Multi", "Poly"):
    targets += [
        os.path.join(THREE, b, "flow_segments_eth.csv"),
        os.path.join(THREE, b, "flow_segments_bnb.csv"),
        os.path.join(THREE, b, "flow_segments_eth_synth.csv"),
        os.path.join(THREE, b, "flow_segments_bnb_synth.csv"),
        os.path.join(THREE, b, "uot", "flow_segments_eth.csv"),
        os.path.join(THREE, b, "uot", "flow_segments_bnb.csv"),
        os.path.join(THREE, b, "uot", "uot_marginals.csv"),
    ]
targets += [
    os.path.join(PIPE, "labels", "flow_segments_eth.csv"),
    os.path.join(PIPE, "labels", "flow_segments_bnb.csv"),
    os.path.join(PIPE, "uot", "uot_marginals.csv"),
    os.path.join(PIPE, "evidence", "evidence_eth.csv"),
    os.path.join(PIPE, "evidence", "evidence_bnb.csv"),
    os.path.join(CH4, "frozen_substrates", "flow_segments_eth.csv"),
    os.path.join(CH4, "frozen_substrates", "flow_segments_bnb.csv"),
]

# src_flows_aggregated / candidate_* -> resolve from discovery.json
import json
with open(os.path.join(HERE, "discovery.json"), encoding="utf-8") as f:
    disc = json.load(f)
for h in disc["strict_column_hits"]:
    b = os.path.basename(h["path"]).lower()
    if b in ("src_flows_aggregated.csv", "candidate_eth_universe_flows.csv",
             "candidate_bnb_universe_flows.csv", "evidence_eth.csv", "evidence_bnb.csv"):
        targets.append(h["path"])

seen = set()
final = []
for t in targets:
    if t not in seen:
        seen.add(t)
        final.append(t)

p("=" * 110)
p("STEP 2/3 - RISK FIELD DISTRIBUTIONS (pandas %s, numpy %s)" % (pd.__version__, np.__version__))
p("=" * 110)
p("")
for t in final:
    analyze(t, group_by_bridge=True)

# ------------------------------------------- data-side negative controls
p("=" * 110)
p("STEP 4 - `data\\` SIDE FILES: full header + risk-column presence check")
p("=" * 110)
p("")
data_files = [
    r"data\Validation\ETH-BNB\Celer\input.csv",
    r"data\Validation\ETH-BNB\Multi\input.csv",
    r"data\Validation\ETH-BNB\Poly\input.csv",
    r"data\Validation\ETH-Polygon\Celer\input.csv",
    r"data\Validation\ETH-Polygon\Multi\input.csv",
    r"data\Validation\ETH-Polygon\Poly\input.csv",
    r"data\Validation\ETH-BNB\Celer\label.csv",
    r"data\Validation\ETH-BNB\Multi\label.csv",
    r"data\Validation\ETH-BNB\Poly\label.csv",
    r"data\Validation\ETH-Polygon\Celer\label.csv",
    r"data\Validation\ETH-Polygon\Multi\label.csv",
    r"data\Validation\ETH-Polygon\Poly\label.csv",
    r"data\label\celer_label.csv",
    r"data\FirstPhrase\Celer.csv",
    r"data\FirstPhrase\Multi.csv",
    r"data\FirstPhrase\Multi_ETH.csv",
    r"data\FirstPhrase\Poly.csv",
    r"data\FirstPhrase\Poly_ETH.csv",
    r"data\FirstPhrase\Celer_ETH.csv",
    r"data\in\Celer_ETH_cun.csv",
    r"data\label\tx\Celer_ETH_cun.csv",
    r"data\label\tx\Celer_BNB_qu.csv",
    r"data\label\tx\Celer_ETH_BNB_completed.csv",
    r"data\Model\model_evaluation_results.csv",
    r"data\Model\combined_resampling_results.csv",
    r"data\Model\normalization_map.csv",
]
for rel in data_files:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        p(f'{rel}\n    FILE NOT FOUND')
        continue
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            hdr = f.readline().rstrip("\n").rstrip("\r")
    except Exception as e:
        p(f"{rel}\n    READ ERROR {e!r}")
        continue
    nrows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="replace")) - 1
    cols = hdr.split(",")
    hits = [c for c in cols if any(k in c.lower() for k in ("risk", "aml", "evidence"))]
    p(f"{rel}   rows={nrows}  ncols={len(cols)}")
    p(f"    HEADER: {hdr}")
    p(f"    risk-like cols: {hits if hits else 'NONE - column NOT present'}")
    p("")

with open(os.path.join(HERE, "stats_report.txt"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("\n[written]", os.path.join(HERE, "stats_report.txt"))
