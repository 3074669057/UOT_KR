"""Decoder-family calibration on seeds 101-103 ONLY (never reads 42-46) and LOCKING.

- Grid: D0 legacy + D1/D2/D3 share thresholds, D4/D5 mutual-rank(+share), D6 cumulative
  row mass with abstention gate (grids pre-registered in decoder_audit.da_common).
- Both plans (UOT = frozen-params RC-UOT-Q, BOT = strictly balanced OT) are decoded with
  EVERY config; the PRIMARY SHARED TRANSPORT DECODER is selected by macro edge F1
  (template -> seed -> bridge -> macro-average over the three bridges), pooled over both
  plan types with equal weight. Bridge-specific / seed-specific thresholds are not allowed.
- Representational gates (checked on calibration for BOTH plan types): the decoder must
  emit >=1 split (a source with >=2 edges), >=1 merge (a target with >=2 incoming edges),
  and >=1 fully abstained source.
- Tie-break (within 0.005 F1): higher precision, then lower FP/template, then higher exact
  recovery, then simpler decoder (pre-registered complexity ordering).
- An RC-UOT-Q-specific best decoder is also computed and marked DIAGNOSTIC / UPPER-BOUND;
  it is never used for the primary comparison.
- Outputs: decoder_grid.csv, DECODER_SELECTION.md, locked_decoder.json (with timestamp,
  git info, code hashes). Once locked_decoder.json exists the decoder is FROZEN.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from decoder_audit.da_common import (  # noqa: E402
    AUDIT, BRIDGES, CALIB_SEEDS, DECODER_COMPLEXITY, decoder_configs, decode, evaluate_edges,
    hier_agg, load_cal_plans,
)


def _git_info() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=REPO, timeout=20)
        diff = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True,
                              cwd=REPO, timeout=60)
        return {"git_head": head.stdout.strip(), "git_diff_stat": diff.stdout.strip()[:1500]}
    except Exception as e:
        return {"git_error": type(e).__name__}


def _code_hash() -> str:
    paths = [Path(__file__), REPO / "scripts" / "multi_bridge" / "decoder_audit" / "da_common.py"]
    h = hashlib.md5()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    cal_dir = AUDIT / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # gate checks helper (existence across the pooled calibration set)
    def gates(edges: list[tuple[str, str]], sids: list[str]) -> dict[str, bool]:
        out_d: dict[str, int] = {}
        in_d: dict[str, int] = {}
        for s, d in edges:
            out_d[s] = out_d.get(s, 0) + 1
            in_d[d] = in_d.get(d, 0) + 1
        return {"split": any(v >= 2 for v in out_d.values()),
                "merge": any(v >= 2 for v in in_d.values()),
                "abstain": len(out_d) < len(sids) or len(edges) == 0}

    rows: list[dict[str, Any]] = []
    frames: dict[str, list[pd.DataFrame]] = {}   # (plan_type, cfg_name) -> [template frames]
    for cfg in decoder_configs():
        key = cfg["name"]
        for plan_type in ("UOT", "BOT"):
            frames[(plan_type, key)] = []
    for bridge in BRIDGES:
        for seed in CALIB_SEEDS:
            inst = load_cal_plans(bridge, seed)
            for plan_type, P in (("UOT", inst["P_uot"]), ("BOT", inst["P_bot"])):
                for cfg in decoder_configs():
                    edges = decode(inst, P, cfg)
                    df, summ = evaluate_edges(inst, edges)
                    df["seed"] = seed
                    df["bridge"] = bridge
                    frames[(plan_type, cfg["name"])].append(df)
                    g = gates(edges, inst["sids"])
                    rows.append({
                        "bridge": bridge, "seed": seed, "plan_type": plan_type,
                        "decoder": cfg["name"], "family": cfg["family"],
                        "n_edges_per_template": summ["n_pred_edges"],
                        "edge_precision": summ["edge_precision"],
                        "edge_recall": summ["edge_recall"], "edge_f1": summ["edge_f1"],
                        "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                        "overall_exact": summ["overall_exact"],
                        "fp_per_template": summ["edge_fp_total"] / 48,
                        "gate_split": g["split"], "gate_merge": g["merge"],
                        "gate_abstain": g["abstain"],
                    })
            print(f"[calib] {bridge} seed {seed} done ({len(decoder_configs())} configs x 2 plans)",
                  flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(cal_dir / "decoder_grid.csv", index=False)

    # macro aggregation: template -> seed -> bridge -> macro over bridges
    def macro_metric(plan_type: str, cfg: str, metric: str) -> dict[str, Any]:
        fr = frames[(plan_type, cfg)]
        per_seed: dict[tuple[str, int], list[float]] = {}
        for f in fr:
            per_seed.setdefault((str(f["bridge"].iloc[0]), int(f["seed"].iloc[0])), []).append(
                float(f[metric].mean()))
        bridge_means: dict[str, list[float]] = {}
        for (b, _s), vals in per_seed.items():
            bridge_means.setdefault(b, []).append(float(np.mean(vals)))
        macro = float(np.mean([float(np.mean(v)) for v in bridge_means.values()])) if bridge_means else float("nan")
        return {"macro": macro, "bridge_means": {b: float(np.mean(v)) for b, v in bridge_means.items()}}

    candidates: list[dict[str, Any]] = []
    for cfg in decoder_configs():
        u = macro_metric("UOT", cfg["name"], "edge_f1")
        b = macro_metric("BOT", cfg["name"], "edge_f1")
        if not np.isfinite(u["macro"]) or not np.isfinite(b["macro"]):
            continue
        shared_f1 = 0.5 * u["macro"] + 0.5 * b["macro"]
        u_p = macro_metric("UOT", cfg["name"], "edge_precision")["macro"]
        b_p = macro_metric("BOT", cfg["name"], "edge_precision")["macro"]
        shared_p = 0.5 * u_p + 0.5 * b_p
        u_fp = macro_metric("UOT", cfg["name"], "edge_fp")["macro"]
        b_fp = macro_metric("BOT", cfg["name"], "edge_fp")["macro"]
        shared_fp = 0.5 * u_fp + 0.5 * b_fp
        u_ex = macro_metric("UOT", cfg["name"], "overall_exact")["macro"]
        b_ex = macro_metric("BOT", cfg["name"], "overall_exact")["macro"]
        shared_ex = 0.5 * u_ex + 0.5 * b_ex
        # representational gates on calibration (existence across the pooled calibration set,
        # per plan type; both plan types must pass)
        gate_ok = True
        for pt in ("UOT", "BOT"):
            sub = grid[(grid["plan_type"] == pt) & (grid["decoder"] == cfg["name"])]
            gate_ok = gate_ok and bool(sub["gate_split"].any()) and bool(sub["gate_merge"].any()) \
                and bool(sub["gate_abstain"].any())
        candidates.append({
            "decoder": cfg["name"], "family": cfg["family"],
            "params": cfg["params"], "complexity": DECODER_COMPLEXITY.get(cfg["family"], 0),
            "shared_macro_f1": shared_f1, "uot_macro_f1": u["macro"], "bot_macro_f1": b["macro"],
            "shared_precision": shared_p, "shared_fp_per_template": shared_fp,
            "shared_exact": shared_ex, "gates_ok": gate_ok,
        })
    cand = pd.DataFrame(candidates)
    cand.to_csv(cal_dir / "decoder_selection_candidates.csv", index=False)
    elig = cand[cand["gates_ok"]].copy()
    if elig.empty:
        raise SystemExit("no decoder config passes the representational gates on calibration!")
    best_f1 = float(elig["shared_macro_f1"].max())
    tie = elig[(elig["shared_macro_f1"] >= best_f1 - 0.005)].copy()
    tie = tie.sort_values(
        ["shared_precision", "shared_fp_per_template", "shared_exact", "complexity", "decoder"],
        ascending=[False, True, False, True, True])
    primary = tie.iloc[0].to_dict()

    # RC-UOT-Q-specific best (DIAGNOSTIC only)
    uot_cand = cand[cand["gates_ok"]].copy()
    uot_best_f1 = float(uot_cand["uot_macro_f1"].max())
    uot_tie = uot_cand[(uot_cand["uot_macro_f1"] >= uot_best_f1 - 0.005)].sort_values(
        ["shared_precision", "shared_fp_per_template", "shared_exact", "complexity", "decoder"],
        ascending=[False, True, False, True, True])
    uot_best = uot_tie.iloc[0].to_dict()

    locked = {
        "decoder_name": primary["decoder"],
        "decoder_family": primary["family"],
        "formula": _formula_text(primary),
        "params": primary["params"],
        "selection_metric": ("macro edge F1: template -> seed -> bridge -> macro-average over "
                             "Celer/Multi/Poly; pooled over UOT and BOT plan types with equal weight"),
        "shared_macro_f1": primary["shared_macro_f1"],
        "uot_macro_f1": primary["uot_macro_f1"],
        "bot_macro_f1": primary["bot_macro_f1"],
        "tie_break": "within 0.005 F1: precision desc, FP/template asc, exact desc, complexity asc",
        "representational_gates": ">=1 split, >=1 merge, >=1 abstained source (both plan types)",
        "applies_to": ["RC-UOT-Q", "Balanced-OT"],  # PRIMARY SHARED TRANSPORT DECODER
        "rc_uot_q_specific_best_decoder_DIAGNOSTIC_ONLY": {
            "decoder": uot_best["decoder"], "uot_macro_f1": uot_best["uot_macro_f1"],
            "note": "DIAGNOSTIC / UPPER-BOUND ANALYSIS — not used for the primary comparison"},
        "calibration_seeds": list(CALIB_SEEDS),
        "bridges": list(BRIDGES),
        "threshold_mm_frozen": {"tau": 0.05, "cutoff": 0.47762288884480164, "source": "previous study, not re-tuned"},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "code_hash_md5": _code_hash(),
        "status": "LOCKED — do not modify after this file is written",
    }
    (cal_dir / "locked_decoder.json").write_text(json.dumps(locked, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")

    md_lines = [
        "# DECODER SELECTION (calibration seeds 101-103 only)", "",
        f"- **Locked primary shared transport decoder:** `{locked['decoder_name']}` "
        f"(family {locked['decoder_family']}, params {locked['params']})",
        f"- **Selection metric:** {locked['selection_metric']}",
        f"- **Calibration macro edge F1:** shared {locked['shared_macro_f1']:.4f} "
        f"(UOT {locked['uot_macro_f1']:.4f}, BOT {locked['bot_macro_f1']:.4f})",
        f"- **Shared precision:** {primary['shared_precision']:.4f}; "
        f"FP/template {primary['shared_fp_per_template']:.2f}; exact {primary['shared_exact']:.4f}",
        f"- **Tie-break:** {locked['tie_break']}",
        f"- **Gates:** {locked['representational_gates']}",
        f"- **RC-UOT-Q-specific best (DIAGNOSTIC ONLY):** `{uot_best['decoder']}` "
        f"(UOT macro F1 {uot_best['uot_macro_f1']:.4f})",
        "- **Threshold-MM frozen** (previous study, NOT re-tuned): tau 0.05, cutoff 0.477623",
        f"- **Locked at:** {locked['timestamp_utc']} (UTC); git head {locked['git'].get('git_head', '?')[:12]}",
        f"- **Code hash (md5):** {locked['code_hash_md5']}",
        "",
        "The decoder is now LOCKED. Test seeds 42-46 were never read by this script.",
    ]
    (cal_dir / "DECODER_SELECTION.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("\n".join(md_lines))
    print("\ntop-12 calibration candidates (shared):")
    print(cand.sort_values("shared_macro_f1", ascending=False).head(12)
          [["decoder", "shared_macro_f1", "uot_macro_f1", "bot_macro_f1", "shared_precision",
            "shared_fp_per_template", "gates_ok"]].to_string(index=False,
                                                             float_format=lambda x: f"{x:.4f}"))
    return 0


def _formula_text(primary: dict[str, Any]) -> str:
    fam = primary["family"]
    p = primary["params"]
    if fam == "D0":
        return "edge iff pi_ij >= 1e-9 (legacy)"
    if fam == "D1":
        return f"edge iff pi_ij / row_mass_i >= {p['tau']}"
    if fam == "D2":
        return f"edge iff min(row_share, col_share) >= {p['tau']}"
    if fam == "D3":
        return f"edge iff sqrt(row_share * col_share) >= {p['tau']}"
    if fam == "D4":
        return f"edge iff row_rank <= {p['k']} AND col_rank <= {p['k']}"
    if fam == "D5":
        return f"edge iff row_rank <= {p['k']} AND col_rank <= {p['k']} AND min_share >= {p['tau']}"
    return (f"per source, keep edges by descending row_share until cumulative >= {p['p']}, "
            f"gated by row_share >= {p['gate']} (abstention preserved)")


if __name__ == "__main__":
    raise SystemExit(main())
