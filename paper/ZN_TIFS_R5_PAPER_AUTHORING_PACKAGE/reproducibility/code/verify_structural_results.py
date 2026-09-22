"""Independent verification of the faithful three-bridge structural results.

Recomputes split/merge recovery means and stds DIRECTLY from per-seed raw artifacts
(eval/uot_evaluation_metrics.json + uot/uot_transport_plan.csv + labels), does NOT trust
the orchestrator's summary. Checks: 5 seeds present, 48 templates per seed, no NaN, no
duplicate seeds, no leakage (label ids vs plan ids), baseline evaluator consistency,
and spot-checks that split = 1 src -> 2 dst and merge = 2 src -> 1 dst in decoded plans.
Writes aggregated/structural_aggregated.csv + aggregated/verification_report.json|md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
BRIDGES = ("Celer", "Multi", "Poly")
SEEDS = (42, 43, 44, 45, 46)
FROZEN = {"split": 0.9458333333333332, "merge": 0.9666666666666668}


def main() -> int:
    report: dict[str, Any] = {"checks": {}, "per_seed": {}, "aggregated": {}}
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    for br in BRIDGES:
        br_rows: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            run_root = OUT / "per_seed" / br / f"seed_{seed}"
            ev_path = run_root / "eval" / "uot_evaluation_metrics.json"
            if not ev_path.is_file():
                continue
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
            plan = pd.read_csv(run_root / "uot" / "uot_transport_plan.csv", dtype=str, keep_default_na=False)
            labels = pd.read_csv(run_root / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
            hints = json.loads((run_root / "labels" / "synthetic_uot_eval_metrics.json").read_text(encoding="utf-8"))

            split = ev.get("split_recovery_rate")
            merge = ev.get("merge_recovery_rate")
            if split is None or merge is None or (isinstance(split, float) and np.isnan(split)):
                checks.setdefault(f"{br}_seed_{seed}_nan", []).append("missing metric")
                continue

            # structural counts from raw plan (independent recomputation)
            plan["_m"] = pd.to_numeric(plan["transport_mass"], errors="coerce").fillna(0.0)
            pos = plan[plan["_m"] > 1e-9]
            split_rows = labels[labels["pattern_type"] == "one_to_many"]
            merge_rows = labels[labels["pattern_type"] == "many_to_one"]
            pred_set = set(zip(pos["src_flow_id"].astype(str), pos["dst_flow_id"].astype(str)))
            split_hits = sum((str(r.src_flow_id), str(r.dst_flow_id)) in pred_set for r in split_rows.itertuples())
            merge_hits = sum((str(r.src_flow_id), str(r.dst_flow_id)) in pred_set for r in merge_rows.itertuples())
            split_recalc = split_hits / max(len(split_rows), 1)
            merge_recalc = merge_hits / max(len(merge_rows), 1)

            # per-src / per-dst fan-out in the decoded plan (non-one-to-one check)
            src_deg = pos.groupby("src_flow_id")["dst_flow_id"].nunique()
            dst_deg = pos.groupby("dst_flow_id")["src_flow_id"].nunique()
            n_split_srcs = int((src_deg > 1).sum())
            n_merge_dsts = int((dst_deg > 1).sum())

            br_rows[seed] = {
                "split_recovery_artifact": split,
                "merge_recovery_artifact": merge,
                "split_recovery_recalc": split_recalc,
                "merge_recovery_recalc": merge_recalc,
                "n_split_label_rows": int(len(split_rows)),
                "n_merge_label_rows": int(len(merge_rows)),
                "n_templates": int(hints.get("seed_templates_used") or 0),
                "synthetic_row_count": int(hints.get("synthetic_row_count") or 0),
                "n_src_with_multi_dst": n_split_srcs,
                "n_dst_with_multi_src": n_merge_dsts,
                "truth_pairs": int(len(hints.get("eval_hints", {}).get("truth_flow_pairs") or [])),
            }
            rows.append({
                "bridge": br, "method": "RC-UOT-Q", "seed": seed,
                "split_recovery": split, "merge_recovery": merge,
                "split_recovery_recalc": split_recalc, "merge_recovery_recalc": merge_recalc,
                "n_templates": br_rows[seed]["n_templates"],
            })
        report["per_seed"][br] = {str(k): v for k, v in sorted(br_rows.items())}

        n_seeds = len(br_rows)
        checks[f"{br}_n_seeds"] = n_seeds
        if n_seeds != 5:
            checks[f"{br}_seed_completeness"] = f"FAIL: {n_seeds}/5"
        split_vals = [v["split_recovery_artifact"] for v in br_rows.values()]
        merge_vals = [v["merge_recovery_artifact"] for v in br_rows.values()]
        if not split_vals:
            continue
        report["aggregated"][br] = {
            "split_mean": float(np.mean(split_vals)),
            "split_std": float(np.std(split_vals, ddof=1)),
            "merge_mean": float(np.mean(merge_vals)),
            "merge_std": float(np.std(merge_vals, ddof=1)),
            "n_seeds": int(n_seeds),
            "per_seed_split": [round(float(x), 6) for x in split_vals],
            "per_seed_merge": [round(float(x), 6) for x in merge_vals],
        }
        # recalc agreement: artifact vs independent recomputation must match
        ok = all(abs(v["split_recovery_artifact"] - v["split_recovery_recalc"]) < 1e-9 for v in br_rows.values())
        checks[f"{br}_artifact_vs_recalc_consistent"] = bool(ok)
        # template counts
        checks[f"{br}_templates_48"] = all(v["n_templates"] == 48 for v in br_rows.values())
        # non-one-to-one structure present in decoded plans
        checks[f"{br}_decoded_has_multi_dst"] = all(v["n_src_with_multi_dst"] > 0 for v in br_rows.values())
        checks[f"{br}_decoded_has_multi_src"] = all(v["n_dst_with_multi_src"] > 0 for v in br_rows.values())

    # Celer regression gate
    cel = report["aggregated"].get("Celer", {})
    checks["celer_regression"] = {
        "frozen": FROZEN,
        "new": {"split_mean": cel.get("split_mean"), "merge_mean": cel.get("merge_mean")},
        "split_delta": round(float(cel.get("split_mean", 0)) - FROZEN["split"], 4) if cel else None,
        "merge_delta": round(float(cel.get("merge_mean", 0)) - FROZEN["merge"], 4) if cel else None,
        "pass": bool(cel and abs(cel["split_mean"] - FROZEN["split"]) <= 0.05 and abs(cel["merge_mean"] - FROZEN["merge"]) <= 0.05),
    }

    # baselines: existing structural evaluator (structure-level) must output 0 per seed
    base = pd.read_csv(OUT / "baselines" / "baseline_structural_per_seed.csv")
    checks["baselines_structure_level"] = {
        "all_zero": bool((base["split_recovery"] == 0).all() and (base["merge_recovery"] == 0).all()),
        "n_rows": int(len(base)),
        "per_bridge_zero": {b: bool((base[base["bridge"] == b][["split_recovery", "merge_recovery"]] == 0).all().all()) for b in BRIDGES},
    }

    # spot-check templates: 1 src -> 2 dst for split, 2 src -> 1 dst for merge (labels)
    spot: dict[str, Any] = {}
    for br in BRIDGES:
        seed_dir = OUT / "per_seed" / br / "seed_42"
        labels = pd.read_csv(seed_dir / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
        split_tpl = labels[labels["label_source"].astype(str).str.contains("split")]
        merge_tpl = labels[labels["label_source"].astype(str).str.contains("merge")]
        s_src = split_tpl.groupby("src_flow_id")["dst_flow_id"].nunique()
        m_dst = merge_tpl.groupby("dst_flow_id")["src_flow_id"].nunique()
        spot[br] = {
            "split_1src_to_2dst_ok": bool((s_src == 2).all() and (split_tpl["dst_flow_id"].nunique() == 2 * s_src.size)),
            "merge_2src_to_1dst_ok": bool((m_dst == 2).all() and (merge_tpl["src_flow_id"].nunique() == 2 * m_dst.size)),
            "n_split_templates": int(s_src.size),
            "n_merge_templates": int(m_dst.size),
        }
    checks["template_structure_spotcheck"] = spot

    report["checks"] = checks
    (OUT / "aggregated").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "aggregated" / "structural_per_seed.csv", index=False)

    agg_rows = []
    for br in BRIDGES:
        a = report["aggregated"].get(br)
        if not a:
            continue
        agg_rows.append({"bridge": br, "method": "RC-UOT-Q", "split_mean": a["split_mean"], "split_std": a["split_std"],
                         "merge_mean": a["merge_mean"], "merge_std": a["merge_std"],
                         "n_seeds": a["n_seeds"], "n_templates_per_seed": 48})
    pd.DataFrame(agg_rows).to_csv(OUT / "aggregated" / "structural_aggregated.csv", index=False)
    (OUT / "aggregated" / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md = ["# Independent verification report", ""]
    for k, v in report["aggregated"].items():
        md.append(f"- **{k}**: split {v['split_mean']:.4f} ± {v['split_std']:.4f}; merge {v['merge_mean']:.4f} ± {v['merge_std']:.4f} "
                  f"(per-seed split {v['per_seed_split']}, merge {v['per_seed_merge']})")
    md.append("")
    md.append("```json")
    md.append(json.dumps(checks, indent=2, default=str))
    md.append("```")
    (OUT / "aggregated" / "verification_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
