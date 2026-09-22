"""Independent verification of the baseline + mechanism study.

Recomputes every headline metric FROM THE RAW PER-TEMPLATE ARTIFACTS
(edges_{method}.csv + template_truth.csv + the frozen per-seed label CSVs), never from
the aggregated summaries. Checks completeness, duplicate/missing seeds and templates,
NaN handling, cross-method template identity, the one-to-one degree bound for
Connector/ABCTracer, the existence of decoded 1->2 / 2->1 instances for the many-match
methods, Balanced-OT convergence, and the consistency of the aggregated tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import (  # noqa: E402
    BRIDGES, FROZEN, FROZEN_PARAMS, METHODS, STUDY, TEST_SEEDS,
    evaluate_template, load_frozen_instance, truth_structure,
)

OUT = STUDY / "verification"
METHOD_FILE = {
    "Connector-style": "edges_Connector_style.csv",
    "ABCTracer-style": "edges_ABCTracer_style.csv",
    "Threshold-MM": "edges_Threshold_MM.csv",
    "Balanced-OT": "edges_Balanced_OT.csv",
    "RC-UOT-Q": "edges_RC_UOT_Q.csv",
}


def recompute_per_template(bridge: str, seed: int, method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute per-template metrics from the raw edge file + frozen labels."""
    root = STUDY / "per_seed" / bridge / f"seed_{seed}"
    edge_path = root / METHOD_FILE[method]
    frozen = load_frozen_instance(bridge, seed)
    truth = truth_structure(frozen["labels"])
    edges = set()
    if edge_path.is_file():
        df = pd.read_csv(edge_path, dtype=str, keep_default_na=False)
        for _, r in df.iterrows():
            edges.add((str(r["src_flow_id"]), str(r["dst_flow_id"])))
    rows = []
    for t, tr in sorted(truth.items()):
        local = {(s, d) for s, d in edges if s.split("__synth")[0] == t}
        m = evaluate_template(tr, local, set(tr["all_src"]))
        m["template_id"] = t
        rows.append(m)
    dfm = pd.DataFrame(rows)
    summ = {c: float(dfm[c].mean()) if not dfm.empty else float("nan") for c in
            ("split_exact", "merge_exact", "deg_acc_split", "deg_acc_merge",
             "edge_precision", "edge_recall", "edge_f1", "coverage", "n_pred_edges")}
    summ["edge_fp_total"] = int(dfm["edge_fp"].sum())
    summ["edge_tp_total"] = int(dfm["edge_tp"].sum())
    summ["n_templates"] = int(len(dfm))
    return dfm, summ


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    issues: list[str] = []
    recomputed: dict[str, dict[str, float]] = {}

    # 1. completeness
    missing: list[str] = []
    for br in BRIDGES:
        for seed in TEST_SEEDS:
            root = STUDY / "per_seed" / br / f"seed_{seed}"
            if not root.is_dir():
                missing.append(f"{br}/seed_{seed}")
                continue
            for m, f in METHOD_FILE.items():
                if not (root / f).is_file():
                    missing.append(f"{br}/seed_{seed}/{f}")
            if not (root / "template_truth.csv").is_file():
                missing.append(f"{br}/seed_{seed}/template_truth.csv")
    checks["artifacts_missing"] = missing
    if missing:
        issues.append(f"missing artifacts: {missing}")

    # 2. template identity + counts, per-seed n=5, no duplicates, 48 templates
    n_seeds_per_bridge: dict[str, int] = {}
    template_sets: dict[str, set[str]] = {}
    nan_found: list[str] = []
    one_to_one_deg_ok = True
    decoded_split_ok: dict[str, bool] = {}
    decoded_merge_ok: dict[str, bool] = {}
    for m in METHODS:
        decoded_split_ok[m] = False
        decoded_merge_ok[m] = False

    for br in BRIDGES:
        n_seeds = 0
        for seed in TEST_SEEDS:
            root = STUDY / "per_seed" / br / f"seed_{seed}"
            if not (root / "template_truth.csv").is_file():
                continue
            n_seeds += 1
            tt = pd.read_csv(root / "template_truth.csv", dtype=str, keep_default_na=False)
            tset = set(tt["template_id"])
            if len(tset) != 48 or len(tt) != 48:
                issues.append(f"{br}/seed_{seed}: template_truth has {len(tset)} unique / {len(tt)} rows (expected 48)")
            # cross-method template identity WITHIN the seed (across-seed template sets
            # legitimately differ: each seed samples its own 48 templates from the pool)
            method_tpl_sets = []
            for m in METHODS:
                dfm, summ = recompute_per_template(br, seed, m)
                method_tpl_sets.append(set(dfm["template_id"]))
                if summ["n_templates"] != 48:
                    issues.append(f"{br}/seed_{seed}/{m}: recomputed {summ['n_templates']} templates (expected 48)")
                if dfm.isna().any().any():
                    nan_found.append(f"{br}/seed_{seed}/{m}: NaN in recomputed metrics")
                    issues.append(f"{br}/seed_{seed}/{m}: NaN in recomputed metrics")
                if not set(dfm["template_id"]) == tset:
                    issues.append(f"{br}/seed_{seed}/{m}: method template set differs from truth set")
                # aggregated artifact consistency (recomputed vs run artifact)
                run_df = pd.read_csv(root / "templates_per_method.csv", dtype={"bridge": str, "method": str})
                run_m = run_df[run_df["method"] == m]
                merged = dfm.merge(run_m, on="template_id", suffixes=("_re", "_run"))
                for col in ("split_exact", "merge_exact", "edge_precision", "edge_recall", "edge_f1",
                            "deg_acc_split", "deg_acc_merge", "edge_fp", "coverage"):
                    if merged.empty:
                        continue
                    diff = (pd.to_numeric(merged[f"{col}_re"], errors="coerce")
                            - pd.to_numeric(merged[f"{col}_run"], errors="coerce")).abs().max()
                    if diff > 1e-9:
                        issues.append(f"{br}/seed_{seed}/{m}/{col}: recomputed vs artifact max diff {diff}")
                # degree audit for one-to-one methods + decoded 1->2 / 2->1
                edges = set()
                ef = pd.read_csv(root / METHOD_FILE[m], dtype=str, keep_default_na=False)
                for _, r in ef.iterrows():
                    edges.add((str(r["src_flow_id"]), str(r["dst_flow_id"])))
                out_d: dict[str, int] = {}
                in_d: dict[str, int] = {}
                for s, d in edges:
                    out_d[s] = out_d.get(s, 0) + 1
                    in_d[d] = in_d.get(d, 0) + 1
                if m in ("Connector-style", "ABCTracer-style"):
                    max_out = max(out_d.values()) if out_d else 0
                    if max_out > 1:
                        one_to_one_deg_ok = False
                        issues.append(f"{br}/seed_{seed}/{m}: max source out-degree {max_out} > 1 (violates one-to-one)")
                if out_d and max(out_d.values()) >= 2:
                    decoded_split_ok[m] = True
                if in_d and max(in_d.values()) >= 2:
                    decoded_merge_ok[m] = True
                recomputed.setdefault(m, {})
                for k, v in summ.items():
                    if k not in ("n_templates",):
                        recomputed[m].setdefault(k, []).append(v)
        n_seeds_per_bridge[br] = n_seeds
        if n_seeds != 5:
            issues.append(f"{br}: {n_seeds} seeds present (expected 5)")
    checks["n_seeds_per_bridge"] = n_seeds_per_bridge
    checks["nan_found"] = nan_found
    checks["one_to_one_max_out_degree_ok"] = one_to_one_deg_ok
    checks["decoded_has_1to2"] = decoded_split_ok
    checks["decoded_has_2to1"] = decoded_merge_ok

    # structural zeros for Connector/ABC must be by degree constraint, not evaluator bug:
    for m in ("Connector-style", "ABCTracer-style"):
        sv = np.asarray(recomputed[m]["split_exact"], dtype=float)
        checks[f"{m}_split_exact_all_zero"] = bool((sv == 0.0).all())
    for m in ("Threshold-MM", "Balanced-OT", "RC-UOT-Q"):
        sv = np.asarray(recomputed[m]["split_exact"], dtype=float)
        checks[f"{m}_split_exact_nonzero_any"] = bool((sv > 0).any())

    # Balanced-OT convergence + threshold cutoff consistency
    bot_ok: list[bool] = []
    cutoff = float(json.loads((STUDY / "calibration" / "selected_threshold.json").read_text(encoding="utf-8"))["global_cutoff_cost"])
    thr_ok = True
    for br in BRIDGES:
        for seed in TEST_SEEDS:
            root = STUDY / "per_seed" / br / f"seed_{seed}"
            if not (root / "seed_summary.json").is_file():
                continue
            s = json.loads((root / "seed_summary.json").read_text(encoding="utf-8"))
            bot_ok.append(bool(s["balanced_ot"]["converged"]))
            # threshold-MM edges must all satisfy C <= cutoff on the frozen cost matrix
            inst = load_frozen_instance(br, seed)
            idx_s = {sid: i for i, sid in enumerate(inst["sids"])}
            idx_t = {tid: j for j, tid in enumerate(inst["tids"])}
            ef = pd.read_csv(root / "edges_Threshold_MM.csv", dtype=str, keep_default_na=False)
            for _, r in ef.iterrows():
                i = idx_s.get(str(r["src_flow_id"])); j = idx_t.get(str(r["dst_flow_id"]))
                if i is None or j is None:
                    continue
                if inst["C"][i, j] > cutoff + 1e-9:
                    thr_ok = False
                    issues.append(f"{br}/seed_{seed}: threshold edge cost {inst['C'][i, j]} > cutoff {cutoff}")
    checks["balanced_ot_all_converged"] = bool(all(bot_ok)) if bot_ok else False
    checks["balanced_ot_n_seeds_checked"] = len(bot_ok)
    checks["threshold_edges_all_within_cutoff"] = thr_ok

    # RC-UOT-Q decoded edges vs frozen plan (mass >= 1e-9)
    rc_ok = True
    for br in BRIDGES:
        for seed in TEST_SEEDS:
            inst = load_frozen_instance(br, seed)
            root = STUDY / "per_seed" / br / f"seed_{seed}"
            ef = pd.read_csv(root / "edges_RC_UOT_Q.csv", dtype=str, keep_default_na=False)
            frozen_edges = sum(1 for i in range(inst["P"].shape[0]) for j in range(inst["P"].shape[1])
                               if inst["P"][i, j] >= FROZEN_PARAMS["uot_decode_threshold"])
            if len(ef) != frozen_edges:
                rc_ok = False
                issues.append(f"{br}/seed_{seed}: RC-UOT-Q edge count {len(ef)} != frozen decoded {frozen_edges}")
    checks["rc_uot_q_frozen_decode_consistent"] = rc_ok

    # aggregated table consistency: recomputed means vs main_structural_comparison.csv
    main_df = pd.read_csv(STUDY / "aggregated" / "main_structural_comparison.csv")
    agg_ok = True
    for br in BRIDGES:
        for m in METHODS:
            row = main_df[(main_df["bridge"] == br) & (main_df["method"] == m)]
            if row.empty:
                continue
            # recomputed per-seed means (mean over templates per seed) -> mean over seeds
            per_seed_means = []
            for seed in TEST_SEEDS:
                root = STUDY / "per_seed" / br / f"seed_{seed}"
                dfm, _ = recompute_per_template(br, seed, m)
                per_seed_means.append(float(dfm["edge_f1"].mean()))
            mean_re = float(np.mean(per_seed_means))
            if abs(mean_re - float(row["edge_f1_mean"].iloc[0])) > 1e-9:
                agg_ok = False
                issues.append(f"{br}/{m}: recomputed edge_f1 mean {mean_re} != table {row['edge_f1_mean'].iloc[0]}")
    checks["aggregated_table_consistent_with_recompute"] = agg_ok
    checks["n_issues"] = len(issues)
    checks["issues"] = issues
    checks["overall_pass"] = len(issues) == 0

    (OUT / "verification_report.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    md_lines = ["# Independent verification report — baseline + mechanism study", "",
                "All metrics recomputed from the raw per-template artifacts (edge CSVs + "
                "frozen label CSVs); aggregated tables were not read as input.", ""]
    for k in ("n_seeds_per_bridge", "artifacts_missing", "nan_found",
              "one_to_one_max_out_degree_ok", "decoded_has_1to2", "decoded_has_2to1",
              "Connector-style_split_exact_all_zero", "ABCTracer-style_split_exact_all_zero",
              "Threshold-MM_split_exact_nonzero_any", "Balanced-OT_split_exact_nonzero_any",
              "RC-UOT-Q_split_exact_nonzero_any", "balanced_ot_all_converged",
              "threshold_edges_all_within_cutoff", "rc_uot_q_frozen_decode_consistent",
              "aggregated_table_consistent_with_recompute", "overall_pass"):
        md_lines.append(f"- **{k}**: `{json.dumps(checks.get(k))}`")
    md_lines += ["", "## Issues", ""]
    if issues:
        for i in issues:
            md_lines.append(f"- {i}")
    else:
        md_lines.append("- none")
    (OUT / "verification_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items() if k != "issues"}, indent=2, default=str))
    print("issues:", issues)
    return 0 if checks["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
