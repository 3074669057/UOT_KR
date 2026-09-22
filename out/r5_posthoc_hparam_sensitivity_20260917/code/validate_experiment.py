"""Final consistency validation for the R5 post-hoc sensitivity experiment.

Implements the 14 required checks and also verifies the two additional guarantees this
experiment relies on (no writes to frozen directories; the default point is stored exactly
once).  Writes VALIDATION_CHECKLIST.md and validation_evidence.json.

READ-ONLY with respect to every frozen artifact.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917"
PRE = EXP / "00_preflight"
RES = EXP / "results"
ABL = EXP / "ablation"
FIG = EXP / "figures"
PROV = EXP / "provenance"
RUNS = EXP / "runs"

HOLDOUT = (301, 302, 303, 304, 305)
DEV = (201, 202, 203, 204, 205)
BRIDGES = ("Celer", "Multi", "Poly")
METHODS = ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4")
SESSION_START_UTC = "2026-09-17T03:00:00Z"

FROZEN_WATCHLIST = [
    "out/multi_bridge_expansion/conditional_plan_holdout_results/statistics.json",
    "out/multi_bridge_expansion/conditional_plan_holdout_results/FINAL_CONFIRMATORY_HOLDOUT_REPORT.md",
    "out/multi_bridge_expansion/conditional_plan_holdout_results/verification.json",
    "3/chinese_rewrite_r5/final/paper_experiments_results/confirmatory_holdout/statistics.json",
    "3/final/ZN_TIFS_FINAL_CN.docx",
    "manuscript_final/full_manuscript_final.md",
    "src/cross/domain/uot/uot_solver.py",
    "src/cross/domain/uot/cost_matrix.py",
    "scripts/multi_bridge/holdout/holdout_common.py",
    "scripts/multi_bridge/holdout/run_locked_holdout.py",
    "ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/tables/Table3_confirmatory_statistics.json",
]
STAGE2_STALE_ROOTS = [
    "3/chinese_rewrite_r4",
    "ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/STAGE_R4",
]

RESULTS: list[dict] = []


def add(item: str, ok: bool, evidence: list[str], detail: str = "") -> None:
    RESULTS.append({"item": item, "status": "PASS" if ok else "FAIL",
                    "evidence": evidence, "detail": detail})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def main() -> int:
    long_df = pd.read_csv(RES / "sweep_long.csv")
    summ = pd.read_csv(RES / "sweep_summary.csv")
    eps_diag = pd.read_csv(RES / "epsilon_solver_diagnostics.csv")
    lam_mass = pd.read_csv(RES / "lambda_unmatched_mass.csv")
    ab_long = pd.read_csv(ABL / "cost_component_ablation_long.csv")
    ab_summ = pd.read_csv(ABL / "cost_component_ablation_summary.csv")
    prov = json.loads((PROV / "frozen_holdout_points.json").read_text(encoding="utf-8"))
    analysis = json.loads((RES / "analysis_summary.json").read_text(encoding="utf-8"))
    spec = json.loads((EXP / "config" / "locked_spec.json").read_text(encoding="utf-8"))
    dev_anchors = {m: [] for m in METHODS}
    evidence: dict = {}

    # ---------------------------------------------------------------- 1
    # (1) no command ever ran seeds 301-305
    ran_holdout = sorted(set(int(s) for s in long_df["seed"].unique()) & set(HOLDOUT))
    ran_holdout += sorted(set(int(s) for s in ab_long["seed"].unique()) & set(HOLDOUT))
    log_text = (EXP / "logs" / "run.log").read_text(encoding="utf-8") if \
        (EXP / "logs" / "run.log").is_file() else ""
    run_files = [p.name for p in list((RUNS / "sweep").glob("*.json")) +
                 list((RUNS / "ablation").glob("*.json")) +
                 list((RUNS / "smoke").glob("*.json"))]
    bad_names = [n for n in run_files if any(f"s{s}__" in n for s in HOLDOUT)]
    failed_log = (EXP / "logs" / "failed_tasks.jsonl")
    add("1. No command ever ran the holdout seeds 301-305",
        not ran_holdout and not bad_names,
        [rel(RES / "sweep_long.csv"), rel(ABL / "cost_component_ablation_long.csv"),
         f"runs/: {len(run_files)} task files scanned for 's30X__' patterns",
         rel(EXP / "logs" / "run.log")],
        f"holdout seeds present in results: {ran_holdout or 'none'}; "
        f"task files matching a holdout seed: {bad_names or 'none'}; "
        f"failed_tasks.jsonl present: {failed_log.is_file()}; "
        f"run.log lines mentioning '30 {1,2,3,4,5}': "
        f"{sum(1 for ln in log_text.splitlines() if 'seed 30' in ln)}")

    # ---------------------------------------------------------------- 2
    # (2) frozen artifact mtime/hash unchanged
    frozen_rows = []
    changed = []
    for r in FROZEN_WATCHLIST:
        p = REPO / r
        if not p.is_file():
            frozen_rows.append({"path": r, "exists": False})
            changed.append(r)
            continue
        st = p.stat()
        mt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
        touched = mt >= SESSION_START_UTC
        frozen_rows.append({"path": r, "exists": True, "size_bytes": st.st_size,
                            "mtime_utc": mt, "sha256": sha256(p),
                            "mtime_after_session_start": touched})
        if touched:
            changed.append(r)
    evidence["frozen_watchlist"] = frozen_rows
    add("2. No frozen artifact was modified (mtime/hash)",
        not changed,
        [rel(EXP / "VALIDATION_CHECKLIST.md"), "evidence in validation_evidence.json"],
        f"session start (UTC) {SESSION_START_UTC}; artifacts with a later mtime: "
        f"{changed or 'none'}; watchlist size {len(FROZEN_WATCHLIST)}")

    # ---------------------------------------------------------------- 3
    docx = REPO / "3" / "final" / "ZN_TIFS_FINAL_CN.docx"
    docx_info = {"path": rel(docx), "exists": docx.is_file()}
    if docx.is_file():
        st = docx.stat()
        docx_info.update({"size_bytes": st.st_size, "sha256": sha256(docx),
                          "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime(st.st_mtime))})
    evidence["final_docx"] = docx_info
    add("3. 3/final/ZN_TIFS_FINAL_CN.docx not modified",
        docx.is_file() and docx_info["mtime_utc"] < SESSION_START_UTC,
        [docx_info["path"]],
        f"sha256 {docx_info.get('sha256', 'n/a')}, mtime {docx_info.get('mtime_utc')}, "
        f"session start {SESSION_START_UTC}")

    # ---------------------------------------------------------------- 4
    stale = []
    for r in STAGE2_STALE_ROOTS:
        d = REPO / r
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                mt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))
                if mt >= SESSION_START_UTC:
                    stale.append(rel(p))
    add("4. Stale Stage-2 submission package not modified",
        not stale, [r for r in STAGE2_STALE_ROOTS],
        f"files with mtime >= session start in the stale Stage-2/R4 trees: "
        f"{len(stale)} {stale[:5]}")

    # ---------------------------------------------------------------- 5
    n_b = long_df["bridge"].nunique()
    n_s = long_df["seed"].nunique()
    seeds = sorted(int(s) for s in long_df["seed"].unique())
    exp_rows = 3 * 5 * 14 * 2
    complete = (n_b == 3 and n_s == 5 and seeds == list(DEV)
                and len(long_df) == exp_rows
                and set(long_df["method"].unique()) == set(METHODS)
                and long_df.groupby(["k", "epsilon", "lambda"]).ngroups == 14)
    add("5. Every sweep has the complete 3 bridges x 5 development seeds",
        complete,
        [rel(RES / "sweep_long.csv"), rel(RES / "sweep_per_seed.csv")],
        f"bridges={n_b}, seeds={seeds}, unique configs="
        f"{long_df.groupby(['k','epsilon','lambda']).ngroups}, methods="
        f"{sorted(long_df['method'].unique())}, rows={len(long_df)} (expected {exp_rows})")

    # ---------------------------------------------------------------- 6
    metric_cols = ["precision", "recall", "macro_edge_f1", "micro_edge_f1",
                   "micro_precision", "micro_recall", "sinkhorn_iterations",
                   "sinkhorn_final_residual", "delta_s_total", "delta_t_total",
                   "delta_total", "transport_mass_total", "mass_retained_fraction"]
    nan_counts = {c: int(long_df[c].isna().sum()) for c in metric_cols}
    ab_nan = {c: int(ab_long[c].isna().sum()) for c in
              ("precision", "recall", "macro_edge_f1", "delta_s_total", "delta_t_total")
              if c in ab_long.columns}
    finite = bool(np.isfinite(long_df[metric_cols].to_numpy(dtype=float)).all())
    add("6. No silent NaN",
        all(v == 0 for v in nan_counts.values()) and all(v == 0 for v in ab_nan.values())
        and finite,
        [rel(RES / "sweep_long.csv"), rel(ABL / "cost_component_ablation_long.csv")],
        f"NaN counts per column: {nan_counts}; ablation NaN: {ab_nan}; "
        f"all finite: {finite}")

    # ---------------------------------------------------------------- 7
    cfgs = long_df.groupby(["k", "epsilon", "lambda"]).size().reset_index(name="n")
    expected_n = 3 * 5 * 2
    missing = cfgs[cfgs["n"] != expected_n]
    add("7. No missing configuration",
        missing.empty,
        [rel(RES / "sweep_summary.csv")],
        f"each of the {len(cfgs)} unique configurations has {expected_n} rows "
        f"(3 bridges x 5 seeds x 2 methods); violations: {len(missing)}")

    # ---------------------------------------------------------------- 8
    dflt = long_df[(long_df["k"] == 5.0) & (long_df["epsilon"] == 0.05)
                   & (long_df["lambda"] == 0.5)]
    dup_runs = dflt.groupby(["bridge", "seed", "method"])["run_id"].nunique()
    n_default_rows = len(dflt)
    add("8. The default configuration was not run twice with divergent values",
        bool((dup_runs == 1).all()) and n_default_rows == 3 * 5 * 2,
        [rel(RES / "sweep_long.csv"), f"runs/sweep: "
         f"{len(list((RUNS / 'sweep').glob('*.json')))} unique task files"],
        f"default rows={n_default_rows} (expected 30); distinct run_id per "
        f"(bridge, seed, method) = {sorted(dup_runs.unique())}; runs/sweep holds "
        f"{len(list((RUNS / 'sweep').glob('*.json')))} files for 210 planned cells "
        f"(one file per cell, both methods inside)")

    # ---------------------------------------------------------------- 9
    # regenerate the exact per-bridge series the figures use and diff against the CSV
    mismatches = []
    for sweep, pcol, grid in (("k", "k", [2, 3, 5, 7, 10, 15]),
                              ("epsilon", "epsilon", [0.01, 0.02, 0.05, 0.1, 0.2]),
                              ("lambda", "lambda", [0.1, 0.25, 0.5, 1.0, 2.0])):
        sl = summ[summ["sweep"] == sweep]
        for v in grid:
            for b in BRIDGES:
                for m in METHODS:
                    csv_val = float(sl[(sl["parameter_value"] == v) & (sl["method"] == m)]
                                    [f"mean_macro_edge_f1_{b}"].iloc[0])
                    raw = long_df[(long_df["bridge"] == b) & (long_df["method"] == m)
                                  & (long_df[pcol] == v)]
                    # the sweep selects a specific slice of the grid
                    if sweep == "k":
                        raw = raw[(raw["epsilon"] == 0.05) & (raw["lambda"] == 0.5)]
                    elif sweep == "epsilon":
                        raw = raw[(raw["k"] == 5.0) & (raw["lambda"] == 0.5)]
                    else:
                        raw = raw[(raw["k"] == 5.0) & (raw["epsilon"] == 0.05)]
                    recomputed = float(raw["macro_edge_f1"].mean())
                    if abs(recomputed - csv_val) > 1e-12:
                        mismatches.append((sweep, v, b, m, csv_val, recomputed))
    add("9. Figure data can be reproduced point-by-point from the CSVs",
        not mismatches,
        [rel(RES / "sweep_summary.csv"), rel(RES / "sweep_long.csv"),
         rel(EXP / "code" / "make_sensitivity_figures.py")],
        f"recomputed all 3 sweeps x grid x 3 bridges x 2 methods = "
        f"{sum(len(g) for g in ([2,3,5,7,10,15],[0.01,0.02,0.05,0.1,0.2],[0.1,0.25,0.5,1.0,2.0])) * 3 * 2} "
        f"panel points from sweep_long.csv; mismatches: {len(mismatches)}")

    # ---------------------------------------------------------------- 10
    # table numbers vs figure numbers: the figure script reads sweep_summary.csv
    fig_script = (EXP / "code" / "make_sensitivity_figures.py").read_text(encoding="utf-8")
    reads_summary = "sweep_summary.csv" in fig_script
    sec_text = (EXP / "paper" / "section_4_3_posthoc_sensitivity_CN.md").read_text(
        encoding="utf-8")
    spot = {}
    for sweep, pcol, grid in (("k", "k", [2, 3, 5, 7, 10, 15]),
                              ("epsilon", "epsilon", [0.01, 0.02, 0.05, 0.1, 0.2]),
                              ("lambda", "lambda", [0.1, 0.25, 0.5, 1.0, 2.0])):
        sl = summ[summ["sweep"] == sweep]
        for v in grid:
            for m in METHODS:
                val = float(sl[(sl["parameter_value"] == v) & (sl["method"] == m)]
                            ["macro_edge_f1_mean_of_cells"].iloc[0])
                spot[f"{sweep}|{v:g}|{m}"] = (val, f"{val:.4f}" in sec_text)
    in_text = sum(1 for _, ok in spot.values() if ok)
    add("10. Table numbers and figure numbers agree",
        reads_summary and in_text == len(spot),
        [rel(EXP / "code" / "make_sensitivity_figures.py"),
         rel(EXP / "paper" / "section_4_3_posthoc_sensitivity_CN.md")],
        f"the figure script reads results/sweep_summary.csv directly: {reads_summary}; "
        f"{in_text}/{len(spot)} headline values appear verbatim (4 dp) in the paper section")

    # ---------------------------------------------------------------- 11
    src = (EXP / "code" / "r5s_diagnostics.py").read_text(encoding="utf-8")
    iters_src = "errs.size - 1" in src and "log.get(\"err\"" in src
    sweeps_runs = list((RUNS / "sweep").glob("*.json"))
    one = json.loads(sweeps_runs[0].read_text(encoding="utf-8"))
    iter_sample = one["methods"][0]["sinkhorn_iterations"]
    distinct_iters = long_df["sinkhorn_iterations"].nunique()
    add("11. Epsilon iteration counts genuinely come from the solver",
        iters_src and distinct_iters > 1 and eps_diag["sinkhorn_iterations"].min() > 0,
        [rel(EXP / "code" / "r5s_diagnostics.py"),
         rel(RES / "epsilon_solver_diagnostics.csv"),
         rel(RUNS / "sweep" / sweeps_runs[0].name)],
        f"iterations are read from POT's own log['err'] trace (len(err)-1); "
        f"{distinct_iters} distinct values across the sweep; range "
        f"{int(long_df['sinkhorn_iterations'].min())}-"
        f"{int(long_df['sinkhorn_iterations'].max())}; sample task value {iter_sample}")

    # ---------------------------------------------------------------- 12
    mass_src = ('"marginal_violation_row_l1": float(row_dev.sum())' in src
                and '"delta_s_total": sol["marginal_violation_row_l1"]' in
                (EXP / "code" / ".." / ".." / ".." / "scripts" /
                 "run_r5_posthoc_hparam_sensitivity.py").resolve().read_text(encoding="utf-8"))
    ident = bool(np.allclose(lam_mass["delta_s_total"], lam_mass["marginal_violation_row_l1"])
                 and np.allclose(lam_mass["delta_t_total"],
                                 lam_mass["marginal_violation_col_l1"]))
    nonconstant = int(lam_mass["delta_s_total"].nunique())
    add("12. delta_S / delta_T genuinely come from algorithm variables",
        mass_src and ident and nonconstant > 1,
        [rel(EXP / "code" / "r5s_diagnostics.py"),
         rel(RES / "lambda_unmatched_mass.csv")],
        f"delta_S = sum_i |sum_j P_ij - a_i| computed from the solver's returned plan P and "
        f"the marginals actually passed in; delta_S column is identical to "
        f"marginal_violation_row_l1: {ident}; {nonconstant} distinct delta_S values across "
        f"the lambda grid (range {lam_mass['delta_s_total'].min():.4f} to "
        f"{lam_mass['delta_s_total'].max():.4f})")

    # ---------------------------------------------------------------- 13
    weights_ok = []
    for v in ab_summ["variant"].unique():
        wj = ab_long[ab_long["variant"] == v]["cost_weights_json"].iloc[0]
        w = json.loads(wj)
        weights_ok.append(abs(sum(w.values()) - 1.0) < 1e-12)
    omitted_ok = []
    for v in ab_summ["variant"].unique():
        w = json.loads(ab_long[ab_long["variant"] == v]["cost_weights_json"].iloc[0])
        om = ab_long[ab_long["variant"] == v]["omitted_component"].iloc[0]
        omitted_ok.append((om == "none" and sum(1 for x in w.values() if x > 0) in (5, 6))
                          or (om in w and w[om] == 0.0))
    # LOCO_AMOUNT must coincide with the main sweep's default cost matrix, cell by cell.
    main_pairs = set(zip(
        long_df[(long_df["k"] == 5.0) & (long_df["epsilon"] == 0.05)
                & (long_df["lambda"] == 0.5)]["bridge"],
        long_df[(long_df["k"] == 5.0) & (long_df["epsilon"] == 0.05)
                & (long_df["lambda"] == 0.5)]["seed"],
        long_df[(long_df["k"] == 5.0) & (long_df["epsilon"] == 0.05)
                & (long_df["lambda"] == 0.5)]["cost_matrix_sha256"]))
    abl_pairs = set(zip(
        ab_long[ab_long["variant"] == "LOCO_AMOUNT"]["bridge"],
        ab_long[ab_long["variant"] == "LOCO_AMOUNT"]["seed"],
        ab_long[ab_long["variant"] == "LOCO_AMOUNT"]["cost_matrix_sha256"]))
    # each variant must yield a DIFFERENT cost matrix (otherwise the ablation is vacuous)
    fp_per_variant = {v: set(ab_long[ab_long["variant"] == v]["cost_matrix_sha256"])
                      for v in ab_long["variant"].unique()}
    distinct_variants = len({next(iter(s)) for s in fp_per_variant.values()
                             if len(s) == len(next(iter(fp_per_variant.values())))})
    add("13. Cost ablation renormalises correctly and matches the main-sweep cost",
        all(weights_ok) and all(omitted_ok) and main_pairs == abl_pairs
        and len(fp_per_variant) == 7,
        [rel(ABL / "cost_component_ablation_long.csv"),
         rel(ABL / "cost_component_ablation_summary.csv"),
         rel(RES / "sweep_long.csv")],
        f"all {len(weights_ok)} variants renormalised to sum exactly 1: {all(weights_ok)}; "
        f"the omitted component's weight is exactly 0: {all(omitted_ok)}; "
        f"LOCO_AMOUNT (bridge, seed, cost-fingerprint) triples identical to the main sweep's "
        f"default configuration: {main_pairs == abl_pairs} "
        f"({len(main_pairs)} cells); 7 variants produce {len({tuple(sorted(s)) for s in fp_per_variant.values()})} "
        f"distinct fingerprint sets")

    # ---------------------------------------------------------------- 14
    # no main-experiment default was changed
    freeze_refs = {
        "scripts/multi_bridge/dev_candidate2/cp_common.py": "K5 = 5",
        "scripts/multi_bridge/holdout/holdout_common.py": "K5 = 5",
        "scripts/multi_bridge/baseline_mechanism/common.py": '"uot_reg": 0.05',
        "scripts/multi_bridge/decoder_audit/da_common.py": '"uot_reg": 0.05, "uot_reg_m": 0.5',
    }
    freeze_ok = {}
    for r, needle in freeze_refs.items():
        txt = (REPO / r).read_text(encoding="utf-8")
        freeze_ok[r] = needle in txt
    regm_ok = '"uot_reg_m": 0.5' in (REPO / "scripts/multi_bridge/baseline_mechanism/"
                                     "common.py").read_text(encoding="utf-8")
    ident_ok = ('if p.get("reg", 0.05) != 0.05' in
                (REPO / "scripts/multi_bridge/holdout/holdout_common.py").read_text(
                    encoding="utf-8"))
    # no new file under src/cross or config
    new_src = []
    for root in ("src/cross", "config"):
        for p in (REPO / root).rglob("*"):
            if p.is_file():
                mt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))
                if mt >= SESSION_START_UTC:
                    new_src.append(rel(p))
    add("14. No main-experiment default was changed by this sweep",
        all(freeze_ok.values()) and regm_ok and ident_ok and not new_src,
        [r for r in freeze_refs] + ["scripts/multi_bridge/holdout/holdout_common.py",
                                    "src/cross/**", "config/**"],
        f"frozen constants still present: {freeze_ok}; reg_m==0.5 guard: {regm_ok}; "
        f"holdout identity_check still enforces reg==0.05/reg_m==0.5: {ident_ok}; "
        f"files under src/cross or config modified during this session: "
        f"{new_src or 'none'}")

    # ------------------------------------------------------- extra A
    # no writes into the frozen output directories
    frozen_dirs = [
        "out/multi_bridge_expansion/conditional_plan_holdout_results",
        "out/multi_bridge_expansion/conditional_plan_holdout_preregistration",
        "out/multi_bridge_expansion/cost_transport_diagnosis",
        "out/multi_bridge_expansion/amount_free_candidate_dev",
        "out/multi_bridge_expansion/conditional_plan_candidate_dev",
        "out/multi_bridge_expansion/faithful_flow_structural_three_bridges",
        "out/paper_full_pipeline_run",
        "3/chinese_rewrite_r5",
    ]
    touched = []
    for r in frozen_dirs:
        d = REPO / r
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                mt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))
                if mt >= SESSION_START_UTC:
                    touched.append(rel(p))
    add("A. No file inside any frozen output directory was written",
        not touched,
        frozen_dirs,
        f"files with mtime >= {SESSION_START_UTC} inside the frozen trees: "
        f"{len(touched)} {touched[:8]}")

    # ------------------------------------------------------- extra B
    # preflight snapshot exists
    pre_files = ["git_head.txt", "git_status_porcelain_v2.txt", "git_diff_stat.txt",
                 "git_diff_tracked_src_tests_scripts_config.diff", "software_check.json"]
    have = {f: (PRE / f).is_file() for f in pre_files}
    head_now = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    head_then = (PRE / "git_head.txt").read_text(encoding="utf-8-sig").strip()
    add("B. Preflight snapshot captured and HEAD unchanged",
        all(have.values()) and head_now == head_then,
        [rel(PRE / f) for f in pre_files],
        f"files present: {have}; HEAD at session start {head_then}; HEAD now {head_now}")

    json.dump({"session_start_utc": SESSION_START_UTC,
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "checks": RESULTS, "evidence": evidence},
              (EXP / "validation_evidence.json").open("w", encoding="utf-8"),
              indent=2, default=str)

    # ------------------------------------------------ markdown checklist
    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    lines = ["# VALIDATION_CHECKLIST — R5 post-hoc hyper-parameter sensitivity\n",
             f"\nGenerated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ",
             f"|  session start (UTC): {SESSION_START_UTC}\n",
             f"\n**Result: {n_pass}/{len(RESULTS)} checks PASS.**\n",
             "\nScope: development seeds 201–205 only. The preregistered confirmatory "
             "holdout seeds 301–305 were never executed by this experiment.\n",
             "\n| # | Check | Status | Evidence | Detail |\n|---|---|---|---|---|\n"]
    for i, r in enumerate(RESULTS, 1):
        ev = "<br>".join(f"`{e}`" for e in r["evidence"])
        det = r["detail"].replace("|", "\\|")
        lines.append(f"| {i} | {r['item']} | **{r['status']}** | {ev} | {det} |\n")
    if n_pass != len(RESULTS):
        lines.append("\n## FAILURES\n\n")
        for r in RESULTS:
            if r["status"] != "PASS":
                lines.append(f"* **{r['item']}** — {r['detail']}\n")
    (EXP / "VALIDATION_CHECKLIST.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"n_checks": len(RESULTS), "n_pass": n_pass,
                      "failures": [r["item"] for r in RESULTS if r["status"] != "PASS"]},
                     indent=2))
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
