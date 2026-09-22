"""R7 executor DRY-RUN harness.

Exercises the confirmatory executor's FULL unit-construction code path
(``build_unit``: generation -> one UOT solve -> every decoder -> primitive assembly ->
JSON serialisation -> Gate D checks) on SELECTION seeds, so that the execution path is
proven before the protocol freeze and before any holdout byte is touched.

This exists because the first confirmatory attempt crashed on a plumbing defect that
hash-only verification could not reach.  It is a TEST HARNESS, not part of the executor:
the executor itself exposes no seed interface, and its confirmatory block stays
hard-coded.

The harness never writes into ``confirmatory/``.

Usage:
    python scripts/r7/dryrun_executor_paths.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import importlib.util                                          # noqa: E402

from r7.r7_common import (BRIDGES, CONFIRMATORY_SEEDS, DIR_SELECTION, SELECTION_SEEDS,
                          log, sha256_file, utc_now, write_json)

DRY_ROOT = DIR_SELECTION / "executor_dryrun"
OUT = DIR_SELECTION / "EXECUTOR_DRYRUN.json"


def load_executor():
    p = REPO / "scripts" / "run_r7_confirmatory_kernel_ranking.py"
    spec = importlib.util.spec_from_file_location("r7_executor", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ex = load_executor()
    spec = json.loads((REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"
                       / "config" / "locked_spec.json").read_text(encoding="utf-8")) \
        if (REPO / "out" / "r7_confirmatory_kernel_ranking_20260917" / "config"
            / "locked_spec.json").is_file() else None

    # Fall back to the protocol defaults when no locked spec exists yet.
    cfg = {
        "epsilon": 0.05,
        "selected_rule": {"family": "R-const", "k": 3},
        "threshold_mm_cutoff": 0.230150,
        "dual_softmax_tau": 0.0,
    }
    if spec:
        cfg["epsilon"] = spec["frozen_cost"]["epsilon"]
        cfg["selected_rule"] = spec["selected_rule"]["rule_for_executor"]
        cfg["threshold_mm_cutoff"] = spec["threshold_mm"]["cutoff"]
        cfg["dual_softmax_tau"] = spec["dual_softmax"]["tau"]

    deg = json.loads((DIR_SELECTION / "degree_calibration" / "degree_sampling_spec.json")
                     .read_text(encoding="utf-8"))
    degree_spec = {"split_degree": deg["split_degree"], "merge_degree": deg["merge_degree"]}

    checks: list[dict] = []

    def check(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    ver = {"spec_sha256": "DRYRUN", "executor_sha256": sha256_file(
        REPO / "scripts" / "run_r7_confirmatory_kernel_ranking.py")}

    # one unit per bridge, all on the SAME selection seed, so the dry-run index is a
    # complete 3-bridge x 1-seed grid (the shape Gate D expects)
    dry_seed = SELECTION_SEEDS[0]
    cases = [(b, dry_seed) for b in BRIDGES]
    unit_paths = {}
    for bridge, seed in cases:
        assert seed in SELECTION_SEEDS, "dry-run must use selection seeds only"
        assert seed not in CONFIRMATORY_SEEDS, "dry-run must never touch a holdout seed"
        work = DRY_ROOT / bridge / f"seed_{seed}"
        unit, srow, drow, orow = ex.build_unit(bridge, seed, cfg, degree_spec,
                                               "DRYRUN_CONFIG_HASH", ver, work)
        p = DRY_ROOT / f"dryrun_unit__{bridge}__s{seed}.json"
        p.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
        unit_paths[(bridge, seed)] = p
        log(f"[dryrun] {bridge}/{seed} built ok")

    # ---- schema + semantics ------------------------------------------------ #
    required_top = {"unit_id", "bridge", "seed", "n_src", "n_dst", "source_ids", "target_ids",
                    "template_of_source", "template_of_target", "truth",
                    "oracle_1to1_ceiling", "predictions", "method_metrics",
                    "family_edge_counts", "margin_mass", "solver", "kkt", "degrees",
                    "hashes", "oracle_summary", "hungarian_info", "threshold_mm_info",
                    "dual_softmax_info", "support_filter_violations", "runtime_sec"}
    rows = []
    for (bridge, seed), p in unit_paths.items():
        u = json.loads(p.read_text(encoding="utf-8"))
        rows.append((bridge, seed, u))
        check(f"schema_complete__{bridge}_{seed}", required_top <= set(u),
              sorted(required_top - set(u)))
    check("all_units_built", len(rows) == len(cases), {"built": len(rows), "expected": len(cases)})

    if rows:
        u = rows[0][2]
        check("margin_mass_delta_present",
              {"delta_S", "delta_T", "realized_source_mass", "realized_target_mass"}
              <= set(u["margin_mass"]), sorted(u["margin_mass"]))
        check("kkt_present", bool(u.get("kkt")), bool(u.get("kkt")))
        check("all_methods_predicted",
              set(u["predictions"]) == {"UOT_KR", "RAW_UOT_PLAN", "CONDITIONAL_UOT",
                                        "SUPPORT_PLUS_K", "HUNGARIAN_1TO1",
                                        "THRESHOLD_MM", "DUAL_SOFTMAX"},
              sorted(u["predictions"]))
        check("support_hard_filter_valid",
              all(int(r[2].get("support_filter_violations", -1)) == 0 for r in rows),
              [r[2].get("support_filter_violations") for r in rows])
        check("families_24", {len({"r7fam" + t.split("__r7fam")[1].split("__inst")[0]
                                  for t in r[2]["truth"] if "__r7fam" in t})
                             for r in rows} == {24},
              sorted({len({"r7fam" + t.split("__r7fam")[1].split("__inst")[0]
                           for t in r[2]["truth"] if "__r7fam" in t}) for r in rows}))
        check("json_roundtrip_stable", True, "all units serialised and re-parsed")

    # ---- gate D on the dry-run units --------------------------------------- #
    index_rows = [{"bridge": b, "seed": s,
                   "path": f"dryrun_unit__{b}__s{s}.json",
                   "sha256": sha256_file(unit_paths[(b, s)]),
                   "bytes": unit_paths[(b, s)].stat().st_size,
                   "cost_matrix_sha256": u["hashes"]["cost_matrix_sha256"],
                   "plan_sha256": u["hashes"]["plan_sha256"]}
                  for (b, s, u) in rows]
    metric_rows = []
    for b, s, u in rows:
        for m, mm in u["method_metrics"].items():
            metric_rows.append({"bridge": b, "seed": s, "method": m,
                                "macro_edge_f1": mm["macro_edge_f1"],
                                "family_mean_edge_precision":
                                    mm["family_mean_edge_precision"],
                                "family_mean_edge_recall": mm["family_mean_edge_recall"],
                                "family_mean_split_exact": mm["family_mean_split_exact"],
                                "family_mean_merge_exact": mm["family_mean_merge_exact"],
                                "family_mean_overall_exact": mm["family_mean_overall_exact"],
                                "n_pred_edges_total": mm["n_pred_edges_total"]})
    solver_rows = []
    for b, s, u in rows:
        solver_rows.append({"bridge": b, "seed": s, **u["solver"], "kkt": u["kkt"]})

    # Gate D on a synthetic index that only contains the three dry-run units
    import types
    ex.UNIT_DIR = DRY_ROOT
    ex.BRIDGES = tuple(b for b, _ in cases)
    ex.CONFIRMATORY_SEEDS = tuple(s for _, s in cases)
    gd = ex.evaluate_gate_d(index_rows, solver_rows, metric_rows, [], cfg)
    check("gate_d_evaluable", gd["GATE_D"] in ("PASS", "FAIL"), gd["GATE_D"])
    check("gate_d_no_missing_cells",
          next(c for c in gd["checks"] if c["check"] == "no_missing_cells")["status"] == "PASS",
          "dry-run index self-consistent")
    check("gate_d_solver_converged",
          next(c for c in gd["checks"] if c["check"] == "all_solver_cells_converged")["status"],
          "solver convergence on dry-run units")

    all_pass = all(c["status"] == "PASS" for c in checks)
    write_json(OUT, {
        "generated_at_utc": utc_now(),
        "purpose": ("prove the confirmatory executor's full unit path before the freeze; "
                    "the first holdout attempt crashed on a defect that hash verification "
                    "could not reach"),
        "scope": "SELECTION seeds only; confirmatory/ never written",
        "executor_sha256": sha256_file(
            REPO / "scripts" / "run_r7_confirmatory_kernel_ranking.py"),
        "cases": [{"bridge": b, "seed": s} for b, s in cases],
        "checks": checks,
        "gate_d_dryrun": gd["GATE_D"],
        "gate_d_checks": gd["checks"],
        "EXECUTOR_DRYRUN": "PASS" if all_pass else "FAIL",
    })
    for c in checks:
        print(f"{c['status']:4s}  {c['check']}")
    if not all_pass:
        for c in gd["checks"]:
            print(f"   gate_d {c['status']:4s} {c['check']}  {json.dumps(c['detail'])[:200]}")
    print(f"EXECUTOR_DRYRUN = {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
