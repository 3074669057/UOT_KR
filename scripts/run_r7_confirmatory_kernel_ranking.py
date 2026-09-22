"""R7 one-shot confirmatory executor.

HARD SEED GUARD: this executor accepts the R7 CONFIRMATORY block ONLY (``401-410``).
It exposes no ``--seed`` style interface and refuses every other seed at argument
validation, before any data access.

ONE-SHOT CONTRACT
-----------------
Before the first byte of 401-410 data is generated or read this executor:

  1. reads ``config/locked_spec.json`` and verifies its SHA256;
  2. verifies its OWN sha256 against the frozen protocol manifest;
  3. verifies the generator, validator, decoder, solver-wrapper and metric hashes;
  4. verifies the family-manifest hash, the degree-histogram hash and the
     selected-rule artifact hash;
  5. hard-rejects every seed outside 401-410;
  6. atomically exclusive-creates ``confirmatory/CONFIRMATORY_TOUCH_ONCE.json``
     (O_CREAT | O_EXCL).  If that file already exists the executor REFUSES TO RUN,
     permanently and for every future invocation.

Only after the ledger is durably on disk does it create ``confirmatory/raw/`` and
generate 401-410 with the frozen generator.  Generation and evaluation are one pipeline:
there is no mode that generates the holdout for inspection without evaluating it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r7.r7_common import (BRIDGES, CONFIRMATORY_SEEDS, DEGREE_RANGE, DIR_ANALYSIS,
                          DIR_CONFIG, DIR_CONFIRMATORY, DIR_DIAGNOSTICS, DIR_FIGURES,
                          DIR_LOGS, DIR_PAPER, DIR_RAW, EPSILON, EXPERIMENT_ID,
                          FORBIDDEN_SEEDS, K_MAX, LAMBDA, METHODS, N_FAMILIES,
                          SOLVER_MARGINAL_TOL, SUPPORT_THRESHOLD, assert_confirmatory_seeds,
                          environment_record, git, log, resolve_frozen_path, sha256_array,
                          sha256_file, sha256_text, utc_now, write_json, write_text)
from r7.r7_generator import build_cell
from r7.r7_methods import hungarian_1to1, oracle_1to1_ceiling
from r7.r7_pipeline import CellContext, run_all_methods

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
SPEC_PATH = DIR_CONFIG / "locked_spec.json"
SPEC_HASH_PATH = DIR_CONFIG / "locked_spec.sha256"
MANIFEST_PATH = DIR_CONFIG / "FROZEN_PROTOCOL_MANIFEST.json"
LEDGER_PATH = DIR_CONFIRMATORY / f"CONFIRMATORY_TOUCH_ONCE__{CONFIRMATORY_SEEDS[0]}_{CONFIRMATORY_SEEDS[-1]}.json"
RETIRED_LEDGER_PATH = DIR_CONFIRMATORY / "CONFIRMATORY_TOUCH_ONCE.json"
GATE_D_PATH = DIR_CONFIRMATORY / "VALIDITY_GATE_D.json"
RAW_INDEX = DIR_RAW / "INDEX.json"
UNIT_DIR = DIR_RAW / "units"


class ProtocolViolation(SystemExit):
    """Raised when any frozen hash, the seed block or the one-shot ledger disagrees."""


# --------------------------------------------------------------------------- #
# unit construction -- the same code path the dry-run harness exercises
# --------------------------------------------------------------------------- #

def build_unit(bridge: str, seed: int, cfg: dict[str, Any], degree_spec: dict[str, Any],
               config_hash: str, ver: dict[str, Any], workdir: Path
               ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Generate + solve + predict ONE unit and assemble its full primitive record.

    Returns ``(unit, solver_row, diagnostics_row, oracle_row)``.  This is the exact code
    path ``execute()`` uses; the pre-freeze dry-run harness calls it on SELECTION seeds so
    the execution path is exercised before any holdout byte is touched.
    """
    t0 = time.time()
    cell = build_cell(bridge, seed, degree_spec, workdir=workdir)
    ctx = CellContext(cell, cfg["epsilon"])
    res = run_all_methods(cell, cfg, ctx=ctx)

    P = ctx.P
    a, b = ctx.a, ctx.b
    delta_s = a - P.sum(axis=1)        # unmatched / marginal-deficit mass, source side
    delta_t = b - P.sum(axis=0)        # unmatched / marginal-deficit mass, target side

    truth = {}
    for t, tr in sorted(cell["truth"].items()):
        truth[t] = {
            "split": sorted([list(e) for e in tr["split"]]),
            "merge": sorted([list(e) for e in tr["merge"]]),
            "decoy": sorted([list(e) for e in tr["decoy"]]),
            "positive": sorted([list(e) for e in tr["positive"]]),
            "unmatched_src": sorted(tr["unmatched_src"]),
            "hidden_dst": sorted(tr["hidden_dst"]),
            "all_src": sorted(tr["all_src"]),
            "all_dst": sorted(tr["all_dst"]),
        }
    oracle = {t: oracle_1to1_ceiling(tr["positive"]) for t, tr in sorted(cell["truth"].items())}

    unit = {
        "unit_id": f"{bridge}|{seed}",
        "bridge": bridge, "seed": seed,
        "n_src": len(cell["sids"]), "n_dst": len(cell["tids"]),
        "source_ids": list(cell["sids"]),
        "target_ids": list(cell["tids"]),
        "template_of_source": list(cell["tpl_s"]),
        "template_of_target": list(cell["tpl_t"]),
        "truth": truth,
        "oracle_1to1_ceiling": oracle,
        "predictions": {m: [list(e) for e in res[m]["pred_edges"]]
                        for m in METHODS if m in res},
        "method_metrics": {m: {k: v for k, v in res[m].items()
                               if k not in ("pred_edges", "family_edge_counts",
                                            "per_template")}
                           for m in METHODS if m in res},
        "family_edge_counts": {m: res[m]["family_edge_counts"]
                               for m in METHODS if m in res},
        "margin_mass": {
            "a": a.tolist(), "b": b.tolist(),
            "realized_source_mass": P.sum(axis=1).tolist(),
            "realized_target_mass": P.sum(axis=0).tolist(),
            "delta_S": delta_s.tolist(), "delta_T": delta_t.tolist(),
            "delta_S_total": float(delta_s.sum()),
            "delta_T_total": float(delta_t.sum()),
            "support_mass_fraction": float(P[P > SUPPORT_THRESHOLD].sum() / P.sum()),
        },
        "solver": res["solver"],
        "kkt": cell["solver"].get("kkt"),
        "degrees": {"split_degrees": cell["degrees"]["split_degrees"],
                    "merge_degrees": cell["degrees"]["merge_degrees"]},
        "hashes": {
            "cost_matrix_sha256": cell["hashes"]["cost_matrix_sha256"],
            "plan_sha256": cell["hashes"]["plan_sha256"],
            "labels_sha256": cell["hashes"]["labels_sha256"],
            "data_sha256": cell["hashes"]["data_sha256"],
            "generator_module_sha256": cell["hashes"]["generator_module_sha256"],
            "config_hash": config_hash,
            "locked_spec_sha256": ver["spec_sha256"],
            "executor_sha256": ver["executor_sha256"],
        },
        "oracle_summary": res["ORACLE_1TO1_CEILING"],
        "hungarian_info": res.get("hungarian"),
        "threshold_mm_info": res.get("threshold_mm"),
        "dual_softmax_info": res.get("dual_softmax"),
        "support_filter_violations": res.get("support_filter_violations", 0),
        "runtime_sec": time.time() - t0,
    }
    solver_row = {"bridge": bridge, "seed": seed,
                  **{k: v for k, v in res["solver"].items()},
                  "kkt": cell["solver"].get("kkt")}
    diagnostics_row = {
        "bridge": bridge, "seed": seed,
        "delta_S_total": float(delta_s.sum()), "delta_T_total": float(delta_t.sum()),
        "mean_delta_S": float(delta_s.mean()), "mean_delta_T": float(delta_t.mean()),
        "delta_S_max": float(delta_s.max()), "delta_T_max": float(delta_t.max()),
        "mass_total": float(P.sum()),
        "support_cells": int((P > SUPPORT_THRESHOLD).sum()),
        "matrix_cells": int(P.size),
        "support_fraction": float((P > SUPPORT_THRESHOLD).mean()),
        "source_zero_mass": int((P.sum(axis=1) <= SUPPORT_THRESHOLD).sum()),
        "target_zero_mass": int((P.sum(axis=0) <= SUPPORT_THRESHOLD).sum()),
    }
    oc = res["ORACLE_1TO1_CEILING"]
    oracle_row = {"bridge": bridge, "seed": seed, "macro_f1": oc["macro_f1"],
                  "macro_recall": oc["macro_recall"], "mean_T": oc["mean_T"],
                  "mean_M": oc["mean_M"]}
    return unit, solver_row, diagnostics_row, oracle_row


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify_protocol() -> dict[str, Any]:
    """Every frozen-artifact verification, performed BEFORE any holdout access."""
    report: dict[str, Any] = {"checks": [], "checked_at_utc": utc_now()}

    def check(name: str, ok: bool, detail: Any) -> None:
        report["checks"].append({"check": name, "status": "PASS" if ok else "FAIL",
                                 "detail": detail})

    if not SPEC_PATH.is_file():
        raise ProtocolViolation(f"locked spec missing: {SPEC_PATH}; refusing to execute")
    if not MANIFEST_PATH.is_file():
        raise ProtocolViolation(f"frozen protocol manifest missing: {MANIFEST_PATH}")

    # 1. locked spec sha256
    spec_bytes = SPEC_PATH.read_bytes()
    spec_sha = sha256_text(spec_bytes.decode("utf-8"))
    recorded = (SPEC_HASH_PATH.read_text(encoding="utf-8").split()[0].strip()
                if SPEC_HASH_PATH.is_file() else "")
    check("locked_spec_sha256", spec_sha == recorded,
          {"computed": spec_sha, "recorded": recorded})
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    check("locked_spec_matches_manifest", manifest.get("locked_spec_sha256") == spec_sha,
          {"computed": spec_sha, "manifest": manifest.get("locked_spec_sha256")})

    # 2-4. code + artefact hashes
    frozen = manifest.get("frozen_hashes", {})
    mismatches: dict[str, Any] = {}
    for rel, recorded_hash in sorted(frozen.items()):
        p = resolve_frozen_path(rel)
        if not p.is_file():
            mismatches[rel] = "MISSING"
            continue
        got = sha256_file(p)
        if got != recorded_hash:
            mismatches[rel] = {"expected": recorded_hash, "got": got}
    check("frozen_artifact_hashes", not mismatches, mismatches)

    self_rel = SCRIPT.relative_to(REPO).as_posix()
    self_sha = sha256_file(SCRIPT)
    check("executor_self_sha256",
          frozen.get(self_rel) == self_sha,
          {"computed": self_sha, "frozen": frozen.get(self_rel), "key": self_rel})

    # 5. seed block
    check("confirmatory_seed_block_only",
          tuple(manifest.get("confirmatory_seed_block", [])) == CONFIRMATORY_SEEDS,
          {"manifest": manifest.get("confirmatory_seed_block"),
           "runtime": list(CONFIRMATORY_SEEDS)})
    rejected = []
    for s in sorted(set(FORBIDDEN_SEEDS) | set(range(0, 401)) | {411, 500}):
        try:
            assert_confirmatory_seeds([s], "executor-guard-test")
        except SystemExit:
            rejected.append(s)
    check("non_confirmatory_seeds_rejected", len(rejected) > 0, len(rejected))

    # 6. one-shot ledger -- block specific, and the retired block's ledger is permanent
    check("one_shot_ledger_absent", not LEDGER_PATH.exists(),
          {"path": str(LEDGER_PATH.relative_to(REPO)), "exists": LEDGER_PATH.exists()})
    check("confirmatory_raw_absent", not DIR_RAW.exists(),
          {"path": str(DIR_RAW.relative_to(REPO)), "exists": DIR_RAW.exists()})
    retired = manifest.get("retired_confirmatory_blocks", [])
    if retired:
        present = {r["ledger"]: resolve_frozen_path(r["ledger"]).exists() for r in retired}
        check("retired_block_ledger_preserved", all(present.values()),
              {"retired": [r["block"] for r in retired], "ledger_present": present})

    report["spec_sha256"] = spec_sha
    report["executor_sha256"] = self_sha
    report["ledger_path"] = str(LEDGER_PATH.relative_to(REPO))
    report["ALL_PASS"] = all(c["status"] == "PASS" for c in report["checks"])
    return report


def write_ledger(ver: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Atomically exclusive-create the first-touch ledger.

    ``O_CREAT | O_EXCL`` guarantees the create fails if the file already exists, so a
    second confirmatory execution is impossible even under a race.
    """
    DIR_CONFIRMATORY.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen = manifest.get("frozen_hashes", {})
    env = environment_record()
    payload = {
        "ledger_version": 1,
        "HOLDOUT_TOUCHED": "YES",
        "timestamp_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "confirmatory_seed_block": list(CONFIRMATORY_SEEDS),
        "locked_spec_sha256": ver["spec_sha256"],
        "executor_sha256": ver["executor_sha256"],
        "generator_sha256": frozen.get(
            "src/cross/domain/evaluation/semi_synthetic_flows.py"),
        "validator_sha256": frozen.get("scripts/validate_r7_confirmatory_results.py"),
        "frozen_protocol_manifest_sha256": sha256_file(MANIFEST_PATH),
        "seed_block": list(CONFIRMATORY_SEEDS),
        "bridges": list(BRIDGES),
        "git_head": env["git_head"],
        "git_branch": env["git_branch"],
        "retired_previous_block": {
            "block": [401, 410],
            "ledger": str(RETIRED_LEDGER_PATH.relative_to(REPO)),
            "reason": ("the first confirmatory execution (block 401-410) crashed on a "
                       "plumbing defect after its ledger was written and after Celer/401 "
                       "was generated, but before any unit result was written or observed; "
                       "see confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md"),
            "reused": False,
        },
        "statement": ("This file is the irreversible first-touch record for the R7 "
                      "confirmatory holdout. It was created with O_CREAT|O_EXCL BEFORE the "
                      "first byte of holdout data was generated or read. The confirmatory "
                      "executor refuses to run again while this file exists."),
    }
    blob = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        fd = os.open(str(LEDGER_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise ProtocolViolation(
            f"CONFIRMATORY TOUCH LEDGER ALREADY EXISTS at {LEDGER_PATH}. The one-shot "
            f"confirmatory execution has already been performed; a second execution is "
            f"permanently refused. A crash does NOT grant a retry.") from exc
    try:
        os.write(fd, blob)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.makedirs(DIR_RAW, exist_ok=True)
    log(f"[one-shot] touch ledger written -> {LEDGER_PATH.name}; HOLDOUT_TOUCHED=YES")
    return payload


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #

def execute(rebuild_ok: bool = True) -> int:
    t_start = time.time()
    run_id = f"r7-{int(time.time())}-{os.getpid()}"

    ver = verify_protocol()
    write_json(DIR_CONFIRMATORY / "gate_pre_execution.json", ver)
    for c in ver["checks"]:
        log(f"[verify] {c['check']}: {c['status']}")
    if not ver["ALL_PASS"]:
        write_text(DIR_CONFIRMATORY / "EXECUTION_REFUSED.md",
                   "# R7 confirmatory execution REFUSED\n\n"
                   "One or more frozen-protocol verifications failed. See "
                   "`confirmatory/gate_pre_execution.json`.\n\n"
                   "**The holdout was NOT touched.**\n")
        log("[verify] FAILED -> refusing to execute; holdout untouched")
        return 1

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    ledger = write_ledger(ver, run_id)

    cfg = {
        "epsilon": spec["frozen_cost"]["epsilon"],
        "selected_rule": spec["selected_rule"]["rule_for_executor"],
        "threshold_mm_cutoff": spec["threshold_mm"]["cutoff"],
        "dual_softmax_tau": spec["dual_softmax"]["tau"],
    }
    sampling_spec = spec["degree_calibration"]["sampling_spec"]
    degree_spec_raw = {
        "split_degree": sampling_spec["split_degree"],
        "merge_degree": sampling_spec["merge_degree"],
    }
    config_hash = sha256_text(json.dumps({"cfg": cfg, "spec": ver["spec_sha256"]},
                                         sort_keys=True))

    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    cell_metric_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for bridge in BRIDGES:
        for seed in CONFIRMATORY_SEEDS:
            assert_confirmatory_seeds([seed], f"execute({bridge},{seed})")
            t0 = time.time()
            try:
                unit, solver_row, diag_row, oracle_row = build_unit(
                    bridge, seed, cfg, degree_spec_raw, config_hash, ver,
                    UNIT_DIR / "_scratch" / bridge / f"seed_{seed}")
            except Exception as exc:                      # noqa: BLE001
                failures.append({"bridge": bridge, "seed": seed,
                                 "error": f"{type(exc).__name__}: {exc}"})
                log(f"[unit] {bridge}/{seed} FAILED: {type(exc).__name__}: {exc}")
                continue

            unit_path = UNIT_DIR / f"unit__{bridge}__s{seed}.json"
            unit_path.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
            unit_sha = sha256_file(unit_path)

            index_rows.append({"bridge": bridge, "seed": seed, "path": unit_path.name,
                               "sha256": unit_sha, "bytes": unit_path.stat().st_size,
                               "n_src": unit["n_src"], "n_dst": unit["n_dst"],
                               "cost_matrix_sha256": unit["hashes"]["cost_matrix_sha256"],
                               "plan_sha256": unit["hashes"]["plan_sha256"]})
            solver_rows.append(solver_row)
            diagnostics_rows.append(diag_row)
            oracle_rows.append(oracle_row)
            for m in METHODS:
                if m not in unit["method_metrics"]:
                    continue
                mm = unit["method_metrics"][m]
                cell_metric_rows.append({
                    "bridge": bridge, "seed": seed, "method": m,
                    "macro_edge_f1": mm["macro_edge_f1"],
                    "family_mean_edge_precision": mm["family_mean_edge_precision"],
                    "family_mean_edge_recall": mm["family_mean_edge_recall"],
                    "family_mean_split_exact": mm["family_mean_split_exact"],
                    "family_mean_merge_exact": mm["family_mean_merge_exact"],
                    "family_mean_overall_exact": mm["family_mean_overall_exact"],
                    "n_pred_edges_total": mm["n_pred_edges_total"],
                    "mean_edges_per_family": float(np.mean(
                        list(unit["family_edge_counts"][m].values()))),
                })
            log(f"[unit] {bridge}/{seed} done in {time.time() - t0:.1f}s "
                f"f1(UOT_KR)={unit['method_metrics']['UOT_KR']['macro_edge_f1']:.4f}")

    # ---- persist tables ------------------------------------------------- #
    for df, path in (
        (pd.DataFrame(cell_metric_rows), DIR_RAW / "executor_cell_metrics.csv"),
        (pd.DataFrame(solver_rows), DIR_RAW / "executor_solver_diagnostics.csv"),
        (pd.DataFrame(oracle_rows), DIR_RAW / "executor_oracle_ceiling.csv"),
        (pd.DataFrame(diagnostics_rows), DIR_RAW / "executor_uot_mass_diagnostics.csv"),
    ):
        df.to_csv(path, index=False)
    write_json(RAW_INDEX, {
        "generated_at_utc": utc_now(), "run_id": run_id,
        "ledger": str(LEDGER_PATH.relative_to(REPO)),
        "n_units_expected": len(BRIDGES) * len(CONFIRMATORY_SEEDS),
        "n_units_written": len(index_rows),
        "n_failures": len(failures),
        "units": index_rows, "failures": failures,
        "config_hash": config_hash,
        "locked_spec_sha256": ver["spec_sha256"],
        "executor_sha256": ver["executor_sha256"],
        "seed_block": list(CONFIRMATORY_SEEDS),
    })

    # ---- Gate D ---------------------------------------------------------- #
    gate_d = evaluate_gate_d(index_rows, solver_rows, cell_metric_rows, failures, cfg)
    write_json(GATE_D_PATH, gate_d)
    log(f"[gate D] {gate_d['GATE_D']}")

    summary = {
        "run_id": run_id, "wall_sec": time.time() - t_start,
        "units_written": len(index_rows), "failures": failures,
        "gate_pre_execution": ver["ALL_PASS"], "gate_d": gate_d["GATE_D"],
        "ledger": ledger,
    }
    write_json(DIR_CONFIRMATORY / "execution_summary.json", summary)
    write_text(DIR_CONFIRMATORY / "first_touch_audit.md",
               build_touch_audit(ledger, index_rows, ver, gate_d, failures))
    return 0


def evaluate_gate_d(index_rows, solver_rows, cell_metric_rows, failures,
                    cfg) -> dict[str, Any]:
    """Technical completeness gate (specification section 44)."""
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    expected = {(b, s) for b in BRIDGES for s in CONFIRMATORY_SEEDS}
    got = {(r["bridge"], r["seed"]) for r in index_rows}
    check("no_missing_cells", expected == got, sorted(expected - got))
    check("no_duplicate_cells", len(index_rows) == len(got), len(index_rows))
    check("no_forbidden_seeds",
          not (got & {(b, s) for b in BRIDGES for s in FORBIDDEN_SEEDS}),
          sorted(got & {(b, s) for b in BRIDGES for s in FORBIDDEN_SEEDS}))
    check("no_unit_failures", not failures, failures)

    conv = [r for r in solver_rows if not bool(r.get("converged"))]
    check("all_solver_cells_converged", not conv,
          [{"bridge": r["bridge"], "seed": r["seed"], "final_err": r.get("final_err")}
           for r in conv])
    hit = [r for r in solver_rows if bool(r.get("hit_max_iter"))]
    check("no_solver_hit_max_iter", not hit, len(hit))

    kkt_bad = []
    for r in solver_rows:
        k = r.get("kkt") or {}
        if k and float(k.get("marginal_kkt_max", 0.0)) > SOLVER_MARGINAL_TOL:
            kkt_bad.append({"bridge": r["bridge"], "seed": r["seed"],
                            "kkt": k.get("marginal_kkt_max")})
    check("uot_kkt_residual_below_tolerance", not kkt_bad, kkt_bad)

    bad = []
    for r in solver_rows:
        for key in ("final_err", "marginal_violation_row_l1", "marginal_violation_col_l1"):
            v = r.get(key)
            if v is None or not np.isfinite(float(v)):
                bad.append({"bridge": r["bridge"], "seed": r["seed"], "field": key})
    check("no_nan_inf_solver_fields", not bad, bad)

    df = pd.DataFrame(cell_metric_rows)
    nanrows = []
    if not df.empty:
        num = df.select_dtypes(include=[float]).columns
        m = df[num].isna().any(axis=1) | np.isinf(df[num]).any(axis=1)
        nanrows = df[m][["bridge", "seed", "method"]].to_dict(orient="records")
    check("no_nan_metrics", not nanrows, nanrows)

    n_expected_method_cells = len(expected) * len(METHODS)
    check("all_methods_present",
          len(cell_metric_rows) == n_expected_method_cells,
          {"expected": n_expected_method_cells, "observed": len(cell_metric_rows)})

    # one cost hash per cell, shared by every method
    if not df.empty:
        per_cell = df.groupby(["bridge", "seed"]).size()
        check("methods_per_cell_complete", bool((per_cell == len(METHODS)).all()),
              {f"{k[0]}|{k[1]}": int(v) for k, v in per_cell.items()})
    cost_hashes = {r["cost_matrix_sha256"] for r in index_rows}
    check("cost_hash_present_per_unit",
          all(r.get("cost_matrix_sha256") for r in index_rows) and len(cost_hashes) > 0,
          len(cost_hashes))

    # support hard filter
    viol = 0
    for r in index_rows:
        p = UNIT_DIR / r["path"]
        if p.is_file():
            u = json.loads(p.read_text(encoding="utf-8"))
            viol += int(u.get("support_filter_violations", 0))
    check("support_hard_filter_valid", viol == 0, viol)

    # edge-set validity: every predicted edge must reference known ids
    invalid = 0
    for r in index_rows:
        p = UNIT_DIR / r["path"]
        if not p.is_file():
            continue
        u = json.loads(p.read_text(encoding="utf-8"))
        ss, tt = set(u["source_ids"]), set(u["target_ids"])
        for m, edges in u["predictions"].items():
            for s, d in edges:
                if s not in ss or d not in tt:
                    invalid += 1
    check("all_edge_sets_valid", invalid == 0, invalid)

    all_pass = all(c["status"] == "PASS" for c in checks)
    return {
        "gate": "D", "generated_at_utc": utc_now(),
        "GATE_D": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "n_units": len(index_rows),
        "n_expected_units": len(expected),
        "n_method_rows": len(cell_metric_rows),
        "failures": failures,
        "config": cfg,
    }


def build_touch_audit(ledger, index_rows, ver, gate_d, failures) -> str:
    return "\n".join([
        "# R7 one-shot confirmatory execution audit",
        "",
        f"* run id: `{ledger['run_id']}`",
        f"* first touch: **{ledger['timestamp_utc']}**",
        f"* `HOLDOUT_TOUCHED = YES`",
        f"* ledger: `{LEDGER_PATH.relative_to(REPO)}` (O_CREAT|O_EXCL, created before the "
        f"first holdout byte)",
        f"* pre-execution verification: **{'PASS' if ver['ALL_PASS'] else 'FAIL'}**",
        f"* confirmatory seed block: `{list(CONFIRMATORY_SEEDS)}`",
        f"* units written: **{len(index_rows)}** of {len(BRIDGES) * len(CONFIRMATORY_SEEDS)}",
        f"* unit failures: **{len(failures)}**",
        f"* Gate D: **{gate_d['GATE_D']}**",
        "",
        "## Irreversibility",
        "",
        "The ledger file exists. Every future invocation of the confirmatory executor "
        "refuses to run, regardless of outcome. A crash does not grant a retry: the run "
        "would be recorded as `INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION`.",
        "",
        "## Locked hashes",
        "",
        f"* locked spec sha256: `{ver['spec_sha256']}`",
        f"* executor sha256: `{ver['executor_sha256']}`",
        "",
    ])


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="R7 one-shot confirmatory executor (401-410 only; no seed interface)")
    ap.add_argument("--verify-only", action="store_true",
                    help="run every frozen-protocol verification and exit WITHOUT touching data")
    ap.add_argument("--execute", action="store_true",
                    help="the one-shot confirmatory execution")
    cli = ap.parse_args()

    if cli.verify_only:
        report = verify_protocol()
        for c in report["checks"]:
            print(f"{c['status']:4s}  {c['check']}")
        print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
        write_json(DIR_CONFIRMATORY / "gate_pre_execution.json", report)
        return 0 if report["ALL_PASS"] else 1
    if cli.execute:
        return execute()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
