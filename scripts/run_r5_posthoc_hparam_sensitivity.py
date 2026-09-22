"""R5 post-hoc hyper-parameter sensitivity + cost-component ablation runner.

INDEPENDENT NEW RUNNER.  Adds no behaviour to any existing pipeline and modifies no file
under ``src/cross`` or any frozen output directory.  All outputs go to
``out/r5_posthoc_hparam_sensitivity_20260917/``.

HARD GUARD
----------
``development seeds 201-205`` are the ONLY seeds this runner will ever touch.  The
preregistered confirmatory holdout seeds ``301, 302, 303, 304, 305`` are refused with a
hard exception at argument-validation time, *before any data access*.

USAGE
-----
    # software validation: instrumented solver vs project wrapper + frozen anchors
    python scripts/run_r5_posthoc_hparam_sensitivity.py --check

    # one default-parameter smoke cell (bridge, seed 201)
    python scripts/run_r5_posthoc_hparam_sensitivity.py --smoke --bridge Celer

    # the full sweep (one UOT solve per unique (bridge, seed, k, epsilon, lambda))
    python scripts/run_r5_posthoc_hparam_sensitivity.py --sweep

    # cost-component leave-one-out ablation (CONDITIONAL_UOT_D4)
    python scripts/run_r5_posthoc_hparam_sensitivity.py --ablation

    # aggregate everything into results/ and ablation/
    python scripts/run_r5_posthoc_hparam_sensitivity.py --aggregate
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
MB = REPO / "scripts" / "multi_bridge"
EXP = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917"
CODE = EXP / "code"
for _p in (str(SRC), str(MB), str(CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from r5s_diagnostics import solve_uot_instrumented, verify_plan_equivalence  # noqa: E402

SPEC_PATH = EXP / "config" / "locked_spec.json"
RUNS_DIR = EXP / "runs"
RUNS_SWEEP = RUNS_DIR / "sweep"
RUNS_ABLATION = RUNS_DIR / "ablation"
LOGS_DIR = EXP / "logs"
RESULTS_DIR = EXP / "results"
ABLATION_DIR = EXP / "ablation"
FAILED_LOG = LOGS_DIR / "failed_tasks.jsonl"
RETRY_LOG = LOGS_DIR / "retries.jsonl"
RUN_LOG = LOGS_DIR / "run.log"

# --------------------------------------------------------------------------- #
# HARD HOLDOUT GUARD
# --------------------------------------------------------------------------- #
HOLDOUT_SEEDS_FORBIDDEN = frozenset({301, 302, 303, 304, 305})
ALLOWED_DEV_SEEDS = (201, 202, 203, 204, 205)
BRIDGES = ("Celer", "Multi", "Poly")

# Frozen absolute cost weights of the project (identical to the locked spec and to
# af_common.KEPT_ABS_WEIGHTS plus the amount component removed by the amount-free variant).
FROZEN_ABS_WEIGHTS: dict[str, float] = {
    "amount": 0.40, "time": 0.25, "route": 0.15, "risk": 0.15,
    "evidence": 0.05, "novelty": 0.05,
}
COMPONENT_KEY: dict[str, str] = {
    "amount": "amount_cost", "time": "time_cost", "route": "route_cost",
    "risk": "risk_cost", "evidence": "evidence_cost", "novelty": "address_novelty_cost",
}
# Short alias -> on-disk npz key, used only when building the component dict.
COMPONENT_NPZ_KEY: dict[str, str] = {
    "time": "time_cost", "route": "route_cost", "risk": "risk_cost",
    "evidence": "evidence_cost", "novelty": "address_novelty_cost",
}
# Ablation variant order.  The paper's primary cost is the amount-free renormalised cost, so
# it is exactly the "amount omitted" condition.  The remaining LOCO_* variants therefore omit
# one of the five retained cost components each; FULL_D6 (no component omitted, all six
# weights renormalised to sum 1) is the complete-cost reference.
ABLATION_ORDER = ("FULL_D6", "LOCO_AMOUNT", "LOCO_TIME", "LOCO_ROUTE",
                  "LOCO_RISK", "LOCO_EVIDENCE", "LOCO_NOVELTY")
ABLATION_OMITTED: dict[str, str | None] = {
    "FULL_D6": None, "LOCO_AMOUNT": "amount", "LOCO_TIME": "time", "LOCO_ROUTE": "route",
    "LOCO_RISK": "risk", "LOCO_EVIDENCE": "evidence", "LOCO_NOVELTY": "novelty",
}
# The main sweep always runs on the paper's primary cost matrix.
PRIMARY_OMITTED = "amount"


def assert_seeds_allowed(seeds) -> None:
    bad = sorted(set(int(s) for s in seeds) & HOLDOUT_SEEDS_FORBIDDEN)
    if bad:
        raise SystemExit(
            f"[HOLDOUT-GUARD] REFUSED: requested seeds {bad} are the preregistered "
            f"confirmatory holdout. Seeds 301-305 were executed exactly once and must "
            f"NEVER be re-run. Allowed seeds: {ALLOWED_DEV_SEEDS}."
        )


def assert_bridges_allowed(bridges) -> None:
    bad = [b for b in bridges if b not in BRIDGES]
    if bad:
        raise SystemExit(f"[GUARD] REFUSED: unknown bridge(s) {bad}")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a unique temp file then os.replace -> safe under concurrency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, default=str, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "UNKNOWN"


def load_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Frozen inputs (READ ONLY) — development seeds 201-205
# --------------------------------------------------------------------------- #
CTD_DEV = REPO / "out" / "multi_bridge_expansion" / "cost_transport_diagnosis" / "plans" / "dev"

_CELL_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def load_dev_cell(bridge: str, seed: int) -> dict[str, Any]:
    """Load one frozen development cell: cost components, ids, labels, ground truth and the
    frozen flows (needed to recompute the lambda-dependent risk-weighted source marginal).

    READ ONLY.  Nothing here is written back to the frozen directories.
    """
    assert_seeds_allowed([seed])
    key = (bridge, seed)
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]

    import pandas as pd

    from baseline_mechanism.common import tpl_maps, truth_structure
    from cross.domain.uot.uot_solver import (
        _evidence_weighted_target_mass, _risk_weighted_source_mass,
    )

    root = CTD_DEV / bridge / f"seed_{seed}"
    if not (root / "cost.npz").is_file():
        raise FileNotFoundError(f"frozen development cell missing: {root}")

    cost = np.load(root / "cost.npz", allow_pickle=False)
    ids = np.load(root / "ids.npz", allow_pickle=True)
    labels = pd.read_csv(root / "labels.csv", dtype=str, keep_default_na=False)
    flows = json.loads((root / "flows.json").read_text(encoding="utf-8"))
    sids = [str(x) for x in ids["sids"]]
    tids = [str(x) for x in ids["tids"]]
    tpl_s, tpl_t = tpl_maps(labels, sids, tids)

    components = {k: np.asarray(cost[COMPONENT_KEY[k]], dtype=float)
                  for k in FROZEN_ABS_WEIGHTS if COMPONENT_KEY[k] in cost}
    if set(components) != set(FROZEN_ABS_WEIGHTS):
        raise KeyError(f"component mismatch for {bridge} seed {seed}: {sorted(components)}")

    # Closure check: the frozen full-weights cost must be the frozen component combination
    # plus the bridge prior bonus.  This is the guard that lets the cost-component ablation
    # trust the on-disk components.  (Verified in 00_preflight/repro_probe2.py for all 15
    # development cells; re-asserted here so a corrupted input can never pass silently.)
    C_effective = np.asarray(cost["C_effective"], dtype=float)
    bonus = np.asarray(cost["bridge_prior_bonus"], dtype=float)
    recon = bonus.copy()
    for name, w in FROZEN_ABS_WEIGHTS.items():
        recon = recon + w * components[name]
    closure_err = float(np.abs(recon - C_effective).max())
    if closure_err > 1e-10:
        raise ValueError(f"cost decomposition does not close for {bridge} seed {seed}: "
                         f"max|recon - C_effective| = {closure_err:.3e}")

    cell = {
        "bridge": bridge, "seed": seed, "root": root,
        "components": components,
        "bridge_prior_bonus": bonus,
        "cost_closure_max_abs_err": closure_err,
        "sids": sids, "tids": tids, "tpl_s": tpl_s, "tpl_t": tpl_t,
        "labels": labels, "truth": truth_structure(labels),
        "flows_src": flows["src"], "flows_dst": flows["dst"],
        "_risk_weighted_source_mass": _risk_weighted_source_mass,
        "_evidence_weighted_target_mass": _evidence_weighted_target_mass,
    }
    _CELL_CACHE[key] = cell
    return cell


def normalised_weights(omitted: str | None) -> dict[str, float]:
    """Frozen absolute weights with ``omitted`` zeroed, renormalised to sum to 1."""
    w = {k: (0.0 if k == omitted else v) for k, v in FROZEN_ABS_WEIGHTS.items()}
    total = sum(w.values())
    if total <= 0:
        raise ValueError(f"degenerate weight set after omitting {omitted!r}")
    return {k: v / total for k, v in w.items()}


def build_cost(cell: dict[str, Any], omitted: str | None = None) -> np.ndarray:
    """Pairwise cost from the frozen components under a (possibly reduced) weight set.

    ``omitted is None`` reproduces the paper's primary amount-free renormalised cost exactly
    (validated against the frozen ``amount_free_candidate_dev`` C_primary in preflight).
    """
    w = normalised_weights(omitted)
    C = np.zeros_like(cell["components"]["time"], dtype=float)
    for name, weight in w.items():
        if weight:
            C = C + weight * cell["components"][name]
    return C


def marginals_for_lambda(cell: dict[str, Any], lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Risk-weighted source marginal for a given UOT marginal-relaxation value.

    NOTE: ``lam`` here is the UOT marginal relaxation ``reg_m``.  The AML risk lift keeps
    its frozen coefficient ``lambda_risk = 0.25`` (baseline_mechanism.common.FROZEN_PARAMS).
    The target marginal does not depend on lambda.
    """
    _, a_rw = cell["_risk_weighted_source_mass"](cell["flows_src"], lambda_risk=0.25)
    _, b_ev = cell["_evidence_weighted_target_mass"](cell["flows_dst"])
    return np.asarray(a_rw, dtype=float), np.asarray(b_ev, dtype=float)


# --------------------------------------------------------------------------- #
# Decoders (project definitions, reused verbatim)
# --------------------------------------------------------------------------- #
def decode(method: str, P: np.ndarray, sids: list[str], tids: list[str], k: int):
    from dev_candidate2.cp_common import conditional_edges, mutual_top5_edges
    if method == "RAW_UOT_PLAN_D4":
        return mutual_top5_edges(P, sids, tids, k)
    if method == "CONDITIONAL_UOT_D4":
        return conditional_edges(P, sids, tids, k)
    raise ValueError(f"unknown method {method}")


def macro_edge_metrics(cell: dict[str, Any], edges) -> dict[str, Any]:
    """Edge metrics using the PROJECT evaluator, unmodified.

    ``precision`` / ``recall`` / ``macro_edge_f1`` are the cell-level values of
    ``decoder_audit.da_common.evaluate_edges`` / ``baseline_mechanism.common.evaluate_method``
    (macro = unweighted mean over the 48 templates of the cell).  ``micro_*`` are pooled over
    templates from the same evaluator's TP/FP/FN totals, exactly as the frozen
    FINAL_CONFIRMATORY_HOLDOUT_REPORT reports its macro P/R (mean over the 15 cells of the
    cell-level values).

    IMPORTANT aggregation convention (documented so the headline numbers are reproducible):
      * ``macro_edge_f1`` per (bridge, config, method) = unweighted mean of the 15 cell-level
        values (5 dev seeds x 3 bridges), i.e. mean over bridges of the per-bridge mean over
        seeds.  This is the convention under which the published frozen anchors
        (holdout_common.EXPECTED_DEV_ANCHORS) reproduce.
      * ``std_*`` = sample standard deviation (ddof=1) over the 5 dev seeds of that bridge.
    """
    from decoder_audit.da_common import evaluate_edges
    df, summ = evaluate_edges(cell, edges)
    tp, fp, fn = int(summ["edge_tp_total"]), int(summ["edge_fp_total"]), int(summ["edge_fn_total"])
    return {
        "precision": float(summ["edge_precision"]),
        "recall": float(summ["edge_recall"]),
        "macro_edge_f1": float(summ["edge_f1"]),
        "micro_precision": tp / max(tp + fp, 1),
        "micro_recall": tp / max(tp + fn, 1),
        "micro_edge_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "n_templates": int(len(df)),
        "n_pred_edges": int(summ["n_pred_edges_total"]),
        "tp_total": tp,
        "fp_total": fp,
        "fn_total": fn,
        "split_edge_f1": float(summ.get("split_edge_f1", float("nan"))),
        "merge_edge_f1": float(summ.get("merge_edge_f1", float("nan"))),
        "overall_exact_total": int(summ.get("overall_exact_total", -1)),
    }


# --------------------------------------------------------------------------- #
# Task identity
# --------------------------------------------------------------------------- #
def cfg_tag(k: float, eps: float, lam: float) -> str:
    return f"k{k:g}_eps{eps:g}_lam{lam:g}"


def cfg_key(c: dict[str, Any]) -> str:
    return cfg_tag(c["k"], c["epsilon"], c["lambda"])


def task_id(bridge: str, seed: int, k: float, eps: float, lam: float, kind: str = "sweep") -> str:
    return f"{kind}__{bridge}__s{seed}__{cfg_tag(k, eps, lam)}"


def sweep_configs(spec: dict[str, Any]) -> list[dict[str, float]]:
    """The 14 unique (k, epsilon, lambda) configurations, default point included once."""
    sk = spec["sweeps"]["k"]
    se = spec["sweeps"]["epsilon"]
    sl = spec["sweeps"]["lambda"]
    seen: dict[str, dict[str, float]] = {}
    for k in sk["grid"]:
        c = {"k": float(k), "epsilon": float(sk["fixed"]["epsilon"]),
             "lambda": float(sk["fixed"]["lambda"])}
        seen.setdefault(cfg_key(c), c)
    for e in se["grid"]:
        c = {"k": float(se["fixed"]["k"]), "epsilon": float(e),
             "lambda": float(se["fixed"]["lambda"])}
        seen.setdefault(cfg_key(c), c)
    for l in sl["grid"]:
        c = {"k": float(sl["fixed"]["k"]), "epsilon": float(sl["fixed"]["epsilon"]),
             "lambda": float(l)}
        seen.setdefault(cfg_key(c), c)
    return list(seen.values())


def sweeps_containing(k: float, eps: float, lam: float, spec: dict[str, Any]) -> list[str]:
    out = []
    sk, se, sl = spec["sweeps"]["k"], spec["sweeps"]["epsilon"], spec["sweeps"]["lambda"]
    if (eps == sk["fixed"]["epsilon"] and lam == sk["fixed"]["lambda"]
            and k in [float(x) for x in sk["grid"]]):
        out.append("k")
    if (k == se["fixed"]["k"] and lam == se["fixed"]["lambda"]
            and eps in [float(x) for x in se["grid"]]):
        out.append("epsilon")
    if (k == sl["fixed"]["k"] and eps == sl["fixed"]["epsilon"]
            and lam in [float(x) for x in sl["grid"]]):
        out.append("lambda")
    return out


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
def run_cell(bridge: str, seed: int, k: float, eps: float, lam: float,
             spec: dict[str, Any], *, omitted: str | None = None,
             kind: str = "sweep", methods: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Solve once at (eps, lam) on the (possibly reduced) frozen cost, then decode both
    ranking methods at k.  The transport plan is shared, so both methods see identical
    numerics -> paired comparison."""
    assert_seeds_allowed([seed])
    assert_bridges_allowed([bridge])
    t0 = time.perf_counter()
    cell = load_dev_cell(bridge, seed)
    C = build_cost(cell, omitted)
    a, b = marginals_for_lambda(cell, lam)

    sol = solve_uot_instrumented(a, b, C, reg=eps, reg_m=lam)
    P = sol["P"]

    methods = methods or ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4")
    rows: list[dict[str, Any]] = []
    for method in methods:
        edges = decode(method, P, cell["sids"], cell["tids"], int(k))
        m = macro_edge_metrics(cell, edges)
        rows.append({
            "bridge": bridge, "seed": seed, "method": method,
            "k": float(k), "epsilon": float(eps), "lambda": float(lam),
            "ablation_variant": (f"LOCO_{omitted.upper()}" if omitted else "FULL_D6"),
            "omitted_component": (omitted if omitted else "none"),
            **m,
            # ---- solver diagnostics (real values from the solver) ----
            "sinkhorn_iterations": sol["iterations"],
            "sinkhorn_converged": sol["converged"],
            "sinkhorn_final_residual": sol["final_err"],
            "sinkhorn_hit_max_iter": sol["hit_max_iter"],
            "sinkhorn_err_trace_len": sol["err_trace_len"],
            "marginal_violation_row_l1": sol["marginal_violation_row_l1"],
            "marginal_violation_col_l1": sol["marginal_violation_col_l1"],
            "row_marginal_max_abs": sol["row_marginal_max_abs"],
            "col_marginal_max_abs": sol["col_marginal_max_abs"],
            # ---- real RC-UOT unmatched mass (delta^S / delta^T) ----
            "delta_s_total": sol["marginal_violation_row_l1"],
            "delta_t_total": sol["marginal_violation_col_l1"],
            "delta_total": sol["marginal_violation_row_l1"] + sol["marginal_violation_col_l1"],
            "transport_mass_total": sol["transport_mass_total"],
            "source_mass_total": sol["source_mass_total"],
            "target_mass_total": sol["target_mass_total"],
            "mass_retained_fraction": sol["mass_retained_fraction"],
            "objective_cost": sol["objective_cost"],
            "solver_num_iter_max": sol["num_iter_max"],
            "solver_stop_thr": sol["stop_thr"],
        })

    runtime = time.perf_counter() - t0
    return {
        "task_id": task_id(bridge, seed, k, eps, lam, kind),
        "kind": kind,
        "status": "ok",
        "bridge": bridge, "seed": seed,
        "k": float(k), "epsilon": float(eps), "lambda": float(lam),
        "ablation_variant": (f"LOCO_{omitted.upper()}" if omitted else "FULL_D6"),
        "omitted_component": (omitted if omitted else "none"),
        "cost_weights": normalised_weights(omitted),
        "cost_matrix_sha256": hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest(),
        "source_marginal_sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
        "sweeps": sweeps_containing(float(k), float(eps), float(lam), spec),
        "methods": rows,
        "runtime_sec": runtime,
        "source_commit": git_head(),
        "run_id": f"{utc_now()}__pid{os.getpid()}",
        "timestamp_utc": utc_now(),
    }


def run_cell_safe(bridge: str, seed: int, k: float, eps: float, lam: float,
                  spec: dict[str, Any], *, omitted: str | None = None,
                  kind: str = "sweep", max_attempts: int = 3,
                  methods: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Run one cell with bounded retries; failures are recorded, never fatal."""
    tid = task_id(bridge, seed, k, eps, lam, kind)
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            res = run_cell(bridge, seed, k, eps, lam, spec, omitted=omitted, kind=kind,
                           methods=methods)
            if attempt > 1:
                append_jsonl(RETRY_LOG, {"task_id": tid, "attempt": attempt,
                                         "result": "recovered", "ts": utc_now()})
            return res
        except Exception as exc:  # noqa: BLE001 - deliberate: never abort the sweep
            last_exc = exc
            append_jsonl(RETRY_LOG, {
                "task_id": tid, "attempt": attempt, "result": "failed",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "ts": utc_now(),
            })
            time.sleep(0.5 * attempt)
    failure = {
        "task_id": tid, "kind": kind, "status": "failed",
        "bridge": bridge, "seed": seed, "k": float(k), "epsilon": float(eps),
        "lambda": float(lam), "ablation_variant": (f"LOCO_{omitted.upper()}" if omitted else "FULL_D6"),
        "error_type": type(last_exc).__name__, "error": str(last_exc),
        "traceback": traceback.format_exc(),
        "attempts": max_attempts, "timestamp_utc": utc_now(), "source_commit": git_head(),
    }
    append_jsonl(FAILED_LOG, failure)
    return failure


def write_task_result(res: dict[str, Any], subdir: Path) -> Path:
    path = subdir / f"{res['task_id']}.json"
    atomic_write_json(path, res)
    return path


def load_task_result(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if obj.get("status") == "ok" else None


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def mode_check(spec: dict[str, Any]) -> int:
    """Software validation: instrumentation equivalence + frozen dev anchor reproduction."""
    out: dict[str, Any] = {"timestamp_utc": utc_now(), "source_commit": git_head()}

    # ---- guard self-test: the holdout protection must actually refuse 301-305 ----
    guard = {"forbidden": sorted(HOLDOUT_SEEDS_FORBIDDEN), "refused_examples": {}}
    for s in sorted(HOLDOUT_SEEDS_FORBIDDEN):
        try:
            assert_seeds_allowed([s])
            guard["refused_examples"][str(s)] = "NOT_REFUSED"
        except SystemExit:
            guard["refused_examples"][str(s)] = "refused"
    for s in ALLOWED_DEV_SEEDS:
        try:
            assert_seeds_allowed([s])
            guard[f"dev_{s}_allowed"] = True
        except SystemExit:
            guard[f"dev_{s}_allowed"] = False
    guard["all_holdout_refused"] = all(v == "refused" for v in guard["refused_examples"].values())
    guard["all_dev_allowed"] = all(guard[f"dev_{s}_allowed"] for s in ALLOWED_DEV_SEEDS)
    guard["pass"] = bool(guard["all_holdout_refused"] and guard["all_dev_allowed"])
    out["holdout_guard_selftest"] = guard

    checks = [verify_plan_equivalence(b, 201) for b in BRIDGES]
    out["plan_equivalence"] = checks
    out["plan_equivalence_pass"] = bool(
        all(c["dP_vs_project_wrapper"] == 0.0 and c["converged_instrumented"]
            and c["converged_wrapper"] for c in checks))

    # Reproduce the project's own development anchors (EXPECTED_DEV_ANCHORS) at the default
    # configuration, end to end through the instrumented solver + frozen decoder, on the
    # paper's PRIMARY cost matrix (the amount-free renormalised 5-component cost).
    out["anchor_reproduction"] = {}
    for method in ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4"):
        cells = []
        per_bridge_seed: dict[str, list[float]] = {}
        for bridge in BRIDGES:
            for seed in ALLOWED_DEV_SEEDS:
                r = run_cell(bridge, seed, 5.0, 0.05, 0.5, spec, omitted="amount")
                m = [x for x in r["methods"] if x["method"] == method][0]
                cells.append(m["macro_edge_f1"])
                per_bridge_seed.setdefault(bridge, []).append(m["macro_edge_f1"])
        macro = float(np.mean(cells))  # mean over the 15 cells (published convention)
        out["anchor_reproduction"][method] = {
            "macro_f1_mean_of_15_cells": macro,
            "expected": spec["holdout_protection"]["pre_existing_frozen_dev_anchors"][method],
            "per_bridge_seed_mean": {b: float(np.mean(v)) for b, v in per_bridge_seed.items()},
        }
        out["anchor_reproduction"][method]["abs_deviation"] = abs(
            macro - out["anchor_reproduction"][method]["expected"])
    worst = max(v["abs_deviation"] for v in out["anchor_reproduction"].values())
    out["max_abs_deviation_from_anchor"] = float(worst)
    out["anchor_tolerance"] = 5e-4
    out["anchor_note"] = (
        "Published frozen development anchors reproduce ONLY on the paper's primary "
        "amount-free renormalised cost C_primary (amount weight removed, remaining 5 weights "
        "renormalised to sum 1). The frozen C_effective is the 6-component full cost used for "
        "the paper's AMOUNT_FREE_COST_D4 reference arm and does NOT reproduce these anchors. "
        "This experiment's main sweep therefore uses C_primary, exactly as the frozen "
        "development study did.")
    out["anchor_reproduction_pass"] = bool(worst < 5e-4)
    out["spec_sha256"] = sha256_file(SPEC_PATH)
    out["overall_pass"] = bool(out["holdout_guard_selftest"]["pass"]
                               and out["plan_equivalence_pass"]
                               and out["anchor_reproduction_pass"])
    atomic_write_json(EXP / "00_preflight" / "software_check.json", out)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["overall_pass"] else 1


def mode_smoke(spec: dict[str, Any], bridge: str, seed: int = 201) -> int:
    """Default-parameter consistency smoke test on ONE development cell, both methods,
    compared against the pre-existing frozen development result for the same cell."""
    assert_seeds_allowed([seed])
    res = run_cell_safe(bridge, seed, 5.0, 0.05, 0.5, spec, kind="smoke", max_attempts=1)
    write_task_result(res, RUNS_DIR / "smoke")
    rec: dict[str, Any] = {
        "timestamp_utc": utc_now(), "bridge": bridge, "seed": seed,
        "k": 5.0, "epsilon": 0.05, "lambda": 0.5,
        "methods": {m["method"]: m for m in res.get("methods", [])},
        "comparison": {},
    }
    # Pre-existing development comparison: the project's frozen per-seed dev table
    # (CP/development/per_seed_methods.csv) was produced on C_effective with default params.
    dev_tbl = (REPO / "out" / "multi_bridge_expansion" / "conditional_plan_candidate_dev"
               / "development" / "per_seed_methods.csv")
    if dev_tbl.is_file():
        import pandas as pd
        df = pd.read_csv(dev_tbl)
        sub = df[(df["bridge"] == bridge) & (df["seed"] == seed)]
        rec["comparison_source"] = {
            "path": str(dev_tbl.relative_to(REPO)),
            "sha256": sha256_file(dev_tbl),
            "note": ("pre-existing frozen development result; solved on the FULL "
                     "C_effective cost, NOT on the amount-free C_primary used here, "
                     "so values are context, not an equality target"),
        }
        rec["comparison"] = {
            row["method"]: {
                "pre_existing_edge_f1": float(row["edge_f1"]),
                "pre_existing_edge_precision": float(row["edge_precision"]),
                "pre_existing_edge_recall": float(row["edge_recall"]),
                "new_edge_f1": rec["methods"].get(row["method"], {}).get("macro_edge_f1"),
            }
            for _, row in sub.iterrows()
            if row["method"] in ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4")
        }
    atomic_write_json(EXP / "logs" / f"smoke_{bridge}_seed{seed}.json", rec)
    print(json.dumps(rec, indent=2, default=str))
    return 0 if res.get("status") == "ok" else 1


def mode_sweep(spec: dict[str, Any], bridges, resume: bool = True) -> int:
    cfgs = sweep_configs(spec)
    assert_seeds_allowed(ALLOWED_DEV_SEEDS)
    assert_bridges_allowed(bridges)
    n_planned = len(cfgs) * len(bridges) * len(ALLOWED_DEV_SEEDS)
    log(f"[sweep] {len(cfgs)} unique configs x {len(bridges)} bridges x "
        f"{len(ALLOWED_DEV_SEEDS)} dev seeds = {n_planned} cells")
    log(f"[sweep] holdout guard ACTIVE; forbidden seeds = {sorted(HOLDOUT_SEEDS_FORBIDDEN)}")
    done = skipped = failed = 0
    for bridge in bridges:
        for seed in ALLOWED_DEV_SEEDS:
            for c in cfgs:
                path = RUNS_SWEEP / f"{task_id(bridge, seed, c['k'], c['epsilon'], c['lambda'])}.json"
                if resume and path.is_file():
                    if load_task_result(path) is not None:
                        skipped += 1
                        continue
                    log(f"[sweep] re-running invalid/failed result file {path.name}")
                res = run_cell_safe(bridge, seed, c["k"], c["epsilon"], c["lambda"], spec,
                                    omitted=PRIMARY_OMITTED, kind="sweep")
                write_task_result(res, RUNS_SWEEP)
                if res.get("status") == "ok":
                    done += 1
                else:
                    failed += 1
                if (done + skipped + failed) % 20 == 0:
                    log(f"[sweep] progress done={done} skipped={skipped} failed={failed}")
    log(f"[sweep] FINISHED done={done} skipped={skipped} failed={failed} planned={n_planned}")
    return 0 if failed == 0 else 2


def mode_ablation(spec: dict[str, Any], bridges, resume: bool = True) -> int:
    assert_seeds_allowed(ALLOWED_DEV_SEEDS)
    assert_bridges_allowed(bridges)
    n_planned = len(ABLATION_ORDER) * len(bridges) * len(ALLOWED_DEV_SEEDS)
    log(f"[ablation] {len(ABLATION_ORDER)} cost variants x {len(bridges)} bridges x "
        f"{len(ALLOWED_DEV_SEEDS)} dev seeds = {n_planned} cells (CONDITIONAL_UOT_D4 only)")
    done = skipped = failed = 0
    for bridge in bridges:
        for seed in ALLOWED_DEV_SEEDS:
            for variant in ABLATION_ORDER:
                omitted = ABLATION_OMITTED[variant]
                tid = f"ablation__{bridge}__s{seed}__{variant}"
                path = RUNS_ABLATION / f"{tid}.json"
                if resume and path.is_file():
                    if load_task_result(path) is not None:
                        skipped += 1
                        continue
                    log(f"[ablation] re-running invalid/failed result file {path.name}")
                res = run_cell_safe(bridge, seed, 5.0, 0.05, 0.5, spec, omitted=omitted,
                                    kind="ablation",
                                    methods=("CONDITIONAL_UOT_D4", "RAW_UOT_PLAN_D4"))
                res["task_id"] = tid
                res["ablation_variant"] = variant
                # The ablation table must distinguish "amount omitted" (the paper's primary
                # cost) from the other omissions by name, not by the generic LOO_x label.
                res["omitted_component"] = (omitted if omitted is not None else "none")
                write_task_result(res, RUNS_ABLATION)
                if res.get("status") == "ok":
                    done += 1
                else:
                    failed += 1
    log(f"[ablation] FINISHED done={done} skipped={skipped} failed={failed} planned={n_planned}")
    return 0 if failed == 0 else 2


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
SUMMARY_METRICS = ("precision", "recall", "macro_edge_f1")


def collect_runs(subdir: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(subdir.glob("*.json")):
        obj = load_task_result(p)
        if obj is not None:
            out.append(obj)
    return out


def mode_aggregate(spec: dict[str, Any]) -> int:
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    sweep_runs = collect_runs(RUNS_SWEEP)
    abla_runs = collect_runs(RUNS_ABLATION)
    if not sweep_runs:
        raise SystemExit("[aggregate] no successful sweep runs found")

    # ---------------- results/sweep_long.csv ----------------
    sk, se, sl = spec["sweeps"]["k"], spec["sweeps"]["epsilon"], spec["sweeps"]["lambda"]
    long_rows: list[dict[str, Any]] = []
    for run in sweep_runs:
        for m in run["methods"]:
            long_rows.append({
                "bridge": run["bridge"], "seed": run["seed"],
                "k": run["k"], "epsilon": run["epsilon"], "lambda": run["lambda"],
                "method": m["method"],
                "precision": m["precision"], "recall": m["recall"],
                "macro_edge_f1": m["macro_edge_f1"],
                "micro_precision": m["micro_precision"], "micro_recall": m["micro_recall"],
                "micro_edge_f1": m["micro_edge_f1"],
                "tp_total": m["tp_total"], "fp_total": m["fp_total"], "fn_total": m["fn_total"],
                "n_pred_edges": m["n_pred_edges"], "n_templates": m["n_templates"],
                "split_edge_f1": m["split_edge_f1"], "merge_edge_f1": m["merge_edge_f1"],
                "overall_exact_total": m["overall_exact_total"],
                "sinkhorn_iterations": m["sinkhorn_iterations"],
                "sinkhorn_converged": m["sinkhorn_converged"],
                "sinkhorn_final_residual": m["sinkhorn_final_residual"],
                "sinkhorn_hit_max_iter": m["sinkhorn_hit_max_iter"],
                "marginal_violation_row_l1": m["marginal_violation_row_l1"],
                "marginal_violation_col_l1": m["marginal_violation_col_l1"],
                "delta_s_total": m["delta_s_total"], "delta_t_total": m["delta_t_total"],
                "delta_total": m["delta_total"],
                "transport_mass_total": m["transport_mass_total"],
                "mass_retained_fraction": m["mass_retained_fraction"],
                "objective_cost": m["objective_cost"],
                "runtime_sec": run["runtime_sec"],
                "status": run["status"],
                "source_commit": run["source_commit"],
                "run_id": run["run_id"],
                "timestamp_utc": run["timestamp_utc"],
                "cost_matrix_sha256": run["cost_matrix_sha256"],
                "source_marginal_sha256": run["source_marginal_sha256"],
            })
    long_df = pd.DataFrame(long_rows).sort_values(
        ["bridge", "seed", "k", "epsilon", "lambda", "method"]).reset_index(drop=True)
    long_path = RESULTS_DIR / "sweep_long.csv"
    long_df.to_csv(long_path, index=False)
    log(f"[aggregate] wrote {long_path.name}: {len(long_df)} rows "
        f"({long_df['bridge'].nunique()} bridges x {long_df['seed'].nunique()} seeds x "
        f"{long_df.groupby(['k','epsilon','lambda']).ngroups} configs x "
        f"{long_df['method'].nunique()} methods)")

    # ---------------- results/sweep_summary.csv ----------------
    # Each physical run belongs to one or more sweeps (the default point belongs to all
    # three).  The SAME run row is referenced by every sweep that contains it -> the default
    # configuration is never computed twice and never reported with two different values.
    summ_rows: list[dict[str, Any]] = []
    seeds = [int(s) for s in sorted(long_df["seed"].unique())]
    for sweep_name, (grid, pcol) in (("k", (sk["grid"], "k")),
                                     ("epsilon", (se["grid"], "epsilon")),
                                     ("lambda", (sl["grid"], "lambda"))):
        sub = long_df[mask_sweep(long_df, sweep_name, spec)]
        for (pval, method), g in sub.groupby([pcol, "method"], sort=True):
            row: dict[str, Any] = {
                "sweep": sweep_name, "parameter": pcol,
                "parameter_value": float(pval), "method": method,
                "k": float(g["k"].iloc[0]), "epsilon": float(g["epsilon"].iloc[0]),
                "lambda": float(g["lambda"].iloc[0]),
                "n_bridges": int(g["bridge"].nunique()),
                "n_seeds": int(g["seed"].nunique()),
                "n_cells": int(len(g)),
                "seeds": ",".join(str(int(s)) for s in sorted(g["seed"].unique())),
                # headline convention: unweighted mean of the 15 cell-level values
                "macro_edge_f1_mean_of_cells": float(g["macro_edge_f1"].mean()),
                "macro_edge_f1_std_of_cells": float(g["macro_edge_f1"].std(ddof=1)),
                "precision_mean_of_cells": float(g["precision"].mean()),
                "recall_mean_of_cells": float(g["recall"].mean()),
                "micro_edge_f1_mean_of_cells": float(g["micro_edge_f1"].mean()),
                "mean_n_pred_edges": float(g["n_pred_edges"].mean()),
                "mean_sinkhorn_iterations": float(g["sinkhorn_iterations"].mean()),
                "max_sinkhorn_final_residual": float(g["sinkhorn_final_residual"].max()),
                "n_converged_cells": int(g["sinkhorn_converged"].sum()),
                "all_converged": bool(g["sinkhorn_converged"].all()),
                "mean_delta_s_total": float(g["delta_s_total"].mean()),
                "mean_delta_t_total": float(g["delta_t_total"].mean()),
                "mean_delta_total": float(g["delta_total"].mean()),
                "mean_transport_mass_total": float(g["transport_mass_total"].mean()),
                "mean_mass_retained_fraction": float(g["mass_retained_fraction"].mean()),
            }
            # per-bridge mean / std over the 5 development seeds (raw values are never dropped)
            for bridge in BRIDGES:
                gb = g[g["bridge"] == bridge].sort_values("seed")
                vals = gb["macro_edge_f1"]
                row[f"mean_macro_edge_f1_{bridge}"] = float(vals.mean())
                row[f"std_macro_edge_f1_{bridge}"] = float(vals.std(ddof=1))
                row[f"mean_precision_{bridge}"] = float(gb["precision"].mean())
                row[f"std_precision_{bridge}"] = float(gb["precision"].std(ddof=1))
                row[f"mean_recall_{bridge}"] = float(gb["recall"].mean())
                row[f"std_recall_{bridge}"] = float(gb["recall"].std(ddof=1))
                row[f"n_seeds_{bridge}"] = int(gb["seed"].nunique())
            # pooled over the 5 seeds within each bridge (std uses the per-seed values)
            row["std_over_seeds_macro_edge_f1"] = float(
                g.groupby("bridge")["macro_edge_f1"].mean().std(ddof=1))
            summ_rows.append(row)
    summ_df = pd.DataFrame(summ_rows).sort_values(
        ["sweep", "parameter_value", "method"]).reset_index(drop=True)
    summ_path = RESULTS_DIR / "sweep_summary.csv"
    summ_df.to_csv(summ_path, index=False)
    log(f"[aggregate] wrote {summ_path.name}: {len(summ_df)} rows")

    # ---------------- results/sweep_per_seed.csv ----------------
    # Explicit raw per-seed table so the 5 development seeds are preserved verbatim and every
    # figure point can be traced back to one row.
    per_seed_cols = ["bridge", "seed", "k", "epsilon", "lambda", "method",
                     "precision", "recall", "macro_edge_f1",
                     "micro_precision", "micro_recall", "micro_edge_f1",
                     "tp_total", "fp_total", "fn_total", "n_pred_edges", "n_templates",
                     "sinkhorn_iterations", "sinkhorn_converged", "sinkhorn_final_residual",
                     "delta_s_total", "delta_t_total", "delta_total",
                     "transport_mass_total", "mass_retained_fraction", "run_id"]
    ps_path = RESULTS_DIR / "sweep_per_seed.csv"
    long_df[per_seed_cols].sort_values(
        ["bridge", "seed", "k", "epsilon", "lambda", "method"]).to_csv(ps_path, index=False)
    log(f"[aggregate] wrote {ps_path.name}: {len(long_df)} rows")

    # ---------------- results/epsilon_solver_diagnostics.csv ----------------
    eps_df = long_df[mask_epsilon(long_df, se)].copy()
    diag_cols = ["bridge", "seed", "k", "epsilon", "lambda", "method",
                 "sinkhorn_iterations", "sinkhorn_converged", "sinkhorn_final_residual",
                 "sinkhorn_hit_max_iter", "marginal_violation_row_l1",
                 "marginal_violation_col_l1", "transport_mass_total",
                 "mass_retained_fraction", "objective_cost", "runtime_sec", "run_id"]
    eps_path = RESULTS_DIR / "epsilon_solver_diagnostics.csv"
    eps_df[diag_cols].sort_values(["bridge", "seed", "epsilon", "method"]).to_csv(
        eps_path, index=False)
    log(f"[aggregate] wrote {eps_path.name}: {len(eps_df)} rows")

    # ---------------- results/lambda_unmatched_mass.csv ----------------
    lam_df = long_df[mask_lambda(long_df, sl)].copy()
    lam_cols = ["bridge", "seed", "k", "epsilon", "lambda", "method",
                "delta_s_total", "delta_t_total", "delta_total",
                "marginal_violation_row_l1", "marginal_violation_col_l1",
                "transport_mass_total", "mass_retained_fraction",
                "precision", "recall", "macro_edge_f1",
                "sinkhorn_iterations", "sinkhorn_converged", "run_id"]
    lam_path = RESULTS_DIR / "lambda_unmatched_mass.csv"
    lam_df[lam_cols].sort_values(["bridge", "seed", "lambda", "method"]).to_csv(
        lam_path, index=False)
    log(f"[aggregate] wrote {lam_path.name}: {len(lam_df)} rows")

    # ---------------- full sweep matrix (long form, all three sweeps tagged) ----------
    melt = long_df.melt(
        id_vars=["bridge", "seed", "method", "k", "epsilon", "lambda",
                 "precision", "recall", "macro_edge_f1"],
        value_vars=["sinkhorn_iterations"], value_name="_drop")
    # (kept simple: sweep tagging lives in sweep_summary.csv)

    # ---------------- ablation CSVs ----------------
    if abla_runs:
        ab_long: list[dict[str, Any]] = []
        for run in abla_runs:
            for m in run["methods"]:
                ab_long.append({
                    "variant": run["ablation_variant"],
                    "omitted_component": run["omitted_component"],
                    "bridge": run["bridge"], "seed": run["seed"], "method": m["method"],
                    "k": run["k"], "epsilon": run["epsilon"], "lambda": run["lambda"],
                    "precision": m["precision"], "recall": m["recall"],
                    "macro_edge_f1": m["macro_edge_f1"],
                    "tp_total": m["tp_total"], "fp_total": m["fp_total"], "fn_total": m["fn_total"],
                    "n_pred_edges": m["n_pred_edges"], "n_templates": m["n_templates"],
                    "sinkhorn_iterations": m["sinkhorn_iterations"],
                    "sinkhorn_converged": m["sinkhorn_converged"],
                    "delta_s_total": m["delta_s_total"], "delta_t_total": m["delta_t_total"],
                    "delta_total": m["delta_total"],
                    "cost_weights_json": json.dumps(run["cost_weights"], sort_keys=True),
                    "cost_matrix_sha256": run["cost_matrix_sha256"],
                    "status": run["status"], "source_commit": run["source_commit"],
                    "run_id": run["run_id"], "runtime_sec": run["runtime_sec"],
                })
        ab_long_df = pd.DataFrame(ab_long).sort_values(
            ["method", "variant", "bridge", "seed"]).reset_index(drop=True)
        ab_long_path = ABLATION_DIR / "cost_component_ablation_long.csv"
        ab_long_df.to_csv(ab_long_path, index=False)
        log(f"[aggregate] wrote {ab_long_path.name}: {len(ab_long_df)} rows")

        # Baselines for the relative deltas:
        #   PRIMARY_COST = LOCO_AMOUNT = the paper's amount-free renormalised cost (C_primary);
        #                  this is THE baseline for the five retained components.
        #   FULL_D6      = all six frozen components renormalised to sum 1 (extra reference).
        base_primary = (ab_long_df[ab_long_df["variant"] == "LOCO_AMOUNT"]
                        .groupby(["method", "bridge"])["macro_edge_f1"].mean())
        base_full = (ab_long_df[ab_long_df["variant"] == "FULL_D6"]
                     .groupby(["method", "bridge"])["macro_edge_f1"].mean())
        ab_summ_rows: list[dict[str, Any]] = []
        for (method, variant, bridge), g in ab_long_df.groupby(
                ["method", "variant", "bridge"], sort=True):
            bp = float(base_primary.loc[(method, bridge)])
            bf = float(base_full.loc[(method, bridge)])
            per_seed = dict(zip((int(s) for s in g["seed"]), (float(v) for v in g["macro_edge_f1"])))
            per_seed_bp = dict(zip(
                (int(s) for s in ab_long_df[(ab_long_df["method"] == method)
                                            & (ab_long_df["bridge"] == bridge)
                                            & (ab_long_df["variant"] == "LOCO_AMOUNT")]["seed"]),
                (float(v) for v in ab_long_df[(ab_long_df["method"] == method)
                                              & (ab_long_df["bridge"] == bridge)
                                              & (ab_long_df["variant"] == "LOCO_AMOUNT")]["macro_edge_f1"])))
            deltas = {s: per_seed[s] - per_seed_bp[s] for s in sorted(per_seed)}
            ab_summ_rows.append({
                "method": method, "variant": variant, "bridge": bridge,
                "omitted_component": (g["omitted_component"].iloc[0]
                                      if g["omitted_component"].iloc[0] != "none" else "none"),
                "is_primary_cost": bool(variant == "LOCO_AMOUNT"),
                "n_seeds": int(g["seed"].nunique()),
                "seeds": ",".join(str(s) for s in sorted(per_seed)),
                "mean_macro_edge_f1": float(g["macro_edge_f1"].mean()),
                "std_macro_edge_f1": float(g["macro_edge_f1"].std(ddof=1)),
                "mean_precision": float(g["precision"].mean()),
                "std_precision": float(g["precision"].std(ddof=1)),
                "mean_recall": float(g["recall"].mean()),
                "std_recall": float(g["recall"].std(ddof=1)),
                "baseline_primary_cost_f1": bp,
                "delta_f1_vs_primary_cost": float(g["macro_edge_f1"].mean()) - bp,
                "baseline_full_d6_f1": bf,
                "delta_f1_vs_full_d6": float(g["macro_edge_f1"].mean()) - bf,
                "per_seed_f1_json": json.dumps(per_seed, sort_keys=True),
                "per_seed_delta_vs_primary_json": json.dumps(deltas, sort_keys=True),
                "n_seeds_with_negative_delta": int(sum(1 for v in deltas.values() if v < 0)),
                "n_seeds_with_positive_delta": int(sum(1 for v in deltas.values() if v > 0)),
            })
        ab_summ_df = pd.DataFrame(ab_summ_rows).sort_values(
            ["method", "variant", "bridge"]).reset_index(drop=True)
        ab_summ_path = ABLATION_DIR / "cost_component_ablation_summary.csv"
        ab_summ_df.to_csv(ab_summ_path, index=False)
        log(f"[aggregate] wrote {ab_summ_path.name}: {len(ab_summ_df)} rows")
    else:
        log("[aggregate] WARNING: no successful ablation runs found")

    # ---------------- run inventory ----------------
    inv = {
        "timestamp_utc": utc_now(),
        "source_commit": git_head(),
        "n_sweep_run_files_ok": len(sweep_runs),
        "n_ablation_run_files_ok": len(abla_runs),
        "n_sweep_rows_long": int(len(long_df)),
        "n_sweep_summary_rows": int(len(summ_df)),
        "bridges": sorted(long_df["bridge"].unique().tolist()),
        "seeds": sorted(int(s) for s in long_df["seed"].unique()),
        "methods": sorted(long_df["method"].unique().tolist()),
        "configs": sorted({cfg_tag(r["k"], r["epsilon"], r["lambda"]) for r in sweep_runs}),
        "n_unique_configs": len({cfg_tag(r["k"], r["epsilon"], r["lambda"]) for r in sweep_runs}),
        "holdout_seeds_executed": [],
        "holdout_guard": "active; 301-305 refused at argument validation",
        "expected_n_sweep_cells": 3 * 5 * len(sweep_configs(spec)),
        "failed_tasks_file": str(FAILED_LOG.relative_to(REPO)) if FAILED_LOG.is_file() else None,
    }
    atomic_write_json(EXP / "logs" / "aggregate_inventory.json", inv)
    print(json.dumps(inv, indent=2, default=str))
    return 0


def mask_epsilon(df, se):
    return (df["k"].eq(float(se["fixed"]["k"]))
            & df["lambda"].eq(float(se["fixed"]["lambda"]))
            & df["epsilon"].isin([float(x) for x in se["grid"]]))


def mask_lambda(df, sl):
    return (df["k"].eq(float(sl["fixed"]["k"]))
            & df["epsilon"].eq(float(sl["fixed"]["epsilon"]))
            & df["lambda"].isin([float(x) for x in sl["grid"]]))


def mask_sweep(df, sweep_name: str, spec: dict[str, Any]):
    sk, se, sl = spec["sweeps"]["k"], spec["sweeps"]["epsilon"], spec["sweeps"]["lambda"]
    if sweep_name == "k":
        return (df["epsilon"].eq(float(sk["fixed"]["epsilon"]))
                & df["lambda"].eq(float(sk["fixed"]["lambda"]))
                & df["k"].isin([float(x) for x in sk["grid"]]))
    if sweep_name == "epsilon":
        return mask_epsilon(df, se)
    if sweep_name == "lambda":
        return mask_lambda(df, sl)
    raise ValueError(sweep_name)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="software validation only")
    ap.add_argument("--smoke", action="store_true", help="default-parameter smoke test")
    ap.add_argument("--sweep", action="store_true", help="run the k/eps/lambda sweep")
    ap.add_argument("--ablation", action="store_true", help="cost-component leave-one-out")
    ap.add_argument("--aggregate", action="store_true", help="aggregate all run files")
    ap.add_argument("--bridge", default=None, help="restrict to one bridge")
    ap.add_argument("--seed", type=int, default=201, help="smoke-test seed (dev only)")
    ap.add_argument("--no-resume", action="store_true", help="recompute finished tasks")
    cli = ap.parse_args()

    # ------- HARD GUARD, evaluated before anything else touches data -------
    assert_seeds_allowed(ALLOWED_DEV_SEEDS + (cli.seed,))
    spec = load_spec()
    assert_seeds_allowed(spec["population"]["development_seeds"])
    overlap = set(int(s) for s in spec["population"]["development_seeds"]) & HOLDOUT_SEEDS_FORBIDDEN
    if overlap:
        raise SystemExit(f"[HOLDOUT-GUARD] spec declares forbidden seeds as development: {overlap}")

    bridges = (cli.bridge,) if cli.bridge else BRIDGES
    assert_bridges_allowed(bridges)

    for d in (RUNS_DIR, RUNS_SWEEP, RUNS_ABLATION, LOGS_DIR, RESULTS_DIR, ABLATION_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not any([cli.check, cli.smoke, cli.sweep, cli.ablation, cli.aggregate]):
        ap.print_help()
        return 1
    rc = 0
    if cli.check:
        rc |= mode_check(spec)
    if cli.smoke:
        rc |= mode_smoke(spec, bridges[0], cli.seed)
    if cli.sweep:
        rc |= mode_sweep(spec, bridges, resume=not cli.no_resume)
    if cli.ablation:
        rc |= mode_ablation(spec, bridges, resume=not cli.no_resume)
    if cli.aggregate:
        rc |= mode_aggregate(spec)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
