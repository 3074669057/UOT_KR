"""R7 Stage 0B / cell builder -- generate one (bridge, seed) cell with the extended
faithful generator, build the frozen amount-free cost and solve UOT exactly once.

The generator code path is the REAL paper generator
(``cross.domain.evaluation.semi_synthetic_flows``), extended in place.  The feature /
cost / marginal construction reuses the same frozen modules the frozen dev and holdout
pipelines used:

  * synthetic segments ....... ``write_synthetic_subgraph_segment_csvs``
  * flow construction ........ ``flows_from_segment_export_csv``
  * cost decomposition ....... ``build_cost_matrix_decomposed`` (weights unchanged)
  * amount-free renormalised . ``dev_candidate.af_common.build_amount_free_costs``
  * transport marginals ...... ``_risk_weighted_source_mass`` / ``_evidence_weighted_target_mass``
  * UOT solve ................ ``ot.unbalanced.sinkhorn_unbalanced`` reg=0.05 reg_m=0.5
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from r7.r7_common import (COST_RENORM_DENOM, DEGREE_RANGE, DIR_GENERATOR, EPSILON,
                          FROZEN_FEATURE_STATS, INSTANCES_PER_FAMILY, K_MAX, LAMBDA,
                          N_FAMILIES, N_TEMPLATES_PER_CELL, REPO, SOLVER_NUM_ITER_MAX,
                          SOLVER_STOP_THR, assert_bridges, assert_seeds_allowed,
                          environment_record, family_of_template, log, sha256_array,
                          sha256_file, sha256_obj, tpl_of)

GENERATOR_MODULE = REPO / "src" / "cross" / "domain" / "evaluation" / "semi_synthetic_flows.py"

# Frozen UOT parameters (identical to the frozen dev/holdout pipeline).
UOT_MAX_DELAY_SEC = 21600.0
UOT_CAUSAL_VIOLATION_PENALTY = 5.0
UOT_LAMBDA_RISK = 0.25
FEATURE_POOL_STATS = {"predominantly_one_to_one": True}


# --------------------------------------------------------------------------- #
# UOT convergence diagnostic
# --------------------------------------------------------------------------- #

def uot_kkt_residual(P: np.ndarray, C: np.ndarray, a: np.ndarray, b: np.ndarray,
                     eps: float, reg_m: float, *, max_iter: int = 20000,
                     tol: float = 1e-15) -> dict[str, Any]:
    """Exact first-order (KKT) residual of the entropic UNBALANCED OT problem
    **as POT actually solves it**.

    POT >= 0.9.5 defaults to ``reg_type='kl'``, so with ``c=None`` it builds

        ``K_eff = exp(-C/reg) * (a outer b)``

    and iterates ``u = (a / (K_eff v))^fi``, ``v = (b / (K_eff^T u))^fi`` with
    ``fi = reg_m / (reg_m + reg)``.  The returned plan is ``P = diag(u) K_eff diag(v)``.

    At that fixed point the relaxed marginal condition is exactly

        ``P 1 = a * u^(-reg/reg_m)``        ``P^T 1 = b * v^(-reg/reg_m)``

    which is the correct marginal condition for an *unbalanced* solve.  Contrast with
    ``|P1 - a|``, a BALANCED-OT criterion that does not vanish here because the marginal
    penalty is finite (reg_m = 0.5) by design.

    Everything is evaluated in the log domain to avoid the ``exp(-C/eps)`` underflow
    (eps = 0.05 makes ``K`` span ~9 orders of magnitude).
    """
    P = np.asarray(P, dtype=float)
    C = np.asarray(C, dtype=float)
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n, m = P.shape

    def _lse_cols(M: np.ndarray) -> np.ndarray:          # logsumexp over axis 1
        mx = np.max(M, axis=1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        return (mx + np.log(np.sum(np.exp(M - mx), axis=1, keepdims=True))).ravel()

    def _lse_rows(M: np.ndarray) -> np.ndarray:          # logsumexp over axis 0
        mx = np.max(M, axis=0, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        return (mx + np.log(np.sum(np.exp(M - mx), axis=0, keepdims=True))).ravel()

    loga = np.log(np.maximum(a, 1e-300))
    logb = np.log(np.maximum(b, 1e-300))
    logK = -C / eps
    fi = reg_m / (reg_m + eps)
    logu = np.zeros(n)
    logv = np.zeros(m)
    for it in range(max_iter):
        # log(K_eff v) = lse_j( logK + loga + logb + logv )
        lse1 = _lse_cols(logK + loga[:, None] + logb[None, :] + logv[None, :])
        logu_new = fi * (loga - lse1)
        lse2 = _lse_rows(logK + loga[:, None] + logb[None, :] + logu_new[:, None])
        logv_new = fi * (logb - lse2)
        delta = max(float(np.max(np.abs(logu_new - logu))),
                    float(np.max(np.abs(logv_new - logv))))
        logu, logv = logu_new, logv_new
        if delta < tol:
            break

    u = np.exp(logu)
    v = np.exp(logv)
    log_phat = (logu[:, None] + loga[:, None] + logK + logb[None, :] + logv[None, :])
    P_hat = np.exp(np.clip(log_phat, -700.0, 700.0))

    r1 = P.sum(axis=1)
    c1 = P.sum(axis=0)
    p1_hat = a * np.exp(np.clip(-logu * eps / reg_m, -700.0, 700.0))
    p2_hat = b * np.exp(np.clip(-logv * eps / reg_m, -700.0, 700.0))
    marg_row = np.abs(r1 - p1_hat)
    marg_col = np.abs(c1 - p2_hat)
    return {
        "definition": ("KKT marginal residual of the entropic UNBALANCED OT optimum as POT "
                       "solves it (reg_type='kl', K_eff = exp(-C/reg) * a outer b): "
                       "max|P1 - a*u^(-reg/reg_m)| and max|P^T1 - b*v^(-reg/reg_m)|"),
        "pot_reg_type": "kl (POT default since 0.9.5)",
        "effective_kernel": "exp(-C/reg) * outer(a, b)",
        "marginal_kkt_row_max": float(marg_row.max()) if marg_row.size else 0.0,
        "marginal_kkt_col_max": float(marg_col.max()) if marg_col.size else 0.0,
        "marginal_kkt_max": float(max(marg_row.max() if marg_row.size else 0.0,
                                      marg_col.max() if marg_col.size else 0.0)),
        "plan_reconstruction_abs_err": float(np.abs(P_hat - P).max()),
        "plan_reconstruction_rel_err": float(np.abs(P_hat - P).max()
                                             / max(float(P.max()), 1e-300)),
        "n_iter": int(it + 1),
        "balanced_marginal_row_max": float(np.abs(r1 - a).max()),
        "balanced_marginal_col_max": float(np.abs(c1 - b).max()),
        "note": ("the 'balanced' figures are reported only to document that a raw "
                 "|P1-a| check is not the right diagnostic for an unbalanced solve"),
    }


# --------------------------------------------------------------------------- #
# degree draws
# --------------------------------------------------------------------------- #

def draw_degrees(sampling_spec: dict[str, Any], *, bridge: str, seed: int,
                 n_templates: int = N_TEMPLATES_PER_CELL) -> dict[str, Any]:
    """Draw one split degree and one merge degree per template from the FROZEN PMFs.

    The RNG is dedicated to the degree draw, so it never perturbs the generator's own
    random stream (which would break the byte-equivalence of the frozen mode).
    """
    def _draw(spec_key: str, rng: random.Random) -> list[int]:
        spec = sampling_spec[spec_key]
        support = [int(x) for x in spec["support"]]
        pmf = np.asarray([float(x) for x in spec["pmf"]], dtype=float)
        if not np.isclose(pmf.sum(), 1.0, atol=1e-9):
            raise ValueError(f"{spec_key} pmf does not sum to 1 (sum={pmf.sum()})")
        idx = list(range(len(support)))
        return [int(rng.choices(support, weights=pmf.tolist(), k=1)[0])
                for _ in range(n_templates)]

    rng_s = random.Random(f"R7_SPLIT|{bridge}|{seed}")
    rng_m = random.Random(f"R7_MERGE|{bridge}|{seed}")
    split = _draw("split_degree", rng_s)
    merge = _draw("merge_degree", rng_m)
    bad = [d for d in split + merge if d < DEGREE_RANGE[0] or d > K_MAX]
    if bad:
        raise ValueError(f"degree draw out of frozen range {DEGREE_RANGE}: {sorted(set(bad))}")
    return {
        "split_degrees": split,
        "merge_degrees": merge,
        "rng": {"split": f"Random('R7_SPLIT|{bridge}|{seed}')",
                "merge": f"Random('R7_MERGE|{bridge}|{seed}')"},
    }


# --------------------------------------------------------------------------- #
# cell builder
# --------------------------------------------------------------------------- #

def build_cell(bridge: str, seed: int, sampling_spec: dict[str, Any],
               workdir: Path | None = None) -> dict[str, Any]:
    """Generate + featurise + solve ONE R7 cell.  Returns primitives, never metrics."""
    assert_bridges([bridge])
    assert_seeds_allowed([seed], f"r7_generator.build_cell({bridge}, {seed})")

    from cross.domain.evaluation.semi_synthetic_flows import (
        build_semi_synthetic_from_flow_labels)
    from cross.domain.evaluation.synthetic_segment_subgraph import (
        write_synthetic_subgraph_segment_csvs)
    from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv
    from cross.domain.uot.cost_matrix import (build_cost_matrix_decomposed,
                                              default_cost_weights)
    from cross.domain.uot.uot_solver import (_evidence_weighted_target_mass,
                                             _risk_weighted_source_mass)
    from dev_candidate.af_common import build_amount_free_costs
    from baseline_mechanism.common import tpl_maps, truth_structure

    pool = FROZEN_FEATURE_STATS / bridge
    for need in ("flow_labels.csv", "flow_segments_eth.csv", "flow_segments_bnb.csv"):
        if not (pool / need).is_file():
            raise FileNotFoundError(f"frozen generator input missing: {pool / need}")

    work = Path(workdir) if workdir is not None else (DIR_GENERATOR / "_cells" / bridge / f"seed_{seed}")
    work.mkdir(parents=True, exist_ok=True)

    # ---- 1. degree draws -------------------------------------------------- #
    deg = draw_degrees(sampling_spec, bridge=bridge, seed=seed)

    # ---- 2. generation through the REAL generator (extended in place) ----- #
    labels_pool = pool / "flow_labels.csv"
    import pandas as _pd
    pool_df = _pd.read_csv(labels_pool, dtype=str, keep_default_na=False)
    (work / "flow_label_stats.json").write_text(
        json.dumps({**FEATURE_POOL_STATS, "n_pool_pairs": int(len(pool_df))}), encoding="utf-8")
    build_semi_synthetic_from_flow_labels(
        labels_pool, work / "flow_label_stats.json", work,
        seed=int(seed), max_seeds=N_TEMPLATES_PER_CELL, force=True,
        r7_family_grid=True,
        r7_split_degrees=deg["split_degrees"],
        r7_merge_degrees=deg["merge_degrees"],
        r7_instances_per_family=INSTANCES_PER_FAMILY)

    hints = json.loads((work / "labels" / "synthetic_uot_eval_metrics.json")
                       .read_text(encoding="utf-8"))
    clones = hints.get("segment_clone_records") or []
    if not clones:
        raise RuntimeError("generator produced no segment clone records")

    se = work / "flow_segments_eth_synth.csv"
    sb = work / "flow_segments_bnb_synth.csv"
    write_synthetic_subgraph_segment_csvs(pool / "flow_segments_eth.csv",
                                          pool / "flow_segments_bnb.csv", clones, se, sb)
    eth = flows_from_segment_export_csv(se, chain="ETH")
    bnb = flows_from_segment_export_csv(sb, chain="BNB")

    # ---- 3. cost decomposition (frozen weights, unchanged) ---------------- #
    decomp = build_cost_matrix_decomposed(
        eth, bnb, weights=default_cost_weights(), use_graph=False,
        max_delay_sec=UOT_MAX_DELAY_SEC,
        causal_violation_penalty=UOT_CAUSAL_VIOLATION_PENALTY)
    C_effective = np.maximum(np.asarray(decomp["C"], dtype=float)
                             + np.asarray(decomp["bridge_prior_bonus"], dtype=float), 0.0)
    comp_keys = ("amount_cost", "time_cost", "route_cost", "risk_cost", "evidence_cost",
                 "address_novelty_cost")
    components = {k: np.asarray(decomp[k], dtype=float) for k in comp_keys if k in decomp}
    for need in ("time_cost", "route_cost", "risk_cost", "evidence_cost",
                 "address_novelty_cost"):
        if need not in components:
            raise RuntimeError(f"cost decomposition missing component {need}")

    afc = build_amount_free_costs(components)
    C_primary = np.asarray(afc["primary"], dtype=float)

    sids = [str(f.get("flow_id")) for f in eth]
    tids = [str(f.get("flow_id")) for f in bnb]

    # ---- 4. marginals + ONE UOT solve ------------------------------------- #
    a0, a_rw = _risk_weighted_source_mass(eth, lambda_risk=UOT_LAMBDA_RISK)
    b0, b_ev = _evidence_weighted_target_mass(bnb)
    a = np.asarray(a_rw, dtype=float).ravel().copy()
    b = np.asarray(b_ev, dtype=float).ravel().copy()
    a = a / a.sum() if a.sum() > 0 else a
    b = b / b.sum() if b.sum() > 0 else b

    import ot
    _t0 = time.time()
    P, lg = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C_primary, reg=float(EPSILON), reg_m=float(LAMBDA),
        numItermax=SOLVER_NUM_ITER_MAX, stopThr=SOLVER_STOP_THR, log=True)
    runtime_sec = time.time() - _t0
    P = np.asarray(P, dtype=float)
    errs = np.asarray(lg.get("err", [1.0]), dtype=float).ravel()
    final_err = float(errs[-1]) if errs.size else float("nan")
    iterations = int(max(errs.size - 1, 0))
    row_dev = np.abs(P.sum(axis=1) - a)
    col_dev = np.abs(P.sum(axis=0) - b)
    kkt = uot_kkt_residual(P, C_primary, a, b, EPSILON, LAMBDA)

    # ---- 5. truth + template maps ----------------------------------------- #
    labels = _pd.read_csv(work / "labels" / "synthetic_flow_labels.csv", dtype=str,
                          keep_default_na=False)
    (work / "labels.csv").write_text(labels.to_csv(index=False), encoding="utf-8")
    tpl_s, tpl_t = tpl_maps(labels, sids, tids)
    truth = truth_structure(labels)
    families = sorted({family_of_template(t) for t in truth})

    np.savez(work / "ids.npz", sids=np.array(sids, dtype=object),
             tids=np.array(tids, dtype=object))
    np.savez(work / "cost.npz", C_effective=C_effective, C_primary=C_primary, **components)
    np.savez(work / "transport_uot.npz", P=P, a_rw=a, b_ev=b, a0=np.asarray(a0, float),
             b0=np.asarray(b0, float))

    solver_meta = {
        "solver_call_id": f"UOT|{bridge}|{seed}|eps{EPSILON}|lam{LAMBDA}",
        "reg": EPSILON, "reg_m": LAMBDA,
        "numItermax": SOLVER_NUM_ITER_MAX, "stopThr": SOLVER_STOP_THR,
        "iterations": iterations, "final_err": final_err,
        "converged": bool(final_err < 1e-7),
        "hit_max_iter": bool(iterations >= SOLVER_NUM_ITER_MAX),
        "runtime_sec": float(runtime_sec),
        "marginal_violation_row_l1": float(row_dev.sum()),
        "marginal_violation_col_l1": float(col_dev.sum()),
        "marginal_violation_row_max": float(row_dev.max()) if row_dev.size else 0.0,
        "marginal_violation_col_max": float(col_dev.max()) if col_dev.size else 0.0,
        "kkt": kkt,
        "transport_mass_total": float(P.sum()),
        "plan_sha256": sha256_array(P),
        "cost_matrix_sha256": sha256_array(C_primary),
        "solver_invocations_for_this_cell": 1,
    }
    (work / "solver.json").write_text(json.dumps(solver_meta, indent=2) + "\n",
                                      encoding="utf-8")

    return {
        "bridge": bridge, "seed": seed, "workdir": work,
        "sids": sids, "tids": tids, "tpl_s": tpl_s, "tpl_t": tpl_t,
        "labels": labels, "truth": truth, "families": families,
        "C_primary": C_primary, "C_effective": C_effective, "components": components,
        "P": P, "a": a, "b": b,
        "solver": solver_meta,
        "degrees": deg,
        "hashes": {
            "cost_matrix_sha256": sha256_array(C_primary),
            "cost_matrix_effective_sha256": sha256_array(C_effective),
            "plan_sha256": sha256_array(P),
            "labels_sha256": sha256_file(work / "labels" / "synthetic_flow_labels.csv"),
            "data_sha256": sha256_file(work / "flow_segments_eth_synth.csv")
                           + sha256_file(work / "flow_segments_bnb_synth.csv"),
            "generator_module_sha256": sha256_file(GENERATOR_MODULE),
            "marginals_sha256": sha256_obj([sha256_array(a), sha256_array(b)]),
        },
        "n_src": len(sids), "n_dst": len(tids),
        "n_templates": len(truth),
        "n_families": len(families),
    }


def cell_workdir(bridge: str, seed: int, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else (DIR_GENERATOR / "_cells")
    return base / bridge / f"seed_{seed}"


def load_cell(bridge: str, seed: int, root: Path | None = None) -> dict[str, Any]:
    """Load a previously built cell's primitives from disk (no regeneration)."""
    assert_bridges([bridge])
    assert_seeds_allowed([seed], f"r7_generator.load_cell({bridge}, {seed})")
    import pandas as _pd
    from baseline_mechanism.common import tpl_maps, truth_structure

    work = cell_workdir(bridge, seed, root)
    for need in ("ids.npz", "cost.npz", "transport_uot.npz", "labels.csv"):
        if not (work / need).is_file():
            raise FileNotFoundError(f"cell primitive missing: {work / need}")
    ids = np.load(work / "ids.npz", allow_pickle=True)
    cz = np.load(work / "cost.npz", allow_pickle=False)
    uz = np.load(work / "transport_uot.npz", allow_pickle=True)
    labels = _pd.read_csv(work / "labels.csv", dtype=str, keep_default_na=False)
    sids = [str(x) for x in ids["sids"]]
    tids = [str(x) for x in ids["tids"]]
    tpl_s, tpl_t = tpl_maps(labels, sids, tids)
    truth = truth_structure(labels)
    comp_keys = ("amount_cost", "time_cost", "route_cost", "risk_cost", "evidence_cost",
                 "address_novelty_cost")
    hints = json.loads((work / "labels" / "synthetic_uot_eval_metrics.json")
                       .read_text(encoding="utf-8"))
    P = np.asarray(uz["P"], dtype=float)
    a = np.asarray(uz["a_rw"], dtype=float)
    b = np.asarray(uz["b_ev"], dtype=float)
    C_primary = np.asarray(cz["C_primary"], dtype=float)
    sm_path = work / "solver.json"
    if sm_path.is_file():
        solver_meta = json.loads(sm_path.read_text(encoding="utf-8"))
        solver_meta["loaded_from_cache"] = True
    else:                                                   # pragma: no cover
        row_dev = np.abs(P.sum(axis=1) - a)
        col_dev = np.abs(P.sum(axis=0) - b)
        solver_meta = {
            "solver_call_id": f"UOT|{bridge}|{seed}|eps{EPSILON}|lam{LAMBDA}",
            "reg": EPSILON, "reg_m": LAMBDA,
            "numItermax": SOLVER_NUM_ITER_MAX, "stopThr": SOLVER_STOP_THR,
            "iterations": -1, "final_err": float("nan"),
            "converged": bool(max(float(row_dev.max()), float(col_dev.max())) < 1e-7),
            "hit_max_iter": False,
            "marginal_violation_row_l1": float(row_dev.sum()),
            "marginal_violation_col_l1": float(col_dev.sum()),
            "marginal_violation_row_max": float(row_dev.max()),
            "marginal_violation_col_max": float(col_dev.max()),
            "kkt": uot_kkt_residual(P, C_primary, a, b, EPSILON, LAMBDA),
            "transport_mass_total": float(P.sum()),
            "plan_sha256": sha256_array(P),
            "cost_matrix_sha256": sha256_array(C_primary),
        }
    families = sorted({family_of_template(t) for t in truth})
    return {
        "bridge": bridge, "seed": seed, "workdir": work,
        "sids": sids, "tids": tids, "tpl_s": tpl_s, "tpl_t": tpl_t,
        "labels": labels, "truth": truth, "families": families,
        "C_primary": C_primary,
        "C_effective": np.asarray(cz["C_effective"], dtype=float),
        "components": {k: np.asarray(cz[k], dtype=float) for k in comp_keys if k in cz},
        "P": P, "a": a, "b": b,
        "solver": solver_meta,
        "degrees": {
            "split_degrees": [int(x) for x in hints.get("r7_split_degrees", [])],
            "merge_degrees": [int(x) for x in hints.get("r7_merge_degrees", [])],
            "rng": {},
        },
        "hashes": {
            "cost_matrix_sha256": sha256_array(C_primary),
            "plan_sha256": sha256_array(P),
            "labels_sha256": sha256_file(work / "labels" / "synthetic_flow_labels.csv"),
            "generator_module_sha256": sha256_file(GENERATOR_MODULE),
        },
        "n_src": len(sids), "n_dst": len(tids),
        "n_templates": len(truth), "n_families": len(families),
        "generator_hints": hints,
    }


def cell_summary(cell: dict[str, Any]) -> dict[str, Any]:
    """Small JSON-safe descriptor of a built cell."""
    return {
        "bridge": cell["bridge"], "seed": cell["seed"],
        "n_src": cell["n_src"], "n_dst": cell["n_dst"],
        "n_templates": cell["n_templates"], "n_families": cell["n_families"],
        "families": cell["families"],
        "solver": cell["solver"],
        "hashes": cell["hashes"],
        "degrees": {"split_degrees": cell["degrees"]["split_degrees"],
                    "merge_degrees": cell["degrees"]["merge_degrees"]},
    }
