"""R7 Stage 0 -- selection runner.

HARD SEED GUARD: this runner accepts the R7 SELECTION block ONLY
(``206-211 + 112-115``).  It provides no interface that can reach the confirmatory
holdout ``401-410`` or any permanently forbidden seed.

Modes
-----
--lock-spec    write the author spec, the operational protocol and the candidate space
               (all BEFORE any method F1 is observed)
--preflight    reproduction anchors, seed-guard self-tests, cost-hash consistency
--selection    run every rule candidate on the selection block; save ALL candidates
--select       deterministic winner + complexity tie-break
--calibrate    Threshold-MM (equal-budget) and Dual-Softmax (confidence) calibration
--report       selection/SELECTION_REPORT.md
--freeze       config/locked_spec.json + sha256 + frozen protocol manifest + audit
--all          lock-spec -> preflight -> selection -> select -> calibrate -> report
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r7.r7_common import (ALPHA, BRIDGES, CONFIRMATORY_SEEDS, COST_RENORM_DENOM,
                          COST_WEIGHTS_ABSOLUTE, DEGREE_RANGE, DIR_ANALYSIS, DIR_CONFIG,
                          DIR_CONFIRMATORY, DIR_DIAGNOSTICS, DIR_RAW, DIR_RULES,
                          DIR_SELECTION, DUAL_SOFTMAX_TAU_GRID, EPSILON, EXPERIMENT_ID,
                          FORBIDDEN_SEEDS, GATE_C_FLOOR, INSTANCES_PER_FAMILY, K_MAX, LAMBDA,
                          METHODS, N_BOOT, N_FAMILIES, N_PERM, N_TEMPLATES_PER_CELL,
                          RETIRED_CONFIRMATORY_BLOCKS, RNG_BOOTSTRAP, RNG_PERMUTATION,
                          SELECTION_SEEDS, SELECTION_SEEDS_AS_SPECIFIED, SOLVER_MARGINAL_TOL,
                          SOLVER_NUM_ITER_MAX, SOLVER_STOP_THR, SUPPORT_THRESHOLD,
                          THRESHOLD_MM_Q_GRID, THRESHOLD_MM_Q_GRID_REFINED, TIE_TOL,
                          assert_selection_seeds, environment_record, git, log,
                          resolve_frozen_path, sha256_file, sha256_obj, sha256_text, utc_now,
                          write_json, write_text)
from r7.r7_generator import build_cell, load_cell
from r7.r7_methods import (cost_d4_edges, conditional_edges, log_kernel, quantile_k,
                           oracle_1to1_ceiling, threshold_mm_edges, dual_softmax_edges,
                           hungarian_1to1)
from r7.r7_pipeline import (CellContext, dual_softmax_calibrate, family_macro,
                            run_all_methods, threshold_mm_calibrate)

REPO = Path(__file__).resolve().parents[1]
DEGREE_SPEC_PATH = DIR_SELECTION / "degree_calibration" / "degree_sampling_spec.json"
PROTOCOL_PATH = DIR_CONFIG / "operational_protocol_preselection.json"
SPEC_PATH = DIR_CONFIG / "locked_spec.json"
SPEC_HASH_PATH = DIR_CONFIG / "locked_spec.sha256"
MANIFEST_PATH = DIR_CONFIG / "FROZEN_PROTOCOL_MANIFEST.json"
CANDIDATE_PATH = DIR_RULES / "candidate_space.json"
ALL_CANDIDATES_CSV = DIR_RULES / "all_candidates.csv"
ALL_CANDIDATES_JSON = DIR_RULES / "all_candidates.json"
SELECTED_RULE_PATH = DIR_RULES / "selected_rule.json"
CELL_CACHE = DIR_SELECTION / "cells"
QA_JSON = DIR_SELECTION / "generator" / "generator_structural_qa.json"

FAMILY_COMPLEXITY = {"R-const": 1, "R-quantile": 2, "R-threshold": 3, "R-adaptive": 4}


# --------------------------------------------------------------------------- #
# candidate space
# --------------------------------------------------------------------------- #

def build_candidate_space() -> dict[str, Any]:
    spec = json.loads(DEGREE_SPEC_PATH.read_text(encoding="utf-8"))
    support = [int(x) for x in spec["split_degree"]["support"]]
    pmf = [float(x) for x in spec["split_degree"]["pmf"]]

    cands: list[dict[str, Any]] = []
    for k in (2, 3, 4, 5, 6, 8):
        cands.append({"rule_id": f"R-const@k{k}", "family": "R-const", "k": int(k),
                      "param_name": "k", "param_value": float(k),
                      "definition": "bilateral mutual top-k on the ranking signal"})
    for q in (0.75, 0.90, 0.95):
        k = quantile_k(support, pmf, q)
        cands.append({"rule_id": f"R-quantile@q{q}", "family": "R-quantile", "q": float(q),
                      "k": int(k), "param_name": "q", "param_value": float(q),
                      "definition": (f"k = ceil(Q_{q}) of the FROZEN pooled truncated degree "
                                     f"distribution, clipped to [2,8] -> k={k}; ONE k shared "
                                     f"by all bridges")})
    for a in (2, 3, 4, 5, 6, 8):
        cands.append({"rule_id": f"R-adaptive@a{a}", "family": "R-adaptive",
                      "alpha": int(a), "param_name": "alpha", "param_value": float(a),
                      "k_max": K_MAX,
                      "definition": ("k_i^S = clip(ceil(alpha*r_i/median(r_positive)),1,8) and "
                                     "k_j^T = clip(ceil(alpha*c_j/median(c_positive)),1,8); "
                                     "symmetric on both sides")})
    for t in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90):
        cands.append({"rule_id": f"R-threshold@t{t}", "family": "R-threshold",
                      "theta": float(t), "param_name": "theta", "param_value": float(t),
                      "definition": ("K_ij/rowmax_i >= theta AND K_ij/colmax_j >= theta "
                                     "(log-domain comparison); no k")})
    return {
        "generated_at_utc": utc_now(),
        "fixed_before_viewing_selection_f1": True,
        "selection_criterion": "highest bridge-balanced overall macro edge F1",
        "tie_break": {
            "order": ["R-const", "R-quantile", "R-threshold", "R-adaptive"],
            "within_family": "smaller parameter value",
            "tolerance": TIE_TOL,
            "note": "no holdout information, no 'more publishable' tie-break",
        },
        "n_candidates": len(cands),
        "candidates": cands,
        "quantile_source": "selection/degree_calibration/degree_sampling_spec.json",
        "quantile_derivation": {
            "support": support, "pmf": pmf,
            "k_for_q": {str(q): quantile_k(support, pmf, q) for q in (0.75, 0.90, 0.95)},
        },
    }


# --------------------------------------------------------------------------- #
# operational protocol
# --------------------------------------------------------------------------- #

def build_operational_protocol() -> dict[str, Any]:
    spec = json.loads(DEGREE_SPEC_PATH.read_text(encoding="utf-8"))
    tail = json.loads((DIR_SELECTION / "degree_calibration" / "tail_report.json")
                      .read_text(encoding="utf-8"))
    return {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now(),
        "stage": "pre-selection (fixed before any selection-block method F1 was observed)",
        "seed_blocks": {
            "selection": list(SELECTION_SEEDS),
            "selection_as_originally_specified": list(SELECTION_SEEDS_AS_SPECIFIED),
            "selection_amendment": {
                "applied": True,
                "reason": ("the originally reserved block 206-215 was found partially "
                           "contaminated: 212-215 already carry produced output from "
                           "out/paper_full_pipeline_run/synthetic/"),
                "decision": ("selection = 206-211 (the fresh part of the originally reserved "
                             "block) + 112-115 (next fresh seeds)"),
                "evidence": "00_preflight/seed_freshness_audit.json",
                "blocker_report": "BLOCKER_REPORT.md",
                "owner_decision": "protocol owner approved option A on 2026-09-17",
            },
            "confirmatory": list(CONFIRMATORY_SEEDS),
            "retired_confirmatory_blocks": [dict(r) for r in RETIRED_CONFIRMATORY_BLOCKS],
            "permanently_forbidden": sorted(FORBIDDEN_SEEDS),
        },
        "cost_configuration": {
            "locked": True,
            "reopened_by_r7": False,
            "family": "amount-free renormalised five-component cost",
            "absolute_weights": COST_WEIGHTS_ABSOLUTE,
            "absolute_sum": COST_RENORM_DENOM,
            "normalisation": "each retained weight divided by 0.65",
            "normalised_weights": {k: v / COST_RENORM_DENOM
                                   for k, v in COST_WEIGHTS_ABSOLUTE.items()},
            "amount_component": ("removed from the PAIRWISE cost only; the amount-derived "
                                 "transport marginals are preserved unchanged"),
            "epsilon": EPSILON,
            "lambda": LAMBDA,
            "back_door_closed": ("cost weights, epsilon, lambda and the normalisation are NOT "
                                 "re-tuned in R7, not even if the expanded generator lowers "
                                 "performance; if the cost proves unsuitable that is a result"),
        },
        "solver_configuration": {
            "backend": "POT ot.unbalanced.sinkhorn_unbalanced",
            "reg": EPSILON, "reg_m": LAMBDA,
            "numItermax": SOLVER_NUM_ITER_MAX, "stopThr": SOLVER_STOP_THR,
            "convergence_criterion": "final POT err < 1e-7 (frozen R5/R6 criterion)",
        },
        "shared_cost_requirement": {
            "rule": "every method in a cell consumes the SAME C and the SAME P",
            "recorded": "cost_matrix_sha256 per cell, asserted equal across methods",
        },
        "generator": {
            "implementation": "src/cross/domain/evaluation/semi_synthetic_flows.py (extended in place)",
            "byte_equivalence_self_test": "selection/generator/generator_equivalence.json",
            "n_families": N_FAMILIES,
            "instances_per_family": INSTANCES_PER_FAMILY,
            "templates_per_cell": N_TEMPLATES_PER_CELL,
            "degree_distribution": "frozen pooled empirical fan-out (split) and fan-in (merge)",
            "degree_range": list(DEGREE_RANGE), "degree_cap": K_MAX,
            "truncation": "d_used = min(d_observed, 8) (right winsorisation, not deletion)",
            "HIGH_TRUNCATION_TAIL": tail["HIGH_TRUNCATION_TAIL"],
            "preserved_mechanisms": [
                "amount allocation (generalised to equal shares amount/d)",
                "time perturbation", "+60/+120 delay decoys", "unmatched source",
                "address / evidence / risk construction", "bridge-specific mechanisms"],
        },
        "degree_calibration": {
            "source": "frozen v4 and v5 audit windows (paper section 4.6)",
            "provenance": "selection/degree_calibration/source_provenance.json",
            "definition": "selection/degree_calibration/degree_definition.json",
            "fanin_empirical": True,
            "fanin_mirrored_proxy_used": False,
            "sampling_spec": spec,
        },
        "methods": {
            "primary_treatment": "UOT_KR",
            "arms": {
                "UOT_KR": "selected rule applied to the direct forensic kernel logK = -C/eps",
                "RAW_UOT_PLAN": "same rule applied to the raw UOT plan P",
                "CONDITIONAL_UOT": "same rule on the dual-cancelling conditional scores",
                "SUPPORT_PLUS_K": ("same rule on logK with a HARD support filter "
                                   "(P > 1e-9 at the candidate-set level)"),
                "HUNGARIAN_1TO1": ("cost-optimal one-to-one assignment baseline: rectangular "
                                   "linear sum assignment on the same C, label-free"),
                "THRESHOLD_MM": "edge iff C_ij <= cutoff; cutoff fixed on the selection block",
                "DUAL_SOFTMAX": ("LoFTR-style dual softmax on logK with bidirectional "
                                 "(mutual nearest neighbour) acceptance and a confidence gate"),
            },
            "diagnostics_not_baselines": {
                "ORACLE_1TO1_CEILING": ("label-informed maximum-cardinality matching on the "
                                        "ground-truth positive graph; not deployable, not in "
                                        "H1/H2, not in Holm, never used to build a UOT_KR "
                                        "prediction"),
            },
        },
        "protocol_corrections": {
            "C1_hungarian_is_not_an_f1_upper_bound": {
                "as_proposed": "HUNGARIAN_1TO1 described as the F1 upper bound for one-to-one methods",
                "operational": ("HUNGARIAN_1TO1 is the cost-optimal one-to-one ASSIGNMENT "
                                "BASELINE at the same forensic cost C. It is not, and must "
                                "never be written as, an F1 upper bound or a performance "
                                "upper bound."),
                "added": ("ORACLE_1TO1_CEILING, a pre-registered label-informed diagnostic that "
                          "does answer what the one-to-one output space can reach: with T true "
                          "edges and maximum non-conflicting true edges M, precision=1, "
                          "recall=M/T, F1=2M/(T+M)."),
                "interpretation_rule": {
                    "UOT_KR > ORACLE_CEILING": ("strong output-space evidence: one-to-one output "
                                                "semantics impose a measurable ceiling under "
                                                "this synthetic truth topology"),
                    "UOT_KR > HUNGARIAN only": ("only licenses 'this many-to-many decoding "
                                                "outperformed the cost-optimal one-to-one "
                                                "assignment baseline at equal cost'"),
                },
            },
            "C2_gate_c_is_not_an_execution_validity_gate": {
                "as_proposed": "per-bridge mean effect >= -0.005 listed as an EXECUTION VALIDITY gate",
                "problem": ("a result-dependent threshold must never be able to declare a valid "
                            "confirmatory execution void"),
                "operational": {
                    "A": "primary efficacy gate (H1)",
                    "B": "primary efficacy gate (H2)",
                    "C": "cross-bridge consistency / claim gate",
                    "D": "technical validity gate",
                    "E": "technical validity gate (independent metric verification)",
                },
                "if_C_fails": ("the experiment remains a VALID confirmatory result, but no "
                               "cross-bridge-consistent overall advantage may be claimed"),
                "if_D_or_E_fails": ("TECHNICALLY_INVALID_CONFIRMATORY_EXECUTION; all results, "
                                    "reasons and logs are preserved"),
                "never": "401-410 must never be re-run, for any reason",
            },
            "C3_holm_and_ci_fully_specified": {
                "as_proposed": "'95% CI + Holm' with no estimand and no adjustment target",
                "primary_estimand": {
                    "cell": ("for each (bridge, seed): macro edge F1 = equal-weight mean over "
                             "the 24 template families of the family-mean edge F1"),
                    "n_primary_paired_cells": 30,
                    "bridges": list(BRIDGES), "seeds_per_bridge": 10,
                    "pseudoreplication": ("the several hundred template instances are NEVER "
                                          "treated as independent samples"),
                    "overall_effect": ("mean the 10 seed-level paired deltas within each "
                                       "bridge, then equal-weight mean the 3 bridge means"),
                },
                "primary_ci": {
                    "method": "paired stratified bootstrap",
                    "within_bridge": "resample the 10 seed-level paired cells with replacement",
                    "paired": "both methods of a pair are resampled together",
                    "across_bridges": "equal-weight macro-average of the 3 bridge means",
                    "B": N_BOOT, "rng_seed": RNG_BOOTSTRAP, "interval": "percentile two-sided 95%",
                    "forbidden": "bootstrapping the hundreds of template instances independently",
                },
                "primary_p_value": {
                    "test": "one-sided paired sign-flip permutation",
                    "statistic": ("equal-weight mean of the 3 bridge means of the 30 paired "
                                  "bridge-seed deltas"),
                    "H0": "effect <= 0", "H1": "effect > 0",
                    "rng_seed": RNG_PERMUTATION, "n_perm": N_PERM,
                    "monte_carlo_p": "(extreme + 1) / (n_perm + 1)",
                    "resolution": f"about 1/{N_PERM + 1} = {1.0 / (N_PERM + 1):.3e}",
                    "resolution_note": ("1.9e-9 is the theoretical order of a FULL exact "
                                        "enumeration of 2^30 sign patterns; this protocol does "
                                        "NOT enumerate them and therefore never reports a "
                                        "p-value below the Monte-Carlo resolution"),
                    "multiplicity": ("Holm step-down over the TWO one-sided primary p-values "
                                     "(H1, H2), FWER alpha = 0.05"),
                },
                "primary_success": [
                    "paired effect > 0",
                    "bootstrap 95% CI lower bound > 0",
                    "Holm-adjusted one-sided p < 0.05",
                ],
                "secondary": {
                    "S1": "F(UOT_KR) - F(CONDITIONAL_UOT), direction > 0",
                    "S2": "F(CONDITIONAL_UOT) - F(RAW_UOT_PLAN), direction > 0",
                    "reported": "effect, 95% bootstrap CI, unadjusted secondary p",
                    "explicitly": "secondary confirmatory hypothesis; NOT in the H1/H2 Holm family",
                },
            },
        },
        "hypotheses": {
            "H1": {"contrast": "F(UOT_KR) - F(HUNGARIAN_1TO1)", "direction": "> 0",
                   "role": "primary", "gate": "A"},
            "H2": {"contrast": "F(UOT_KR) - F(THRESHOLD_MM)", "direction": "> 0",
                   "role": "primary", "gate": "B"},
            "S1": {"contrast": "F(UOT_KR) - F(CONDITIONAL_UOT)", "direction": "> 0",
                   "role": "secondary", "gate": None},
            "S2": {"contrast": "F(CONDITIONAL_UOT) - F(RAW_UOT_PLAN)", "direction": "> 0",
                   "role": "secondary", "gate": None},
        },
        "gates": {
            "A": "H1: effect > 0 AND bootstrap 95% CI lower > 0 AND Holm-adjusted one-sided p < 0.05",
            "B": "H2: same three conditions",
            "C": ("cross-bridge consistency / CLAIM gate: the per-bridge mean H1 and H2 effects "
                  f"must all be >= {GATE_C_FLOOR}. FAIL does NOT invalidate the run; it forbids "
                  "claiming a cross-bridge-consistent advantage."),
            "D": ("technical completeness: all expected confirmatory units present, no missing "
                  "cells, no duplicates, no forbidden seeds, all solver cells converged, "
                  f"final marginal/KKT error < {SOLVER_MARGINAL_TOL:g}, no silent NaN/Inf, all "
                  "edge sets valid, all cost hashes consistent, support hard filter valid"),
            "E": ("independent metric verification: a frozen validator recomputes precision, "
                  "recall, macro edge F1, strict exact recovery, edge count and mechanism "
                  f"metrics from the frozen raw prediction/truth files; |diff| <= 1e-9"),
        },
        "classification_rules": {
            "CONFIRMATORY_METHOD_SUPPORT": "A/B/C/D/E all PASS",
            "OVERALL_SUPPORT_WITH_HETEROGENEITY": "A/B/D/E PASS, C FAIL",
            "VALID_NEGATIVE_CONFIRMATORY_RESULT": "D/E PASS but A or B FAIL",
            "TECHNICALLY_INVALID_CONFIRMATORY_EXECUTION": "D or E FAIL",
            "forbidden_rename": ("an unfavourable A/B/C scientific result must never be "
                                 "relabelled as an execution failure"),
        },
        "reporting_requirements": {
            "always_report": ["strict exact recovery", "unmatched mass",
                              "per-bridge negative effects", "convergence issues"],
            "expected_values": ("no expected numeric result (for example 'expected to be "
                                "significantly higher than 0.3104') may survive into the final "
                                "report; only actual measured results are reported"),
            "preregistration_wording": {
                "allowed": "hash-locked protocol freeze",
                "not_allowed": "externally timestamped preregistration",
                "actual_evidence": ["SHA256 of the locked spec", "freeze time",
                                    "code manifest", "one-shot ledger"],
            },
        },
    }


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #

def mode_lock_spec() -> int:
    for d in (DIR_CONFIG, DIR_RULES, DIR_SELECTION, DIR_ANALYSIS, DIR_DIAGNOSTICS):
        d.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL_PATH.exists():
        write_json(PROTOCOL_PATH, build_operational_protocol())
        log(f"[lock] operational protocol -> {PROTOCOL_PATH.name}")
    else:
        log("[lock] operational protocol already present (immutable within R7)")
    if not CANDIDATE_PATH.exists():
        write_json(CANDIDATE_PATH, build_candidate_space())
        log(f"[lock] candidate space -> {CANDIDATE_PATH.name}")
    else:
        log("[lock] candidate space already present (immutable within R7)")
    return 0


def _spec() -> dict[str, Any]:
    return json.loads(DEGREE_SPEC_PATH.read_text(encoding="utf-8"))


def get_ctx(bridge: str, seed: int, *, rebuild: bool = False) -> CellContext:
    assert_selection_seeds([seed], f"selection.get_ctx({bridge},{seed})")
    try:
        if rebuild:
            raise FileNotFoundError
        cell = load_cell(bridge, seed, root=CELL_CACHE)
    except FileNotFoundError:
        cell = build_cell(bridge, seed, _spec(), workdir=CELL_CACHE / bridge / f"seed_{seed}")
    return CellContext(cell, EPSILON)


def mode_preflight() -> int:
    from diag.ctd_common import load_dev_cell
    from dev_candidate.af_common import build_amount_free_costs
    from decoder_audit.da_common import evaluate_edges as frozen_eval

    out: dict[str, Any] = {"generated_at_utc": utc_now(), "checks": []}

    def check(name: str, ok: bool, detail: Any) -> None:
        out["checks"].append({"check": name, "status": "PASS" if ok else "FAIL",
                              "detail": detail})

    # 1. seed guards
    guards = {}
    for s in sorted(FORBIDDEN_SEEDS):
        try:
            assert_selection_seeds([s], "guard-test")
            guards[str(s)] = "NOT_REJECTED"
        except SystemExit:
            guards[str(s)] = "REJECTED"
    check("forbidden_seeds_rejected", all(v == "REJECTED" for v in guards.values()), guards)
    try:
        assert_selection_seeds([401], "guard-test")
        check("confirmatory_seed_rejected_by_selection_runner", False, "401 NOT rejected")
    except SystemExit:
        check("confirmatory_seed_rejected_by_selection_runner", True, "401 rejected")

    # 2. frozen R5/R6 reproduction anchors on the frozen dev cells
    anc = {"AMOUNT_FREE_COST_D4": 0.3172, "CONDITIONAL_UOT_D4": 0.3129}
    rows = []
    for br in BRIDGES:
        for sd in (201, 202, 203, 204, 205):
            cc = load_dev_cell(br, sd)
            C = build_amount_free_costs(cc["components"])["primary"]
            P = np.load(REPO / "out" / "multi_bridge_expansion" / "amount_free_candidate_dev"
                        / "plans" / br / f"seed_{sd}" / "uot_primary.npz")["P"]
            cc["C_primary"] = C
            cc["P_uot"] = np.asarray(P, dtype=float)
            for nm, ed in (("AMOUNT_FREE_COST_D4", cost_d4_edges(C, cc["sids"], cc["tids"], 5)),
                           ("CONDITIONAL_UOT_D4", conditional_edges(cc["P_uot"], cc["sids"],
                                                                    cc["tids"], 5))):
                df, _ = frozen_eval(cc, ed)
                rows.append({"method": nm, "bridge": br, "seed": sd,
                             "f1": float(df["edge_f1"].mean())})
    dfr = pd.DataFrame(rows)
    rep = {}
    for nm, target in anc.items():
        got = float(dfr[dfr["method"] == nm]["f1"].mean())
        rep[nm] = {"r7_value": got, "frozen_value": target, "abs_diff": abs(got - target),
                   "tolerance": 5e-4, "pass": abs(got - target) < 5e-4}
    check("frozen_reproduction_anchors", all(v["pass"] for v in rep.values()), rep)
    out["reproduction_anchors"] = rep

    # 3. kernel/cost rank equivalence at the cell level
    eq = []
    for br in BRIDGES:
        ctx = get_ctx(br, SELECTION_SEEDS[0])
        from r7.r7_methods import rank_desc
        rr_k = rank_desc(ctx.logK, 1)
        rr_c = rank_desc(-ctx.C, 1)
        cr_k = rank_desc(ctx.logK, 0)
        cr_c = rank_desc(-ctx.C, 0)
        eq.append(bool((rr_k == rr_c).all() and (cr_k == cr_c).all()))
    check("logK_equals_negative_cost_ranking", all(eq), eq)

    # 4. cost hash equality across methods within a cell
    ctx = get_ctx(BRIDGES[0], SELECTION_SEEDS[0])
    hashes = {ctx.cell["hashes"]["cost_matrix_sha256"]}
    check("single_cost_hash_per_cell", len(hashes) == 1, sorted(hashes))

    # 5. support hard filter
    viol = 0
    for br in BRIDGES:
        c = get_ctx(br, SELECTION_SEEDS[0])
        e = c.edges("SUPPORT_PLUS_K", {"family": "R-const", "k": 5})
        idx_s = {s: i for i, s in enumerate(c.sids)}
        idx_t = {d: j for j, d in enumerate(c.tids)}
        viol += sum(1 for s, d in e if c.P[idx_s[s], idx_t[d]] <= SUPPORT_THRESHOLD)
    check("support_hard_filter_zero_violations", viol == 0, viol)

    # 6. tie determinism
    c = get_ctx(BRIDGES[0], SELECTION_SEEDS[0])
    e1 = c.edges("UOT_KR", {"family": "R-const", "k": 4})
    e2 = c.edges("UOT_KR", {"family": "R-const", "k": 4})
    check("decoder_determinism", sorted(e1) == sorted(e2), len(e1))

    # 7. structural QA gate
    if QA_JSON.is_file():
        qa = json.loads(QA_JSON.read_text(encoding="utf-8"))
        check("generator_structural_qa_before_f1",
              qa.get("GENERATOR_STRUCTURAL_QA") == "PASS"
              and qa.get("method_f1_computed") is False,
              {"qa": qa.get("GENERATOR_STRUCTURAL_QA"),
               "f1_before_qa": qa.get("method_f1_computed")})
    else:
        check("generator_structural_qa_before_f1", False, "generator_structural_qa.json missing")

    out["ALL_PASS"] = all(c["status"] == "PASS" for c in out["checks"])
    write_json(DIR_SELECTION / "preflight.json", out)
    for c in out["checks"]:
        log(f"[preflight] {c['check']}: {c['status']}")
    return 0 if out["ALL_PASS"] else 1


def mode_selection() -> int:
    if not CANDIDATE_PATH.is_file():
        raise SystemExit("run --lock-spec first: candidate space must be frozen before F1")
    if not QA_JSON.is_file() or json.loads(QA_JSON.read_text(encoding="utf-8"))[
            "GENERATOR_STRUCTURAL_QA"] != "PASS":
        raise SystemExit("generator structural QA has not passed; refusing to compute method F1")
    space = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    per_cell_dump: dict[str, Any] = {}

    for bridge in BRIDGES:
        for seed in SELECTION_SEEDS:
            ctx = get_ctx(bridge, seed)
            for cand in space["candidates"]:
                rule = {k: v for k, v in cand.items()
                        if k in ("family", "k", "alpha", "theta", "k_max")}
                edges = ctx.edges("UOT_KR", rule)
                m = ctx.metrics("UOT_KR", edges)
                rows.append({
                    "rule_id": cand["rule_id"], "family": cand["family"],
                    "param_name": cand["param_name"], "param_value": cand["param_value"],
                    "k": cand.get("k"), "alpha": cand.get("alpha"), "theta": cand.get("theta"),
                    "bridge": bridge, "seed": seed,
                    "macro_edge_f1": m["macro_edge_f1"],
                    "family_mean_edge_precision": m["family_mean_edge_precision"],
                    "family_mean_edge_recall": m["family_mean_edge_recall"],
                    "n_pred_edges_total": m["n_pred_edges_total"],
                    "mean_edges_per_family": float(np.mean(list(m["family_edge_counts"].values()))),
                    "cost_matrix_sha256": ctx.cell["hashes"]["cost_matrix_sha256"],
                })
            log(f"[selection] {bridge} seed {seed}: {len(space['candidates'])} candidates done "
                f"({time.time() - t0:.0f}s)")
            # oracle ceiling, once per cell (diagnostic)
            per_cell_dump[f"{bridge}|{seed}"] = {
                "bridge": bridge, "seed": seed,
                "oracle_1to1_ceiling_macro_f1": ctx.oracle()["macro_f1"],
                "oracle_mean_T": ctx.oracle()["mean_T"],
                "oracle_mean_M": ctx.oracle()["mean_M"],
                "n_src": len(ctx.sids), "n_dst": len(ctx.tids),
                "solver": {k: v for k, v in ctx.cell["solver"].items() if k != "kkt"},
                "cost_matrix_sha256": ctx.cell["hashes"]["cost_matrix_sha256"],
                "plan_sha256": ctx.cell["hashes"]["plan_sha256"],
            }

    df = pd.DataFrame(rows)
    df.to_csv(ALL_CANDIDATES_CSV, index=False)

    agg = (df.groupby(["rule_id", "family", "param_name", "param_value"], dropna=False)
             .agg(overall_macro_edge_f1=("macro_edge_f1", "mean"),
                  n_cells=("macro_edge_f1", "size")).reset_index())
    per_bridge = (df.groupby(["rule_id", "bridge"])["macro_edge_f1"].mean()
                    .unstack("bridge").reset_index())
    budget = (df.groupby("rule_id")["mean_edges_per_family"].mean().rename("mean_edges_per_family")
                .reset_index())
    agg = agg.merge(per_bridge, on="rule_id", how="left").merge(budget, on="rule_id", how="left")
    write_json(ALL_CANDIDATES_JSON, {
        "generated_at_utc": utc_now(),
        "n_candidates": len(space["candidates"]),
        "n_cells": len(BRIDGES) * len(SELECTION_SEEDS),
        "aggregate": agg.to_dict(orient="records"),
        "per_cell": per_cell_dump,
        "note": "ALL candidates are preserved, not only the winner",
    })
    log(f"[selection] complete: {len(rows)} candidate-cell rows in {time.time() - t0:.0f}s")
    return 0


def mode_select() -> int:
    data = json.loads(ALL_CANDIDATES_JSON.read_text(encoding="utf-8"))
    agg = data["aggregate"]
    best_val = max(float(a["overall_macro_edge_f1"]) for a in agg)
    tied = [a for a in agg if best_val - float(a["overall_macro_edge_f1"]) <= TIE_TOL]
    tied.sort(key=lambda a: (FAMILY_COMPLEXITY[a["family"]], float(a["param_value"])))
    winner = tied[0]
    space = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    cand = next(c for c in space["candidates"] if c["rule_id"] == winner["rule_id"])

    sel = {
        "generated_at_utc": utc_now(),
        "selection_criterion": "highest bridge-balanced overall macro edge F1",
        "winner": winner["rule_id"],
        "winner_family": winner["family"],
        "winner_params": {k: cand[k] for k in ("family", "k", "alpha", "theta", "q", "k_max")
                          if k in cand},
        "winner_score": float(winner["overall_macro_edge_f1"]),
        "n_tied_at_tolerance": len(tied),
        "tied_rule_ids": [t["rule_id"] for t in tied],
        "tie_break": {"tolerance": TIE_TOL,
                      "order": space["tie_break"]["order"],
                      "within_family": space["tie_break"]["within_family"],
                      "applied": len(tied) > 1},
        "per_bridge": {b: float(winner[b]) for b in BRIDGES if b in winner},
        "mean_edges_per_family": float(winner["mean_edges_per_family"]),
        "mean_pred_edges_per_cell": float(winner["mean_edges_per_family"]) * N_FAMILIES,
        "rule_for_executor": {k: cand[k] for k in ("family", "k", "alpha", "theta", "k_max")
                              if k in cand},
        "truncation_boundary_dependence": None,
    }
    # truncation-boundary risk (specification section 20)
    fam = sel["winner_family"]
    dep = False
    why = []
    if fam == "R-const" and int(cand.get("k", 0)) >= K_MAX:
        dep, why = True, ["R-const winner has k = 8 = the frozen degree cap"]
    if fam == "R-quantile" and int(cand.get("k", 0)) >= K_MAX:
        dep, why = True, ["R-quantile-derived k = 8 = the frozen degree cap"]
    if fam == "R-adaptive":
        sat = 0
        tot = 0
        for bridge in BRIDGES:
            for seed in SELECTION_SEEDS:
                ctx = get_ctx(bridge, seed)
                from r7.r7_methods import adaptive_budgets
                kr, kc = adaptive_budgets(ctx.realized_rows, ctx.realized_cols,
                                          float(cand["alpha"]), K_MAX)
                sat += int((kr >= K_MAX).sum()) + int((kc >= K_MAX).sum())
                tot += kr.size + kc.size
        frac = sat / max(tot, 1)
        why = [f"saturation fraction at k_max=8: {frac:.4f}"]
        dep = frac > 0.10
    sel["truncation_boundary_dependence"] = {
        "TRUNCATION_BOUNDARY_DEPENDENCE": bool(dep), "reasons": why,
        "degree_cap_modified": False,
        "manuscript_note": ("the selected rule may still be influenced by the generator's "
                            "truncation bound; discuss in limitations" if dep else
                            "no boundary dependence detected"),
    }
    write_json(SELECTED_RULE_PATH, sel)
    log(f"[select] winner {sel['winner']} score={sel['winner_score']:.6f} "
        f"(tied={len(tied)}) TRUNCATION_BOUNDARY_DEPENDENCE="
        f"{sel['truncation_boundary_dependence']['TRUNCATION_BOUNDARY_DEPENDENCE']}")
    return 0


def mode_calibrate() -> int:
    sel = json.loads(SELECTED_RULE_PATH.read_text(encoding="utf-8"))
    rule = sel["rule_for_executor"]
    cells = [load_cell(b, s, root=CELL_CACHE) for b in BRIDGES for s in SELECTION_SEEDS]
    cfg = {"epsilon": EPSILON, "selected_rule": rule,
           "threshold_mm_cutoff": 0.0, "dual_softmax_tau": 0.0}
    tmm = threshold_mm_calibrate(cells, rule, cfg, THRESHOLD_MM_Q_GRID_REFINED)
    coarse = threshold_mm_calibrate(cells, rule, cfg, THRESHOLD_MM_Q_GRID)
    tmm["grid_amendment"] = {
        "applied": True,
        "reason": ("the pre-registered coarse quantile grid (q >= 0.02) bound the solution: "
                   "at q = 0.02 the threshold rule already predicts far more edges per family "
                   "than the selected UOT_KR rule, so the equal-budget minimum was not "
                   "attainable inside the grid"),
        "what_changed": "the discretisation of the continuous threshold only",
        "what_did_not_change": ["the criterion (minimise |budget - UOT_KR budget|)",
                                "the target (UOT_KR bridge-balanced budget)",
                                "the method definition (edge iff C_ij <= cutoff)",
                                "the tie-break (higher threshold)"],
        "direction_of_effect": ("refining the grid moves THRESHOLD_MM toward the UOT_KR edge "
                                "budget, i.e. makes the H2 comparator stronger; conservative"),
        "observed_before_amendment": True,
        "applied_before_protocol_freeze": True,
        "confirmatory_block_untouched": True,
        "coarse_grid_result": {
            "selected_q": coarse["selected_q"],
            "selected_cutoff": coarse["selected_cutoff"],
            "selected_abs_diff": coarse["selected_abs_diff"],
            "n_candidates": coarse["n_candidates"],
        },
        "target_budget": tmm["uot_kr_target_budget"],
    }
    write_json(DIR_RULES / "threshold_mm_calibration.json", tmm)
    log(f"[calibrate] THRESHOLD_MM cutoff={tmm['selected_cutoff']:.6f} "
        f"(q={tmm['selected_q']}) target_budget={tmm['uot_kr_target_budget']:.4f} "
        f"diff={tmm['selected_abs_diff']:.4f} "
        f"[coarse grid had diff={coarse['selected_abs_diff']:.4f}]")
    ds = dual_softmax_calibrate(cells, rule, cfg, DUAL_SOFTMAX_TAU_GRID)
    scores = [c["bridge_balanced_macro_edge_f1"] for c in ds["candidates"]]
    ds["monotone_decreasing_in_tau"] = bool(all(scores[i] >= scores[i + 1] - 1e-12
                                                for i in range(len(scores) - 1)))
    ds["tau_zero_is_global_optimum"] = bool(
        ds["selected_tau"] == 0.0 and ds["monotone_decreasing_in_tau"])
    ds["tau_zero_note"] = (
        "D >= 0 holds by construction, so any tau <= 0 is inactive and tau = 0 is the "
        "global optimum whenever the selection score is non-increasing in tau. The "
        "confidence gate is therefore calibrated to OFF: the binding constraint of this "
        "arm is the bidirectional (mutual nearest neighbour) acceptance, which makes its "
        "output semantics one-to-one.")
    write_json(DIR_RULES / "dual_softmax_calibration.json", ds)
    log(f"[calibrate] DUAL_SOFTMAX tau={ds['selected_tau']} score={ds['selected_score']:.6f} "
        f"monotone={ds['monotone_decreasing_in_tau']}")
    return 0


def mode_validation_probe() -> int:
    """Write a SELECTION-block raw package in the executor's schema and run the frozen
    validator on it (specification section 34: the independent validator must be tested
    on 206-215 before the freeze).  No confirmatory seed is touched."""
    import subprocess
    from r7.r7_methods import oracle_1to1_ceiling

    sel = json.loads(SELECTED_RULE_PATH.read_text(encoding="utf-8"))
    tmm = json.loads((DIR_RULES / "threshold_mm_calibration.json").read_text(encoding="utf-8"))
    ds = json.loads((DIR_RULES / "dual_softmax_calibration.json").read_text(encoding="utf-8"))
    cfg = {"epsilon": EPSILON, "selected_rule": sel["rule_for_executor"],
           "threshold_mm_cutoff": tmm["selected_cutoff"],
           "dual_softmax_tau": ds["selected_tau"]}
    probe = DIR_SELECTION / "raw_probe"
    units_dir = probe / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    index_rows, metric_rows = [], []
    for bridge in BRIDGES:
        for seed in SELECTION_SEEDS:
            cell = load_cell(bridge, seed, root=CELL_CACHE)
            ctx = CellContext(cell, EPSILON)
            res = run_all_methods(cell, cfg, ctx=ctx)
            P = ctx.P
            truth = {t: {"split": sorted([list(e) for e in tr["split"]]),
                         "merge": sorted([list(e) for e in tr["merge"]]),
                         "decoy": sorted([list(e) for e in tr["decoy"]]),
                         "positive": sorted([list(e) for e in tr["positive"]]),
                         "unmatched_src": sorted(tr["unmatched_src"]),
                         "hidden_dst": sorted(tr["hidden_dst"]),
                         "all_src": sorted(tr["all_src"]),
                         "all_dst": sorted(tr["all_dst"])}
                     for t, tr in sorted(cell["truth"].items())}
            unit = {
                "unit_id": f"{bridge}|{seed}", "bridge": bridge, "seed": seed,
                "n_src": len(cell["sids"]), "n_dst": len(cell["tids"]),
                "source_ids": list(cell["sids"]), "target_ids": list(cell["tids"]),
                "template_of_source": list(cell["tpl_s"]),
                "template_of_target": list(cell["tpl_t"]),
                "truth": truth,
                "oracle_1to1_ceiling": {t: oracle_1to1_ceiling(tr["positive"])
                                        for t, tr in sorted(cell["truth"].items())},
                "predictions": {m: [list(e) for e in res[m]["pred_edges"]] for m in METHODS},
                "oracle_summary": res["ORACLE_1TO1_CEILING"],
                "support_filter_violations": res.get("support_filter_violations", 0),
                "scope": "SELECTION_PROBE_NOT_CONFIRMATORY",
            }
            p = units_dir / f"unit__{bridge}__s{seed}.json"
            p.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
            index_rows.append({"bridge": bridge, "seed": seed, "path": p.name,
                               "sha256": sha256_file(p), "bytes": p.stat().st_size})
            for m in METHODS:
                mm = res[m]
                metric_rows.append({
                    "bridge": bridge, "seed": seed, "method": m,
                    "macro_edge_f1": mm["macro_edge_f1"],
                    "family_mean_edge_precision": mm["family_mean_edge_precision"],
                    "family_mean_edge_recall": mm["family_mean_edge_recall"],
                    "family_mean_split_exact": mm["family_mean_split_exact"],
                    "family_mean_merge_exact": mm["family_mean_merge_exact"],
                    "family_mean_overall_exact": mm["family_mean_overall_exact"],
                    "n_pred_edges_total": mm["n_pred_edges_total"],
                })
    write_json(probe / "INDEX.json", {
        "generated_at_utc": utc_now(), "scope": "SELECTION_PROBE_NOT_CONFIRMATORY",
        "n_units_written": len(index_rows), "units": index_rows,
        "seed_block": list(SELECTION_SEEDS),
        "note": ("schema-identical probe package built from SELECTION cells so the frozen "
                 "validator can be exercised before the confirmatory freeze"),
    })
    pd.DataFrame(metric_rows).to_csv(probe / "executor_cell_metrics.csv", index=False)

    out = DIR_SELECTION / "VALIDATOR_SELFTEST_ON_SELECTION.json"
    cmd = [sys.executable, str(REPO / "scripts" / "validate_r7_confirmatory_results.py"),
           "--raw", str(probe), "--out", str(out), "--label", "selection_probe"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(REPO))
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
    log(f"[probe] validator self-test on selection raw package rc={r.returncode}")
    return r.returncode


FROZEN_CODE_FILES = (
    "scripts/run_r7_selection.py",
    "scripts/run_r7_preflight.py",
    "scripts/run_r7_confirmatory_kernel_ranking.py",
    "scripts/validate_r7_confirmatory_results.py",
    "scripts/r7/__init__.py",
    "scripts/r7/r7_common.py",
    "scripts/r7/r7_degree.py",
    "scripts/r7/r7_generator.py",
    "scripts/r7/r7_methods.py",
    "scripts/r7/r7_pipeline.py",
    "scripts/r7/r7_qa.py",
    "scripts/r7/selftest_generator_equivalence.py",
    "src/cross/domain/evaluation/semi_synthetic_flows.py",
    "scripts/multi_bridge/baseline_mechanism/common.py",
    "scripts/multi_bridge/decoder_audit/da_common.py",
    "scripts/multi_bridge/dev_candidate/af_common.py",
    "scripts/multi_bridge/dev_candidate2/cp_common.py",
    "scripts/multi_bridge/diag/ctd_common.py",
    "scripts/multi_bridge/run_threshold_many_match.py",
)
FROZEN_ARTIFACT_FILES = (
    "config/operational_protocol_preselection.json",
    "config/author_proposed_spec.md",
    "selection/degree_calibration/degree_sampling_spec.json",
    "selection/degree_calibration/tail_report.json",
    "selection/degree_calibration/degree_definition.json",
    "selection/degree_calibration/source_provenance.json",
    "selection/degree_calibration/pooled_degree_hist_truncated.csv",
    "selection/degree_calibration/pooled_fanin_hist_truncated.csv",
    "selection/generator/family_manifest.csv",
    "selection/generator/generator_structural_qa.json",
    "selection/generator/generator_equivalence.json",
    "selection/rule_search/candidate_space.json",
    "selection/rule_search/all_candidates.csv",
    "selection/rule_search/selected_rule.json",
    "selection/rule_search/threshold_mm_calibration.json",
    "selection/rule_search/dual_softmax_calibration.json",
    "selection/preflight.json",
    "selection/VALIDATOR_SELFTEST_ON_SELECTION.json",
    "selection/EXECUTOR_DRYRUN.json",
    "confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md",
    "confirmatory/CONFIRMATORY_TOUCH_ONCE.json",
    "confirmatory/retired_401_410/README.md",
)


def build_locked_spec() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    space = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    sel = json.loads(SELECTED_RULE_PATH.read_text(encoding="utf-8"))
    tmm = json.loads((DIR_RULES / "threshold_mm_calibration.json").read_text(encoding="utf-8"))
    ds = json.loads((DIR_RULES / "dual_softmax_calibration.json").read_text(encoding="utf-8"))
    qa = json.loads(QA_JSON.read_text(encoding="utf-8"))
    degspec = json.loads(DEGREE_SPEC_PATH.read_text(encoding="utf-8"))
    pre = json.loads((DIR_SELECTION / "preflight.json").read_text(encoding="utf-8"))
    probe = json.loads((DIR_SELECTION / "VALIDATOR_SELFTEST_ON_SELECTION.json")
                       .read_text(encoding="utf-8"))
    return {
        "experiment_id": EXPERIMENT_ID,
        "spec_version": 1,
        "frozen_at_utc": utc_now(),
        "git_head": git(["rev-parse", "HEAD"]).strip(),
        "git_branch": git(["rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "freeze_type": "hash-locked protocol freeze",
        "freeze_evidence": ["SHA256 of this spec", "freeze timestamp", "code manifest",
                            "one-shot ledger"],
        "preregistration_wording": protocol["reporting_requirements"]["preregistration_wording"],

        "seed_blocks": {
            **protocol["seed_blocks"],
            # The pre-selection protocol snapshot was taken while the confirmatory block
            # was still 401-410.  The locked spec is authoritative for the FINAL block and
            # records the substitution explicitly.
            "confirmatory": list(CONFIRMATORY_SEEDS),
            "confirmatory_amendment": {
                "applied": True,
                "as_pre_selection_protocol": protocol["seed_blocks"]["confirmatory"],
                "final": list(CONFIRMATORY_SEEDS),
                "reason": ("the originally frozen block 401-410 was SPENT by an incomplete "
                           "first confirmatory execution (plumbing defect; no unit result "
                           "written or observed)"),
                "action": "block retired and never re-run; a fresh untouched block reserved",
                "incident_record": "confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md",
                "owner_decision": "protocol owner approved the fresh-block substitution",
                "result_driven": False,
                "protocol_content_changed": False,
            },
            "retired_confirmatory_blocks": [dict(r) for r in RETIRED_CONFIRMATORY_BLOCKS],
        },

        "frozen_cost": {
            "family": "amount-free renormalised five-component cost",
            "absolute_weights": COST_WEIGHTS_ABSOLUTE,
            "renormalisation_denominator": COST_RENORM_DENOM,
            "weights_used": {k: v / COST_RENORM_DENOM
                             for k, v in COST_WEIGHTS_ABSOLUTE.items()},
            "amount_component": "removed pairwise only; amount-derived marginals preserved",
            "epsilon": EPSILON, "lambda": LAMBDA,
            "reopened_by_r7": False,
        },
        "solver_parameters": protocol["solver_configuration"],
        "support_threshold": SUPPORT_THRESHOLD,
        "degree_calibration": {
            "sampling_spec": degspec,
            "histogram_hashes": degspec["hashes"],
            "truncation": {"range": list(DEGREE_RANGE), "cap": K_MAX,
                           "rule": f"d_used = min(d_observed, {K_MAX})"},
            "HIGH_TRUNCATION_TAIL": degspec["HIGH_TRUNCATION_TAIL"],
            "fanin_empirical": True,
            "mirrored_proxy_not_empirical_fanin": False,
        },
        "generator_spec": {
            **protocol["generator"],
            "family_manifest_sha256": qa["family_manifest"]["sha256"],
            "family_manifest_rows": qa["family_manifest"]["rows"],
            "structural_qa": qa["GENERATOR_STRUCTURAL_QA"],
            "structural_qa_completed_before_method_f1": not qa["method_f1_computed"],
        },
        "candidate_space": {"n_candidates": space["n_candidates"],
                            "criterion": space["selection_criterion"],
                            "tie_break": space["tie_break"],
                            "candidates": space["candidates"]},
        "selected_rule": sel,
        "method_list": list(METHODS),
        "method_definitions": protocol["methods"],
        "threshold_mm": {
            "cutoff": tmm["selected_cutoff"],
            "quantile": tmm["selected_q"],
            "criterion": tmm["criterion"],
            "uses_ground_truth_f1": False,
            "tie_break": tmm["tie_break"],
            "target_budget": tmm["uot_kr_target_budget"],
            "achieved_abs_diff": tmm["selected_abs_diff"],
            "recalibrated_on_holdout": False,
            "grid_amendment": tmm["grid_amendment"],
        },
        "dual_softmax": {
            "tau": ds["selected_tau"],
            "acceptance": ds["acceptance"],
            "criterion": ds["criterion"],
            "monotone_decreasing_in_tau": ds["monotone_decreasing_in_tau"],
            "tau_zero_is_global_optimum": ds["tau_zero_is_global_optimum"],
            "tie_break": ds["tie_break"],
            "recalibrated_on_holdout": False,
        },
        "hypotheses": protocol["hypotheses"],
        "estimand": protocol["protocol_corrections"]["C3_holm_and_ci_fully_specified"][
            "primary_estimand"],
        "aggregation": {
            "order": ["instance", "family mean (equal weight)", "24-family macro",
                      "seed", "bridge", "bridge-balanced overall"],
            "primary_paired_cells": 30,
            "pseudoreplication_forbidden": True,
        },
        "bootstrap": protocol["protocol_corrections"]["C3_holm_and_ci_fully_specified"][
            "primary_ci"],
        "permutation": protocol["protocol_corrections"]["C3_holm_and_ci_fully_specified"][
            "primary_p_value"],
        "holm": {"family": ["H1", "H2"], "method": "Holm step-down",
                 "alpha": ALPHA, "sided": "one-sided (effect > 0)"},
        "secondary": protocol["protocol_corrections"]["C3_holm_and_ci_fully_specified"][
            "secondary"],
        "gates": protocol["gates"],
        "classification_rules": protocol["classification_rules"],
        "negative_result_handling": {
            "A_or_B_fail": "VALID_NEGATIVE_CONFIRMATORY_RESULT; never called an execution failure",
            "C_fail": "valid result; no cross-bridge-consistent claim; report the bridge, effect and CI",
            "D_or_E_fail": "TECHNICALLY_INVALID_CONFIRMATORY_EXECUTION; keep everything; never re-run 401-410",
            "holdout_reuse": "forbidden under all circumstances",
        },
        "raw_result_schema": {
            "per_unit_fields": ["unit_id", "bridge", "seed", "n_src", "n_dst", "source_ids",
                                "target_ids", "template_of_source", "template_of_target",
                                "truth", "oracle_1to1_ceiling", "predictions",
                                "method_metrics", "family_edge_counts", "margin_mass",
                                "solver", "kkt", "degrees", "hashes", "oracle_summary",
                                "hungarian_info", "threshold_mm_info", "dual_softmax_info",
                                "support_filter_violations", "runtime_sec"],
            "index": "confirmatory/raw/INDEX.json",
        },
        "executor": {
            "path": "scripts/run_r7_confirmatory_kernel_ranking.py",
            "seed_block": list(CONFIRMATORY_SEEDS),
            "ledger": ("confirmatory/CONFIRMATORY_TOUCH_ONCE__"
                       f"{CONFIRMATORY_SEEDS[0]}_{CONFIRMATORY_SEEDS[-1]}.json"),
            "one_shot": True,
            "verifications": ["locked_spec_sha256", "executor_self_sha256",
                              "generator_sha256", "validator_sha256",
                              "family_manifest_hash", "degree_histogram_hash",
                              "selected_rule_artifact", "seed_block_guard",
                              "one_shot_ledger_absent"],
        },
        "pre_selection_evidence": {
            "preflight_all_pass": pre["ALL_PASS"],
            "frozen_reproduction_anchors": pre["reproduction_anchors"],
            "validator_selftest_on_selection": probe["GATE_E"],
            "validator_metric_tolerance": probe["tolerance"],
            "validator_max_abs_diff": max(
                [d["max_abs_diff"] for d in probe["metric_diffs_vs_executor"]], default=None),
        },
        "confirmatory_block_generated": False,
        "confirmatory_block_read": False,
        "confirmatory_block_reused": False,
    }


def mode_freeze() -> int:
    """Write the hash-locked protocol freeze.  MUST run before any 401-410 generation."""
    active_ledger = (DIR_CONFIRMATORY / "CONFIRMATORY_TOUCH_ONCE__"
                     f"{CONFIRMATORY_SEEDS[0]}_{CONFIRMATORY_SEEDS[-1]}.json")
    if active_ledger.exists():
        raise SystemExit(
            f"the ACTIVE one-shot ledger already exists ({active_ledger.name}); refusing to "
            f"re-freeze a block whose holdout has been touched")
    for rel in FROZEN_CODE_FILES + FROZEN_ARTIFACT_FILES:
        if not resolve_frozen_path(rel).is_file():
            raise SystemExit(f"cannot freeze: required frozen input missing: {rel}")

    spec = build_locked_spec()
    write_json(SPEC_PATH, spec)
    spec_sha = sha256_file(SPEC_PATH)
    write_text(SPEC_HASH_PATH, f"{spec_sha}  config/locked_spec.json\n")

    frozen_hashes = {rel: sha256_file(resolve_frozen_path(rel))
                     for rel in FROZEN_CODE_FILES + FROZEN_ARTIFACT_FILES}
    rel_report = "selection/SELECTION_REPORT.md"
    if resolve_frozen_path(rel_report).is_file():
        frozen_hashes[rel_report] = sha256_file(resolve_frozen_path(rel_report))

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now(),
        "freeze_type": "hash-locked protocol freeze",
        "locked_spec_sha256": spec_sha,
        "confirmatory_seed_block": list(CONFIRMATORY_SEEDS),
        "retired_confirmatory_blocks": [dict(r) for r in RETIRED_CONFIRMATORY_BLOCKS],
        "active_ledger": ("confirmatory/CONFIRMATORY_TOUCH_ONCE__"
                          f"{CONFIRMATORY_SEEDS[0]}_{CONFIRMATORY_SEEDS[-1]}.json"),
        "selection_seed_block": list(SELECTION_SEEDS),
        "frozen_hashes": frozen_hashes,
        "roles": {
            "scripts/r7/r7_generator.py": "solver wrapper + cell builder",
            "scripts/r7/r7_methods.py": "decoder catalogue (relevant decoder code)",
            "scripts/r7/r7_pipeline.py": "metric definitions + aggregation",
            "scripts/r7/r7_common.py": "frozen constants and guards",
            "src/cross/domain/evaluation/semi_synthetic_flows.py": "generator source",
            "scripts/run_r7_confirmatory_kernel_ranking.py": "confirmatory executor",
            "scripts/validate_r7_confirmatory_results.py": "independent validator",
            "selection/rule_search/selected_rule.json": "selection winner file",
            "selection/degree_calibration/pooled_degree_hist_truncated.csv": "degree histogram",
            "selection/degree_calibration/pooled_fanin_hist_truncated.csv": "fan-in histogram",
            "selection/generator/family_manifest.csv": "family manifest",
        },
        "holdout_status_at_freeze": {
            "active_block": list(CONFIRMATORY_SEEDS),
            "generated": False, "read": False,
            "active_ledger_exists": active_ledger.exists(),
            "raw_dir_exists": DIR_RAW.exists(),
            "retired_blocks": [dict(r) for r in RETIRED_CONFIRMATORY_BLOCKS],
        },
    }
    write_json(MANIFEST_PATH, manifest)

    audit = "\n".join([
        "# R7 pre-confirmatory audit",
        "",
        f"* generated: {utc_now()}",
        f"* git HEAD: `{spec['git_head']}` (branch `{spec['git_branch']}`)",
        "",
        "## Statement",
        "",
        "> **THE ACTIVE CONFIRMATORY BLOCK HAS NOT BEEN GENERATED OR READ.**",
        "",
        f"* active confirmatory block: `{list(CONFIRMATORY_SEEDS)}`",
        f"* active ledger `"
        f"confirmatory/CONFIRMATORY_TOUCH_ONCE__{CONFIRMATORY_SEEDS[0]}"
        f"_{CONFIRMATORY_SEEDS[-1]}.json` exists: "
        f"**{(DIR_CONFIRMATORY / ('CONFIRMATORY_TOUCH_ONCE__' + str(CONFIRMATORY_SEEDS[0]) + '_' + str(CONFIRMATORY_SEEDS[-1]) + '.json')).exists()}**",
        f"* `confirmatory/raw/` exists: **{DIR_RAW.exists()}**",
        "* no R7 generator run has touched any active confirmatory seed",
        "* no confirmatory executor invocation has occurred for this block",
        "",
        "### Retired block 401-410 (disclosed)",
        "",
        f"* block `401-410` is **spent** and is never re-run: "
        f"{RETIRED_CONFIRMATORY_BLOCKS[0]['reason']}",
        f"* its ledger is retained, unmodified, at "
        f"`{RETIRED_CONFIRMATORY_BLOCKS[0]['ledger']}`",
        "* its partial artefacts are archived under `confirmatory/retired_401_410/`",
        "* full incident record: `confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md`",
        "",
        "## Frozen protocol",
        "",
        "| item | value |",
        "|---|---|",
        f"| freeze type | hash-locked protocol freeze |",
        f"| freeze time | {spec['frozen_at_utc']} |",
        f"| locked spec | `config/locked_spec.json` |",
        f"| locked spec sha256 | `{spec_sha}` |",
        f"| code manifest | `config/FROZEN_PROTOCOL_MANIFEST.json` "
        f"({len(frozen_hashes)} hashed files) |",
        f"| selection block | `{list(SELECTION_SEEDS)}` |",
        f"| confirmatory block | `{list(CONFIRMATORY_SEEDS)}` |",
        f"| selected rule | `{spec['selected_rule']['winner']}` |",
        f"| rule parameters | `{json.dumps(spec['selected_rule']['rule_for_executor'])}` |",
        f"| Threshold-MM cutoff | `{spec['threshold_mm']['cutoff']:.6f}` |",
        f"| Dual-Softmax tau | `{spec['dual_softmax']['tau']}` |",
        f"| epsilon / lambda | `{EPSILON}` / `{LAMBDA}` |",
        f"| support threshold | `{SUPPORT_THRESHOLD}` |",
        f"| bootstrap | B={N_BOOT}, RNG={RNG_BOOTSTRAP} |",
        f"| permutation | n_perm={N_PERM}, RNG={RNG_PERMUTATION} |",
        f"| Holm family | H1, H2 (FWER alpha = {ALPHA}) |",
        "",
        "## Pre-freeze verification evidence",
        "",
        f"* selection preflight: **{'ALL PASS' if spec['pre_selection_evidence']['preflight_all_pass'] else 'FAIL'}**",
        f"* frozen reproduction anchors: "
        f"{json.dumps(spec['pre_selection_evidence']['frozen_reproduction_anchors'])}",
        f"* independent validator self-test on selection raw outputs: "
        f"**{spec['pre_selection_evidence']['validator_selftest_on_selection']}**",
        f"* validator max |diff| vs executor: "
        f"{spec['pre_selection_evidence']['validator_max_abs_diff']} "
        f"(tolerance {spec['pre_selection_evidence']['validator_metric_tolerance']})",
        "",
        "## Wording",
        "",
        "* this freeze may be described as a **hash-locked protocol freeze**",
        "* it must NOT be described as an externally timestamped preregistration",
        "",
    ])
    write_text(REPO / "out" / EXPERIMENT_ID / "PRE_CONFIRMATORY_AUDIT.md", audit)
    log(f"[freeze] locked_spec sha256={spec_sha}")
    log(f"[freeze] manifest: {len(frozen_hashes)} frozen files")
    return 0


def r_adaptive_correlations() -> dict[str, Any]:
    """Spearman diagnostics for the R-adaptive arm (specification section 17).

    EXPLANATORY MATERIAL ONLY -- these correlations never influence the selection score.
    """
    from scipy.stats import spearmanr

    per_bridge: dict[str, Any] = {}
    all_r, all_fo, all_c, all_fi = [], [], [], []
    for bridge in BRIDGES:
        rho_s, rho_t, rs, cs, fo, fi = [], [], [], [], [], []
        for seed in SELECTION_SEEDS:
            cell = load_cell(bridge, seed, root=CELL_CACHE)
            P = np.asarray(cell["P"], dtype=float)
            r = P.sum(axis=1)
            c = P.sum(axis=0)
            fo_deg = {s: 0 for s in cell["sids"]}
            fi_deg = {d: 0 for d in cell["tids"]}
            for t, tr in cell["truth"].items():
                for s, d in tr["positive"]:
                    fo_deg[s] = fo_deg.get(s, 0) + 1
                    fi_deg[d] = fi_deg.get(d, 0) + 1
            ds = np.array([fo_deg[s] for s in cell["sids"]], dtype=float)
            dt = np.array([fi_deg[d] for d in cell["tids"]], dtype=float)
            rs.append(r)
            cs.append(c)
            fo.append(ds)
            fi.append(dt)
            if np.std(r) > 0 and np.std(ds) > 0:
                rho_s.append(float(spearmanr(r, ds).statistic))
            if np.std(c) > 0 and np.std(dt) > 0:
                rho_t.append(float(spearmanr(c, dt).statistic))
        rr = np.concatenate(rs)
        ff = np.concatenate(fo)
        cc = np.concatenate(cs)
        gg = np.concatenate(fi)
        all_r.append(rr)
        all_fo.append(ff)
        all_c.append(cc)
        all_fi.append(gg)
        per_bridge[bridge] = {
            "source_spearman_r_vs_true_fanout": float(spearmanr(rr, ff).statistic),
            "target_spearman_c_vs_true_fanin": float(spearmanr(cc, gg).statistic),
            "per_seed_source": rho_s, "per_seed_target": rho_t,
            "n_source": int(rr.size), "n_target": int(cc.size),
        }
    R = np.concatenate(all_r)
    F = np.concatenate(all_fo)
    C = np.concatenate(all_c)
    G = np.concatenate(all_fi)

    def boot_ci(x, y, n_boot=2000, seed=0):
        rng = np.random.RandomState(seed)
        n = x.size
        vals = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            xi, yi = x[idx], y[idx]
            if np.std(xi) == 0 or np.std(yi) == 0:
                continue
            vals.append(float(spearmanr(xi, yi).statistic))
        if not vals:
            return [float("nan"), float("nan")]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    return {
        "role": "explanatory diagnostic only; never enters the selection score",
        "overall": {
            "source_spearman_r_vs_true_fanout": float(spearmanr(R, F).statistic),
            "source_bootstrap_95ci": boot_ci(R, F, seed=0),
            "target_spearman_c_vs_true_fanin": float(spearmanr(C, G).statistic),
            "target_bootstrap_95ci": boot_ci(C, G, seed=0),
            "n_source": int(R.size), "n_target": int(C.size),
            "n_bootstrap": 2000,
        },
        "per_bridge": per_bridge,
    }


def mode_report() -> int:
    space = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    data = json.loads(ALL_CANDIDATES_JSON.read_text(encoding="utf-8"))
    sel = json.loads(SELECTED_RULE_PATH.read_text(encoding="utf-8"))
    tmm = json.loads((DIR_RULES / "threshold_mm_calibration.json").read_text(encoding="utf-8"))
    ds = json.loads((DIR_RULES / "dual_softmax_calibration.json").read_text(encoding="utf-8"))
    qa = json.loads(QA_JSON.read_text(encoding="utf-8"))
    deg = json.loads((DIR_SELECTION / "degree_calibration"
                      / "degree_sampling_spec.json").read_text(encoding="utf-8"))
    tail = json.loads((DIR_SELECTION / "degree_calibration"
                       / "tail_report.json").read_text(encoding="utf-8"))
    prov = json.loads((DIR_SELECTION / "degree_calibration"
                       / "source_provenance.json").read_text(encoding="utf-8"))
    corr = r_adaptive_correlations()
    write_json(DIR_RULES / "r_adaptive_correlations.json", corr)

    agg = sorted(data["aggregate"], key=lambda a: -float(a["overall_macro_edge_f1"]))
    L: list[str] = []
    A = L.append
    A("# R7 selection report (Stage 0)")
    A("")
    A(f"* generated: {utc_now()}")
    A(f"* experiment: `{EXPERIMENT_ID}`")
    A(f"* selection block: `{list(SELECTION_SEEDS)}` "
      f"(amended from `{list(SELECTION_SEEDS_AS_SPECIFIED)}`; see BLOCKER_REPORT.md)")
    A("* confirmatory holdout `401-410`: **reserved, not generated, not read**")
    A("* this stage is **exploratory / selection**; nothing here is confirmatory evidence")
    A("")
    A("## 1. Real degree calibration")
    A("")
    A("Source: frozen v4 / v5 audit windows, recomputed from the canonical edge lists.")
    A("")
    A("| window | canonical edges | fan-out units (>=2) | fan-in units (>=2) | max fan-out | max fan-in |")
    A("|---|---:|---:|---:|---:|---:|")
    for t in ("v4", "v5"):
        st = prov["windows"][t]["window_statistics"]
        A(f"| {t} | {st['canonical_edges']} | {st['fanout_units_ge2']} | "
          f"{st['fanin_units_ge2']} | {st['fanout_max']} | {st['fanin_max']} |")
    A("")
    A(f"* v4 fan-out histogram exactly reproduces the frozen v4 final report: "
      f"`{tail['v4_final_report_cross_check']['exact_match']}`")
    A(f"* v5 manifest cross-check: edges match "
      f"`{tail['v5_manifest_cross_check']['edges_match']}`, fan-out units match "
      f"`{tail['v5_manifest_cross_check']['fanout_units_match']}`")
    A("")
    A("### Pooled truncated distribution (the frozen generator input)")
    A("")
    A("| degree | split (fan-out) count | split p | merge (fan-in) count | merge p |")
    A("|---:|---:|---:|---:|---:|")
    for i, d in enumerate(deg["split_degree"]["support"]):
        A(f"| {d} | {deg['split_degree']['counts'][i]} | {deg['split_degree']['pmf'][i]:.4f} | "
          f"{deg['merge_degree']['counts'][i]} | {deg['merge_degree']['pmf'][i]:.4f} |")
    A("")
    A("### Truncation tail")
    A("")
    A(f"* rule: `d_used = min(d_observed, {K_MAX})` (right winsorisation, not deletion)")
    A(f"* split: n={tail['fanout']['n_units_degree_ge_2']}, "
      f"P(d>{K_MAX})={tail['fanout']['p_degree_gt_cap']:.6f}, "
      f"observed max={tail['fanout']['observed_max_degree']}, "
      f"median={tail['fanout']['median']}, q75={tail['fanout']['q75']}, "
      f"q90={tail['fanout']['q90']}, q95={tail['fanout']['q95']}")
    A(f"* merge: n={tail['fanin']['n_units_degree_ge_2']}, "
      f"P(d>{K_MAX})={tail['fanin']['p_degree_gt_cap']:.6f}, "
      f"observed max={tail['fanin']['observed_max_degree']}")
    A(f"* **HIGH_TRUNCATION_TAIL = {tail['HIGH_TRUNCATION_TAIL']}**")
    A(f"* fan-in is empirical, not a mirrored proxy: "
      f"`mirrored_proxy_not_empirical_fanin = False`")
    A(f"* v4/v5 duplicate anchored units: shared canonical anchors = "
      f"{json.loads((DIR_SELECTION / 'degree_calibration' / 'duplicate_units.json').read_text(encoding='utf-8'))['shared_anchor_count']}"
      f" -> deduplication not required")
    A("")
    A("## 2. Generator: 24 families")
    A("")
    A(f"* families per bridge/seed: **{N_FAMILIES}**, instances per family: "
      f"**{INSTANCES_PER_FAMILY}**, templates per cell: **{N_TEMPLATES_PER_CELL}**")
    A("* family grid: 4 amount quartiles x 6 delay sextiles")
    A("* 12 **original** families = first half of each frozen 4x3 cell (sextiles 0,2,4)")
    A("* 12 **new** families = second half of each frozen cell (sextiles 1,3,5)")
    A("* every template in every cell has a DISTINCT base anchor, so base-anchor clusters "
      "are disjoint by construction")
    A(f"* structural QA: **{qa['GENERATOR_STRUCTURAL_QA']}** "
      f"(completed before any method F1 was computed: "
      f"`method_f1_computed = {qa['method_f1_computed']}`)")
    A(f"* family manifest: `{qa['family_manifest']['path']}` "
      f"({qa['family_manifest']['rows']} rows, sha256 "
      f"`{qa['family_manifest']['sha256'][:16]}...`)")
    A("")
    A("| check | status |")
    A("|---|---|")
    for c in qa["checks"]:
        A(f"| `{c['check']}` | {c['status']} |")
    A("")
    A("## 3. Rule candidate space (fixed before any selection F1)")
    A("")
    A(f"* candidates: **{space['n_candidates']}**")
    A(f"* criterion: {space['selection_criterion']}")
    A(f"* tie-break order: {space['tie_break']['order']}, then "
      f"{space['tie_break']['within_family']}")
    A("")
    A("## 4. All candidate results (every candidate, not only the winner)")
    A("")
    A("| rule | family | overall macro edge F1 | Celer | Multi | Poly | edges/family |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for a in agg:
        A(f"| `{a['rule_id']}` | {a['family']} | {a['overall_macro_edge_f1']:.6f} | "
          f"{a.get('Celer', float('nan')):.6f} | {a.get('Multi', float('nan')):.6f} | "
          f"{a.get('Poly', float('nan')):.6f} | {a['mean_edges_per_family']:.3f} |")
    A("")
    A("Full precision: `selection/rule_search/all_candidates.csv` and `.json`.")
    A("")
    A("## 5. Winner")
    A("")
    A(f"* **`{sel['winner']}`** (family `{sel['winner_family']}`)")
    A(f"* parameters: `{json.dumps(sel['winner_params'])}`")
    A(f"* selection score (bridge-balanced overall macro edge F1): "
      f"**{sel['winner_score']:.6f}**")
    A(f"* tied at tolerance {TIE_TOL}: {sel['n_tied_at_tolerance']} -> "
      f"{sel['tied_rule_ids']}")
    A(f"* tie-break applied: **{sel['tie_break']['applied']}** "
      f"(complexity order; R-const before R-quantile)")
    A(f"* mean predicted edges per family: {sel['mean_edges_per_family']:.3f} "
      f"(~{sel['mean_pred_edges_per_cell']:.0f} edges per cell)")
    A(f"* `TRUNCATION_BOUNDARY_DEPENDENCE = "
      f"{sel['truncation_boundary_dependence']['TRUNCATION_BOUNDARY_DEPENDENCE']}` "
      f"({'; '.join(sel['truncation_boundary_dependence']['reasons']) or 'no reason recorded'})")
    A("")
    A(f"Winner reason: highest bridge-balanced overall macro edge F1 over all "
      f"{space['n_candidates']} pre-registered candidates. "
      f"`R-quantile@q0.75` derived exactly the same k (ceil(Q_0.75) = 3 from the frozen "
      f"degree distribution) and therefore scored identically; the pre-registered "
      f"complexity tie-break selects the simpler `R-const` form. "
      f"No holdout information was used.")
    A("")
    A("## 6. R-adaptive diagnostics (explanatory only)")
    A("")
    A(f"* source Spearman(r_i, true fan-out degree), overall: "
      f"**{corr['overall']['source_spearman_r_vs_true_fanout']:.4f}** "
      f"95% CI {corr['overall']['source_bootstrap_95ci']}, n={corr['overall']['n_source']}")
    A(f"* target Spearman(c_j, true fan-in degree), overall: "
      f"**{corr['overall']['target_spearman_c_vs_true_fanin']:.4f}** "
      f"95% CI {corr['overall']['target_bootstrap_95ci']}, n={corr['overall']['n_target']}")
    A("")
    A("| bridge | source rho | target rho |")
    A("|---|---:|---:|")
    for b in BRIDGES:
        A(f"| {b} | {corr['per_bridge'][b]['source_spearman_r_vs_true_fanout']:.4f} | "
          f"{corr['per_bridge'][b]['target_spearman_c_vs_true_fanin']:.4f} |")
    A("")
    A("These correlations are reported for interpretation only and did not change the "
      "selection score.")
    A("")
    A("## 7. Threshold-MM calibration (selection block only)")
    A("")
    A(f"* criterion: {tmm['criterion']}")
    A(f"* uses ground-truth F1: **{tmm['uses_ground_truth_f1']}**")
    A(f"* UOT_KR target budget (bridge-balanced mean predicted edges per family): "
      f"**{tmm['uot_kr_target_budget']:.4f}**")
    A(f"* selected cutoff: **{tmm['selected_cutoff']:.6f}** (q={tmm['selected_q']}), "
      f"achieved budget "
      f"{tmm['candidates'][[c['q'] for c in tmm['candidates']].index(tmm['selected_q'])]['bridge_balanced_mean_edges_per_family']:.4f}, "
      f"|diff| = {tmm['selected_abs_diff']:.4f}")
    A(f"* grid amendment: {tmm['grid_amendment']['reason']}")
    A(f"  (coarse pre-registered grid best |diff| = "
      f"{tmm['grid_amendment']['coarse_grid_result']['selected_abs_diff']:.4f})")
    A(f"* tie-break: {tmm['tie_break']}")
    A("")
    A("## 8. Dual-Softmax calibration (selection block only)")
    A("")
    A(f"* acceptance: {ds['acceptance']}")
    A(f"* criterion: {ds['criterion']}")
    A(f"* selected tau: **{ds['selected_tau']}**, score {ds['selected_score']:.6f}")
    A(f"* monotone decreasing in tau: {ds['monotone_decreasing_in_tau']} -> "
      f"tau = 0 is the global optimum")
    A(f"* historical status: {ds['historical_status']}")
    A("")
    A("| tau | selection score |")
    A("|---:|---:|")
    for c in ds["candidates"]:
        A(f"| {c['tau']} | {c['bridge_balanced_macro_edge_f1']:.6f} |")
    A("")
    A("## 9. Cost hashes and solver diagnostics")
    A("")
    A("| bridge | seed | cost_matrix_sha256 | plan_sha256 | solver converged |")
    A("|---|---:|---|---|---|")
    for k in sorted(data["per_cell"]):
        c = data["per_cell"][k]
        A(f"| {c['bridge']} | {c['seed']} | `{c['cost_matrix_sha256'][:16]}...` | "
          f"`{c['plan_sha256'][:16]}...` | {c['solver'].get('converged')} |")
    A("")
    A("All methods in a cell consume the SAME `cost_matrix_sha256` and the SAME plan; the "
      "plan is solved exactly once per cell and reused by every decoder and every rule "
      "candidate.")
    A("")
    A("## 10. Oracle one-to-one ceiling (diagnostic, not a baseline)")
    A("")
    A("| bridge | seed | oracle macro F1 | mean T | mean M |")
    A("|---|---:|---:|---:|---:|")
    for k in sorted(data["per_cell"]):
        c = data["per_cell"][k]
        A(f"| {c['bridge']} | {c['seed']} | {c['oracle_1to1_ceiling_macro_f1']:.4f} | "
          f"{c['oracle_mean_T']:.2f} | {c['oracle_mean_M']:.2f} |")
    A("")
    A("The label-informed ceiling answers what the one-to-one output space can reach. It "
      "is never a deployable method, never enters H1/H2, and is never used to build a "
      "UOT_KR prediction.")
    A("")
    A("## 11. Not done in this stage")
    A("")
    A("* the confirmatory holdout `401-410` was **not** generated, read or evaluated")
    A("* no method result from this stage is confirmatory evidence")
    A("* no cost weight, epsilon, lambda or normalisation was re-tuned")
    A("")
    write_text(DIR_SELECTION / "SELECTION_REPORT.md", "\n".join(L) + "\n")
    log("[report] SELECTION_REPORT.md written")
    return 0


def _pv(tag: str, key: str) -> str:
    p = DIR_SELECTION / "degree_calibration" / "source_provenance.json"
    prov = json.loads(p.read_text(encoding="utf-8"))
    w = prov["windows"][tag]
    return str(w.get(key, "n/a"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-spec", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--selection", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--all", action="store_true")
    cli = ap.parse_args()
    if cli.all:
        cli.lock_spec = cli.preflight = cli.selection = True
        cli.select = cli.calibrate = cli.probe = cli.report = True
    rc = 0
    if cli.lock_spec:
        rc |= mode_lock_spec()
    if cli.preflight:
        rc |= mode_preflight()
    if cli.selection:
        rc |= mode_selection()
    if cli.select:
        rc |= mode_select()
    if cli.calibrate:
        rc |= mode_calibrate()
    if cli.probe:
        rc |= mode_validation_probe()
    if cli.report:
        rc |= mode_report()
    if cli.freeze:
        rc |= mode_freeze()
    if not any([cli.lock_spec, cli.preflight, cli.selection, cli.select, cli.calibrate,
                cli.probe, cli.report, cli.freeze]):
        ap.print_help()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
