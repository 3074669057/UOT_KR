"""S9 post-hoc degree-stratification -- analysis, figures, paper patches, reporting.

POST-HOC / NON-PREREGISTERED / ARCHIVED-PREDICTION RE-STRATIFICATION ANALYSIS.

This script reads ONLY ``results/template_degree_joined.csv`` (produced by
``run_s9_degree_stratification.py``).  It never touches the frozen confirmatory raw
package, never runs a generator, and never executes any prediction method.

Modes
-----
--lock-spec   lock the post-hoc analysis design BEFORE any stratified H1 is viewed
--analyze     strata summaries, bootstrap, trend test, oracle ceiling, sensitivity
--figures     the two S9 figures
--paper       S9 section + section 4.3 cross-reference patches
--manifest    S9-only MANIFEST.json + PROVENANCE.md
--checklist   VALIDATION_CHECKLIST.md
--report      FINAL_POSTHOC_REPORT.md
--all         everything in order (lock-spec first)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R7 = Path(__file__).resolve().parents[2] / "out" / "r7_confirmatory_kernel_ranking_20260917"
REPO = R7.parents[1]
S9 = R7 / "posthoc_s9_degree_stratification_20260918"
FEAS = S9 / "00_feasibility"
CONFIG = S9 / "config"
RESULTS = S9 / "results"
FIGURES = S9 / "figures"
PAPER = S9 / "paper"
SENS = S9 / "sensitivity"

JOINED = RESULTS / "template_degree_joined.csv"
SPEC_PATH = CONFIG / "locked_posthoc_spec.json"
SPEC_HASH = CONFIG / "locked_posthoc_spec.sha256"

BRIDGES = ("Celer", "Multi", "Poly")
B_BOOT = 4000
RNG_BOOT = 20240101
N_PERM = 20000
RNG_PERM = 20240103          # locked here; R7's 20240102 was the PRIMARY H1/H2 test RNG
TIE_TOL = 1e-12

EXPECTED_H1 = 0.025531674679475275


class NoRerunViolation(SystemExit):
    pass


def assert_no_method_imports(stage: str) -> None:
    bad = [m for m in ("r7.r7_pipeline", "r7.r7_methods") if m in sys.modules]
    if bad:
        raise NoRerunViolation(
            f"S9 NO-RERUN GUARD at '{stage}': {bad} imported. The post-hoc analysis must "
            f"not be able to execute any prediction method.")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(p: Path, o: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, ensure_ascii=False, default=str) + "\n",
                 encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- #
# locked spec
# --------------------------------------------------------------------------- #

def build_spec() -> dict[str, Any]:
    return {
        "experiment_id": "r7_posthoc_degree_stratification_20260918",
        "classification": ("POST-HOC / NON-PREREGISTERED / ARCHIVED-PREDICTION "
                           "RE-STRATIFICATION ANALYSIS"),
        "locked_at_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     __import__("time").gmtime()),
        "locked_before_viewing_stratified_h1": True,

        "no_rerun": {
            "prediction_methods_executed": False,
            "solver_calls": 0, "decoder_calls": 0,
            "generator_calls": ("truth-only replay only; no cost construction, no solver, "
                                "no decoder"),
            "forbidden_seeds_never_executed": sorted(
                list(range(42, 47)) + list(range(201, 206)) + list(range(301, 306))
                + list(range(401, 411))),
            "confirmatory_block_used": list(range(411, 421)),
            "block_401_410_used": False,
        },

        "r7_primary_unit_statement": {
            "r7_primary_resampling_unit": "bridge x seed",
            "r7_n_primary_paired_cells": 30,
            "r7_uses_template_instances_as_independent_samples": False,
            "consequence": ("S9 re-stratifies archived paired results at the TEMPLATE "
                            "INSTANCE level. This is NOT the seed-level primary estimand of "
                            "the R7 confirmatory test, so S9 is reported only as a "
                            "structural-heterogeneity diagnostic."),
            "forbidden_wording": [
                "template instances match the resampling unit of the main analysis",
                "preregistered", "new confirmatory hypothesis", "prospective subgroup "
                "analysis", "second holdout execution"],
        },

        "d_max_definition": {
            "primary": ("d_max = max over source nodes i of |{j : (i,j) is a NON-DECOY "
                        "ground-truth truth edge}|, i.e. over split union merge"),
            "decoy_edges_excluded": True,
            "unmatched_source_degree_zero": True,
            "target_ids_deduplicated": True,
            "computed_from": "ground-truth edges ONLY (never from predictions, never from "
                             "transport support)",
            "cross_check": ("d_max_including_decoys is recorded and must equal d_max because "
                            "every decoy pair is 1-to-1"),
            "sampled_split_degree_cross_check": ("compared against the archived generator "
                                                 "metadata; truth-derived d_max is "
                                                 "authoritative if they differ"),
        },

        "strata": {
            "binary_primary": {"low": "d_max == 2", "higher": "d_max >= 3"},
            "three_bin": {"A": "d_max == 2", "B": "d_max in {3,4}", "C": "d_max >= 5"},
            "exact_degree": "each observed d_max value separately (2,3,4,5,6,7,8, ...)",
            "cut_points_locked_before_viewing_effects": True,
            "no_post_hoc_cut_point_adjustment": True,
            "values_above_8_kept_unmerged": True,
        },

        "estimands": {
            "per_template_effect": "delta_H1(t) = F1_UOT_KR(t) - F1_HUNGARIAN(t)",
            "source": ("Tier A: recomputed independently from ARCHIVED prediction edge sets "
                       "and archived truth edges"),
            "stratum_macro_effect": ("within each bridge, mean delta_H1 over that bridge's "
                                     "template instances in the stratum; then equal-weight "
                                     "mean over the three bridges"),
            "interaction_contrast": "effect(d_max>=3) - effect(d_max=2), descriptive",
        },

        "bootstrap": {
            "B": B_BOOT, "rng_seed": RNG_BOOT,
            "design": ("within each bridge, resample that bridge's stratum template "
                       "instances WITH replacement; UOT_KR and HUNGARIAN move together as a "
                       "paired observation; per-bridge mean; equal-weight mean over the "
                       "three bridges"),
            "interval": "percentile 2.5 / 97.5",
            "applied_to": ["d_max=2", "d_max>=3", "A", "B", "C", "interaction"],
        },

        "missing_bridge_rule": {
            "trigger": "a bridge has n = 0 in a stratum",
            "action": ["report the stratum raw counts",
                       "flag 'bridge-balanced three-bridge estimand unavailable'",
                       "optionally report an available-bridge descriptive effect",
                       "never compare it directly with a complete three-bridge stratum"],
            "silent_two_bridge_macro_average_forbidden": True,
        },

        "dependence_sensitivity": {
            "required_because_primary_unit_is_bridge_x_seed": True,
            "design": ("within each bridge resample SEED clusters with replacement, then "
                       "keep the stratum template instances belonging to the sampled seeds"),
            "purpose": ("check whether the template-level bootstrap is over-optimistic "
                        "because of family/template dependence"),
            "may_not_change_the_conclusion": True,
        },

        "oracle_ceiling": {
            "definition": ("per template, maximum-cardinality bipartite matching on the "
                           "ground-truth POSITIVE graph"),
            "T_t": "truth positive edge count", "M_t": "maximum one-to-one truth matching",
            "ceiling_f1": "2*M_t/(T_t+M_t)", "precision": 1.0, "recall": "M_t/T_t",
            "status": "label-informed one-to-one semantic ceiling; NOT a deployable baseline",
            "must_not_be_conflated_with": "HUNGARIAN_1TO1",
            "monotonicity_assumed": False,
            "smoothing_forbidden": ["monotonic smoothing", "isotonic regression",
                                    "dropping points"],
        },

        "trend_test": {
            "question": "does delta_H1 change linearly with d_max?",
            "observed_slope": ("within each bridge, center d_max and center delta_H1, then "
                               "ordinary least squares slope; beta_macro = equal-weight mean "
                               "of the three bridge slopes"),
            "null": "degree-association slope = 0, while each bridge may keep a non-zero mean H1",
            "permutation": ("per bridge, take the centered paired-difference residuals "
                            "delta_H1 - bridge_mean_delta; multiply each independently by "
                            "random +/-1; add the original bridge mean back; keep d_max "
                            "fixed; recompute the three-bridge macro slope"),
            "n_perm": N_PERM, "rng_seed": RNG_PERM, "sided": "two-sided",
            "p_value": "(extreme + 1) / (n_perm + 1), extreme = count(|beta_perm| >= |beta_obs|)",
            "why_not_spearman": ("the analysis object is a PAIRED METHOD DIFFERENCE; the "
                                 "question is whether that paired effect changes with "
                                 "structural degree, so a locked paired-difference slope "
                                 "permutation is used. Any Spearman value is diagnostics "
                                 "only."),
        },

        "decision_wording": {
            "P_positive_trend": {
                "condition": "beta_macro > 0 AND two-sided permutation p < 0.05",
                "text": ("事后分层分析显示，UOT-KR 相对一对一指派基线的优势随真值源级扇出"
                         "复杂度增加而增强。R7 全体样本中的总体 H1 效应因此部分受到大量低度"
                         "模板的稀释。该结果为结构复杂度依赖的收益提供了事后证据，但不改变"
                         "原预注册 H1 判定。"),
                "extra": ("if the exact-degree means are not pointwise monotone, write "
                          "'positive overall trend', never 'strictly monotonic increase'"),
            },
            "N_no_detectable_trend": {
                "condition": "p >= 0.05 (whatever the sign of beta_macro)",
                "text": ("本事后分析未检测到 H1 效应随源级最大扇出度系统变化的证据。因此，"
                         "“总体 +0.0255 主要被低度模板稀释”的解释未得到本分析支持。"),
                "extra": "p >= 0.05 must NOT be written as proof that the effect is flat",
            },
            "D_negative_trend": {
                "condition": "beta_macro < 0 AND p < 0.05",
                "text": ("H1 效应随真值扇出复杂度增加反而下降，与预期的结构稀释解释方向相反。"
                         "该事后结果需作为局限和进一步机制研究对象明确报告。"),
            },
        },

        "claim_limits": {
            "always_labelled": ["post-hoc", "non-preregistered",
                                "archived-result re-stratification",
                                "mechanism / heterogeneity diagnostic"],
            "allowed_if_P": ["consistent with a dilution interpretation",
                             "supports the hypothesis that benefits are larger under higher "
                             "fan-out complexity"],
            "forbidden": ["proves that many-to-many output semantics caused the H1 gain",
                          "preregistered evidence", "new confirmatory hypothesis",
                          "new H1"],
            "h1_confound_note": ("H1 still compares UOT-KR against a cost-optimal one-to-one "
                                 "baseline and therefore mixes a decoder difference with an "
                                 "output-constraint difference"),
            "oracle_wording": ("if the ceiling decreases with degree, one may say 'the "
                               "representational capacity of a one-to-one output space "
                               "decreases as the truth graph becomes more one-to-many'; one "
                               "must NOT say Hungarian performance must equal the ceiling"),
        },

        "unchanged_by_s9": ["R7 H1/H2/S1/S2", "Gate A-E", "analysis/DECISION.json",
                            "Table 3", "original confirmatory metrics",
                            "frozen protocol", "success classification"],
    }


# --------------------------------------------------------------------------- #
# analysis helpers
# --------------------------------------------------------------------------- #

def stratum_macro(df: pd.DataFrame, col: str = "delta_H1") -> dict[str, Any]:
    """Within-bridge mean then equal-weight mean over bridges, with the missing-bridge rule."""
    per_bridge, counts = {}, {}
    for b in BRIDGES:
        v = df.loc[df["bridge"] == b, col].to_numpy(dtype=float)
        counts[b] = int(v.size)
        per_bridge[b] = float(v.mean()) if v.size else float("nan")
    available = [b for b in BRIDGES if counts[b] > 0]
    complete = len(available) == 3
    return {
        "effect": float(np.mean([per_bridge[b] for b in available])) if available
        else float("nan"),
        "per_bridge": per_bridge, "per_bridge_n": counts,
        "n_templates": int(len(df)),
        "three_bridge_estimand_available": complete,
        "estimand_note": ("bridge-balanced three-bridge estimand" if complete else
                          "bridge-balanced three-bridge estimand unavailable "
                          "(one or more bridges have n = 0); the reported effect is an "
                          "available-bridge descriptive effect and must not be compared "
                          "directly with a complete three-bridge stratum"),
        "available_bridges": available,
    }


def paired_bootstrap(df: pd.DataFrame, *, col: str = "delta_H1", B: int = B_BOOT,
                     seed: int = RNG_BOOT) -> dict[str, Any]:
    """Within-bridge template-level paired bootstrap, then equal-weight bridge macro."""
    rng = np.random.RandomState(int(seed))
    by_bridge = {b: df.loc[df["bridge"] == b, col].to_numpy(dtype=float) for b in BRIDGES}
    avail = [b for b in BRIDGES if by_bridge[b].size > 0]
    if not avail:
        return {"effect": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    stats = np.empty(int(B), dtype=float)
    for k in range(int(B)):
        means = []
        for b in avail:
            v = by_bridge[b]
            means.append(float(v[rng.randint(0, v.size, v.size)].mean()))
        stats[k] = float(np.mean(means))
    obs = float(np.mean([by_bridge[b].mean() for b in avail]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"effect": obs, "ci_lower": float(lo), "ci_upper": float(hi),
            "bootstrap_mean": float(stats.mean()), "B": int(B), "rng_seed": int(seed),
            "distribution_sha256": hashlib.sha256(
                np.ascontiguousarray(stats).tobytes()).hexdigest()}


def seed_cluster_bootstrap(df: pd.DataFrame, *, col: str = "delta_H1", B: int = B_BOOT,
                           seed: int = RNG_BOOT) -> dict[str, Any]:
    """Sensitivity: resample SEED clusters within each bridge, keep their templates."""
    rng = np.random.RandomState(int(seed))
    per_bridge = {b: df[df["bridge"] == b] for b in BRIDGES}
    avail = [b for b in BRIDGES if len(per_bridge[b])]
    if not avail:
        return {"effect": float("nan")}
    seeds_by_bridge = {b: sorted(per_bridge[b]["seed"].unique()) for b in avail}
    obs = float(np.mean([per_bridge[b][col].mean() for b in avail]))
    stats = np.empty(int(B), dtype=float)
    for k in range(int(B)):
        means = []
        for b in avail:
            s = np.array(seeds_by_bridge[b])
            draw = s[rng.randint(0, s.size, s.size)]
            vals = np.concatenate([per_bridge[b].loc[per_bridge[b]["seed"] == sd, col]
                                   .to_numpy(dtype=float) for sd in draw])
            means.append(float(vals.mean()) if vals.size else np.nan)
        stats[k] = float(np.nanmean(means))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"effect": obs, "ci_lower": float(lo), "ci_upper": float(hi),
            "B": int(B), "rng_seed": int(seed), "resampling_unit": "seed cluster"}


def bridge_centered_slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    den = float(np.sum(xc * xc))
    return float(np.sum(xc * yc) / den) if den > 0 else float("nan")


def trend_test(df: pd.DataFrame, *, n_perm: int = N_PERM, seed: int = RNG_PERM
               ) -> dict[str, Any]:
    betas, xs, ys, resids = {}, {}, {}, {}
    for b in BRIDGES:
        s = df[df["bridge"] == b]
        x = s["d_max"].to_numpy(dtype=float)
        y = s["delta_H1"].to_numpy(dtype=float)
        if x.size < 2 or np.unique(x).size < 2:
            betas[b] = float("nan")
            continue
        xs[b], ys[b] = x, y
        betas[b] = bridge_centered_slope(x, y)
        resids[b] = y - y.mean()
    avail = [b for b in BRIDGES if b in xs and np.isfinite(betas[b])]
    if not avail:
        return {"available": False}
    beta_macro = float(np.mean([betas[b] for b in avail]))

    rng = np.random.RandomState(int(seed))
    perm = np.empty(int(n_perm), dtype=float)
    means = {b: ys[b].mean() for b in avail}
    for k in range(int(n_perm)):
        bs = []
        for b in avail:
            sign = rng.choice([-1.0, 1.0], size=resids[b].size)
            y_star = means[b] + resids[b] * sign
            bs.append(bridge_centered_slope(xs[b], y_star))
        perm[k] = float(np.mean(bs))
    extreme = int(np.sum(np.abs(perm) >= abs(beta_macro)))
    p = (extreme + 1) / (int(n_perm) + 1)
    # OPTIONAL QA ONLY (specification section 21): Spearman is never primary evidence.
    from scipy.stats import spearmanr
    sp = {}
    for b in avail:
        rho = spearmanr(xs[b], ys[b])
        sp[b] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": int(xs[b].size)}
    rho_all = spearmanr(df["d_max"].to_numpy(float), df["delta_H1"].to_numpy(float))
    return {
        "available": True,
        "beta_per_bridge": betas, "beta_macro": beta_macro,
        "n_perm": int(n_perm), "rng_seed": int(seed), "sided": "two-sided",
        "extreme_count": extreme, "p_value": float(p),
        "resolution": 1.0 / (int(n_perm) + 1),
        "permutation_distribution_sha256": hashlib.sha256(
            np.ascontiguousarray(perm).tobytes()).hexdigest(),
        "n_templates": int(len(df)),
        "bridges_used": avail,
        "spearman_qa_only": {
            "warning": ("DIAGNOSTIC ONLY. The primary trend evidence is the paired-"
                        "difference slope permutation above; Spearman is reported for QA "
                        "and must never be presented as the main statistical evidence."),
            "per_bridge": sp,
            "pooled": {"rho": float(rho_all.statistic), "p": float(rho_all.pvalue),
                       "n": int(len(df))},
        },
    }


# --------------------------------------------------------------------------- #

def mode_lock_spec() -> int:
    if SPEC_PATH.exists():
        print(f"[lock] post-hoc spec already present (immutable): {SPEC_PATH.name}")
        return 0
    spec = build_spec()
    write_json(SPEC_PATH, spec)
    h = sha256_file(SPEC_PATH)
    write_text(SPEC_HASH, f"{h}  config/locked_posthoc_spec.json\n")
    print(f"[lock] locked_posthoc_spec.json sha256={h}")
    return 0


def mode_analyze() -> int:
    assert_no_method_imports("analyze")
    df = pd.read_csv(JOINED)
    spec_hash = sha256_file(SPEC_PATH)

    binary, three, exact = [], [], []
    for label, sub in (("d_max=2", df[df["d_max"] == 2]),
                       ("d_max>=3", df[df["d_max"] >= 3])):
        sm = stratum_macro(sub)
        bs = paired_bootstrap(sub)
        binary.append({"stratum": label, **sm, "bootstrap": bs,
                       "UOT_KR_f1_mean": float(sub["UOT_KR_f1"].mean()),
                       "HUNGARIAN_f1_mean": float(sub["HUNGARIAN_f1"].mean()),
                       "UOT_KR_f1_per_bridge": {b: float(sub.loc[sub["bridge"] == b,
                                                               "UOT_KR_f1"].mean())
                                                for b in BRIDGES},
                       "HUNGARIAN_f1_per_bridge": {b: float(sub.loc[sub["bridge"] == b,
                                                                   "HUNGARIAN_f1"].mean())
                                                   for b in BRIDGES}})
    for label, sub in (("A_d2", df[df["d_max"] == 2]),
                       ("B_d3_4", df[df["d_max"].isin([3, 4])]),
                       ("C_d5plus", df[df["d_max"] >= 5])):
        three.append({"stratum": label, **stratum_macro(sub), "bootstrap": paired_bootstrap(sub),
                      "UOT_KR_f1_mean": float(sub["UOT_KR_f1"].mean()),
                      "HUNGARIAN_f1_mean": float(sub["HUNGARIAN_f1"].mean())})
    for d in sorted(df["d_max"].unique()):
        sub = df[df["d_max"] == d]
        exact.append({"d_max": int(d), **stratum_macro(sub), "bootstrap": paired_bootstrap(sub),
                      "UOT_KR_f1_mean": float(sub["UOT_KR_f1"].mean()),
                      "HUNGARIAN_f1_mean": float(sub["HUNGARIAN_f1"].mean()),
                      "oracle_ceiling_mean": float(sub["oracle_1to1_ceiling_f1"].mean()),
                      "oracle_ceiling_std": float(sub["oracle_1to1_ceiling_f1"].std(ddof=1)),
                      "n_above_8": int(d > 8)})

    # interaction contrast
    lo = df[df["d_max"] == 2]["delta_H1"].to_numpy(float)
    hi = df[df["d_max"] >= 3]["delta_H1"].to_numpy(float)
    rng = np.random.RandomState(RNG_BOOT)
    b2 = {b: df[(df["d_max"] == 2) & (df["bridge"] == b)]["delta_H1"].to_numpy(float)
          for b in BRIDGES}
    b3 = {b: df[(df["d_max"] >= 3) & (df["bridge"] == b)]["delta_H1"].to_numpy(float)
          for b in BRIDGES}
    inter = np.empty(B_BOOT)
    for k in range(B_BOOT):
        m2 = [b2[b][rng.randint(0, b2[b].size, b2[b].size)].mean()
              for b in BRIDGES if b2[b].size]
        m3 = [b3[b][rng.randint(0, b3[b].size, b3[b].size)].mean()
              for b in BRIDGES if b3[b].size]
        inter[k] = float(np.mean(m3) - np.mean(m2))
    interaction = {
        "definition": "effect(d_max>=3) - effect(d_max=2)",
        "role": "descriptive interaction contrast; NOT a pre-locked primary significance test",
        "effect": float(np.mean([b3[b].mean() for b in BRIDGES if b3[b].size])
                        - np.mean([b2[b].mean() for b in BRIDGES if b2[b].size])),
        "ci_lower": float(np.percentile(inter, 2.5)),
        "ci_upper": float(np.percentile(inter, 97.5)),
        "B": B_BOOT, "rng_seed": RNG_BOOT,
    }

    # oracle ceiling strata
    ceil_rows = []
    for label, sub in (("d_max=2", df[df["d_max"] == 2]),
                       ("d_max>=3", df[df["d_max"] >= 3]),
                       ("A_d2", df[df["d_max"] == 2]),
                       ("B_d3_4", df[df["d_max"].isin([3, 4])]),
                       ("C_d5plus", df[df["d_max"] >= 5])):
        v = sub["oracle_1to1_ceiling_f1"].to_numpy(float)
        rr = np.random.RandomState(RNG_BOOT)
        bs = np.array([v[rr.randint(0, v.size, v.size)].mean() for _ in range(2000)]) \
            if v.size else np.array([np.nan])
        ceil_rows.append({"grouping": "binary" if label.startswith("d_max") else "three_bin",
                          "stratum": label, "n_templates": int(v.size),
                          "mean_ceiling": float(v.mean()) if v.size else float("nan"),
                          "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                          "q025": float(np.percentile(bs, 2.5)),
                          "q975": float(np.percentile(bs, 97.5)),
                          "n_Celer": int((sub["bridge"] == "Celer").sum()),
                          "n_Multi": int((sub["bridge"] == "Multi").sum()),
                          "n_Poly": int((sub["bridge"] == "Poly").sum())})
    ceil_df = pd.DataFrame(ceil_rows)
    ceil_exact_rows = []
    for d in sorted(df["d_max"].unique()):
        v = df.loc[df["d_max"] == d, "oracle_1to1_ceiling_f1"].to_numpy(float)
        rr = np.random.RandomState(RNG_BOOT)
        bs = (np.array([v[rr.randint(0, v.size, v.size)].mean() for _ in range(2000)])
              if v.size else np.array([np.nan]))
        ceil_exact_rows.append({
            "d_max": int(d), "n_templates": int(v.size),
            "mean_ceiling": float(v.mean()) if v.size else float("nan"),
            "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "boot_ci_lower": float(np.percentile(bs, 2.5)),
            "boot_ci_upper": float(np.percentile(bs, 97.5)),
            "value_q025": float(np.percentile(v, 2.5)) if v.size else float("nan"),
            "value_q975": float(np.percentile(v, 97.5)) if v.size else float("nan"),
            "n_Celer": int(((df["d_max"] == d) & (df["bridge"] == "Celer")).sum()),
            "n_Multi": int(((df["d_max"] == d) & (df["bridge"] == "Multi")).sum()),
            "n_Poly": int(((df["d_max"] == d) & (df["bridge"] == "Poly")).sum()),
            "mean_T": float(df.loc[df["d_max"] == d, "truth_edge_count"].mean()),
            "mean_M": float(df.loc[df["d_max"] == d, "oracle_matching_size"].mean()),
        })
    ceil_exact = pd.DataFrame(ceil_exact_rows)
    pd.concat([ceil_exact.assign(grouping="exact_degree"), ceil_df], ignore_index=True) \
        .to_csv(RESULTS / "oracle_ceiling_by_degree.csv", index=False)

    trend = trend_test(df)
    if trend.get("available"):
        if trend["beta_macro"] > 0 and trend["p_value"] < 0.05:
            trend["outcome"] = "P_positive_trend"
        elif trend["beta_macro"] < 0 and trend["p_value"] < 0.05:
            trend["outcome"] = "D_negative_trend"
        else:
            trend["outcome"] = "N_no_detectable_trend"
    write_json(RESULTS / "degree_trend_test.json", {
        "locked_spec_sha256": spec_hash, **trend,
        "expected_h1_full_precision": EXPECTED_H1,
    })

    # dependence sensitivity
    SENS.mkdir(parents=True, exist_ok=True)
    sens_rows = []
    for label, sub in (("d_max=2", df[df["d_max"] == 2]), ("d_max>=3", df[df["d_max"] >= 3]),
                       ("A_d2", df[df["d_max"] == 2]),
                       ("B_d3_4", df[df["d_max"].isin([3, 4])]),
                       ("C_d5plus", df[df["d_max"] >= 5])):
        a = paired_bootstrap(sub)
        s = seed_cluster_bootstrap(sub)
        sens_rows.append({"stratum": label,
                          "template_bootstrap_effect": a["effect"],
                          "template_ci_lower": a["ci_lower"],
                          "template_ci_upper": a["ci_upper"],
                          "seed_cluster_effect": s["effect"],
                          "seed_cluster_ci_lower": s["ci_lower"],
                          "seed_cluster_ci_upper": s["ci_upper"],
                          "ci_width_template": a["ci_upper"] - a["ci_lower"],
                          "ci_width_seed_cluster": s["ci_upper"] - s["ci_lower"],
                          "conclusion_consistent": bool(
                              (a["ci_lower"] > 0) == (s["ci_lower"] > 0))})
    pd.DataFrame(sens_rows).to_csv(SENS / "seed_cluster_bootstrap.csv", index=False)

    # per-template raw dump already exists; add exact-degree H1 table
    pd.DataFrame(exact).to_csv(RESULTS / "h1_effect_by_exact_degree.csv", index=False)
    pd.DataFrame(binary).drop(columns=["bootstrap"]).to_csv(
        RESULTS / "h1_effect_binary_strata.csv", index=False)
    pd.DataFrame(three).drop(columns=["bootstrap"]).to_csv(
        RESULTS / "h1_effect_three_bin_strata.csv", index=False)

    write_json(RESULTS / "s9_analysis.json", {
        "locked_spec_sha256": spec_hash,
        "n_templates": int(len(df)),
        "binary": binary, "three_bin": three, "exact_degree": exact,
        "interaction": interaction, "trend": trend,
        "oracle_ceiling_exact": ceil_exact.to_dict(orient="records"),
        "oracle_ceiling_strata": ceil_df.to_dict(orient="records"),
        "dependence_sensitivity": sens_rows,
        "sample_structure": {
            "total_templates": int(len(df)),
            "per_bridge": {b: int((df["bridge"] == b).sum()) for b in BRIDGES},
            "per_seed": {str(s): int((df["seed"] == s).sum()) for s in sorted(df["seed"].unique())},
            "per_degree": {str(d): int((df["d_max"] == d).sum())
                           for d in sorted(df["d_max"].unique())},
            "per_degree_per_bridge": {
                str(d): {b: int(((df["d_max"] == d) & (df["bridge"] == b)).sum())
                         for b in BRIDGES} for d in sorted(df["d_max"].unique())},
            "d_max_equals_sampled_split_degree":
                int(df["d_max_equals_sampled_split_degree"].sum()),
            "d_max_equals_including_decoys":
                int((df["d_max"] == df["d_max_including_decoys"]).sum()),
        },
    })
    print(json.dumps({"binary": [{k: v for k, v in b.items()
                                  if k in ("stratum", "effect", "n_templates")}
                                 for b in binary],
                      "three_bin": [{k: v for k, v in t.items()
                                     if k in ("stratum", "effect", "n_templates")}
                                    for t in three],
                      "trend": {k: trend.get(k) for k in
                                ("beta_macro", "p_value", "outcome")},
                      "interaction": interaction["effect"]}, indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    for f in ("lock_spec", "analyze", "figures", "paper", "manifest", "checklist",
              "report", "all"):
        ap.add_argument(f"--{f.replace('_', '-')}", action="store_true")
    cli = ap.parse_args()
    if cli.all:
        cli.lock_spec = cli.analyze = cli.figures = cli.paper = True
        cli.manifest = cli.checklist = cli.report = True
    rc = 0
    if cli.lock_spec:
        rc |= mode_lock_spec()
    if cli.analyze:
        rc |= mode_analyze()
    if cli.figures:
        from r7.s9_figures import build_all
        build_all()
    if cli.paper:
        from r7.s9_paper import build_paper
        build_paper()
    if cli.manifest:
        from r7.s9_report import build_feasibility_report, build_manifest, build_provenance
        build_feasibility_report(); build_manifest(); build_provenance()
    if cli.checklist:
        from r7.s9_report import build_checklist
        build_checklist()
    if cli.report:
        from r7.s9_report import build_report
        build_report()
    if not any(vars(cli).values()):
        ap.print_help()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
