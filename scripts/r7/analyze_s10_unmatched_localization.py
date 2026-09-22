"""S10 post-hoc unmatched-mass localization -- analysis, figures, paper, reporting.

POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS.

Reads ONLY the archived frozen transport representation (per-node delta^S / delta^T) and
the archived truth labels of the successful confirmatory block 411-420.  Never executes any
prediction method, never re-solves UOT, never touches 401-410.

Modes: --lock-spec --analyze --figures --paper --manifest --checklist --report --all
"""
from __future__ import annotations

import argparse
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
S10 = R7 / "posthoc_s10_unmatched_mass_localization_20260919"
FEAS = S10 / "00_feasibility"
CONFIG = S10 / "config"
RESULTS = S10 / "results"
FIGURES = S10 / "figures"
PAPER = S10 / "paper"
SENS = S10 / "sensitivity"
VALID = S10 / "VALIDATION"
RAW = R7 / "confirmatory" / "raw"

BRIDGES = ("Celer", "Multi", "Poly")
CONFIRMATORY_SEEDS = (411, 412, 413, 414, 415, 416, 417, 418, 419, 420)
FORBIDDEN_SEEDS = (set(range(42, 47)) | set(range(201, 206)) | set(range(301, 306))
                   | set(range(401, 411)))

SPEC_PATH = CONFIG / "locked_posthoc_spec.json"
SPEC_HASH = CONFIG / "locked_posthoc_spec.sha256"

B_BOOT = 4000
RNG_BOOT = 20240105          # seed-cluster AUC bootstrap
N_PERM_SRC = 20000
RNG_PERM_SRC = 20240104      # source truth-label permutation
N_PERM_TGT = 20000
RNG_PERM_TGT = 20240106      # target random-identity permutation
ZERO_TOL = 1e-15             # locked zero-total-delta threshold
FORBIDDEN_MODULES = ("r7.r7_pipeline", "r7.r7_methods")


class NoRerunViolation(SystemExit):
    pass


def assert_no_method_imports(stage: str) -> None:
    bad = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    if bad:
        raise NoRerunViolation(f"S10 NO-RERUN GUARD at '{stage}': {bad} imported.")


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
        "experiment_id": "r7_posthoc_unmatched_mass_localization_20260919",
        "classification": "POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS",
        "locked_at_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     __import__("time").gmtime()),
        "locked_before_viewing_localization_results": True,
        "provenance_mode": "A_ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS",
        "tier": "A",
        "no_rerun": {
            "prediction_methods_executed": False, "uot_resolved": False,
            "confirmatory_block": list(range(411, 421)),
            "block_401_410_touched": False,
            "forbidden_seeds_never_executed": sorted(FORBIDDEN_SEEDS),
        },
        "delta_definition": {
            "delta_S": "a_i - sum_j P_ij   (a = risk-weighted source mass, normalised)",
            "delta_T": "b_j - sum_i P_ij   (b = evidence-weighted target mass, normalised)",
            "source": "frozen scripts/run_r7_confirmatory_kernel_ranking.py build_unit "
                      "lines 90-91; normalization frozen scripts/r7/r7_generator.py "
                      "build_cells lines 258-259",
            "verified_from_archived_plan": True,
            "verification_tolerance": 1e-12,
        },
        "truth_labels": {
            "TRUE_UNMATCHED_SOURCE": "the single generator-injected __synth_unmatched_src",
            "DECOY_TARGET": "the two generator-injected __synth_noise_dst_* targets",
            "MATCHED_SOURCE": "any source with at least one positive truth edge",
            "MATCHED_TARGET": "any other target",
            "identity_from_predictions": False,
            "identity_from_delta_maximum": False,
        },
        "top1_rule": {
            "ranking": "delta descending",
            "tie_break": ("ties by ascending canonical source index -- the EXISTING frozen "
                          "engineering rule (r7_methods.rank_desc uses "
                          "np.lexsort((index, -score))); no new rule was invented"),
            "source_top1_hit": "1 iff the true unmatched source is the deterministic rank 1",
            "tie_aware_top1": ("diagnostic: 1 iff the true unmatched source attains the "
                               "maximum delta (any tie counts)"),
            "primary_metric_uses_deterministic_rule": True,
        },
        "zero_total_rule": {
            "tau": ZERO_TOL,
            "if_sum_delta_S_le_tau": ["source_delta_informative = False",
                                      "source Top-1 counts as a MISS",
                                      "source share is NaN",
                                      "the template is NOT silently dropped",
                                      "zero-total rate is reported separately"],
            "same_for_targets": True,
        },
        "chance_baselines": {
            "source_top1": "per template 1 / n_sources_t, then bridge-balanced",
            "target_two_target": ("per template the share of two targets drawn uniformly "
                                  "WITHOUT replacement, then bridge-balanced"),
            "size_adjusted": True,
        },
        "primary_discrimination_metric": {
            "name": "template_stratified_source_AUC",
            "definition": ("per informative template, the probability that the true "
                           "unmatched source's delta exceeds a uniformly drawn matched "
                           "source's delta, with ties counted 0.5: AUC_t = (#(d_u > d_m) + "
                           "0.5 #(d_u = d_m)) / N_matched.  Template -> bridge/seed -> "
                           "three-bridge equal weight."),
            "avoids_template_size_weighting": True,
            "pooled_source_AUC": "DIAGNOSTIC ONLY",
        },
        "uncertainty": {
            "primary": "bridge-balanced seed-cluster bootstrap (resample the 10 seeds within "
                       "each bridge; keep all templates and nodes inside a cluster; "
                       "three-bridge equal weight)",
            "B": B_BOOT, "rng_seed": RNG_BOOT, "interval": "percentile 2.5 / 97.5",
            "naive_template_bootstrap": "sensitivity only, never the primary CI",
            "pseudoreplication_forbidden": "nodes are never treated as independent samples",
        },
        "permutation_tests": {
            "source_top1": {"design": ("keep each template's delta vector; assign the "
                                       "'unmatched' label to a uniformly random source in "
                                       "that template; recompute the bridge-balanced Top-1 "
                                       "hit rate"),
                            "n_perm": N_PERM_SRC, "rng_seed": RNG_PERM_SRC,
                            "sided": "one-sided (observed > null)"},
            "target_decoy_share": {"design": ("keep each template's delta_T vector; label "
                                              "two uniformly random distinct targets as "
                                              "'decoy'; recompute the bridge-balanced "
                                              "combined share"),
                                   "n_perm": N_PERM_TGT, "rng_seed": RNG_PERM_TGT,
                                   "sided": "one-sided (observed > null)"},
            "role": "post-hoc mechanism evidence; NOT confirmatory; NOT in any Holm family",
        },
        "decision_rules": {
            "L1_strong_localization": {
                "conditions": [
                    "source Top-1 hit is significantly above the size-adjusted random "
                    "baseline (permutation p < 0.05)",
                    "source stratified AUC 95% CI lower bound > 0.5",
                    "target true-decoy combined delta_T share exceeds the random two-target "
                    "baseline with permutation p < 0.05"],
                "conclusion": ("UOT 的未匹配质量不仅在形式上存在，而且在该半合成数据中显著集中"
                               "到生成器已知的真实未匹配源与诱饵目标上。"),
                "still_called": "post-hoc evidence, never pre-registered confirmation",
            },
            "L2_source_localization_only": {
                "conditions": ["source conditions met", "target condition NOT met"],
                "conclusion": ("delta^S 对真实未匹配源表现出定位能力，但 delta^T 对诱饵目标的"
                               "对应证据不足。"),
                "must_not": "generalise to two-sided localization",
            },
            "L3_weak_or_no_separation": {
                "conditions": ["source AUC CI contains 0.5", "or Top-1 not above the random "
                               "baseline"],
                "conclusion": ("本分析未发现足够证据证明未匹配质量能够可靠定位真实未匹配源；"
                               "因此 UOT 的贡献仍应限定为表示未匹配质量，而不能升级为定位能力。"),
            },
            "L4_reversed_localization": {
                "conditions": ["source AUC clearly < 0.5"],
                "conclusion": "report the reversal explicitly as a limitation",
                "must_not": ["change the delta definition", "delete anomalous templates"],
            },
        },
        "claim_boundaries": {
            "do_not_write": ["任何一对一方法在构造上都无法做到这一点",
                             "UOT causes the R7 H1 gain",
                             "calibration curve (ROC/AUC is discrimination, not calibration)"],
            "accurate_wording": ("the cost-optimal one-to-one assignment baseline used in R7 "
                                 "does not expose distributed source/target unmatched-mass "
                                 "variables analogous to delta^S / delta^T; assignment "
                                 "variants with dummy/null states can express rejection, but "
                                 "with different semantics from UOT's continuous mass "
                                 "relaxation"),
            "figure_title": "Unmatched-mass localization and discrimination",
        },
        "unchanged_by_s10": ["R7 H1/H2/S1/S2", "Gate A-E", "R7 success classification",
                             "Table 3", "S9", "frozen confirmatory outputs"],
        "post_hoc_label_required_on": ["Top-1", "target share", "AUC"],
    }


def mode_lock_spec() -> int:
    if SPEC_PATH.exists():
        print(f"[lock] spec already present: {SPEC_PATH.name}")
        return 0
    write_json(SPEC_PATH, build_spec())
    write_text(SPEC_HASH, f"{sha256_file(SPEC_PATH)}  config/locked_posthoc_spec.json\n")
    print(f"[lock] locked_posthoc_spec.json sha256={sha256_file(SPEC_PATH)}")
    return 0


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def build_long_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-node long tables + per-template summary, read-only from the frozen archive."""
    src_rows, tgt_rows, tpl_rows = [], [], []
    for bridge in BRIDGES:
        assert bridge in BRIDGES
        for seed in CONFIRMATORY_SEEDS:
            assert seed not in FORBIDDEN_SEEDS
            u = json.loads((RAW / "units" / f"unit__{bridge}__s{seed}.json")
                           .read_text(encoding="utf-8"))
            sids = list(u["source_ids"])
            tids = list(u["target_ids"])
            dS = np.asarray(u["margin_mass"]["delta_S"], dtype=float)
            dT = np.asarray(u["margin_mass"]["delta_T"], dtype=float)
            idx_s = {s: i for i, s in enumerate(sids)}
            idx_t = {t: i for i, t in enumerate(tids)}
            for t, tr in sorted(u["truth"].items()):
                tsrc = [s for s in sids if s.startswith(t + "__")]
                ttgt = [d for d in tids if d in {str(x) for _, x in tr["positive"]}
                        or d in {str(x) for x in tr["hidden_dst"]}]
                ttgt = sorted(set(ttgt))
                unmatched = sorted(tr["unmatched_src"])
                decoys = sorted({str(d) for _, d in tr["decoy"]})
                pos = {(str(a), str(b)) for a, b in tr["positive"]}
                srcs_pos = {s for s, _ in pos}

                Ds = float(dS[[idx_s[s] for s in tsrc]].sum())
                Dt = float(dT[[idx_t[d] for d in ttgt]].sum())
                zs, zt = Ds <= ZERO_TOL, Dt <= ZERO_TOL

                # ---- source side ----
                order = sorted(tsrc, key=lambda s: (-dS[idx_s[s]], idx_s[s]))
                rank_of = {s: r + 1 for r, s in enumerate(order)}
                maxS = max((dS[idx_s[s]] for s in tsrc), default=float("nan"))
                tied_at_max = [s for s in tsrc if dS[idx_s[s]] == maxS]
                u_src = unmatched[0] if len(unmatched) == 1 else None
                top1 = int((not zs) and u_src is not None and rank_of.get(u_src) == 1)
                tie_aware = int((not zs) and u_src is not None and u_src in tied_at_max)
                n_matched = len([s for s in tsrc if s in srcs_pos])
                if (not zs) and u_src is not None and n_matched > 0:
                    du = dS[idx_s[u_src]]
                    dm = np.array([dS[idx_s[s]] for s in tsrc if s in srcs_pos])
                    auc_t = float(((du > dm).sum() + 0.5 * (du == dm).sum()) / n_matched)
                else:
                    auc_t = float("nan")
                for s in tsrc:
                    src_rows.append({
                        "bridge": bridge, "seed": int(seed), "template_id": t,
                        "source_id": s,
                        "truth_unmatched": int(s == u_src),
                        "has_positive_truth_edge": int(s in srcs_pos),
                        "delta_s": float(dS[idx_s[s]]),
                        "total_delta_s": Ds,
                        "delta_s_share": (float(dS[idx_s[s]]) / Ds) if not zs else float("nan"),
                        "rank_delta_s": int(rank_of[s]),
                        "is_top1": int(rank_of[s] == 1),
                        "informative_total": int(not zs),
                        "provenance": "frozen confirmatory/raw margin_mass.delta_S",
                    })

                # ---- target side ----
                order_t = sorted(ttgt, key=lambda d: (-dT[idx_t[d]], idx_t[d]))
                rank_t = {d: r + 1 for r, d in enumerate(order_t)}
                decoy_share = (float(sum(dT[idx_t[d]] for d in decoys)) / Dt) if not zt \
                    else float("nan")
                top2 = set(order_t[:2])
                top2_both = int((not zt) and set(decoys) == top2 and len(decoys) == 2)
                top2_any = int((not zt) and bool(top2 & set(decoys)))
                for d in ttgt:
                    tgt_rows.append({
                        "bridge": bridge, "seed": int(seed), "template_id": t,
                        "target_id": d,
                        "truth_decoy": int(d in set(decoys)),
                        "delta_t": float(dT[idx_t[d]]),
                        "total_delta_t": Dt,
                        "delta_t_share": (float(dT[idx_t[d]]) / Dt) if not zt else float("nan"),
                        "rank_delta_t": int(rank_t[d]),
                        "informative_total": int(not zt),
                        "provenance": "frozen confirmatory/raw margin_mass.delta_T",
                    })

                n_tgt = len(ttgt)
                tpl_rows.append({
                    "bridge": bridge, "seed": int(seed), "template_id": t,
                    "n_sources": len(tsrc), "n_targets": n_tgt,
                    "true_unmatched_source_id": u_src,
                    "unmatched_source_delta_share":
                        (float(dS[idx_s[u_src]]) / Ds) if ((not zs) and u_src) else float("nan"),
                    "source_top1_hit": top1,
                    "source_tie_aware_top1": tie_aware,
                    "source_auc_template": auc_t,
                    "n_matched_sources": n_matched,
                    "chance_top1": 1.0 / len(tsrc) if tsrc else float("nan"),
                    "tied_at_max_count": len(tied_at_max),
                    "decoy_target_1": decoys[0] if len(decoys) > 0 else None,
                    "decoy_target_2": decoys[1] if len(decoys) > 1 else None,
                    "decoy_combined_delta_share": decoy_share,
                    "decoy_top2_both_hit": top2_both,
                    "decoy_top2_any_hit": top2_any,
                    "chance_decoy_share":
                        (2.0 / n_tgt) if n_tgt >= 2 else float("nan"),
                    "total_delta_s": Ds, "total_delta_t": Dt,
                    "zero_delta_s": int(zs), "zero_delta_t": int(zt),
                })
    return pd.DataFrame(src_rows), pd.DataFrame(tgt_rows), pd.DataFrame(tpl_rows)


def bridge_macro(df: pd.DataFrame, valcol: str) -> float:
    """Within-bridge mean then equal-weight mean over bridges."""
    ms = [df.loc[df["bridge"] == b, valcol].mean() for b in BRIDGES
          if (df["bridge"] == b).any()]
    return float(np.mean(ms)) if ms else float("nan")


def seed_cluster_bootstrap(df: pd.DataFrame, valcol: str, *, B: int = B_BOOT,
                           seed: int = RNG_BOOT, agg: str = "mean") -> dict[str, Any]:
    rng = np.random.RandomState(int(seed))
    by = {b: df[df["bridge"] == b] for b in BRIDGES}
    avail = [b for b in BRIDGES if len(by[b])]
    if not avail:
        return {"effect": float("nan")}
    seeds = {b: sorted(by[b]["seed"].unique()) for b in avail}
    obs = bridge_macro(df, valcol)
    stats = np.empty(int(B), dtype=float)
    for k in range(int(B)):
        means = []
        for b in avail:
            s = np.array(seeds[b])
            draw = s[rng.randint(0, s.size, s.size)]
            vals = pd.concat([by[b].loc[by[b]["seed"] == sd, valcol] for sd in draw])
            vals = vals.dropna()
            means.append(float(vals.mean()) if len(vals) else np.nan)
        stats[k] = float(np.nanmean(means))
    return {"effect": obs, "ci_lower": float(np.percentile(stats, 2.5)),
            "ci_upper": float(np.percentile(stats, 97.5)),
            "B": int(B), "rng_seed": int(seed), "resampling_unit": "seed cluster",
            "distribution_sha256": hashlib.sha256(
                np.ascontiguousarray(stats).tobytes()).hexdigest()}


def template_bootstrap(df: pd.DataFrame, valcol: str, *, B: int = B_BOOT,
                       seed: int = RNG_BOOT) -> dict[str, Any]:
    """Sensitivity only: resample templates inside each bridge."""
    rng = np.random.RandomState(int(seed))
    by = {b: df.loc[df["bridge"] == b, valcol].dropna().to_numpy(float) for b in BRIDGES}
    avail = [b for b in BRIDGES if by[b].size]
    stats = np.empty(int(B), dtype=float)
    for k in range(int(B)):
        stats[k] = float(np.mean([by[b][rng.randint(0, by[b].size, by[b].size)].mean()
                                  for b in avail]))
    return {"effect": bridge_macro(df, valcol),
            "ci_lower": float(np.percentile(stats, 2.5)),
            "ci_upper": float(np.percentile(stats, 97.5)), "B": int(B)}


def summarise(v: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(v, errors="coerce").dropna().to_numpy(float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
            "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
            "std": float(x.std(ddof=1)) if x.size > 1 else float("nan")}


# --------------------------------------------------------------------------- #

def mode_analyze() -> int:
    assert_no_method_imports("analyze")
    spec_hash = sha256_file(SPEC_PATH)
    src, tgt, tpl = build_long_tables()
    src.to_csv(RESULTS / "source_localization_long.csv", index=False)
    tgt.to_csv(RESULTS / "target_localization_long.csv", index=False)
    tpl.to_csv(RESULTS / "template_localization_summary.csv", index=False)

    # ---- source Top-1 + permutation ----
    t = tpl.dropna(subset=["source_top1_hit"])
    obs_top1 = bridge_macro(t, "source_top1_hit")
    chance_top1 = bridge_macro(t, "chance_top1")
    tie_aware = bridge_macro(t, "source_tie_aware_top1")
    n_tied = int((t["tied_at_max_count"] > 1).sum())

    rng = np.random.RandomState(RNG_PERM_SRC)
    rank1 = t["is_top1"].to_numpy() if "is_top1" in t else None
    rank1_idx, ns = [], []
    for _, r in t.iterrows():
        sub = src[(src["template_id"] == r["template_id"]) & (src["seed"] == r["seed"])
                  & (src["bridge"] == r["bridge"])]
        rank1_idx.append(int((sub["rank_delta_s"] == 1).to_numpy().argmax()))
        ns.append(int(len(sub)))
    rank1_idx = np.array(rank1_idx)
    ns = np.array(ns)
    br = t["bridge"].to_numpy()
    perm = np.empty(N_PERM_SRC, dtype=float)
    for k in range(N_PERM_SRC):
        pick = np.floor(rng.random(ns.size) * ns).astype(int)
        hit = (pick == rank1_idx).astype(float)
        perm[k] = float(np.mean([hit[br == b].mean() for b in BRIDGES if (br == b).any()]))
    p_src = (int(np.sum(perm >= obs_top1)) + 1) / (N_PERM_SRC + 1)

    # ---- target decoy share + permutation ----
    tt = tpl.dropna(subset=["decoy_combined_delta_share"])
    obs_decoy = bridge_macro(tt, "decoy_combined_delta_share")
    chance_decoy = bridge_macro(tt, "chance_decoy_share")
    top2_both = bridge_macro(tpl, "decoy_top2_both_hit")
    top2_any = bridge_macro(tpl, "decoy_top2_any_hit")

    rng2 = np.random.RandomState(RNG_PERM_TGT)
    keys = list(tt[["bridge", "seed", "template_id"]].itertuples(index=False, name=None))
    groups = [tgt[(tgt["bridge"] == b) & (tgt["seed"] == s) & (tgt["template_id"] == tm)]
              for b, s, tm in keys]
    vecs = [g["delta_t"].to_numpy(float) for g in groups]
    tots = [float(g["total_delta_t"].iloc[0]) for g in groups]
    gbr = np.array([b for b, _, _ in keys])
    perm2 = np.empty(N_PERM_TGT, dtype=float)
    for k in range(N_PERM_TGT):
        shares = np.empty(len(vecs), dtype=float)
        for i, v in enumerate(vecs):
            n = v.size
            if n < 2:
                shares[i] = np.nan
                continue
            pick = np.argsort(rng2.random(n))[:2]
            shares[i] = v[pick].sum() / tots[i]
        perm2[k] = float(np.mean([np.nanmean(shares[gbr == b])
                                  for b in BRIDGES if (gbr == b).any()]))
    p_tgt = (int(np.sum(perm2 >= obs_decoy)) + 1) / (N_PERM_TGT + 1)

    # ---- AUC with seed-cluster bootstrap ----
    auc_ci = seed_cluster_bootstrap(tpl, "source_auc_template")
    auc_ci_template = template_bootstrap(tpl, "source_auc_template")
    decoy_ci = seed_cluster_bootstrap(tpl, "decoy_combined_delta_share")
    top1_ci = seed_cluster_bootstrap(tpl, "source_top1_hit")

    # pooled ROC (diagnostic only)
    from sklearn.metrics import roc_auc_score
    lab = src["truth_unmatched"].to_numpy(int)
    sc = src["delta_s"].to_numpy(float)
    pooled_auc = float(roc_auc_score(lab, sc)) if lab.min() != lab.max() else float("nan")

    # ---- zero totals ----
    zero_s = int(tpl["zero_delta_s"].sum())
    zero_t = int(tpl["zero_delta_t"].sum())

    # ---- ADDITIONAL POST-HOC NULL DIAGNOSTIC: is delta just the marginal? ----
    # This does NOT redefine any locked metric; it tests whether the observed
    # discrimination could have been produced by the marginals alone.
    marg_rows = []
    for bridge in BRIDGES:
        for seed in CONFIRMATORY_SEEDS:
            u = json.loads((RAW / "units" / f"unit__{bridge}__s{seed}.json")
                           .read_text(encoding="utf-8"))
            sids, tids = u["source_ids"], u["target_ids"]
            a = np.asarray(u["margin_mass"]["a"], float)
            b = np.asarray(u["margin_mass"]["b"], float)
            dS = np.asarray(u["margin_mass"]["delta_S"], float)
            dT = np.asarray(u["margin_mass"]["delta_T"], float)
            for t, tr in sorted(u["truth"].items()):
                si = [i for i, s in enumerate(sids) if s.startswith(t + "__")]
                tt = {str(d) for _, d in tr["positive"]} | {str(d) for d in tr["hidden_dst"]}
                ti = [i for i, d in enumerate(tids) if d in tt]
                def is_func_of_marg(dv, mv):
                    g: dict[float, set] = {}
                    for x, y in zip(mv, dv):
                        g.setdefault(round(float(x), 15), set()).add(round(float(y), 15))
                    return all(len(v) == 1 for v in g.values())
                unmatched = tr["unmatched_src"]
                decoys = sorted({str(d) for _, d in tr["decoy"]})
                pos = {(str(x), str(y)) for x, y in tr["positive"]}
                srcs_pos = {s for s, _ in pos}
                usrc = unmatched[0] if len(unmatched) == 1 else None
                # amount-only null AUC for the source side
                dm = np.array([a[i] for i in si if sids[i] in srcs_pos])
                du = a[si[[sids[i] for i in si].index(usrc)]] if usrc else np.nan
                auc_null = float(((du > dm).sum() + 0.5 * (du == dm).sum()) / dm.size) \
                    if dm.size else float("nan")
                dtsum = float(dT[ti].sum()) if ti else float("nan")
                dec_share_null = (float(sum(b[tids.index(d)] for d in decoys)) / float(b[ti].sum())) \
                    if ti and float(b[ti].sum()) > 0 else float("nan")
                marg_rows.append({
                    "bridge": bridge, "seed": int(seed), "template_id": t,
                    "delta_S_is_function_of_a_only": int(is_func_of_marg(dS[si], a[si])),
                    "delta_T_is_function_of_b_only": int(is_func_of_marg(dT[ti], b[ti])),
                    "n_distinct_delta_S_in_template": int(len({round(float(x), 15)
                                                               for x in dS[si]})),
                    "n_distinct_delta_T_in_template": int(len({round(float(x), 15)
                                                               for x in dT[ti]})),
                    "source_auc_amount_only_null": auc_null,
                    "decoy_share_amount_only_null": dec_share_null,
                    "delta_S_equals_a_exactly": int(bool(np.array_equal(dS, a))),
                })
    marg = pd.DataFrame(marg_rows)
    marg.to_csv(RESULTS / "marginal_null_diagnostic.csv", index=False)
    tpl_m = tpl.merge(marg, on=["bridge", "seed", "template_id"], how="left")
    obs_auc_vals = tpl_m["source_auc_template"].to_numpy(float)
    null_auc_vals = tpl_m["source_auc_amount_only_null"].to_numpy(float)
    ok = np.isfinite(obs_auc_vals) & np.isfinite(null_auc_vals)
    per_gap = np.abs(obs_auc_vals[ok] - null_auc_vals[ok])
    n_identical = int(np.sum(per_gap <= 1e-12))
    obs_auc_bb = bridge_macro(tpl_m, "source_auc_template")
    null_auc_bb = bridge_macro(tpl_m, "source_auc_amount_only_null")
    null_decoy = bridge_macro(tpl_m, "decoy_share_amount_only_null")
    marginal_diag = {
        "role": ("ADDITIONAL POST-HOC NULL DIAGNOSTIC (not a locked metric, does not "
                 "redefine anything): tests whether the observed discrimination could have "
                 "been produced by the frozen marginals alone"),
        "delta_S_is_function_of_a_only_templates":
            int(marg["delta_S_is_function_of_a_only"].sum()),
        "delta_T_is_function_of_b_only_templates":
            int(marg["delta_T_is_function_of_b_only"].sum()),
        "n_templates": int(len(marg)),
        "distinct_delta_S_per_template":
            {str(k): int(v) for k, v in
             marg["n_distinct_delta_S_in_template"].value_counts().items()},
        "distinct_delta_T_per_template":
            {str(k): int(v) for k, v in
             marg["n_distinct_delta_T_in_template"].value_counts().items()},
        "source_auc_observed_bridge_balanced": obs_auc_bb,
        "source_auc_amount_only_null_bridge_balanced": null_auc_bb,
        "source_auc_bridge_balanced_gap": obs_auc_bb - null_auc_bb,
        "templates_with_identical_auc_under_the_null": n_identical,
        "templates_compared": int(ok.sum()),
        "mean_abs_per_template_auc_gap": float(per_gap.mean()) if per_gap.size else float("nan"),
        "max_abs_per_template_auc_gap": float(per_gap.max()) if per_gap.size else float("nan"),
        "source_auc_is_essentially_reproduced_by_the_marginal":
            bool(abs(obs_auc_bb - null_auc_bb) <= 0.005),
        "decoy_share_amount_only_null": null_decoy,
        "decoy_share_observed": obs_decoy,
        "decoy_share_gap_vs_amount_only_null": obs_decoy - null_decoy,
        "target_side_is_fully_explained_by_the_marginal":
            bool(abs(obs_decoy - null_decoy) <= 1e-12),
        "interpretation": ("on the SOURCE side the frozen delta^S vector is, in every "
                           "template, a deterministic function of the source marginal "
                           "a_i alone (2 distinct values per template), and replacing "
                           "delta^S by a_i reproduces the source AUC almost exactly -- so "
                           "the source-side discrimination reflects the generator's amount "
                           "allocation, not transport geometry. On the TARGET side "
                           "delta^T is NOT determined by b_j and the observed decoy "
                           "concentration is NOT reproduced by the marginal."),
    }

    # ---- L1..L4 ----
    c1 = bool(p_src < 0.05 and obs_top1 > chance_top1)
    c2 = bool(auc_ci["ci_lower"] > 0.5)
    c3 = bool(p_tgt < 0.05 and obs_decoy > chance_decoy)
    reversed_auc = bool(auc_ci["ci_upper"] < 0.5)
    if c1 and c2 and c3:
        outcome = "L1_strong_localization"
    elif c1 and c2 and not c3:
        outcome = "L2_source_localization_only"
    elif reversed_auc:
        outcome = "L4_reversed_localization"
    else:
        outcome = "L3_weak_or_no_separation"

    per_bridge = {}
    for b in BRIDGES:
        sb = tpl[tpl["bridge"] == b]
        per_bridge[b] = {
            "n_templates": int(len(sb)),
            "source_top1_hit": float(sb["source_top1_hit"].mean()),
            "chance_top1": float(sb["chance_top1"].mean()),
            "source_tie_aware_top1": float(sb["source_tie_aware_top1"].mean()),
            "unmatched_source_delta_share": summarise(sb["unmatched_source_delta_share"]),
            "source_auc": float(sb["source_auc_template"].mean()),
            "decoy_combined_delta_share":
                summarise(sb["decoy_combined_delta_share"]),
            "chance_decoy_share": float(sb["chance_decoy_share"].mean()),
            "decoy_top2_both_hit": float(sb["decoy_top2_both_hit"].mean()),
            "decoy_top2_any_hit": float(sb["decoy_top2_any_hit"].mean()),
            "zero_delta_s": int(sb["zero_delta_s"].sum()),
            "zero_delta_t": int(sb["zero_delta_t"].sum()),
        }

    out = {
        "locked_spec_sha256": spec_hash,
        "provenance_mode": "A_ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS",
        "tier": "A",
        "n_templates": int(len(tpl)),
        "n_units": len(BRIDGES) * len(CONFIRMATORY_SEEDS),
        "source_side": {
            "top1_hit": obs_top1, "chance_baseline": chance_top1,
            "top1_ci_seed_cluster": top1_ci,
            "permutation_p": float(p_src), "n_perm": N_PERM_SRC,
            "rng_seed": RNG_PERM_SRC,
            "permutation_null_mean": float(perm.mean()),
            "tie_aware_top1": tie_aware,
            "templates_with_top_score_ties": n_tied,
            "unmatched_source_delta_share": summarise(tpl["unmatched_source_delta_share"]),
            "matched_source_delta_share": summarise(
                src.loc[src["truth_unmatched"] == 0, "delta_s_share"]),
            "template_stratified_auc": auc_ci,
            "template_stratified_auc_naive_template_bootstrap": auc_ci_template,
            "pooled_source_auc_DIAGNOSTIC_ONLY": pooled_auc,
        },
        "target_side": {
            "decoy_combined_delta_share": summarise(tpl["decoy_combined_delta_share"]),
            "chance_baseline": chance_decoy,
            "seed_cluster_bootstrap": decoy_ci,
            "permutation_p": float(p_tgt), "n_perm": N_PERM_TGT,
            "rng_seed": RNG_PERM_TGT,
            "permutation_null_mean": float(perm2.mean()),
            "top2_both_hit": top2_both, "top2_any_hit": top2_any,
            "nondecoy_target_delta_share": summarise(
                tgt.loc[tgt["truth_decoy"] == 0, "delta_t_share"]),
        },
        "zero_totals": {"zero_delta_s_templates": zero_s,
                        "zero_delta_t_templates": zero_t,
                        "tau": ZERO_TOL,
                        "rate_s": zero_s / len(tpl), "rate_t": zero_t / len(tpl)},
        "marginal_null_diagnostic": marginal_diag,
        "per_bridge": per_bridge,
        "outcome": outcome,
        "outcome_conditions": {"source_top1_above_chance_and_p_lt_05": c1,
                               "source_auc_ci_lower_gt_0.5": c2,
                               "target_decoy_above_chance_and_p_lt_05": c3,
                               "auc_ci_upper_lt_0.5": reversed_auc},
        "post_hoc_label": ("POST-HOC mechanism evidence; not pre-registered; not in any "
                           "Holm family; does not modify R7 Gate A-E or DECISION"),
    }
    write_json(RESULTS / "s10_analysis.json", out)

    print(json.dumps({
        "outcome": outcome,
        "source_top1": obs_top1, "chance_top1": chance_top1,
        "top1_p": float(p_src), "tie_aware": tie_aware,
        "unmatched_share_mean": out["source_side"]["unmatched_source_delta_share"].get("mean"),
        "auc": auc_ci["effect"], "auc_ci": [auc_ci["ci_lower"], auc_ci["ci_upper"]],
        "pooled_auc": pooled_auc,
        "decoy_share": obs_decoy, "chance_decoy": chance_decoy, "decoy_p": float(p_tgt),
        "top2_both": top2_both, "zero_s": zero_s, "zero_t": zero_t,
        "marginal_null": {
            "delta_S_func_of_a_only": marginal_diag["delta_S_is_function_of_a_only_templates"],
            "delta_T_func_of_b_only": marginal_diag["delta_T_is_function_of_b_only_templates"],
            "source_auc_observed": marginal_diag["source_auc_observed_bridge_balanced"],
            "source_auc_null": marginal_diag["source_auc_amount_only_null_bridge_balanced"],
            "source_auc_gap": marginal_diag["source_auc_bridge_balanced_gap"],
            "templates_identical_under_null":
                marginal_diag["templates_with_identical_auc_under_the_null"],
            "decoy_null": marginal_diag["decoy_share_amount_only_null"],
            "decoy_gap": marginal_diag["decoy_share_gap_vs_amount_only_null"],
        },
    }, indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    for f in ("lock_spec", "analyze", "figures", "paper", "manifest", "checklist",
              "report", "all"):
        ap.add_argument(f"--{f.replace('_','-')}", action="store_true")
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
        from r7.s10_figures import build_all
        build_all()
    if cli.paper:
        from r7.s10_paper import build_paper
        build_paper()
    if cli.manifest:
        from r7.s10_report import (build_feasibility_report, build_manifest,
                                   build_provenance)
        build_feasibility_report(); build_manifest(); build_provenance()
    if cli.checklist:
        from r7.s10_report import build_checklist
        build_checklist()
    if cli.report:
        from r7.s10_report import build_report
        build_report()
    if not any(vars(cli).values()):
        ap.print_help()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
