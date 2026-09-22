"""R7 confirmatory statistics.

Implements the frozen protocol exactly:

  * primary estimand: per (bridge, seed) the equal-weight mean over the 24 template
    families of the family-mean edge F1 (30 paired cells; template instances are NEVER
    independent samples)
  * overall effect: mean the 10 seed-level paired deltas within each bridge, then
    equal-weight mean the 3 bridge means
  * primary CI: paired stratified bootstrap, B = 4000, RNG 20240101, percentile two-sided
  * primary p: one-sided paired sign-flip permutation, n_perm = 20000, RNG 20240102,
    Monte-Carlo p = (extreme + 1) / (n_perm + 1), Holm step-down over {H1, H2}
  * per-bridge tests: EXACT paired sign-flip enumeration over 2^10 = 1024 patterns
  * base-anchor cluster sensitivity: reseed the bootstrap on base-anchor clusters
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r7.r7_common import (ALPHA, BRIDGES, CONFIRMATORY_SEEDS, DIR_ANALYSIS, DIR_RAW,
                          GATE_C_FLOOR, METHODS, N_BOOT, N_PERM, RNG_BOOTSTRAP,
                          RNG_PERMUTATION, utc_now, write_json, write_text)

CONTRASTS = {
    "H1": ("UOT_KR", "HUNGARIAN_1TO1", "primary"),
    "H2": ("UOT_KR", "THRESHOLD_MM", "primary"),
    "S1": ("UOT_KR", "CONDITIONAL_UOT", "secondary"),
    "S2": ("CONDITIONAL_UOT", "RAW_UOT_PLAN", "secondary"),
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_units(raw_dir: Path = DIR_RAW) -> list[dict[str, Any]]:
    index = json.loads((raw_dir / "INDEX.json").read_text(encoding="utf-8"))
    return [json.loads((raw_dir / "units" / r["path"]).read_text(encoding="utf-8"))
            for r in index["units"]]


def cell_table(units: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per (bridge, seed, method) with the family-macro metrics."""
    rows = []
    for u in units:
        for m in METHODS:
            mm = u["method_metrics"][m]
            rows.append({
                "bridge": u["bridge"], "seed": u["seed"], "method": m,
                "macro_edge_f1": float(mm["macro_edge_f1"]),
                "family_mean_edge_precision": float(mm["family_mean_edge_precision"]),
                "family_mean_edge_recall": float(mm["family_mean_edge_recall"]),
                "family_mean_split_exact": float(mm["family_mean_split_exact"]),
                "family_mean_merge_exact": float(mm["family_mean_merge_exact"]),
                "family_mean_overall_exact": float(mm["family_mean_overall_exact"]),
                "n_pred_edges_total": float(mm["n_pred_edges_total"]),
                "mean_edges_per_family": float(np.mean(
                    list(u["family_edge_counts"][m].values()))),
            })
    return pd.DataFrame(rows)


def delta_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Paired bridge-seed deltas for every contrast (the 30 primary paired cells)."""
    wide = cells.pivot_table(index=["bridge", "seed"], columns="method",
                             values="macro_edge_f1").reset_index()
    out = wide[["bridge", "seed"]].copy()
    for name, (a, b, _role) in CONTRASTS.items():
        out[name] = wide[a] - wide[b]
    out["UOT_KR"] = wide["UOT_KR"]
    for m in METHODS:
        if m != "UOT_KR":
            out[m] = wide[m]
    return out


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #

def bridge_balanced(values: np.ndarray, bridges: Sequence[str]) -> float:
    """Mean within each bridge, then equal-weight mean over the 3 bridges."""
    means = []
    for b in BRIDGES:
        sel = values[np.asarray(bridges) == b]
        if sel.size:
            means.append(float(np.mean(sel)))
    return float(np.mean(means)) if means else float("nan")


def paired_stratified_bootstrap(deltas: np.ndarray, bridges: Sequence[str],
                                *, n_boot: int = N_BOOT, seed: int = RNG_BOOTSTRAP
                                ) -> dict[str, Any]:
    """Resample the 10 seed-level paired cells WITHIN each bridge, then macro-average.

    Both methods of a pair move together because the bootstrap is applied to the paired
    delta itself; template instances are never resampled independently.
    """
    rng = np.random.RandomState(int(seed))
    idx_by_bridge = {b: np.flatnonzero(np.asarray(bridges) == b) for b in BRIDGES}
    stats = np.empty(int(n_boot), dtype=float)
    for k in range(int(n_boot)):
        means = []
        for b in BRIDGES:
            idx = idx_by_bridge[b]
            if idx.size == 0:
                continue
            draw = rng.choice(idx, size=idx.size, replace=True)
            means.append(float(np.mean(deltas[draw])))
        stats[k] = float(np.mean(means)) if means else float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        "observed_effect": float(bridge_balanced(deltas, bridges)),
        "bootstrap_mean": float(np.mean(stats)),
        "ci_lower_2.5": float(lo), "ci_upper_97.5": float(hi),
        "B": int(n_boot), "rng_seed": int(seed),
        "bootstrap_distribution_sha256": _sha(stats),
        "bootstrap_distribution_std": float(np.std(stats)),
    }


def _sha(a: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def signflip_permutation(deltas: np.ndarray, bridges: Sequence[str], *,
                         n_perm: int = N_PERM, seed: int = RNG_PERMUTATION,
                         one_sided: bool = True) -> dict[str, Any]:
    """One-sided paired sign-flip permutation of the bridge-balanced mean effect."""
    rng = np.random.RandomState(int(seed))
    observed = bridge_balanced(deltas, bridges)
    n = deltas.size
    order = {b: np.flatnonzero(np.asarray(bridges) == b) for b in BRIDGES}
    signs = rng.choice([-1.0, 1.0], size=(int(n_perm), n))
    perm = np.empty(int(n_perm), dtype=float)
    for k in range(int(n_perm)):
        d = deltas * signs[k]
        perm[k] = float(np.mean([np.mean(d[order[b]]) for b in BRIDGES if order[b].size]))
    extreme = int(np.sum(perm >= observed)) if one_sided else int(
        np.sum(np.abs(perm) >= abs(observed)))
    p = (extreme + 1) / (int(n_perm) + 1)
    return {
        "observed_effect": float(observed),
        "n_perm": int(n_perm), "rng_seed": int(seed),
        "alternative": "effect > 0" if one_sided else "two-sided",
        "extreme_count": extreme,
        "raw_p": float(p),
        "resolution": 1.0 / (int(n_perm) + 1),
        "permutation_distribution_sha256": _sha(perm),
    }


def exact_signflip(deltas: np.ndarray, *, one_sided: bool = True) -> dict[str, Any]:
    """EXACT paired sign-flip enumeration over all 2^n sign patterns (n = 10 per bridge)."""
    d = np.asarray(deltas, dtype=float)
    n = d.size
    if n > 20:
        return {"available": False, "reason": f"2^{n} too large"}
    signs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    perm = (signs * d[None, :]).mean(axis=1)
    observed = float(d.mean())
    if one_sided:
        extreme = int(np.sum(perm >= observed - 1e-15))
    else:
        extreme = int(np.sum(np.abs(perm) >= abs(observed) - 1e-15))
    return {
        "available": True, "n_patterns": int(1 << n),
        "observed_effect": observed,
        "extreme_count": extreme,
        "exact_p": extreme / float(1 << n),
        "alternative": "effect > 0" if one_sided else "two-sided",
        "min_attainable_p": 1.0 / float(1 << n),
    }


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, Any]:
    """Holm step-down over the given family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    steps = []
    rejected_upto = -1
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        if i == 0 or all(s["rejected"] for s in steps[:i]):
            rej = bool(p < thresh)
        else:
            rej = False
        steps.append({"hypothesis": name, "raw_p": float(p), "rank": i + 1,
                      "threshold": float(thresh), "rejected": rej})
        if rej:
            rejected_upto = i
    return {
        "family": [k for k, _ in items],
        "alpha": float(alpha),
        "method": "Holm step-down",
        "steps": steps,
        "adjusted_p": {s["hypothesis"]: float(min(1.0, s["raw_p"] * (m - s["rank"] + 1)))
                       for s in steps},
        "rejection_order": [s["hypothesis"] for s in steps if s["rejected"]],
        "any_rejected": any(s["rejected"] for s in steps),
    }


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gate_ab(boot: dict[str, Any], perm: dict[str, Any], adj_p: float) -> dict[str, Any]:
    c1 = boot["observed_effect"] > 0
    c2 = boot["ci_lower_2.5"] > 0
    c3 = adj_p < ALPHA
    return {
        "effect_gt_zero": bool(c1), "ci_lower_gt_zero": bool(c2),
        "holm_adjusted_p_lt_alpha": bool(c3),
        "effect": boot["observed_effect"], "ci": [boot["ci_lower_2.5"], boot["ci_upper_97.5"]],
        "holm_adjusted_p": float(adj_p),
        "PASS": bool(c1 and c2 and c3),
    }


def gate_c(deltas: pd.DataFrame) -> dict[str, Any]:
    per_bridge = {}
    ok = True
    for b in BRIDGES:
        sub = deltas[deltas["bridge"] == b]
        h1 = float(sub["H1"].mean())
        h2 = float(sub["H2"].mean())
        good = (h1 >= GATE_C_FLOOR) and (h2 >= GATE_C_FLOOR)
        ok &= good
        per_bridge[b] = {"H1_mean_effect": h1, "H2_mean_effect": h2,
                         "n_seeds": int(len(sub)), "meets_floor": bool(good)}
    return {
        "gate": "C", "role": "cross-bridge consistency / CLAIM gate",
        "floor": GATE_C_FLOOR,
        "per_bridge": per_bridge,
        "PASS": bool(ok),
        "if_fail": ("the run remains a VALID confirmatory result; no cross-bridge "
                    "consistent advantage may be claimed"),
    }


# --------------------------------------------------------------------------- #
# base-anchor cluster sensitivity
# --------------------------------------------------------------------------- #

def base_anchor_clusters(raw_dir: Path = DIR_RAW) -> dict[tuple[str, int], list[str]]:
    """Per (bridge, seed): family -> base anchor id, read from the raw units."""
    index = json.loads((raw_dir / "INDEX.json").read_text(encoding="utf-8"))
    out: dict[tuple[str, int], dict[str, str]] = {}
    for r in index["units"]:
        u = json.loads((raw_dir / "units" / r["path"]).read_text(encoding="utf-8"))
        fam_anchor = {}
        for t in u["truth"]:
            if "__r7fam" not in t:
                continue
            fam = "r7fam" + t.split("__r7fam", 1)[1].split("__inst", 1)[0]
            anchor = t.split("__r7fam", 1)[0]
            fam_anchor[fam] = anchor
        out[(u["bridge"], u["seed"])] = fam_anchor
    return out


def cluster_sensitivity(units: Sequence[dict[str, Any]], *, n_boot: int = N_BOOT,
                        seed: int = RNG_BOOTSTRAP) -> dict[str, Any]:
    """Resample BASE-ANCHOR CLUSTERS instead of seeds, purely as a sensitivity analysis.

    Per (bridge, seed) the 24 families sit on 24 disjoint base anchors.  For each
    contrast we rebuild the family-level paired deltas, then bootstrap over the pooled
    cluster set within each bridge.
    """
    rng = np.random.RandomState(int(seed))
    rows = []
    for u in units:
        fams = {}
        for t, tr in u["truth"].items():
            if "__r7fam" not in t:
                continue
            fam = "r7fam" + t.split("__r7fam", 1)[1].split("__inst", 1)[0]
            anchor = t.split("__r7fam", 1)[0]
            fams.setdefault(fam, {"anchor": anchor, "templates": []})["templates"].append(t)
        rows.append((u, fams))

    out: dict[str, Any] = {
        "role": "sensitivity analysis only; does NOT replace the primary seed-level protocol",
        "resampling_unit": "base-anchor cluster",
        "n_clusters_per_cell": 24,
        "B": int(n_boot), "rng_seed": int(seed),
        "contrasts": {},
    }
    # rebuild family metrics independently from primitives
    from validate_r7_confirmatory_results import template_metrics, template_key, family_key
    fam_deltas: dict[str, dict[str, list[tuple[str, float, str]]]] = {}
    for u, fams in rows:
        pred_by_tpl = {}
        for m, edges in u["predictions"].items():
            per = {}
            for s, d in edges:
                per.setdefault(template_key(s), set()).add((str(s), str(d)))
            pred_by_tpl[m] = per
        per_fam_method: dict[str, dict[str, float]] = {}
        for fam, info in fams.items():
            vals: dict[str, list[float]] = {m: [] for m in METHODS}
            for t in info["templates"]:
                tr = u["truth"][t]
                for m in METHODS:
                    met = template_metrics(
                        {(str(a), str(b)) for a, b in tr["positive"]},
                        {(str(a), str(b)) for a, b in tr["split"]},
                        {(str(a), str(b)) for a, b in tr["merge"]},
                        pred_by_tpl[m].get(t, set()))
                    vals[m].append(met["edge_f1"])
            per_fam_method[fam] = {m: float(np.mean(v)) for m, v in vals.items()}
        for name, (a, b, _r) in CONTRASTS.items():
            for fam, per in per_fam_method.items():
                anchor = fams[fam]["anchor"]
                fam_deltas.setdefault(name, {}).setdefault(u["bridge"], []).append(
                    (f"{u['seed']}|{anchor}", per[a] - per[b], anchor))

    for name in CONTRASTS:
        per_bridge_means = []
        boot_stats = []
        for b in BRIDGES:
            arr = fam_deltas.get(name, {}).get(b, [])
            vals = np.array([x[1] for x in arr], dtype=float)
            if vals.size == 0:
                continue
            per_bridge_means.append(float(vals.mean()))
            local = []
            for _ in range(int(n_boot)):
                draw = rng.randint(0, vals.size, vals.size)
                local.append(float(vals[draw].mean()))
            boot_stats.append(np.array(local))
        if not boot_stats:
            continue
        n = min(a.size for a in boot_stats)
        macro = np.mean([a[:n] for a in boot_stats], axis=0)
        obs = float(np.mean(per_bridge_means))
        out["contrasts"][name] = {
            "observed_effect": obs,
            "ci_lower_2.5": float(np.percentile(macro, 2.5)),
            "ci_upper_97.5": float(np.percentile(macro, 97.5)),
            "n_clusters_total": int(sum(len(fam_deltas.get(name, {}).get(b, []))
                                        for b in BRIDGES)),
        }
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def analyse(raw_dir: Path = DIR_RAW, gate_e_path: Path | None = None) -> dict[str, Any]:
    DIR_ANALYSIS.mkdir(parents=True, exist_ok=True)
    units = load_units(raw_dir)
    cells = cell_table(units)
    deltas = delta_table(cells)
    bridges = deltas["bridge"].to_numpy()

    cells.to_csv(DIR_ANALYSIS / "confirmatory_cell_level.csv", index=False)

    # bridge summary + overall summary
    bridge_rows = []
    for b in BRIDGES:
        sub = cells[cells["bridge"] == b]
        for m in METHODS:
            v = sub[sub["method"] == m]
            bridge_rows.append({
                "bridge": b, "method": m, "n_seeds": int(len(v)),
                "macro_edge_f1_mean": float(v["macro_edge_f1"].mean()),
                "macro_edge_f1_std": float(v["macro_edge_f1"].std(ddof=1)),
                "precision_mean": float(v["family_mean_edge_precision"].mean()),
                "recall_mean": float(v["family_mean_edge_recall"].mean()),
                "split_exact_mean": float(v["family_mean_split_exact"].mean()),
                "merge_exact_mean": float(v["family_mean_merge_exact"].mean()),
                "overall_exact_mean": float(v["family_mean_overall_exact"].mean()),
                "edges_per_family_mean": float(v["mean_edges_per_family"].mean()),
                "n_pred_edges_mean": float(v["n_pred_edges_total"].mean()),
            })
    pd.DataFrame(bridge_rows).to_csv(DIR_ANALYSIS / "confirmatory_bridge_summary.csv",
                                     index=False)

    overall_rows = []
    for m in METHODS:
        per_bridge = [np.mean([r["macro_edge_f1_mean"] for r in bridge_rows
                               if r["bridge"] == b and r["method"] == m]) for b in BRIDGES]
        base = [r for r in bridge_rows if r["method"] == m]
        overall_rows.append({
            "method": m,
            "macro_edge_f1_bridge_balanced": float(np.mean(per_bridge)),
            "macro_edge_f1_per_bridge": {b: float(p) for b, p in zip(BRIDGES, per_bridge)},
            "precision_bridge_balanced": float(np.mean([r["precision_mean"] for r in base])),
            "recall_bridge_balanced": float(np.mean([r["recall_mean"] for r in base])),
            "split_exact_bridge_balanced": float(np.mean([r["split_exact_mean"] for r in base])),
            "merge_exact_bridge_balanced": float(np.mean([r["merge_exact_mean"] for r in base])),
            "overall_exact_bridge_balanced": float(np.mean([r["overall_exact_mean"] for r in base])),
            "edges_per_family_bridge_balanced": float(np.mean([r["edges_per_family_mean"] for r in base])),
        })
    pd.DataFrame(overall_rows).to_csv(DIR_ANALYSIS / "confirmatory_overall_summary.csv",
                                      index=False)

    # oracle ceiling
    oracle_rows = []
    for u in units:
        oc = u["oracle_summary"]
        oracle_rows.append({"bridge": u["bridge"], "seed": u["seed"],
                            "oracle_macro_f1": oc["macro_f1"],
                            "oracle_recall": oc["macro_recall"],
                            "mean_T": oc["mean_T"], "mean_M": oc["mean_M"],
                            "UOT_KR_macro_f1": u["method_metrics"]["UOT_KR"]["macro_edge_f1"],
                            "UOT_KR_above_oracle": bool(
                                u["method_metrics"]["UOT_KR"]["macro_edge_f1"] > oc["macro_f1"])})
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(DIR_ANALYSIS.parent / "diagnostics" / "oracle_1to1_ceiling.csv",
                     index=False)

    # primary bootstrap + permutation
    boot: dict[str, Any] = {}
    perm: dict[str, Any] = {}
    for name in CONTRASTS:
        boot[name] = paired_stratified_bootstrap(deltas[name].to_numpy(), bridges)
        perm[name] = signflip_permutation(deltas[name].to_numpy(), bridges)
    write_json(DIR_ANALYSIS / "primary_bootstrap.json", {
        "generated_at_utc": utc_now(), "contrasts": boot,
        "estimand": "bridge-balanced mean of the family-macro edge F1 paired delta",
        "n_primary_paired_cells": int(len(deltas)),
    })
    holm_res = holm({k: perm[k]["raw_p"] for k in ("H1", "H2")})

    gate_a = gate_ab(boot["H1"], perm["H1"], holm_res["adjusted_p"]["H1"])
    gate_b = gate_ab(boot["H2"], perm["H2"], holm_res["adjusted_p"]["H2"])
    gate_c_res = gate_c(deltas)

    write_json(DIR_ANALYSIS / "primary_holm_tests.json", {
        "generated_at_utc": utc_now(),
        "family": "H1, H2 (primary)", "holm": holm_res,
        "raw_permutation": {k: perm[k] for k in ("H1", "H2")},
        "gate_A": gate_a, "gate_B": gate_b,
        "resolution_note": ("Monte-Carlo resolution 1/20001; no p below the resolution is "
                            "reported, and 1.9e-9 is NOT claimed"),
    })

    # per-bridge tests
    pb = {}
    for b in BRIDGES:
        sub = deltas[deltas["bridge"] == b]
        pb[b] = {}
        for name in CONTRASTS:
            v = sub[name].to_numpy()
            pb[b][name] = {
                "effect": float(v.mean()), "n": int(v.size),
                "exact_two_sided": exact_signflip(v, one_sided=False),
                "exact_one_sided": exact_signflip(v, one_sided=True),
                "descriptive_ci_95": paired_stratified_bootstrap(
                    v, np.array([b] * v.size), n_boot=2000, seed=RNG_BOOTSTRAP),
            }
    write_json(DIR_ANALYSIS / "per_bridge_tests.json", {
        "generated_at_utc": utc_now(),
        "note": ("per-bridge tests are NOT part of the Holm primary family; with 10 pairs "
                 "the smallest attainable exact two-sided p is 2/1024"),
        "per_bridge": pb,
    })

    # base-anchor cluster sensitivity
    sens = cluster_sensitivity(units)
    write_json(DIR_ANALYSIS / "base_anchor_cluster_sensitivity.json", {
        "generated_at_utc": utc_now(), **sens,
        "primary_comparison": {k: {"primary_effect": boot[k]["observed_effect"],
                                   "primary_ci": [boot[k]["ci_lower_2.5"],
                                                  boot[k]["ci_upper_97.5"]]}
                               for k in CONTRASTS},
    })

    # UOT representation diagnostics
    rep_rows = []
    for u in units:
        mm = u["margin_mass"]
        rep_rows.append({
            "bridge": u["bridge"], "seed": u["seed"],
            "delta_S_total": mm["delta_S_total"], "delta_T_total": mm["delta_T_total"],
            "mean_delta_S": float(np.mean(mm["delta_S"])),
            "mean_delta_T": float(np.mean(mm["delta_T"])),
            "support_mass_fraction": mm["support_mass_fraction"],
            "mass_total": float(np.sum(mm["realized_source_mass"])),
            "UOT_KR_macro_f1": u["method_metrics"]["UOT_KR"]["macro_edge_f1"],
            "UOT_KR_edges_per_family": float(np.mean(
                list(u["family_edge_counts"]["UOT_KR"].values()))),
            "mean_split_degree": float(np.mean(u["degrees"]["split_degrees"])),
            "mean_merge_degree": float(np.mean(u["degrees"]["merge_degrees"])),
        })
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(DIR_ANALYSIS / "uot_representation_diagnostics.csv", index=False)

    # gate E
    gate_e = None
    if gate_e_path and gate_e_path.is_file():
        gate_e = json.loads(gate_e_path.read_text(encoding="utf-8"))
    gate_d = json.loads((DIR_RAW.parent / "VALIDITY_GATE_D.json").read_text(encoding="utf-8"))

    return {
        "units": units, "cells": cells, "deltas": deltas,
        "bridge_rows": bridge_rows, "overall_rows": overall_rows,
        "oracle": oracle_df, "bootstrap": boot, "permutation": perm, "holm": holm_res,
        "gate_A": gate_a, "gate_B": gate_b, "gate_C": gate_c_res,
        "gate_D": gate_d, "gate_E": gate_e,
        "per_bridge": pb, "sensitivity": sens, "representation": rep_df,
    }


if __name__ == "__main__":
    res = analyse()
    print(json.dumps({"gate_A": res["gate_A"], "gate_B": res["gate_B"],
                      "gate_C": res["gate_C"]["PASS"],
                      "gate_D": res["gate_D"]["GATE_D"],
                      "gate_E": (res["gate_E"] or {}).get("GATE_E")}, indent=2))
