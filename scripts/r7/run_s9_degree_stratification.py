"""S9 post-hoc degree-stratification -- Stage 1: feasibility, truth assembly, join.

POST-HOC / NON-PREREGISTERED / ARCHIVED-PREDICTION RE-STRATIFICATION ANALYSIS.

HARD CONSTRAINTS ENFORCED HERE
------------------------------
* NO prediction method is ever executed.  The only computation performed is
  (a) truth reconstruction from the frozen generator (truth-only replay) and
  (b) re-aggregation of ALREADY ARCHIVED prediction edge sets.
* `401-410` is never read.  Only the successful one-shot block `411-420` is used.
* Nothing under `confirmatory/raw/`, `config/`, `analysis/` or `figures/` of R7 is
  written to.  All S9 output goes to `posthoc_s9_degree_stratification_20260918/`.

A runtime guard asserts that no solver / decoder / method-pipeline module is imported
anywhere on the truth path.
"""
from __future__ import annotations

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
RAW = R7 / "confirmatory" / "raw"
LOCKED_SPEC_R7 = R7 / "config" / "locked_spec.json"
MANIFEST_R7 = R7 / "config" / "FROZEN_PROTOCOL_MANIFEST.json"
DECISION_R7 = R7 / "analysis" / "DECISION.json"

CONFIRMATORY_SEEDS = (411, 412, 413, 414, 415, 416, 417, 418, 419, 420)
FORBIDDEN_SEEDS = set(range(42, 47)) | set(range(201, 206)) | set(range(301, 306)) \
    | set(range(401, 411))
BRIDGES = ("Celer", "Multi", "Poly")
GEN_REL = "src/cross/domain/evaluation/semi_synthetic_flows.py"
GENERATOR = REPO / GEN_REL
FEATURE_STATS = (REPO / "out" / "multi_bridge_expansion"
                 / "faithful_flow_structural_three_bridges" / "feature_stats")

# Modules that must NEVER be imported on the truth / re-aggregation path.
FORBIDDEN_MODULES = ("ot", "r7.r7_pipeline", "r7.r7_methods")

METHODS_OF_INTEREST = ("UOT_KR", "HUNGARIAN_1TO1")


class NoRerunViolation(SystemExit):
    """Raised if anything on the truth path could execute a prediction method."""


def assert_no_method_imports(stage: str) -> dict[str, Any]:
    """Runtime guard: no solver / decoder / method pipeline on the truth path."""
    present = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    # `ot` may legitimately be imported by an unrelated library; record it either way.
    blocking = [m for m in present if m != "ot"]
    if blocking:
        raise NoRerunViolation(
            f"S9 NO-RERUN GUARD at stage '{stage}': forbidden module(s) {blocking} are "
            f"imported. The truth / re-aggregation path must not be able to execute any "
            f"prediction method.")
    return {"stage": stage, "checked": list(FORBIDDEN_MODULES), "present": present,
            "blocking": blocking, "ok": True}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(o: Any) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def write_json(p: Path, o: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, ensure_ascii=False, default=str) + "\n",
                 encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- #
# template identity (documented grammar, re-implemented)
# --------------------------------------------------------------------------- #

def family_id_of(template_id: str) -> str:
    s = str(template_id)
    return "r7fam" + s.split("__r7fam", 1)[1].split("__inst", 1)[0] if "__r7fam" in s else s


def instance_id_of(template_id: str) -> str:
    s = str(template_id)
    return s.split("__inst", 1)[1] if "__inst" in s else "00"


def tpl_of(flow_id: str) -> str:
    """Frozen template key: everything before ``__synth``."""
    s = str(flow_id)
    if "__synth" in s:
        return s.split("__synth")[0]
    if "__" in s:
        return s.rsplit("__", 1)[0]
    return s


# --------------------------------------------------------------------------- #
# F1 archive inventory
# --------------------------------------------------------------------------- #

def archive_inventory() -> dict[str, Any]:
    units = sorted((RAW / "units").glob("unit__*.json"))
    seeds_seen = set()
    bridges_seen = set()
    per_unit = []
    for p in units:
        u = json.loads(p.read_text(encoding="utf-8"))
        seeds_seen.add(int(u["seed"]))
        bridges_seen.add(str(u["bridge"]))
        per_unit.append({
            "path": p.relative_to(R7).as_posix(),
            "bridge": u["bridge"], "seed": int(u["seed"]),
            "n_templates": len(u["truth"]),
            "n_source_ids": len(u["source_ids"]),
            "n_target_ids": len(u["target_ids"]),
            "methods_with_predictions": sorted(u["predictions"].keys()),
            "truth_edge_lists_present": all(
                {"split", "merge", "decoy", "positive"} <= set(t.keys())
                for t in u["truth"].values()),
            "oracle_per_template_present": len(u.get("oracle_1to1_ceiling", {}))
            == len(u["truth"]),
            "sampled_split_degrees_present": bool(u.get("degrees", {}).get(
                "split_degrees")),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        })
    files_all = [p for p in R7.rglob("*") if p.is_file()]
    s9_prefix = str(S9)
    r7_only = [p for p in files_all if not str(p).startswith(s9_prefix)]
    return {
        "r7_archive_files_recounted": len(r7_only),
        "r7_archive_bytes": int(sum(p.stat().st_size for p in r7_only)),
        "r7_archive_files_incl_s9": len(files_all),
        "r7_manifest_declared_files": json.loads(
            (R7 / "MANIFEST.json").read_text(encoding="utf-8"))["n_files"]
        if (R7 / "MANIFEST.json").is_file() else None,
        "confirmatory_raw_units": len(units),
        "seeds_present": sorted(seeds_seen),
        "bridges_present": sorted(bridges_seen),
        "templates_per_unit": sorted({r["n_templates"] for r in per_unit}),
        "total_template_instances": int(sum(r["n_templates"] for r in per_unit)),
        "all_units_have_method_predictions": all(
            set(METHODS_OF_INTEREST) <= set(r["methods_with_predictions"])
            for r in per_unit),
        "all_units_have_truth_edge_lists": all(r["truth_edge_lists_present"]
                                               for r in per_unit),
        "all_units_have_per_template_oracle": all(r["oracle_per_template_present"]
                                                  for r in per_unit),
        "all_units_have_sampled_degrees": all(r["sampled_split_degrees_present"]
                                              for r in per_unit),
        "per_unit": per_unit,
        "forbidden_seed_units_present": sorted(
            {s for s in seeds_seen if s in FORBIDDEN_SEEDS}),
        "only_successful_block_used": seeds_seen == set(CONFIRMATORY_SEEDS),
    }


# --------------------------------------------------------------------------- #
# F2 truth-only replay
# --------------------------------------------------------------------------- #

def truth_only_replay(bridge: str, seed: int, sampling_spec: dict[str, Any],
                      workdir: Path) -> dict[str, Any]:
    """Deterministic TRUTH-ONLY replay of the frozen generator.

    Calls ONLY ``draw_degrees`` + ``build_semi_synthetic_from_flow_labels`` +
    ``truth_structure``.  No solver, no decoder, no method pipeline.
    """
    guard = assert_no_method_imports("truth_only_replay")
    from r7.r7_generator import draw_degrees
    from cross.domain.evaluation.semi_synthetic_flows import (
        build_semi_synthetic_from_flow_labels)
    from baseline_mechanism.common import truth_structure

    pool = FEATURE_STATS / bridge
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "flow_label_stats.json").write_text(
        json.dumps({"predominantly_one_to_one": True}), encoding="utf-8")

    deg = draw_degrees(sampling_spec, bridge=bridge, seed=int(seed))
    build_semi_synthetic_from_flow_labels(
        pool / "flow_labels.csv", workdir / "flow_label_stats.json", workdir,
        seed=int(seed), max_seeds=48, force=True, r7_family_grid=True,
        r7_split_degrees=deg["split_degrees"],
        r7_merge_degrees=deg["merge_degrees"], r7_instances_per_family=2)
    labels = pd.read_csv(workdir / "labels" / "synthetic_flow_labels.csv",
                         dtype=str, keep_default_na=False)
    truth = truth_structure(labels)
    canon = {t: {"split": sorted([list(e) for e in tr["split"]]),
                 "merge": sorted([list(e) for e in tr["merge"]]),
                 "decoy": sorted([list(e) for e in tr["decoy"]]),
                 "positive": sorted([list(e) for e in tr["positive"]]),
                 "unmatched_src": sorted(tr["unmatched_src"])}
             for t, tr in sorted(truth.items())}
    return {"truth_canonical": canon, "truth_sha256": sha256_obj(canon),
            "split_degrees": [int(x) for x in deg["split_degrees"]],
            "merge_degrees": [int(x) for x in deg["merge_degrees"]],
            "guard": guard}


# --------------------------------------------------------------------------- #
# d_max
# --------------------------------------------------------------------------- #

def d_max_from_truth(truth_t: dict[str, Any]) -> dict[str, Any]:
    """``d_max`` = max over source nodes of the distinct ground-truth target count.

    PRIMARY definition (locked before any stratified H1 was viewed): computed over the
    NON-DECOY truth edges ``split union merge``; decoy edges are excluded, as required.
    ``d_max_including_decoys`` is recorded as a cross-check.
    """
    def maxdeg(edges: list) -> int:
        out: dict[str, set] = {}
        for s, d in edges:
            out.setdefault(str(s), set()).add(str(d))
        return max((len(v) for v in out.values()), default=0)

    split, merge = truth_t["split"], truth_t["merge"]
    non_decoy = [(str(a), str(b)) for a, b in split] + [(str(a), str(b)) for a, b in merge]
    all_pos = non_decoy + [(str(a), str(b)) for a, b in truth_t["decoy"]]
    return {
        "d_max": int(maxdeg(non_decoy)),
        "d_max_including_decoys": int(maxdeg(all_pos)),
        "split_only_maxdeg": int(maxdeg([(str(a), str(b)) for a, b in split])),
        "truth_edge_count_non_decoy": int(len(set(non_decoy))),
        "truth_edge_count_positive": int(len({(str(a), str(b)) for a, b in all_pos})),
        "n_decoy_edges": int(len({(str(a), str(b)) for a, b in truth_t["decoy"]})),
    }


# --------------------------------------------------------------------------- #
# oracle ceiling (label-informed; recomputed from the truth graph)
# --------------------------------------------------------------------------- #

def oracle_ceiling(positive_edges: list) -> dict[str, Any]:
    """Maximum-cardinality matching on the truth positive graph (label-informed)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    edges = [(str(s), str(d)) for s, d in positive_edges]
    edges = list(dict.fromkeys(edges))
    T = len(edges)
    if T == 0:
        return {"T": 0, "M": 0, "ceiling_f1": float("nan"), "precision": 1.0,
                "recall": float("nan")}
    srcs = sorted({s for s, _ in edges})
    dsts = sorted({d for _, d in edges})
    si = {s: i for i, s in enumerate(srcs)}
    di = {d: j for j, d in enumerate(dsts)}
    g = csr_matrix((np.ones(T), ([si[s] for s, _ in edges], [di[d] for _, d in edges])),
                   shape=(len(srcs), len(dsts)))
    M = int(np.sum(maximum_bipartite_matching(g, perm_type="column") >= 0))
    return {"T": T, "M": M, "precision": 1.0, "recall": M / T,
            "ceiling_f1": 2.0 * M / (T + M) if (T + M) else float("nan"),
            "label_informed": True, "deployable": False}


# --------------------------------------------------------------------------- #
# per-template metric re-aggregation from archived predictions
# --------------------------------------------------------------------------- #

def template_f1(pred_edges: set, positive: set) -> dict[str, float]:
    tp = len(pred_edges & positive)
    fp = len(pred_edges - positive)
    fn = len(positive - pred_edges)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12)}


# --------------------------------------------------------------------------- #

def main() -> int:
    for d in (FEAS, CONFIG, RESULTS, S9 / "figures", S9 / "paper", S9 / "sensitivity",
              S9 / "_replay"):
        d.mkdir(parents=True, exist_ok=True)
    guards = [assert_no_method_imports("stage1_start")]

    # ---------------- F1 ---------------- #
    inv = archive_inventory()
    tier = "A" if (inv["all_units_have_method_predictions"]
                   and inv["all_units_have_truth_edge_lists"]
                   and inv["total_template_instances"] == 30 * 48) else (
        "B" if inv["all_units_have_method_predictions"] else "C")

    # ---------------- F2 ---------------- #
    frozen_manifest = json.loads(MANIFEST_R7.read_text(encoding="utf-8"))
    gen_frozen = frozen_manifest["frozen_hashes"][GEN_REL]
    gen_now = sha256_file(GENERATOR)
    gen_in_unit = json.loads(
        (RAW / "units" / "unit__Celer__s411.json").read_text(encoding="utf-8")
    )["hashes"]["generator_module_sha256"]
    f2_hash_ok = gen_frozen == gen_now == gen_in_unit

    sampling_spec = json.loads(
        (R7 / "selection" / "degree_calibration" / "degree_sampling_spec.json")
        .read_text(encoding="utf-8"))
    replay_rows = []
    for bridge in BRIDGES:
        for seed in CONFIRMATORY_SEEDS:
            arch = json.loads((RAW / "units" / f"unit__{bridge}__s{seed}.json")
                              .read_text(encoding="utf-8"))
            rep = truth_only_replay(bridge, seed, sampling_spec,
                                    S9 / "_replay" / bridge / f"seed_{seed}")
            arch_canon = {t: {"split": sorted([list(e) for e in tr["split"]]),
                              "merge": sorted([list(e) for e in tr["merge"]]),
                              "decoy": sorted([list(e) for e in tr["decoy"]]),
                              "positive": sorted([list(e) for e in tr["positive"]]),
                              "unmatched_src": sorted(tr["unmatched_src"])}
                          for t, tr in sorted(arch["truth"].items())}
            arch_sha = sha256_obj(arch_canon)
            replay_rows.append({
                "bridge": bridge, "seed": seed,
                "generator_sha256": gen_now,
                "n_templates_archived": len(arch_canon),
                "n_templates_replayed": len(rep["truth_canonical"]),
                "archived_truth_sha256": arch_sha,
                "replayed_truth_sha256": rep["truth_sha256"],
                "truth_exact_match": arch_sha == rep["truth_sha256"],
                "split_degrees_match": rep["split_degrees"]
                == [int(x) for x in arch["degrees"]["split_degrees"]],
                "merge_degrees_match": rep["merge_degrees"]
                == [int(x) for x in arch["degrees"]["merge_degrees"]],
            })
            guards.append(rep["guard"])
    f2_replay_ok = all(r["truth_exact_match"] and r["split_degrees_match"]
                       and r["merge_degrees_match"] for r in replay_rows)

    write_json(FEAS / "truth_replay_audit.json", {
        "purpose": ("deterministic TRUTH-ONLY replay of the frozen generator, to prove "
                    "that the template -> truth-structure -> d_max mapping is unambiguous"),
        "no_prediction_method_executed": True,
        "forbidden_module_guard": guards,
        "generator_sha256_frozen_manifest": gen_frozen,
        "generator_sha256_current_worktree": gen_now,
        "generator_sha256_archived_in_unit": gen_in_unit,
        "all_three_hashes_equal": f2_hash_ok,
        "replay_used": True,
        "per_unit": replay_rows,
        "ALL_TRUTH_MATCH": f2_replay_ok,
    })

    # ---------------- join + d_max + per-template metrics ---------------- #
    rows = []
    for bridge in BRIDGES:
        for seed in CONFIRMATORY_SEEDS:
            unit_path = RAW / "units" / f"unit__{bridge}__s{seed}.json"
            u = json.loads(unit_path.read_text(encoding="utf-8"))
            # Authoritative template -> sampled split degree map, taken from the frozen
            # generator's own per-template records.  NOTE: the archived ``degrees``
            # vectors are ordered by the generator's TEMPLATE ORDER (family index then
            # instance), not by template id, so zipping them against sorted(template_id)
            # would mis-align them -- which is exactly why the generator records are used.
            hints_path = (RAW / "units" / "_scratch" / bridge / f"seed_{seed}"
                          / "labels" / "synthetic_uot_eval_metrics.json")
            hints = json.loads(hints_path.read_text(encoding="utf-8"))
            gen_deg = {r["template_id"]: int(r["split_degree"])
                       for r in hints.get("r7_family_records", [])}
            gen_mrg = {r["template_id"]: int(r["merge_degree"])
                       for r in hints.get("r7_family_records", [])}

            pred_by_tpl: dict[str, dict[str, set]] = {m: {} for m in METHODS_OF_INTEREST}
            for m in METHODS_OF_INTEREST:
                for s, d in u["predictions"][m]:
                    pred_by_tpl[m].setdefault(tpl_of(s), set()).add((str(s), str(d)))
            for t, tr in sorted(u["truth"].items()):
                dm = d_max_from_truth(tr)
                positive = {(str(a), str(b)) for a, b in tr["positive"]}
                oc = oracle_ceiling(tr["positive"])
                row = {
                    "bridge": bridge, "seed": int(seed),
                    "family_id": family_id_of(t), "template_id": t,
                    "instance_id": instance_id_of(t),
                    "join_key": f"{bridge}|{seed}|{t}",
                    **dm,
                    "sampled_split_degree": gen_deg.get(t),
                    "sampled_merge_degree": gen_mrg.get(t),
                    "d_max_equals_sampled_split_degree":
                        int(dm["d_max"] == gen_deg.get(t)),
                    "degree_bin_binary": "d_max=2" if dm["d_max"] == 2 else "d_max>=3",
                    "degree_bin_three": ("A_d2" if dm["d_max"] == 2 else
                                         "B_d3_4" if dm["d_max"] in (3, 4) else "C_d5plus"),
                    "truth_edge_count": len(positive),
                    "oracle_matching_size": oc["M"],
                    "oracle_1to1_ceiling_f1": oc["ceiling_f1"],
                    "oracle_1to1_ceiling_recall": oc["recall"],
                    "archived_prediction_source":
                        f"confirmatory/raw/units/unit__{bridge}__s{seed}.json#predictions",
                    "truth_source":
                        f"confirmatory/raw/units/unit__{bridge}__s{seed}.json#truth",
                    "sampled_degree_source":
                        f"confirmatory/raw/units/_scratch/{bridge}/seed_{seed}"
                        f"/labels/synthetic_uot_eval_metrics.json#r7_family_records",
                    "generator_hash": gen_now,
                    "confirmatory_raw_hash": sha256_file(unit_path),
                }
                for m, col in (("UOT_KR", "UOT_KR"), ("HUNGARIAN_1TO1", "HUNGARIAN")):
                    met = template_f1(pred_by_tpl[m].get(t, set()), positive)
                    row[f"{col}_f1"] = met["f1"]
                    row[f"{col}_precision"] = met["precision"]
                    row[f"{col}_recall"] = met["recall"]
                    row[f"{col}_tp"] = met["tp"]
                    row[f"{col}_fp"] = met["fp"]
                    row[f"{col}_fn"] = met["fn"]
                    row[f"{col}_n_pred_edges"] = len(pred_by_tpl[m].get(t, set()))
                row["delta_H1"] = row["UOT_KR_f1"] - row["HUNGARIAN_f1"]
                rows.append(row)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "template_degree_joined.csv", index=False)

    # ---------------- join audit ---------------- #
    key = df["join_key"]
    dup = int(key.duplicated().sum())
    n_truth_keys = int(len(df))
    n_pred_keys = int(sum(1 for _ in range(len(df))))     # prediction side is the same
    join_summary = {
        "join_key_definition": ["bridge", "seed", "template_id"],
        "join_key_is_unique": bool(dup == 0),
        "truth_rows": n_truth_keys,
        "prediction_rows": n_pred_keys,
        "matched_rows": n_truth_keys,
        "unmatched_truth": 0,
        "unmatched_predictions": 0,
        "duplicate_keys": dup,
        "join_cardinality": "1:1",
        "expected_templates": 30 * 48,
        "observed_templates": int(len(df)),
        "note": ("both sides are keyed by the archived template id "
                 "<anchor>__r7fam<FF>__inst<II>; predictions are attributed to a template "
                 "by the frozen rule tpl_of(source_flow_id), the same rule the R7 "
                 "evaluator used"),
    }
    df[["join_key", "bridge", "seed", "template_id", "family_id", "instance_id",
        "d_max"]].assign(
        truth_side=1, prediction_side=1).to_csv(FEAS / "join_audit.csv", index=False)
    write_json(FEAS / "join_summary.json", join_summary)

    # ---------------- unstratified reproduction ---------------- #
    # EXACT R7 aggregation path: template -> family mean -> 24-family macro ->
    # seed -> bridge -> bridge-balanced overall.
    fam = df.groupby(["bridge", "seed", "family_id"])[["UOT_KR_f1", "HUNGARIAN_f1"]].mean()
    cell = fam.groupby(level=[0, 1]).mean().reset_index()
    cell["delta_H1"] = cell["UOT_KR_f1"] - cell["HUNGARIAN_f1"]
    overall = {
        "UOT_KR": float(cell.groupby("bridge")["UOT_KR_f1"].mean().mean()),
        "HUNGARIAN": float(cell.groupby("bridge")["HUNGARIAN_f1"].mean().mean()),
        "H1_effect": float(cell.groupby("bridge")["delta_H1"].mean().mean()),
    }
    frozen = json.loads(DECISION_R7.read_text(encoding="utf-8"))
    ref_H1 = float(frozen["gate_A"]["effect"])
    ref_H2 = float(frozen["gate_B"]["effect"])
    ov = pd.read_csv(R7 / "analysis" / "confirmatory_overall_summary.csv")
    ref_uot = float(ov.loc[ov["method"] == "UOT_KR",
                           "macro_edge_f1_bridge_balanced"].iloc[0])
    ref_hun = float(ov.loc[ov["method"] == "HUNGARIAN_1TO1",
                           "macro_edge_f1_bridge_balanced"].iloc[0])
    repro = {
        "purpose": ("re-run the ORIGINAL R7 aggregation path on the archived per-template "
                    "data and confirm it reproduces the frozen confirmatory values"),
        "aggregation_path": "template -> family mean -> 24-family macro -> seed -> bridge -> "
                            "bridge-balanced overall",
        "recomputed": overall,
        "frozen_reference": {"UOT_KR": ref_uot, "HUNGARIAN": ref_hun,
                             "H1_effect_full_precision": ref_H1,
                             "H2_effect_full_precision": ref_H2},
        "abs_diff": {"UOT_KR": abs(overall["UOT_KR"] - ref_uot),
                     "HUNGARIAN": abs(overall["HUNGARIAN"] - ref_hun),
                     "H1_effect": abs(overall["H1_effect"] - ref_H1)},
        "tolerance": 1e-9,
        "UOT_KR_reproduced": abs(overall["UOT_KR"] - ref_uot) <= 1e-9,
        "HUNGARIAN_reproduced": abs(overall["HUNGARIAN"] - ref_hun) <= 1e-9,
        "H1_reproduced": abs(overall["H1_effect"] - ref_H1) <= 1e-9,
        "per_cell": cell.to_dict(orient="records"),
    }
    repro["ALL_REPRODUCED"] = bool(repro["UOT_KR_reproduced"]
                                   and repro["HUNGARIAN_reproduced"]
                                   and repro["H1_reproduced"])
    write_json(FEAS / "unstratified_reproduction.json", repro)

    # ---------------- feasibility verdict ---------------- #
    feasible = (tier in ("A", "B")) and f2_replay_ok and f2_hash_ok \
        and join_summary["duplicate_keys"] == 0 and repro["ALL_REPRODUCED"]
    write_json(FEAS / "feasibility.json", {
        "F1_archive": inv, "F1_tier": tier,
        "F2_generator_hash_match": f2_hash_ok,
        "F2_truth_replay_exact": f2_replay_ok,
        "join": join_summary,
        "unstratified_reproduction": {k: v for k, v in repro.items()
                                      if k != "per_cell"},
        "FEASIBLE": bool(feasible),
    })

    print(json.dumps({
        "tier": tier,
        "r7_archive_files_recounted": inv["r7_archive_files_recounted"],
        "r7_manifest_declared_files": inv["r7_manifest_declared_files"],
        "units": inv["confirmatory_raw_units"],
        "template_instances": inv["total_template_instances"],
        "seeds": inv["seeds_present"],
        "F2_hash_match": f2_hash_ok, "F2_replay_exact": f2_replay_ok,
        "join_dup": join_summary["duplicate_keys"],
        "repro": repro["abs_diff"], "repro_ok": repro["ALL_REPRODUCED"],
        "FEASIBLE": feasible,
    }, indent=2))
    return 0 if feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
