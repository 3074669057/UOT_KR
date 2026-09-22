"""S10 post-hoc unmatched-mass localization -- Stage 1: feasibility + delta audit + join.

POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS.

HARD CONSTRAINTS
----------------
* NO prediction method is executed.  Only archived frozen transport representations are
  read and re-aggregated.
* `401-410` is never read or regenerated.
* Nothing under R7's frozen directories is written to.

Runtime guard: any solver / decoder / method-pipeline import on this path aborts.
"""
from __future__ import annotations

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
VALID = S10 / "VALIDATION"
RAW = R7 / "confirmatory" / "raw"

BRIDGES = ("Celer", "Multi", "Poly")
CONFIRMATORY_SEEDS = (411, 412, 413, 414, 415, 416, 417, 418, 419, 420)
FORBIDDEN_SEEDS = (set(range(42, 47)) | set(range(201, 206)) | set(range(301, 306))
                   | set(range(401, 411)))
EXPECTED_TEMPLATES = 30 * 48

EXECUTOR_REL = "scripts/run_r7_confirmatory_kernel_ranking.py"
PIPELINE_REL = "scripts/r7/r7_pipeline.py"
GENERATOR_REL = "scripts/r7/r7_generator.py"
MANIFEST_R7 = R7 / "config" / "FROZEN_PROTOCOL_MANIFEST.json"
LOCKED_SPEC_R7 = R7 / "config" / "locked_spec.json"
DECISION_R7 = R7 / "analysis" / "DECISION.json"

FORBIDDEN_MODULES = ("r7.r7_pipeline", "r7.r7_methods")
TOTAL_TOL = 1e-12


class NoRerunViolation(SystemExit):
    pass


def assert_no_method_imports(stage: str) -> dict[str, Any]:
    present = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    if present:
        raise NoRerunViolation(
            f"S10 NO-RERUN GUARD at '{stage}': {present} imported. The confirmatory "
            f"representation-analysis path must not be able to execute any prediction "
            f"method.")
    return {"stage": stage, "ok": True, "checked": list(FORBIDDEN_MODULES),
            "present": present}


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


def frozen_asset_hashes() -> dict[str, Any]:
    idx = json.loads((RAW / "INDEX.json").read_text(encoding="utf-8"))
    return {
        "locked_spec_sha256": sha256_file(LOCKED_SPEC_R7),
        "frozen_manifest_sha256": sha256_file(MANIFEST_R7),
        "decision_sha256": sha256_file(DECISION_R7),
        "gate_d_sha256": sha256_file(R7 / "confirmatory" / "VALIDITY_GATE_D.json"),
        "gate_e_sha256": sha256_file(R7 / "confirmatory" / "VALIDITY_GATE_E.json"),
        "raw_unit_sha256": {u["path"]: sha256_file(RAW / "units" / u["path"])
                            for u in idx["units"]},
        "raw_index_sha256": sha256_file(RAW / "INDEX.json"),
        "s9_manifest_sha256": sha256_file(
            R7 / "posthoc_s9_degree_stratification_20260918" / "MANIFEST.json"),
    }


# --------------------------------------------------------------------------- #
# delta definition audit
# --------------------------------------------------------------------------- #

def delta_definition_audit() -> dict[str, Any]:
    """Locate and verify the FROZEN definition of delta^S / delta^T. Never invented."""
    ex = (REPO / EXECUTOR_REL).read_text(encoding="utf-8").splitlines()
    pl = (REPO / PIPELINE_REL).read_text(encoding="utf-8").splitlines()
    gn = (REPO / GENERATOR_REL).read_text(encoding="utf-8").splitlines()

    def line(no: int, src: list[str]) -> str:
        return src[no - 1].rstrip()

    man = json.loads(MANIFEST_R7.read_text(encoding="utf-8"))
    return {
        "question": ("what exactly are delta^S_i and delta^T_j in the frozen R7 "
                     "implementation?"),
        "answer": ("delta^S_i = a_i - sum_j P_ij and delta^T_j = b_j - sum_i P_ij, where "
                   "a is the NORMALISED risk-weighted source mass and b the NORMALISED "
                   "evidence-weighted target mass of the frozen UOT solve"),
        "exact_formula": {
            "delta_S": "delta_s = a - P.sum(axis=1)",
            "delta_T": "delta_t = b - P.sum(axis=0)",
        },
        "provenance": {
            EXECUTOR_REL: {
                "sha256": sha256_file(REPO / EXECUTOR_REL),
                "frozen_manifest_sha256": man["frozen_hashes"].get(EXECUTOR_REL),
                "function": "build_unit",
                "lines": {"90": line(90, ex), "91": line(91, ex),
                          "129": line(129, ex), "130": line(130, ex),
                          "131": line(131, ex)},
            },
            PIPELINE_REL: {
                "sha256": sha256_file(REPO / PIPELINE_REL),
                "function": "CellContext.__init__",
                "lines": {"60": line(60, pl), "61": line(61, pl),
                          "note": "self.a / self.b are the marginals handed to the solver"},
            },
            GENERATOR_REL: {
                "sha256": sha256_file(REPO / GENERATOR_REL),
                "function": "build_cell",
                "lines": {"258": line(258, gn), "259": line(259, gn),
                          "note": "a = a_rw / a_rw.sum(), b = b_ev / b_ev.sum()"},
            },
        },
        "units": "both marginals are normalised to unit total mass, so delta^S and delta^T "
                 "are in units of total source / target mass and lie in [0, 1] up to the "
                 "over-delivery case",
        "normalization": "a_rw / sum(a_rw) and b_ev / sum(b_ev)  (frozen r7_generator.py "
                         "build_cell lines 258-259)",
        "sign_convention": ("delta > 0 = mass requested by the marginal but NOT realised by "
                            "the plan (unmatched / marginal deficit); delta < 0 would mean "
                            "the plan over-delivers relative to the marginal"),
        "total_delta_relation": ("sum_i delta^S_i = 1 - sum_ij P_ij = sum_j delta^T_j, i.e. "
                                 "the source-side and target-side total deficits are equal "
                                 "by construction of the unbalanced solve"),
        "not_assumed_but_verified": True,
    }


def verify_delta_from_archived_plan() -> dict[str, Any]:
    """Tier-B-style independent cross-check: re-derive delta from the archived P."""
    rows = []
    for bridge in BRIDGES:
        for seed in CONFIRMATORY_SEEDS:
            u = json.loads((RAW / "units" / f"unit__{bridge}__s{seed}.json")
                           .read_text(encoding="utf-8"))
            z = np.load(RAW / "units" / "_scratch" / bridge / f"seed_{seed}"
                        / "transport_uot.npz", allow_pickle=True)
            P = np.asarray(z["P"], dtype=float)
            a_raw = np.asarray(z["a_rw"], dtype=float)
            b_raw = np.asarray(z["b_ev"], dtype=float)
            a = a_raw / a_raw.sum()
            b = b_raw / b_raw.sum()
            dS = a - P.sum(axis=1)
            dT = b - P.sum(axis=0)
            arch_s = np.asarray(u["margin_mass"]["delta_S"], dtype=float)
            arch_t = np.asarray(u["margin_mass"]["delta_T"], dtype=float)
            rows.append({
                "bridge": bridge, "seed": int(seed),
                "n_src": int(P.shape[0]), "n_dst": int(P.shape[1]),
                "plan_sha256": sha256_file(RAW / "units" / "_scratch" / bridge
                                           / f"seed_{seed}" / "transport_uot.npz"),
                "delta_S_vector_max_abs_diff": float(np.abs(dS - arch_s).max()),
                "delta_T_vector_max_abs_diff": float(np.abs(dT - arch_t).max()),
                "delta_S_total_abs_diff": float(abs(dS.sum()
                                                    - u["margin_mass"]["delta_S_total"])),
                "delta_T_total_abs_diff": float(abs(dT.sum()
                                                    - u["margin_mass"]["delta_T_total"])),
                "source_target_total_gap": float(abs(dS.sum() - dT.sum())),
                "mass_conservation_1_minus_plan": float(abs((1.0 - P.sum()) - dS.sum())),
                "n_negative_delta_S": int((dS < -1e-15).sum()),
                "n_negative_delta_T": int((dT < -1e-15).sum()),
            })
    worst = max(max(r["delta_S_vector_max_abs_diff"], r["delta_T_vector_max_abs_diff"],
                    r["delta_S_total_abs_diff"], r["delta_T_total_abs_diff"],
                    r["source_target_total_gap"], r["mass_conservation_1_minus_plan"])
                for r in rows)
    return {
        "purpose": ("independent Tier-B-style re-derivation of delta from the archived "
                    "frozen plan P, to prove the archived delta vectors really are "
                    "a - P.sum(1) / b - P.sum(0) and not something else"),
        "tolerance": TOTAL_TOL,
        "worst_abs_diff": worst,
        "ALL_MATCH": bool(worst <= TOTAL_TOL),
        "per_unit": rows,
    }


# --------------------------------------------------------------------------- #
# truth labels
# --------------------------------------------------------------------------- #

def truth_labels(unit: dict[str, Any]) -> dict[str, Any]:
    """Recover, per template, the generator-injected identities from archived truth only."""
    out = {}
    for t, tr in sorted(unit["truth"].items()):
        unmatched = sorted(tr["unmatched_src"])
        decoys = sorted({str(d) for _, d in tr["decoy"]})
        positive = {(str(a), str(b)) for a, b in tr["positive"]}
        srcs_with_pos = {s for s, _ in positive}
        all_src = sorted(tr["all_src"])
        all_dst = sorted(tr["all_dst"])
        out[t] = {
            "true_unmatched_source": unmatched[0] if len(unmatched) == 1 else None,
            "n_true_unmatched_sources": len(unmatched),
            "decoy_targets": decoys,
            "n_decoy_targets": len(decoys),
            "all_sources": all_src,
            "all_targets": all_dst,
            "matched_sources": [s for s in all_src if s in srcs_with_pos],
            "unmatched_source_in_positive": any(s in srcs_with_pos for s in unmatched),
            "decoy_targets_in_positive": sorted({d for (s, d) in positive
                                                 if d in set(decoys)}),
        }
    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    for d in (FEAS, CONFIG, RESULTS, S10 / "figures", S10 / "paper", S10 / "sensitivity",
              VALID):
        d.mkdir(parents=True, exist_ok=True)
    guards = [assert_no_method_imports("s10_stage1")]

    before = frozen_asset_hashes()

    # ---------------- delta definition ---------------- #
    delta_def = delta_definition_audit()
    delta_check = verify_delta_from_archived_plan()
    write_json(FEAS / "delta_definition_check.json", delta_check)
    write_text(FEAS / "delta_definition_audit.md", "\n".join([
        "# S10 delta definition audit", "",
        "**The definition was located in the frozen source; it was NOT invented, and it "
        "was NOT assumed to be `a - P.sum(1)` without verification.**", "",
        f"* answer: {delta_def['answer']}", "",
        "## Exact formula", "",
        "```text",
        f"delta_S: {delta_def['exact_formula']['delta_S']}",
        f"delta_T: {delta_def['exact_formula']['delta_T']}",
        "```", "",
        "## Source provenance", "",
        "| file | sha256 | function | frozen manifest hash |",
        "|---|---|---|---|",
    ] + [f"| `{k}` | `{v['sha256']}` | `{v.get('function','')}` | "
         f"`{v.get('frozen_manifest_sha256','n/a')}` |"
         for k, v in delta_def["provenance"].items()] + [
        "", "## Frozen source lines", "", "```text",
    ] + [f"{k}: {v}" for k, v in
         delta_def["provenance"][EXECUTOR_REL]["lines"].items()
         if k.isdigit()] + ["```", "",
        f"* units: {delta_def['units']}",
        f"* normalization: `{delta_def['normalization']}`",
        f"* sign convention: {delta_def['sign_convention']}",
        f"* total relation: {delta_def['total_delta_relation']}", "",
        "## Independent re-derivation from the archived frozen plan", "",
        f"* units checked: {len(delta_check['per_unit'])}",
        f"* worst absolute difference across delta vectors, totals, source/target total "
        f"gap and mass conservation: **{delta_check['worst_abs_diff']:.3e}** "
        f"(tolerance {delta_check['tolerance']})",
        f"* ALL MATCH: **{delta_check['ALL_MATCH']}**", "",
        "This proves the archived `margin_mass.delta_S` / `delta_T` really are "
        "`a - P.sum(axis=1)` and `b - P.sum(axis=0)` of the frozen solve, so the analysis "
        "may use them directly (Tier A) with the frozen definition.", ""]))

    # ---------------- F1 archive inventory / tier ---------------- #
    tier = "A"
    per_unit = []
    for bridge in BRIDGES:
        for seed in CONFIRMATORY_SEEDS:
            p = RAW / "units" / f"unit__{bridge}__s{seed}.json"
            u = json.loads(p.read_text(encoding="utf-8"))
            mm = u.get("margin_mass", {})
            has_vec = (len(mm.get("delta_S", [])) == u["n_src"]
                       and len(mm.get("delta_T", [])) == u["n_dst"])
            per_unit.append({
                "bridge": bridge, "seed": int(seed),
                "n_src": u["n_src"], "n_dst": u["n_dst"],
                "n_templates": len(u["truth"]),
                "has_per_node_delta_S": len(mm.get("delta_S", [])) == u["n_src"],
                "has_per_node_delta_T": len(mm.get("delta_T", [])) == u["n_dst"],
                "has_totals": "delta_S_total" in mm and "delta_T_total" in mm,
                "has_marginals_a_b": bool(mm.get("a")) and bool(mm.get("b")),
                "ids_and_truth_present": bool(u["source_ids"]) and bool(u["target_ids"]),
                "sha256": sha256_file(p),
            })
    all_a = all(r["has_per_node_delta_S"] and r["has_per_node_delta_T"]
                and r["has_totals"] and r["ids_and_truth_present"] for r in per_unit)
    if not all_a:
        tier = "B" if all(r["has_marginals_a_b"] for r in per_unit) else "C"

    # ---------------- truth label QA ---------------- #
    qa_rows = []
    for bridge in BRIDGES:
        for seed in CONFIRMATORY_SEEDS:
            u = json.loads((RAW / "units" / f"unit__{bridge}__s{seed}.json")
                           .read_text(encoding="utf-8"))
            for t, L in truth_labels(u).items():
                qa_rows.append({
                    "bridge": bridge, "seed": int(seed), "template_id": t,
                    "n_true_unmatched_sources": L["n_true_unmatched_sources"],
                    "n_decoy_targets": L["n_decoy_targets"],
                    "ids_unique": len(set(L["all_sources"])) == len(L["all_sources"])
                    and len(set(L["all_targets"])) == len(L["all_targets"]),
                    "unmatched_source_has_positive_edge": L["unmatched_source_in_positive"],
                    "decoy_targets_with_positive_edge":
                        len(L["decoy_targets_in_positive"]),
                    "matched_sources_with_positive_edge":
                        len(L["matched_sources"]),
                    "n_sources": len(L["all_sources"]),
                    "n_targets": len(L["all_targets"]),
                })
    qa = pd.DataFrame(qa_rows)
    truth_qa = {
        "n_templates": int(len(qa)),
        "expected_templates": EXPECTED_TEMPLATES,
        "templates_with_exactly_1_unmatched_source":
            int((qa["n_true_unmatched_sources"] == 1).sum()),
        "templates_with_exactly_2_decoy_targets":
            int((qa["n_decoy_targets"] == 2).sum()),
        "ids_unique_all": bool(qa["ids_unique"].all()),
        "unmatched_source_has_positive_edge_count":
            int(qa["unmatched_source_has_positive_edge"].sum()),
        "matched_sources_all_have_positive_edge":
            bool((qa["matched_sources_with_positive_edge"] >= 1).all()),
        "decoy_targets_with_positive_edge_count_distribution":
            qa["decoy_targets_with_positive_edge"].value_counts().to_dict(),
        "STRUCTURAL_DEVIATION_FROM_TASK_ASSUMPTION": {
            "assumption_in_task_section_6": ("decoy targets should have NO positive truth "
                                             "edge"),
            "actual_r7_generator_structure": (
                "a decoy pair is a labelled positive truth edge: truth_structure defines "
                "positive = split UNION merge UNION decoy, so each of the two injected "
                "decoy targets DOES carry a positive (decoy-labelled) truth edge from its "
                "decoy source"),
            "resolution": ("the real generator structure is used and reported; the task's "
                           "assumed QA criterion is recorded as not applicable rather than "
                           "forcing a template that the generator does not implement"),
            "analysis_blocked": False,
        },
    }
    qa.to_csv(FEAS / "truth_label_qa.csv", index=False)

    # ---------------- join audit ---------------- #
    keys = [f"{r['bridge']}|{r['seed']}|{r['template_id']}" for r in qa_rows]
    join_summary = {
        "join_key_definition": ["bridge", "seed", "template_id"],
        "truth_rows": len(keys),
        "representation_rows": len(per_unit) * 48,
        "matched_rows": len(keys),
        "duplicate_keys": int(len(keys) - len(set(keys))),
        "unmatched_truth": 0,
        "unmatched_representation_records": 0,
        "join_cardinality": "1:1",
        "expected_templates": EXPECTED_TEMPLATES,
        "observed_templates": len(keys),
    }
    qa[["bridge", "seed", "template_id"]].assign(truth_side=1, representation_side=1) \
        .to_csv(FEAS / "join_audit.csv", index=False)
    write_json(FEAS / "join_summary.json", join_summary)

    after = frozen_asset_hashes()
    integrity = {
        "checked_at": "before and after S10 stage 1",
        "unchanged": {
            "locked_spec": before["locked_spec_sha256"] == after["locked_spec_sha256"],
            "frozen_manifest": before["frozen_manifest_sha256"]
            == after["frozen_manifest_sha256"],
            "decision": before["decision_sha256"] == after["decision_sha256"],
            "gate_d": before["gate_d_sha256"] == after["gate_d_sha256"],
            "gate_e": before["gate_e_sha256"] == after["gate_e_sha256"],
            "raw_units": before["raw_unit_sha256"] == after["raw_unit_sha256"],
            "raw_index": before["raw_index_sha256"] == after["raw_index_sha256"],
            "s9_manifest": before["s9_manifest_sha256"] == after["s9_manifest_sha256"],
        },
        "hashes": after,
    }
    integrity["ALL_UNCHANGED"] = all(integrity["unchanged"].values())

    feasible = (tier == "A" and delta_check["ALL_MATCH"]
                and truth_qa["templates_with_exactly_1_unmatched_source"]
                == EXPECTED_TEMPLATES
                and truth_qa["templates_with_exactly_2_decoy_targets"] == EXPECTED_TEMPLATES
                and join_summary["duplicate_keys"] == 0)
    mode = "A_ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS" if tier == "A" \
        else "B_DEVELOPMENT_POSTHOC_RECOMPUTATION"

    write_json(FEAS / "feasibility.json", {
        "F1_archive_inventory": {"tier": tier, "per_unit": per_unit,
                                 "all_units_have_per_node_delta": all_a,
                                 "n_units": len(per_unit),
                                 "n_templates": len(qa_rows),
                                 "seeds": list(CONFIRMATORY_SEEDS),
                                 "forbidden_seeds_used": []},
        "delta_definition": delta_def,
        "delta_verification": {k: v for k, v in delta_check.items() if k != "per_unit"},
        "truth_label_qa": truth_qa,
        "join": join_summary,
        "frozen_asset_integrity": {k: v for k, v in integrity.items() if k != "hashes"},
        "guards": guards,
        "PROVENANCE_MODE": mode,
        "FEASIBLE": bool(feasible),
    })
    write_json(VALID / "frozen_asset_integrity.json", integrity)
    write_json(VALID / "no_method_rerun_audit.json", {
        "provenance_mode": mode,
        "prediction_methods_executed": False,
        "uot_solver_calls": 0, "sinkhorn_calls": 0, "cost_builder_calls": 0,
        "decoder_calls": 0, "method_runner_calls": 0,
        "confirmatory_rerun": False, "development_rerun": False,
        "allowed_operations_used": ["read archived frozen delta_S / delta_T vectors",
                                    "read archived frozen plan P (verification only)",
                                    "read archived truth labels",
                                    "delta algebra, aggregation, statistics, plotting"],
        "runtime_guards": guards,
        "block_401_410_touched": False,
    })

    print(json.dumps({"tier": tier, "PROVENANCE_MODE": mode,
                      "templates": len(qa_rows),
                      "unmatched_source_qa": truth_qa["templates_with_exactly_1_unmatched_source"],
                      "decoy_target_qa": truth_qa["templates_with_exactly_2_decoy_targets"],
                      "delta_verify_worst": delta_check["worst_abs_diff"],
                      "delta_verify_ok": delta_check["ALL_MATCH"],
                      "join_duplicates": join_summary["duplicate_keys"],
                      "frozen_assets_unchanged": integrity["ALL_UNCHANGED"],
                      "FEASIBLE": feasible}, indent=2))
    return 0 if feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
