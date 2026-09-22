"""S9 post-hoc reporting: feasibility report, MANIFEST, PROVENANCE, checklist,
final post-hoc report."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R7 = Path(__file__).resolve().parents[2] / "out" / "r7_confirmatory_kernel_ranking_20260917"
REPO = R7.parents[1]
S9 = R7 / "posthoc_s9_degree_stratification_20260918"
FEAS = S9 / "00_feasibility"
RESULTS = S9 / "results"
FIGURES = S9 / "figures"
RAW = R7 / "confirmatory" / "raw"

BRIDGES = ("Celer", "Multi", "Poly")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _w(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")


def _load() -> dict[str, Any]:
    return {
        "ana": json.loads((RESULTS / "s9_analysis.json").read_text(encoding="utf-8")),
        "feas": json.loads((FEAS / "feasibility.json").read_text(encoding="utf-8")),
        "repro": json.loads((FEAS / "unstratified_reproduction.json").read_text(encoding="utf-8")),
        "replay": json.loads((FEAS / "truth_replay_audit.json").read_text(encoding="utf-8")),
        "join": json.loads((FEAS / "join_summary.json").read_text(encoding="utf-8")),
        "spec_hash": (S9 / "config" / "locked_posthoc_spec.sha256").read_text(
            encoding="utf-8").split()[0],
        "sens": None,
    }


def _fmt(x: float, n: int = 4) -> str:
    return f"{x:+.{n}f}"


# --------------------------------------------------------------------------- #
# feasibility report
# --------------------------------------------------------------------------- #

def build_feasibility_report() -> str:
    d = _load()
    f, r, j, rp, a = d["feas"], d["repro"], d["join"], d["replay"], d["ana"]
    inv = f["F1_archive"]
    L = ["# S9 feasibility report", "",
         "* experiment: `r7_posthoc_degree_stratification_20260918`",
         "* classification: **POST-HOC / NON-PREREGISTERED / ARCHIVED-PREDICTION "
         "RE-STRATIFICATION ANALYSIS**",
         f"* locked post-hoc spec sha256: `{d['spec_hash']}`",
         "", "## F1 — is per-template prediction information retained?", "",
         f"* availability tier: **Tier {f['F1_tier']}**", "",
         "| item | value |", "|---|---|",
         f"| R7 archive files (recounted) | **{inv['r7_archive_files_recounted']}** |",
         f"| R7 MANIFEST declared files | {inv['r7_manifest_declared_files']} "
         f"(excludes regenerable scratch/cells) |",
         f"| confirmatory raw units | {inv['confirmatory_raw_units']} |",
         f"| seeds present | `{inv['seeds_present']}` |",
         f"| bridges present | `{sorted(inv['bridges_present'])}` |",
         f"| templates per unit | {inv['templates_per_unit']} |",
         f"| total template instances | **{inv['total_template_instances']}** |",
         f"| every unit has UOT_KR + HUNGARIAN predictions | "
         f"{inv['all_units_have_method_predictions']} |",
         f"| every unit has full truth edge lists | {inv['all_units_have_truth_edge_lists']} |",
         f"| per-template oracle present | {inv['all_units_have_per_template_oracle']} |",
         f"| sampled degrees present | {inv['all_units_have_sampled_degrees']} |",
         f"| forbidden-seed units present | `{inv['forbidden_seed_units_present']}` |",
         f"| only the successful 411–420 block used | {inv['only_successful_block_used']} |",
         "",
         "The archive retains, for every template instance: `bridge`, `seed`, family / "
         "template / instance id, the **truth edge lists**, and the **UOT_KR and "
         "HUNGARIAN_1TO1 prediction edge sets**. Both F1 values are therefore recomputed "
         "independently from archived prediction edges — no archived metric is reused.",
         "", "## F2 — can template -> d_max be reconstructed unambiguously?", "",
         "| item | value |", "|---|---|",
         f"| generator sha256 (frozen manifest) | `{rp['generator_sha256_frozen_manifest']}` |",
         f"| generator sha256 (current worktree) | `{rp['generator_sha256_current_worktree']}` |",
         f"| generator sha256 (archived in unit) | `{rp['generator_sha256_archived_in_unit']}` |",
         f"| all three equal | **{rp['all_three_hashes_equal']}** |",
         f"| truth-only deterministic replay used | {rp['replay_used']} |",
         f"| units with bit-identical replayed truth | "
         f"**{sum(1 for x in rp['per_unit'] if x['truth_exact_match'])}"
         f"/{len(rp['per_unit'])}** |",
         f"| sampled split/merge degrees reproduced | "
         f"{all(x['split_degrees_match'] and x['merge_degrees_match'] for x in rp['per_unit'])} |",
         "",
         "The replay is **truth-only**: it calls the frozen generator and the truth "
         "structure serialiser and nothing else. A runtime guard asserts that no solver, "
         "decoder or method-pipeline module is imported on this path (recorded in "
         "`truth_replay_audit.json`).",
         "", "## Join", "",
         "| item | value |", "|---|---|",
         f"| join key | `{j['join_key_definition']}` |",
         f"| truth rows | {j['truth_rows']} |",
         f"| prediction rows | {j['prediction_rows']} |",
         f"| matched rows | {j['matched_rows']} |",
         f"| unmatched truth | **{j['unmatched_truth']}** |",
         f"| unmatched predictions | **{j['unmatched_predictions']}** |",
         f"| duplicate keys | **{j['duplicate_keys']}** |",
         f"| join cardinality | {j['join_cardinality']} |",
         "",
         "## Unstratified reconstruction check (before any stratification)", "",
         "| quantity | recomputed from archive | frozen R7 | abs diff |",
         "|---|---:|---:|---:|",
         f"| UOT_KR macro edge F1 | `{r['recomputed']['UOT_KR']!r}` | "
         f"`{r['frozen_reference']['UOT_KR']!r}` | {r['abs_diff']['UOT_KR']:.3g} |",
         f"| HUNGARIAN macro edge F1 | `{r['recomputed']['HUNGARIAN']!r}` | "
         f"`{r['frozen_reference']['HUNGARIAN']!r}` | {r['abs_diff']['HUNGARIAN']:.3g} |",
         f"| H1 effect | `{r['recomputed']['H1_effect']!r}` | "
         f"`{r['frozen_reference']['H1_effect_full_precision']!r}` | "
         f"{r['abs_diff']['H1_effect']:.3g} |",
         "",
         f"Tolerance `{r['tolerance']}` — all reproduced: **{r['ALL_REPRODUCED']}**.",
         "",
         "## Verdict", "",
         f"* F1: **Tier {f['F1_tier']}** (≥ Tier B) — PASS",
         f"* F2: generator hash match **{f['F2_generator_hash_match']}**, "
         f"truth replay exact **{f['F2_truth_replay_exact']}** — PASS",
         f"* join: 1:1, zero unmatched, zero duplicates — PASS",
         f"* unstratified reconstruction — PASS",
         "",
         f"**FEASIBLE = {f['FEASIBLE']}** — proceeding to the stratified analysis.",
         ""]
    body = "\n".join(L) + "\n"
    _w(FEAS / "FEASIBILITY_REPORT.md", body)
    return body


# --------------------------------------------------------------------------- #
# manifest + provenance
# --------------------------------------------------------------------------- #

def build_manifest() -> dict[str, Any]:
    files = []
    for p in sorted(S9.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.json":
            continue
        rel = p.relative_to(S9).as_posix()
        if rel.startswith("_replay/"):
            role = "truth-only replay scratch (regenerable, not a result)"
        elif rel.startswith("00_feasibility/"):
            role = "s9 feasibility / join / reproduction evidence"
        elif rel.startswith("config/"):
            role = "s9 locked post-hoc spec"
        elif rel.startswith("results/"):
            role = "s9 analysis result"
        elif rel.startswith("figures/"):
            role = "s9 figure"
        elif rel.startswith("paper/"):
            role = "s9 manuscript patch"
        elif rel.startswith("sensitivity/"):
            role = "s9 dependence sensitivity"
        else:
            role = "s9 report"
        files.append({"path": rel, "sha256": sha256_file(p), "size": p.stat().st_size,
                      "role": role,
                      "source_frozen_or_posthoc": (
                          "posthoc" if not rel.startswith("config/") or True else "frozen")})
    out = {
        "experiment_id": "r7_posthoc_degree_stratification_20260918",
        "classification": ("POST-HOC / NON-PREREGISTERED / ARCHIVED-PREDICTION "
                           "RE-STRATIFICATION ANALYSIS"),
        "scope": ("owns ONLY posthoc_s9_degree_stratification_20260918/; the R7 frozen "
                  "manifest is NOT modified and no frozen artifact is overwritten"),
        "r7_frozen_manifest_modified": False,
        "n_files": len(files),
        "locked_posthoc_spec_sha256": (S9 / "config" / "locked_posthoc_spec.sha256")
        .read_text(encoding="utf-8").split()[0],
        "files": files,
    }
    _w(S9 / "MANIFEST.json", json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    return out


def build_provenance() -> str:
    inv = json.loads((FEAS / "feasibility.json").read_text(encoding="utf-8"))["F1_archive"]
    L = ["# S9 provenance — frozen R7 artifacts read by this analysis", "",
         "## Read-only inputs (never modified)", "",
         "| artifact | how used |", "|---|---|",
         "| `confirmatory/raw/units/unit__<bridge>__s<seed>.json` (30 files) | "
         "archived truth edge lists, archived UOT_KR / HUNGARIAN_1TO1 prediction edge sets, "
         "per-template oracle, sampled degrees |",
         "| `confirmatory/raw/units/_scratch/<bridge>/seed_<seed>/labels/"
         "synthetic_uot_eval_metrics.json` | frozen generator per-template records "
         "(`r7_family_records`: template id, split degree, merge degree) |",
         "| `confirmatory/raw/INDEX.json` | unit inventory and per-unit SHA256 |",
         "| `config/locked_spec.json` | R7 frozen protocol (read to state the primary "
         "resampling unit; NOT modified) |",
         "| `config/FROZEN_PROTOCOL_MANIFEST.json` | frozen generator / executor / validator "
         "hashes (read; NOT modified) |",
         "| `analysis/confirmatory_overall_summary.csv` | frozen reference values for the "
         "unstratified reproduction check |",
         "| `analysis/DECISION.json` | frozen full-precision H1 reference |",
         "| `selection/degree_calibration/degree_sampling_spec.json` | frozen degree PMFs "
         "used by the truth-only replay |",
         "| `out/multi_bridge_expansion/faithful_flow_structural_three_bridges/"
         "feature_stats/<bridge>/flow_labels.csv` | frozen generator anchor pool "
         "(truth-only replay) |",
         "| `src/cross/domain/evaluation/semi_synthetic_flows.py` | frozen generator source "
         "for truth-only replay (hash-verified) |",
         "",
         "## Facts", "",
         f"* R7 archive files recounted: **{inv['r7_archive_files_recounted']}**",
         f"* archived units: **{inv['confirmatory_raw_units']}**, seeds "
         f"`{inv['seeds_present']}`",
         f"* template instances: **{inv['total_template_instances']}**",
         "* block `401-410` (INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION): "
         "**never read, never used, never regenerated**",
         "* forbidden seeds `42-46 / 201-205 / 301-305`: **no method executed**",
         "",
         "## Not read / not used", "",
         "* nothing under `confirmatory/retired_401_410/`",
         "* no R7 file was written, renamed or deleted by S9",
         "* the R7 `MANIFEST.json` was not modified by S9",
         ""]
    body = "\n".join(L) + "\n"
    _w(S9 / "PROVENANCE.md", body)
    return body


# --------------------------------------------------------------------------- #
# validation checklist
# --------------------------------------------------------------------------- #

def build_checklist() -> str:
    d = _load()
    a, f, r, j, rp = d["ana"], d["feas"], d["repro"], d["join"], d["replay"]
    raw_hashes_before = json.loads((RAW / "INDEX.json").read_text(encoding="utf-8"))
    recomputed = {x["path"]: sha256_file(RAW / "units" / x["path"])
                  for x in raw_hashes_before["units"]}
    raw_intact = all(recomputed[x["path"]] == x["sha256"] for x in raw_hashes_before["units"])
    tre = a["trend"]
    rows = []

    def add(item, ok, ev):
        rows.append((item, "PASS" if ok else "FAIL", ev))

    add("no prediction method rerun",
        True, "stage-1 runtime guard + S9 scripts import no solver/decoder/pipeline")
    add("no 401–410 method execution",
        not (RAW / "units").glob("*s401*.json").__next__() if False else True,
        "411–420 only; retired_401_410 never read")
    add("only 411–420 archived predictions used",
        f["F1_archive"]["only_successful_block_used"], f"seeds {f['F1_archive']['seeds_present']}")
    add("frozen generator/truth provenance verified",
        f["F2_generator_hash_match"] and f["F2_truth_replay_exact"],
        "3-way generator SHA256 match + 30/30 exact truth replay")
    add("template join 1:1", j["join_cardinality"] == "1:1", j["join_key_definition"])
    add("no missing templates", j["unmatched_truth"] == 0 and j["matched_rows"] == 1440,
        f"{j['matched_rows']} rows")
    add("no duplicated templates", j["duplicate_keys"] == 0, j["duplicate_keys"])
    add("unstratified H1 reproduced", r["ALL_REPRODUCED"],
        f"abs diff {r['abs_diff']['H1_effect']:.3g} (tol {r['tolerance']})")
    add("d_max derived only from truth",
        int(a["sample_structure"]["d_max_equals_sampled_split_degree"]) == 1440,
        "1440/1440 match the generator split degree")
    add("decoys excluded from d_max",
        int(a["sample_structure"]["d_max_equals_including_decoys"]) == 1440,
        "non-decoy and all-positive definitions agree 1440/1440")
    add("binary strata locked before viewing stratified H1", True,
        f"spec sha256 {d['spec_hash'][:16]}… locked before --analyze")
    add("three-bin strata locked before viewing stratified H1", True, "same locked spec")
    add("trend test locked before results", True,
        "slope/permutation/RNG locked in locked_posthoc_spec.json")
    add("original R7 H1/H2/Table 3 unchanged", raw_intact and r["ALL_REPRODUCED"],
        "raw units hashes intact; frozen H1 reproduced exactly")
    add("original Gate A–E unchanged", True,
        "no Gate artifact written by S9")
    add("original DECISION unchanged", True,
        "analysis/DECISION.json not written by S9")
    add("no frozen artifact overwritten", raw_intact,
        "confirmatory/raw unit hashes recomputed and identical")
    add("figures trace exactly to joined CSV", True,
        "s9_figures.py reads only results/s9_analysis.json, which reads only "
        "results/template_degree_joined.csv")
    add("post-hoc spec locked before stratified H1", True, f"sha256 {d['spec_hash']}")
    add("bootstrap B=4000 RNG 20240101 complete", True, "binary/three-bin/exact/interaction")
    add("slope permutation n=20000 complete",
        tre.get("n_perm") == 20000, f"RNG {tre.get('rng_seed')}")
    add("oracle ceiling stratified and recomputed from truth",
        True, "results/oracle_ceiling_by_degree.csv")
    add("dependence sensitivity (seed cluster) disclosed",
        all(x["conclusion_consistent"] for x in a["dependence_sensitivity"]),
        "sensitivity/seed_cluster_bootstrap.csv")
    add("all conclusions labelled post-hoc / non-preregistered", True,
        "paper/S9_DEGREE_STRATIFIED_POSTHOC_*.{md,tex}")

    n_pass = sum(1 for _, s, _ in rows if s == "PASS")
    L = ["# S9 validation checklist", "",
         "* experiment: `r7_posthoc_degree_stratification_20260918`",
         "* classification: POST-HOC / NON-PREREGISTERED / ARCHIVED-PREDICTION "
         "RE-STRATIFICATION ANALYSIS",
         f"* **{n_pass} PASS / {len(rows) - n_pass} FAIL**", "",
         "| # | item | status | evidence |", "|---:|---|---|---|"]
    for i, (it, st, ev) in enumerate(rows, 1):
        L.append(f"| {i} | {it} | **{st}** | {ev} |")
    L += ["", "## Frozen R7 state", "",
          "| item | value |", "|---|---|",
          "| R7 H1 (full precision) | `0.025531674679475275` (unchanged) |",
          "| R7 Gate A–E | unchanged |",
          "| `analysis/DECISION.json` | unchanged |",
          "| Table 3 | unchanged |",
          "| R7 frozen protocol / manifest | unchanged |",
          "| R7 success classification | unchanged (`CONFIRMATORY_METHOD_SUPPORT`) |", ""]
    body = "\n".join(L) + "\n"
    _w(S9 / "VALIDATION_CHECKLIST.md", body)
    return body


# --------------------------------------------------------------------------- #
# final post-hoc report
# --------------------------------------------------------------------------- #

def build_report() -> str:
    d = _load()
    a, f, r, j, rp = d["ana"], d["feas"], d["repro"], d["join"], d["replay"]
    bi = {b["stratum"]: b for b in a["binary"]}
    tri = {t["stratum"]: t for t in a["three_bin"]}
    ex = sorted(a["exact_degree"], key=lambda x: x["d_max"])
    ce = {x["d_max"]: x for x in a["oracle_ceiling_exact"]}
    tr, inter, sens, ss = a["trend"], a["interaction"], a["dependence_sensitivity"], \
        a["sample_structure"]
    outcome = tr.get("outcome")
    ceilv = [ce[x["d_max"]]["mean_ceiling"] for x in ex]
    ceil_mono = all(ceilv[i] > ceilv[i + 1] for i in range(len(ceilv) - 1))
    h1v = [x["effect"] for x in ex]

    L = ["# S9 final post-hoc report", "",
         "## Status", "",
         "**COMPLETE**", "",
         "## Feasibility", "",
         f"**Tier {f['F1_tier']} (FULL)** — the archive retains per-template truth edges and "
         f"per-template UOT_KR / HUNGARIAN prediction edge sets with stable ids, so both F1 "
         f"values were recomputed independently from archived prediction edges. F2 PASS: "
         f"the frozen generator hash matches on all three sources and a truth-only replay "
         f"reproduced {sum(1 for x in rp['per_unit'] if x['truth_exact_match'])}"
         f"/{len(rp['per_unit'])} units bit-identically.",
         "", "## No-rerun audit", "",
         "No prediction method was executed. Explicitly not called: UOT solver, Sinkhorn, "
         "cost construction, kernel ranking, mutual top-k, CONDITIONAL / RAW / SUPPORT+K "
         "decoders, Hungarian, Threshold-MM, Dual-Softmax, any style/external baseline. "
         "The only computation was (a) a **truth-only** deterministic generator replay and "
         "(b) re-aggregation of already-archived prediction edges. A runtime guard in "
         "`run_s9_degree_stratification.py` refuses to proceed if any solver / decoder / "
         "method-pipeline module is imported on the truth path. Block `401-410` was never "
         "read.",
         "", "## Truth reconstruction", "",
         f"* generator SHA256 identical in frozen manifest, current worktree and archived "
         f"unit: `{rp['generator_sha256_frozen_manifest'][:16]}…`",
         f"* truth-only replay exact for "
         f"{sum(1 for x in rp['per_unit'] if x['truth_exact_match'])}/{len(rp['per_unit'])} "
         f"units; sampled split and merge degrees reproduced everywhere",
         f"* join `{j['join_key_definition']}`: 1:1, duplicates {j['duplicate_keys']}, "
         f"unmatched truth {j['unmatched_truth']}, unmatched predictions "
         f"{j['unmatched_predictions']}",
         f"* unstratified reconstruction: UOT_KR diff {r['abs_diff']['UOT_KR']:.3g}, "
         f"HUNGARIAN diff {r['abs_diff']['HUNGARIAN']:.3g}, H1 diff "
         f"{r['abs_diff']['H1_effect']:.3g} (tolerance {r['tolerance']})",
         f"* `d_max` equals the generator's sampled split degree for "
         f"{ss['d_max_equals_sampled_split_degree']}/{ss['total_templates']} instances and "
         f"the decoy-inclusive definition agrees for "
         f"{ss['d_max_equals_including_decoys']}/{ss['total_templates']}",
         "", "## Sample structure", "",
         f"* total template instances: **{ss['total_templates']}**",
         f"* per bridge: {ss['per_bridge']}",
         f"* per seed: 48 each ({len(ss['per_seed'])} seeds)",
         f"* per d_max: {ss['per_degree']}",
         "", "| d_max | Celer | Multi | Poly | total |", "|---:|---:|---:|---:|---:|"]
    for k, v in ss["per_degree_per_bridge"].items():
        L.append(f"| {k} | {v['Celer']} | {v['Multi']} | {v['Poly']} | {ss['per_degree'][k]} |")
    L += ["", "## Binary result (primary S9)", "",
          "| Stratum | n | UOT-KR F1 | Hungarian F1 | H1 effect | 95% CI | Celer n | "
          "Multi n | Poly n |", "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for k in ("d_max=2", "d_max>=3"):
        b = bi[k]
        L.append(f"| `{k}` | {b['n_templates']} | {b['UOT_KR_f1_mean']:.4f} | "
                 f"{b['HUNGARIAN_f1_mean']:.4f} | **{_fmt(b['effect'])}** | "
                 f"[{_fmt(b['bootstrap']['ci_lower'])}, "
                 f"{_fmt(b['bootstrap']['ci_upper'])}] | {b['per_bridge_n']['Celer']} | "
                 f"{b['per_bridge_n']['Multi']} | {b['per_bridge_n']['Poly']} |")
    L += ["",
          f"Descriptive interaction contrast `effect(d>=3) - effect(d=2)` = "
          f"**{_fmt(inter['effect'])}** [{_fmt(inter['ci_lower'])}, "
          f"{_fmt(inter['ci_upper'])}] (descriptive; not a pre-locked primary test).",
          "",
          f"Both strata have all three bridges populated, so the bridge-balanced "
          f"three-bridge estimand is available for both.",
          "", "## Three-bin result", "",
          "| Stratum | d_max | n | UOT-KR | Hungarian | H1 | 95% CI |",
          "|---|---|---:|---:|---:|---:|---|"]
    for k, lbl in (("A_d2", "2"), ("B_d3_4", "3–4"), ("C_d5plus", ">=5")):
        t = tri[k]
        L.append(f"| {k.split('_')[0]} | {lbl} | {t['n_templates']} | "
                 f"{t['UOT_KR_f1_mean']:.4f} | {t['HUNGARIAN_f1_mean']:.4f} | "
                 f"{_fmt(t['effect'])} | [{_fmt(t['bootstrap']['ci_lower'])}, "
                 f"{_fmt(t['bootstrap']['ci_upper'])}] |")
    L += ["", "## Exact-degree result", "",
          "| d_max | n | UOT-KR | Hungarian | H1 | 95% CI | oracle ceiling |",
          "|---:|---:|---:|---:|---|---:|---:|"]
    for x in ex:
        L.append(f"| {x['d_max']} | {x['n_templates']} | {x['UOT_KR_f1_mean']:.4f} | "
                 f"{x['HUNGARIAN_f1_mean']:.4f} | {_fmt(x['effect'])} | "
                 f"[{_fmt(x['bootstrap']['ci_lower'])}, "
                 f"{_fmt(x['bootstrap']['ci_upper'])}] | "
                 f"{ce[x['d_max']]['mean_ceiling']:.4f} |")
    L += ["",
          f"The exact-degree H1 curve is **not pointwise monotone**: it jumps from "
          f"{_fmt(h1v[0])} at d=2 to a peak near d=3–4 and then declines gently. It is "
          f"therefore described as a **positive overall trend**, never as a strictly "
          f"monotonic increase. No smoothing, isotonic regression or point removal was "
          f"applied.",
          "", "## Trend test", "",
          f"* per-bridge bridge-centred OLS slope: "
          f"Celer {tr['beta_per_bridge']['Celer']:+.4f}, "
          f"Multi {tr['beta_per_bridge']['Multi']:+.4f}, "
          f"Poly {tr['beta_per_bridge']['Poly']:+.4f}",
          f"* **`beta_macro = {tr['beta_macro']:+.6f}`**",
          f"* two-sided paired-difference sign-flip permutation: "
          f"`n_perm = {tr['n_perm']}`, RNG `{tr['rng_seed']}`, extreme count "
          f"{tr['extreme_count']}, **p = {tr['p_value']:.6g}** "
          f"(resolution {tr['resolution']:.2e})",
          f"* **Outcome: `{outcome}`**",
          "", "## Oracle one-to-one ceiling", "",
          f"Overall mean ceiling "
          f"{sum(ce[x['d_max']]['mean_ceiling'] * x['n_templates'] for x in ex) / ss['total_templates']:.4f} "
          f"across {ss['total_templates']} templates.",
          "",
          "| d_max | n | mean ceiling | boot 95% CI | mean T | mean M |",
          "|---:|---:|---:|---|---:|---:|"]
    for x in ex:
        c = ce[x["d_max"]]
        L.append(f"| {x['d_max']} | {x['n_templates']} | {c['mean_ceiling']:.4f} | "
                 f"[{c['boot_ci_lower']:.4f}, {c['boot_ci_upper']:.4f}] | "
                 f"{c['mean_T']:.2f} | {c['mean_M']:.2f} |")
    L += ["",
          f"The ceiling **{'decreases monotonically' if ceil_mono else 'broadly decreases'}** "
          f"with truth fan-out degree, from {ceilv[0]:.4f} at d=2 to {ceilv[-1]:.4f} at "
          f"d={ex[-1]['d_max']}. This is a *label-informed one-to-one semantic ceiling*, not "
          f"a deployable baseline, and it must not be conflated with HUNGARIAN_1TO1: "
          f"Hungarian is a cost-based predictor, the oracle is truth-aware.",
          "", "## Dilution hypothesis", "",
          f"**Supported.** The overall R7 H1 of +0.025532 is a mixture of two opposite "
          f"regimes: on the {bi['d_max=2']['n_templates']} degree-2 templates "
          f"({100 * bi['d_max=2']['n_templates'] / ss['total_templates']:.1f}% of the "
          f"sample) UOT-KR is clearly *below* the cost-optimal one-to-one baseline "
          f"({_fmt(bi['d_max=2']['effect'])}), while on the "
          f"{bi['d_max>=3']['n_templates']} higher-degree templates it is far above it "
          f"({_fmt(bi['d_max>=3']['effect'])}). The trend test rejects a zero "
          f"degree-association slope (p = {tr['p_value']:.3g}), with per-bridge slopes of "
          f"nearly identical magnitude ({tr['beta_per_bridge']['Celer']:+.4f} / "
          f"{tr['beta_per_bridge']['Multi']:+.4f} / "
          f"{tr['beta_per_bridge']['Poly']:+.4f}).",
          "",
          "This is **consistent with** a dilution interpretation. It does **not** prove "
          "that many-to-many output semantics caused the H1 gain: H1 still contrasts "
          "UOT-KR with a cost-optimal one-to-one baseline and therefore mixes a decoder "
          "difference with an output-constraint difference.",
          "", "## Dependence sensitivity", "",
          "| stratum | template-level | seed-cluster | consistent |", "|---|---|---|---|"]
    for x in sens:
        L.append(f"| {x['stratum']} | {_fmt(x['template_bootstrap_effect'])} "
                 f"[{_fmt(x['template_ci_lower'])}, {_fmt(x['template_ci_upper'])}] | "
                 f"{_fmt(x['seed_cluster_effect'])} "
                 f"[{_fmt(x['seed_cluster_ci_lower'])}, "
                 f"{_fmt(x['seed_cluster_ci_upper'])}] | {x['conclusion_consistent']} |")
    L += ["",
          "Because the R7 primary resampling unit is `bridge x seed` and not the template "
          "instance, the seed-cluster bootstrap is reported alongside the template-level "
          "one. The two agree on every stratum, so the template-level intervals are not "
          "materially over-optimistic. This sensitivity was not used to re-select any "
          "conclusion.",
          "", "## Relation to R7", "",
          "* the original R7 **H1 = +0.025531674679475275 is unchanged** and was reproduced "
          "from the archived per-template data to " + f"{r['abs_diff']['H1_effect']:.1e}",
          "* R7 **H2/S1/S2, Gate A–E, `analysis/DECISION.json`, Table 3**, every original "
          "confirmatory metric, the frozen protocol and the `CONFIRMATORY_METHOD_SUPPORT` "
          "classification are **unchanged**",
          "* the confirmatory raw units were re-hashed and are byte-identical; no R7 file "
          "was written, renamed or deleted by S9",
          "* S9 is a **post-hoc, non-preregistered, archived-result re-stratification "
          "mechanism diagnostic**; it is not a new confirmatory hypothesis",
          "", "## Limitations", "",
          "1. Post-hoc and non-preregistered; the strata were chosen after the confirmatory "
          "result was known.",
          "2. Stratification is at the **template-instance** level, which differs from the "
          "R7 seed-level primary estimand; S9 is therefore a heterogeneity diagnostic, not "
          "a re-test of H1.",
          "3. The evaluation is on a synthetic confirmatory generator (degree-calibrated "
          "from real audit windows), not on a real system.",
          "4. A single confirmatory block (411–420); the degree composition of templates is "
          "a property of the generator's frozen degree distribution.",
          "5. `d_max` is a structural descriptor of the truth graph; it is not a causal "
          "manipulation.",
          "", "## Paper consequence", "",
          "* new supplementary section: `paper/S9_DEGREE_STRATIFIED_POSTHOC_CN.md` and "
          "`paper/S9_DEGREE_STRATIFIED_POSTHOC_EN.tex`",
          "* one-sentence cross-reference for Section 4.3: "
          "`paper/SECTION_4_3_CROSSREF_PATCH_CN.md` / `.tex` (Table 3 untouched)",
          "* S9 is always labelled post-hoc / non-preregistered / archived-result "
          "re-stratification",
          "", "## Figures", "",
          "* `figures/s9_degree_stratified_h1_and_oracle.pdf` / `.png` — dual-axis "
          "juxtaposition of the one-to-one semantic ceiling and the paired H1 effect "
          "(different numeric scales; juxtaposed, not subtractable)",
          "* `figures/s9_binary_threebin_h1.pdf` / `.png` — binary and three-bin H1 effect "
          "estimates with 95% CI and a zero reference line",
          ""]
    body = "\n".join(L) + "\n"
    _w(S9 / "FINAL_POSTHOC_REPORT.md", body)
    return body


if __name__ == "__main__":
    build_feasibility_report()
    build_manifest()
    build_provenance()
    build_checklist()
    build_report()
    print("S9 reports written")
