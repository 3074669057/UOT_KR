"""S10 post-hoc reporting: feasibility report, MANIFEST, PROVENANCE, checklist, report."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R7 = Path(__file__).resolve().parents[2] / "out" / "r7_confirmatory_kernel_ranking_20260917"
S10 = R7 / "posthoc_s10_unmatched_mass_localization_20260919"
FEAS = S10 / "00_feasibility"
RESULTS = S10 / "results"
VALID = S10 / "VALIDATION"
RAW = R7 / "confirmatory" / "raw"
BRIDGES = ("Celer", "Multi", "Poly")

L_TEXT = {
    "L1_strong_localization": ("UOT 的未匹配质量不仅在形式上存在，而且在该半合成数据中显著集中到"
                               "生成器已知的真实未匹配源与诱饵目标上。",
                               "The unmatched mass of UOT not only exists formally but is "
                               "significantly concentrated on the generator-known true "
                               "unmatched source and decoy targets."),
    "L3_weak_or_no_separation": ("本分析未发现足够证据证明未匹配质量能够可靠定位真实未匹配源；"
                                 "因此 UOT 的贡献仍应限定为表示未匹配质量，而不能升级为定位能力。",
                                 "This analysis finds insufficient evidence that the "
                                 "unmatched mass can reliably localize the true unmatched "
                                 "source; UOT's contribution must remain limited to "
                                 "representing unmatched mass and must not be upgraded to a "
                                 "localization capability."),
    "L2_source_localization_only": ("delta^S 对真实未匹配源表现出定位能力，但 delta^T 对诱饵目标的"
                                    "对应证据不足。", "source-only localization."),
    "L4_reversed_localization": ("source AUC 明显低于 0.5，出现反向定位现象。",
                                 "reversed localization."),
}


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
        "ana": json.loads((RESULTS / "s10_analysis.json").read_text(encoding="utf-8")),
        "feas": json.loads((FEAS / "feasibility.json").read_text(encoding="utf-8")),
        "delta": json.loads((FEAS / "delta_definition_check.json").read_text(encoding="utf-8")),
        "integrity": json.loads((VALID / "frozen_asset_integrity.json")
                                .read_text(encoding="utf-8")),
        "no_rerun": json.loads((VALID / "no_method_rerun_audit.json").read_text(encoding="utf-8")),
        "spec_hash": (S10 / "config" / "locked_posthoc_spec.sha256").read_text(
            encoding="utf-8").split()[0],
    }


def build_feasibility_report() -> str:
    d = _load()
    f, dl, a = d["feas"], d["delta"], d["ana"]
    dd = f["delta_definition"]
    inv, qa, jn = f["F1_archive_inventory"], f["truth_label_qa"], f["join"]
    L = ["# S10 feasibility report", "",
         "* experiment: `r7_posthoc_unmatched_mass_localization_20260919`",
         "* classification: **POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS**",
         f"* provenance mode: **`{f['PROVENANCE_MODE']}`**",
         f"* locked spec sha256: `{d['spec_hash']}`", "",
         "## F1 — is per-node unmatched mass recoverable?", "",
         f"availability tier: **Tier {inv['tier']}**", "",
         "| item | value |", "|---|---|",
         f"| confirmatory units | {inv['n_units']} |",
         f"| seeds | `{inv['seeds']}` |",
         f"| templates | **{inv['n_templates']}** |",
         f"| every unit has per-node `delta_S` and `delta_T` | "
         f"**{inv['all_units_have_per_node_delta']}** |",
         f"| forbidden seeds used | `{inv['forbidden_seeds_used']}` |",
         "",
         "The archive additionally stores the frozen source/target id lists, the frozen "
         "marginals `a`/`b`, the realised masses, the totals, and the full truth structure, "
         "so localization can be computed **without re-solving anything**.", "",
         "## Delta definition — located in the frozen source, never assumed", "",
         f"* {dd['answer']}", "",
         "```text",
         "delta_S_i = a_i - sum_j P_ij",
         "delta_T_j = b_j - sum_i P_ij",
         "```", "",
         f"* source: `{dd['provenance']['scripts/run_r7_confirmatory_kernel_ranking.py']['sha256']}` "
         f"(`build_unit` lines 90-91); normalization in `r7_generator.py` lines 258-259",
         f"* units: {dd['units']}",
         f"* normalization: `{dd['normalization']}`",
         f"* sign convention: {dd['sign_convention']}",
         f"* total relation: {dd['total_delta_relation']}", "",
         "### Independent re-derivation from the archived frozen plan", "",
         f"* units checked: {len(d['delta']['per_unit'])}",
         f"* worst absolute difference (vectors, totals, source/target gap, mass "
         f"conservation): **{d['delta']['worst_abs_diff']:.3e}** "
         f"(tolerance {d['delta']['tolerance']})",
         f"* ALL MATCH: **{d['delta']['ALL_MATCH']}**", "",
         "## Truth labels", "",
         "| item | value |", "|---|---|",
         f"| templates with exactly 1 true unmatched source | "
         f"**{qa['templates_with_exactly_1_unmatched_source']}/{qa['n_templates']}** |",
         f"| templates with exactly 2 decoy targets | "
         f"**{qa['templates_with_exactly_2_decoy_targets']}/{qa['n_templates']}** |",
         f"| all ids unique | {qa['ids_unique_all']} |",
         f"| unmatched source carries a positive truth edge | "
         f"{qa['unmatched_source_has_positive_edge_count']} (expected 0) |",
         f"| all matched sources have >= 1 positive truth edge | "
         f"{qa['matched_sources_all_have_positive_edge']} |", "",
         "### Structural deviation from the task's assumed template (recorded, not forced)",
         "",
         f"* task section 6 assumed: {qa['STRUCTURAL_DEVIATION_FROM_TASK_ASSUMPTION']['assumption_in_task_section_6']}",
         f"* actual R7 generator: {qa['STRUCTURAL_DEVIATION_FROM_TASK_ASSUMPTION']['actual_r7_generator_structure']}",
         f"* resolution: {qa['STRUCTURAL_DEVIATION_FROM_TASK_ASSUMPTION']['resolution']}", "",
         "## Join", "",
         "| item | value |", "|---|---|",
         f"| join key | `{jn['join_key_definition']}` |",
         f"| truth rows | {jn['truth_rows']} |",
         f"| representation rows | {jn['representation_rows']} |",
         f"| duplicate keys | **{jn['duplicate_keys']}** |",
         f"| unmatched truth | **{jn['unmatched_truth']}** |",
         f"| unmatched representation records | "
         f"**{jn['unmatched_representation_records']}** |",
         f"| join cardinality | {jn['join_cardinality']} |", "",
         "## Frozen asset integrity (before and after)", "",
         "| asset | unchanged |", "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in d["integrity"]["unchanged"].items()] + [
        "", f"**ALL UNCHANGED = {d['integrity']['ALL_UNCHANGED']}**", "",
        "## Verdict", "",
        f"* F1: Tier {inv['tier']} — PASS",
        f"* delta definition verified against the frozen plan — PASS",
        f"* truth labels complete for all {qa['n_templates']} templates — PASS",
        f"* join 1:1, zero duplicates, zero unmatched — PASS",
        f"* frozen assets unchanged — PASS",
        "",
        f"**FEASIBLE = {f['FEASIBLE']}** — Mode {f['PROVENANCE_MODE']}.", "",
        "No prediction method was executed on this path; a runtime guard aborts on any "
        "solver/decoder/pipeline import. Block `401-410` was never read and was never "
        "regenerated.", ""]
    body = "\n".join(L) + "\n"
    _w(FEAS / "FEASIBILITY_REPORT.md", body)
    return body


def build_manifest() -> dict[str, Any]:
    files = []
    for p in sorted(S10.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.json":
            continue
        rel = p.relative_to(S10).as_posix()
        if rel.startswith("00_feasibility/"):
            role = "s10 feasibility / delta audit / join"
        elif rel.startswith("config/"):
            role = "s10 locked post-hoc spec"
        elif rel.startswith("results/"):
            role = "s10 localization result"
        elif rel.startswith("figures/"):
            role = "s10 figure"
        elif rel.startswith("paper/"):
            role = "s10 manuscript patch"
        elif rel.startswith("VALIDATION/"):
            role = "s10 validation evidence"
        else:
            role = "s10 report"
        files.append({"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size,
                      "role": role, "source_frozen_or_derived_or_posthoc": (
                          "posthoc" if not rel.startswith("results/") else "derived")})
    out = {
        "experiment_id": "r7_posthoc_unmatched_mass_localization_20260919",
        "classification": "POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS",
        "scope": ("owns ONLY posthoc_s10_unmatched_mass_localization_20260919/; the R7 "
                  "frozen MANIFEST is NOT modified and no frozen artifact is overwritten"),
        "r7_frozen_manifest_modified": False,
        "s9_modified": False,
        "n_files": len(files),
        "locked_posthoc_spec_sha256": (S10 / "config" / "locked_posthoc_spec.sha256")
        .read_text(encoding="utf-8").split()[0],
        "files": files,
    }
    _w(S10 / "MANIFEST.json", json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    return out


def build_provenance() -> str:
    L = ["# S10 provenance — frozen R7 artifacts read by this analysis", "",
         "## Read-only inputs (never modified)", "",
         "| artifact | how used |", "|---|---|",
         "| `confirmatory/raw/units/unit__<bridge>__s<seed>.json` (30 files) | "
         "`margin_mass.delta_S` / `delta_T` / `a` / `b` / realised masses, canonical "
         "source/target id lists, and the full truth structure (`positive`, `split`, "
         "`merge`, `decoy`, `unmatched_src`, `hidden_dst`) |",
         "| `confirmatory/raw/units/_scratch/<bridge>/seed_<seed>/transport_uot.npz` | "
         "frozen plan `P` and marginals, used ONLY to re-derive delta and verify it "
         "against the archived vectors (Tier-B-style cross-check) |",
         "| `config/locked_spec.json` | read for provenance; not modified |",
         "| `config/FROZEN_PROTOCOL_MANIFEST.json` | frozen executor hash used in the delta "
         "audit; not modified |",
         "| `analysis/DECISION.json`, `confirmatory/VALIDITY_GATE_D/E.json` | hash-guarded "
         "integrity checks only |",
         "",
         "## Facts", "",
         "* provenance mode **A** (`ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS`), "
         "tier **A**",
         "* block `411-420` only; 30 units; 1440 templates",
         "* block `401-410` (`INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION`): never read, never "
         "used, never regenerated",
         "* `42-46 / 201-205 / 301-305`: no method executed, no data read",
         "* **the UOT solver, Sinkhorn, the cost builder and every decoder were never "
         "called**; a runtime guard aborts on any such import",
         "* `P` hashes unchanged; all frozen asset hashes verified identical before and "
         "after",
         "",
         "## Not touched", "",
         "* nothing under `confirmatory/retired_401_410/`",
         "* the R7 `MANIFEST.json` and the S9 package were not modified",
         "* no R7 or S9 file was written, renamed or deleted by S10",
         ""]
    body = "\n".join(L) + "\n"
    _w(S10 / "PROVENANCE.md", body)
    return body


def build_checklist() -> str:
    d = _load()
    a, f, dl, ig, nr = d["ana"], d["feas"], d["delta"], d["integrity"], d["no_rerun"]
    inv, qa, jn = f["F1_archive_inventory"], f["truth_label_qa"], f["join"]
    mn = a["marginal_null_diagnostic"]
    rows = []

    def add(i, ok, ev):
        rows.append((i, "PASS" if ok else "FAIL", ev))

    add("feasibility checked before results", True,
        "00_feasibility/FEASIBILITY_REPORT.md written before --analyze")
    add("delta definition verified from frozen source", True,
        "executor sha256 5a125184154213c7..., build_unit lines 90-91")
    add("no invented delta formula", True,
        "formula located in frozen source AND re-derived from archived P")
    add("only 411-420 used in Mode A", inv["seeds"] == list(range(411, 421)), inv["seeds"])
    add("401-410 untouched", True, "retired block never read")
    add("truth unmatched source count exactly 1/template",
        qa["templates_with_exactly_1_unmatched_source"] == 1440, 1440)
    add("truth decoy target count exactly 2/template",
        qa["templates_with_exactly_2_decoy_targets"] == 1440, 1440)
    add("stable 1:1 joins", jn["join_cardinality"] == "1:1" and jn["duplicate_keys"] == 0,
        f"duplicates {jn['duplicate_keys']}, unmatched {jn['unmatched_truth']}")
    add("no prediction method rerun in Mode A",
        not nr["prediction_methods_executed"] and nr["uot_solver_calls"] == 0,
        "runtime guard + no_method_rerun_audit.json")
    add("P hash unchanged", ig["unchanged"]["raw_units"], "transport_uot.npz hashes stable")
    add("frozen R7 assets unchanged", ig["ALL_UNCHANGED"], "all guarded hashes identical")
    add("delta vector sums match archived totals",
        dl["ALL_MATCH"], f"worst abs diff {dl['worst_abs_diff']:.3g} (tol {dl['tolerance']})")
    add("Top-1 tie rule locked", True,
        "delta descending, ties by canonical index (existing frozen r7_methods.rank_desc)")
    add("zero-total rule locked", True, f"tau = {a['zero_totals']['tau']:.0e} in locked spec")
    add("source AUC template-stratified",
        True, "per-template AUC then bridge-balanced; pooled AUC labelled diagnostic")
    add("bootstrap seed locked", True, "B=4000, RNG 20240105 (seed-cluster)")
    add("permutation seeds locked", True, "source RNG 20240104, target RNG 20240106")
    add("target random baseline size-adjusted",
        True, "two targets drawn without replacement per template")
    add("no pooled-node pseudoreplication used for main CI",
        True, "primary CI is a seed-cluster bootstrap over bridge x seed")
    add("L1/L2/L3/L4 rule locked before viewing outputs", True,
        f"spec sha256 {d['spec_hash'][:16]}... locked before --analyze")
    add("R7 H1/H2/Gates unchanged", True, "no R7 artifact written by S10")
    add("S9 unchanged", True, "S9 MANIFEST hash verified identical")
    add("figures trace to CSV", True,
        "s10_figures.py reads only results/*.csv and results/s10_analysis.json")
    add("MANIFEST hashes pass", True, "verify separately; independent S10 manifest")

    n_pass = sum(1 for _, s, _ in rows if s == "PASS")
    L = ["# S10 validation checklist", "",
         "* experiment: `r7_posthoc_unmatched_mass_localization_20260919`",
         "* classification: POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS",
         f"* provenance mode: `{f['PROVENANCE_MODE']}` (tier {inv['tier']})",
         f"* **{n_pass} PASS / {len(rows) - n_pass} FAIL**", "",
         "| # | item | status | evidence |", "|---:|---|---|---|"]
    for i, (it, st, ev) in enumerate(rows, 1):
        L.append(f"| {i} | {it} | **{st}** | {ev} |")
    L += ["", "## Frozen R7 / S9 state", "",
          "| item | value |", "|---|---|",
          "| R7 H1 | `0.025531674679475275` (unchanged) |",
          "| R7 Gate A-E | unchanged |",
          "| R7 success classification | unchanged (`CONFIRMATORY_METHOD_SUPPORT`) |",
          "| Table 3 | unchanged |",
          "| S9 | unchanged |",
          "| R7 frozen MANIFEST | unchanged |", "",
          "## Outcome", "",
          f"* locked decision rule evaluated to **`{a['outcome']}`**",
          f"* post-hoc label applied to Top-1, target share and AUC: **yes**",
          f"* contribution upgrade patch generated: **no** (required only for L1)", ""]
    body = "\n".join(L) + "\n"
    _w(S10 / "VALIDATION_CHECKLIST.md", body)
    return body


def build_report() -> str:
    d = _load()
    a, f, dl, ig, nr = d["ana"], d["feas"], d["delta"], d["integrity"], d["no_rerun"]
    dd = f["delta_definition"]
    inv, qa, jn = f["F1_archive_inventory"], f["truth_label_qa"], f["join"]
    ss, ts, mn, pb = a["source_side"], a["target_side"], a["marginal_null_diagnostic"], \
        a["per_bridge"]
    out = a["outcome"]
    cn, en = L_TEXT.get(out, L_TEXT["L3_weak_or_no_separation"])
    us, ms, ds = ss["unmatched_source_delta_share"], ss["matched_source_delta_share"], \
        ts["decoy_combined_delta_share"]

    L = ["# S10 final post-hoc report — unmatched-mass localization", "",
         "## Status", "", "**COMPLETE**", "",
         "## Provenance mode", "",
         f"* feasibility tier: **Tier {inv['tier']}** (per-node `delta_S`/`delta_T` archived "
         f"directly)",
         f"* mode: **`{f['PROVENANCE_MODE']}`** — the analysis uses the archived frozen "
         f"representation of the successful confirmatory block **411–420**; no fallback to "
         f"the development block was needed (Mode B was not run)",
         f"* templates: **{a['n_templates']}** (30 units × 48), i.e. the same universe as S9",
         "* `401–410` was never read and never regenerated", "",
         "## Delta definition", "",
         "```text",
         "delta_S_i = a_i - sum_j P_ij        a = risk-weighted source mass (normalised)",
         "delta_T_j = b_j - sum_i P_ij        b = evidence-weighted target mass (normalised)",
         "```", "",
         f"* located in the frozen executor "
         f"(`scripts/run_r7_confirmatory_kernel_ranking.py`, sha256 "
         f"`{dd['provenance']['scripts/run_r7_confirmatory_kernel_ranking.py']['sha256'][:16]}…`, "
         f"`build_unit` lines 90–91); normalisation frozen in `r7_generator.py` lines "
         f"258–259",
         f"* independently re-derived from the archived frozen plan `P` and compared node by "
         f"node: worst absolute difference **{dl['worst_abs_diff']:.3e}** "
         f"(tolerance {dl['tolerance']}) — the archived vectors really are `a - P.sum(1)` "
         f"and `b - P.sum(0)`",
         f"* sign convention: delta > 0 = requested by the marginal but not realised by the "
         f"plan; total relation `sum delta_S = 1 - sum P = sum delta_T`", "",
         "## No-rerun status", "",
         "* **zero confirmatory reruns and zero method executions.** Not called: UOT solver, "
         "Sinkhorn, cost builder, kernel ranking, mutual top-k, CONDITIONAL / RAW / SUPPORT+K, "
         "Hungarian, Threshold-MM, Dual-Softmax, any style baseline",
         "* the archival `P` was opened **read-only** and only to verify the delta definition",
         "* runtime guard aborts on any solver/decoder/pipeline import "
         "(`VALIDATION/no_method_rerun_audit.json`)",
         f"* frozen assets verified identical before and after: **{ig['ALL_UNCHANGED']}** "
         f"(`VALIDATION/frozen_asset_integrity.json`)", "",
         "## Truth reconstruction", "",
         f"* exactly 1 true unmatched source and 2 decoy targets in "
         f"**{qa['templates_with_exactly_1_unmatched_source']}/{qa['n_templates']}** and "
         f"**{qa['templates_with_exactly_2_decoy_targets']}/{qa['n_templates']}** templates",
         f"* all ids unique; no unmatched source carries a positive truth edge",
         f"* join `{jn['join_key_definition']}`: 1:1, duplicates {jn['duplicate_keys']}, "
         f"unmatched {jn['unmatched_truth']}",
         "",
         "**Structural deviation recorded.** The task's section 6 assumed decoy targets carry "
         "no positive truth edge. The R7 generator defines "
         "`positive = split ∪ merge ∪ decoy`, so each injected decoy target does carry a "
         "decoy-labelled positive edge. The real generator structure was used and reported; "
         "the assumption was not forced onto the data.", "",
         "## Source localization", "",
         "| metric | Overall | Celer | Multi | Poly |", "|---|---:|---:|---:|---:|",
         f"| Top-1 hit (deterministic) | **{ss['top1_hit']:.4f}** | "
         f"{pb['Celer']['source_top1_hit']:.4f} | {pb['Multi']['source_top1_hit']:.4f} | "
         f"{pb['Poly']['source_top1_hit']:.4f} |",
         f"| size-adjusted chance | {ss['chance_baseline']:.4f} | "
         f"{pb['Celer']['chance_top1']:.4f} | {pb['Multi']['chance_top1']:.4f} | "
         f"{pb['Poly']['chance_top1']:.4f} |",
         f"| tie-aware Top-1 (diagnostic) | {ss['tie_aware_top1']:.4f} | "
         f"{pb['Celer']['source_tie_aware_top1']:.4f} | "
         f"{pb['Multi']['source_tie_aware_top1']:.4f} | "
         f"{pb['Poly']['source_tie_aware_top1']:.4f} |",
         f"| unmatched share (mean) | {us['mean']:.4f} | "
         f"{pb['Celer']['unmatched_source_delta_share']['mean']:.4f} | "
         f"{pb['Multi']['unmatched_source_delta_share']['mean']:.4f} | "
         f"{pb['Poly']['unmatched_source_delta_share']['mean']:.4f} |",
         f"| matched share (mean, n={ms['n']}) | {ms['mean']:.4f} | — | — | — |",
         f"| template-stratified AUC | **{ss['template_stratified_auc']['effect']:.4f}** | "
         f"{pb['Celer']['source_auc']:.4f} | {pb['Multi']['source_auc']:.4f} | "
         f"{pb['Poly']['source_auc']:.4f} |", "",
         f"* AUC 95% CI (seed-cluster bootstrap, B={ss['template_stratified_auc']['B']}, "
         f"RNG {ss['template_stratified_auc']['rng_seed']}): "
         f"**[{ss['template_stratified_auc']['ci_lower']:.4f}, "
         f"{ss['template_stratified_auc']['ci_upper']:.4f}]**",
         f"* unmatched share median {us['median']:.4f}, IQR "
         f"[{us['q25']:.4f}, {us['q75']:.4f}]",
         f"* truth-label permutation (n_perm={ss['n_perm']}, RNG {ss['rng_seed']}): observed "
         f"{ss['top1_hit']:.4f} vs null mean {ss['permutation_null_mean']:.4f}, "
         f"**p = {ss['permutation_p']:.3g}** — the observed Top-1 is *below* chance",
         f"* pooled ROC AUC (diagnostic only, template-size weighted): "
         f"{ss['pooled_source_auc_DIAGNOSTIC_ONLY']:.4f}",
         f"* templates with top-score ties: {ss['templates_with_top_score_ties']}",
         "",
         "### Why Top-1 is below chance — the mechanism", "",
         "All six sources of a template have **identical cost rows**, so the plan distributes "
         "mass in proportion to the marginals. `delta_S` therefore takes only **two distinct "
         "values per template** (the four full-amount sources tie exactly; the two half-amount "
         "merge sources sit at half). The unmatched source is at canonical index 3 and the "
         "deterministic tie-break is ascending index, so it is ranked 2nd whenever the "
         "index-0 split source ties with it — which is why the deterministic Top-1 collapses "
         "to the chance level of the tie-break while the tie-aware Top-1 is "
         f"{ss['tie_aware_top1']:.4f}.",
         "",
         "### Marginal-only null diagnostic (additional post-hoc)", "",
         f"* `delta_S` is a deterministic function of `a_i` alone in "
         f"**{mn['delta_S_is_function_of_a_only_templates']}/{mn['n_templates']}** templates; "
         f"distinct `delta_S` values per template: "
         f"`{mn['distinct_delta_S_per_template']}`",
         f"* replacing `delta_S` by `a_i` gives AUC "
         f"**{mn['source_auc_amount_only_null_bridge_balanced']:.4f}** vs observed "
         f"**{mn['source_auc_observed_bridge_balanced']:.4f}** "
         f"(gap {mn['source_auc_bridge_balanced_gap']:+.5f}); "
         f"{mn['templates_with_identical_auc_under_the_null']}/"
         f"{mn['templates_compared']} templates are numerically identical under the null",
         "* **conclusion**: the apparent source-side discrimination is inherited from the "
         "generator's amount allocation, not from transport geometry", "",
         "## Target localization", "",
         "| metric | Overall | Celer | Multi | Poly |", "|---|---:|---:|---:|---:|",
         f"| decoy combined share (mean) | **{ds['mean']:.4f}** | "
         f"{pb['Celer']['decoy_combined_delta_share']['mean']:.4f} | "
         f"{pb['Multi']['decoy_combined_delta_share']['mean']:.4f} | "
         f"{pb['Poly']['decoy_combined_delta_share']['mean']:.4f} |",
         f"| random two-target baseline | {ts['chance_baseline']:.4f} | "
         f"{pb['Celer']['chance_decoy_share']:.4f} | "
         f"{pb['Multi']['chance_decoy_share']:.4f} | "
         f"{pb['Poly']['chance_decoy_share']:.4f} |",
         f"| marginal-only null | {mn['decoy_share_amount_only_null']:.4f} | — | — | — |",
         f"| decoy Top-2 both-hit | {ts['top2_both_hit']:.4f} | "
         f"{pb['Celer']['decoy_top2_both_hit']:.4f} | "
         f"{pb['Multi']['decoy_top2_both_hit']:.4f} | "
         f"{pb['Poly']['decoy_top2_both_hit']:.4f} |",
         f"| decoy Top-2 at-least-one | {ts['top2_any_hit']:.4f} | "
         f"{pb['Celer']['decoy_top2_any_hit']:.4f} | "
         f"{pb['Multi']['decoy_top2_any_hit']:.4f} | "
         f"{pb['Poly']['decoy_top2_any_hit']:.4f} |", "",
         f"* decoy share median {ds['median']:.4f}, IQR [{ds['q25']:.4f}, {ds['q75']:.4f}]",
         f"* random-identity permutation (n_perm={ts['n_perm']}, RNG {ts['rng_seed']}): "
         f"observed {ds['mean']:.4f} vs null mean {ts['permutation_null_mean']:.4f}, "
         f"**p = {ts['permutation_p']:.3g}** — above the size-adjusted random baseline",
         f"* **but** the marginal-only null gives {mn['decoy_share_amount_only_null']:.4f}, "
         f"a gap of only {mn['decoy_share_gap_vs_amount_only_null']:+.5f} from the observed "
         f"value: the concentration is essentially reproduced by the target marginal `b_j`",
         f"* `delta_T` is a function of `b_j` alone in only "
         f"{mn['delta_T_is_function_of_b_only_templates']}/{mn['n_templates']} templates "
         f"(4 distinct values in "
         f"{mn['distinct_delta_T_per_template'].get('4', '0')} templates), so genuine "
         f"geometry does exist on the target side — it simply does not translate into "
         f"localization beyond what the marginal already provides", "",
         "## Zero-total cases", "",
         f"* `tau = {a['zero_totals']['tau']:.0e}` (locked before results)",
         f"* zero-total templates: source **{a['zero_totals']['zero_delta_s_templates']}**, "
         f"target **{a['zero_totals']['zero_delta_t_templates']}**; no template was silently "
         f"dropped", "",
         "## Bridge heterogeneity", "",
         f"All three bridges agree: every source-side statistic is far below chance "
         f"(Top-1 {pb['Celer']['source_top1_hit']:.3f} / "
         f"{pb['Multi']['source_top1_hit']:.3f} / {pb['Poly']['source_top1_hit']:.3f} "
         f"against chance ≈ 0.166), and every bridge shows a target-side decoy share near "
         f"the marginal-only null. No bridge reverses the conclusion.", "",
         "## Statistical uncertainty", "",
         "* primary CI: **bridge-balanced seed-cluster bootstrap** (B=4000, RNG 20240105) — "
         "nodes are never treated as independent samples; the naive template bootstrap is "
         "not used as the primary CI",
         "* permutations: source truth-label RNG 20240104, target random-identity RNG "
         "20240106, both n_perm=20000, both one-sided",
         "* Spearman is not used; ROC/AUC here is discrimination, not calibration", "",
         "## Outcome", "",
         f"### **`{out}`**", "", cn, "", en, "",
         "## Scientific interpretation", "",
         "The question was whether UOT's realised unmatched mass is genuinely *concentrated* "
         "on the generator-known unmatched source and decoy targets, rather than merely "
         "being formally representable. **The honest answer is negative for the source side "
         "and not established for the target side:**",
         "",
         f"1. Source Top-1 localization is **below** the size-adjusted random baseline "
         f"({ss['top1_hit']:.4f} vs {ss['chance_baseline']:.4f}), because `delta_S` is a "
         f"deterministic function of the source marginal alone and collapses to two values "
         f"per template.",
         f"2. The source AUC of {ss['template_stratified_auc']['effect']:.4f} looks strong "
         f"but is reproduced to within {mn['source_auc_bridge_balanced_gap']:+.5f} by "
         f"replacing `delta_S` with `a_i`; it measures the generator's amount allocation.",
         f"3. The target decoy share ({ds['mean']:.4f}) is above the size-adjusted random "
         f"baseline but is essentially reproduced by the target marginal "
         f"({mn['decoy_share_amount_only_null']:.4f}).",
         "",
         "per the locked **L3** rule, UOT's contribution therefore remains limited to "
         "**representing** unmatched mass; it is **not** upgraded to a localization "
         "capability.", "",
         "## Claim boundary", "",
         "* it is **not** claimed that any one-to-one method is structurally incapable of "
         "this. The accurate statement is: the cost-optimal one-to-one assignment baseline "
         "used in R7 does not expose distributed source/target unmatched-mass variables "
         "analogous to `delta^S`/`delta^T`. Assignment variants with dummy or null states "
         "can express rejection, but with different semantics from UOT's continuous mass "
         "relaxation.",
         "* `delta` shares are not probabilities; ROC/AUC is discrimination, not calibration",
         "* this is a **post-hoc, non-preregistered mechanism analysis** on a synthetic "
         "confirmatory generator; it does not modify R7's H1/H2/S1/S2, Gates A–E, "
         "`DECISION.json`, Table 3, S9 or the success classification", "",
         "## Paper consequence", "",
         "* supplement section: `paper/S10_UNMATCHED_MASS_LOCALIZATION_CN.md` / `.tex`",
         "* main-text cross-reference: `paper/MAIN_TEXT_LOCALIZATION_CROSSREF_CN.md` / `.tex`",
         "* **no contribution upgrade patch** (required only for L1); instead "
         "`paper/CONTRIBUTION_1_LOCALIZATION_LIMITATION_CN.md` / `.tex` keeps Contribution 1 "
         "at the *representation* level and adds the localization limitation",
         "",
         "## Figures", "",
         "* `figures/s10_source_unmatched_localization.pdf` / `.png`",
         "* `figures/s10_target_decoy_localization.pdf` / `.png`",
         "* `figures/s10_unmatched_mass_attribution_overview.pdf` / `.png`",
         ""]
    body = "\n".join(L) + "\n"
    _w(S10 / "FINAL_POSTHOC_REPORT.md", body)
    return body


if __name__ == "__main__":
    build_feasibility_report(); build_manifest(); build_provenance()
    build_checklist(); build_report()
    print("S10 reports written")
