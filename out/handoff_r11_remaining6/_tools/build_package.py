"""R11 remaining-evidence handoff builder (copy-only, byte-for-byte) — single pass.

Guarantees:
  * every source file is only READ (shutil.copy2); nothing in the repo outside
    out/handoff_r11_remaining6/ is written, moved or deleted
  * no manuscript / docx / tex / bib / frozen artifact is modified
  * no experiment executed, no number recomputed, no dependency installed,
    no network access
  * secrets are never copied (filter below) and every skip is logged
  * destination is rebuilt from scratch, so the layout is a pure function of
    this script
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(r"D:\trae\tool\a\cross")
PKG = REPO / "out" / "handoff_r11_remaining6"
TOOLS = PKG / "_tools"
CORE = PKG / "core"
E1 = PKG / "e1"

SECRET_RE = re.compile(
    r"(^|[\\/])(\.env[^\\/]*|local\.json|local\.[^\\/]*\.json|secrets?|credentials?|"
    r"cookies?|id_rsa[^\\/]*|.*\.pem|.*\.key|.*\.p12|keystore[^\\/]*)$",
    re.IGNORECASE,
)
GROUP_DIR = {
    "manuscript_authority": CORE / "manuscript_authority",
    "pending1_cost_weights": CORE / "pending1_cost_weights",
    "pending2_coverage_tiers": CORE / "pending2_coverage_tiers",
    "pending3_baselines": CORE / "pending3_baselines",
    "pending4_generator": CORE / "pending4_generator",
    "pending5_holdout_d4": CORE / "pending5_holdout_d4",
    "pending5_repository_link": CORE / "repository_link_status",
    "pending6_references": CORE / "pending6_references",
}

copied: list[dict] = []
skipped: list[dict] = []
_taken: dict[str, set[str]] = defaultdict(set)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flat_name(repo_path: str, d: Path) -> str:
    p = Path(repo_path.replace("/", "\\"))
    base = p.name
    key = str(d)
    if base.lower() not in _taken[key]:
        _taken[key].add(base.lower())
        return base
    parts = [x for x in p.parts[:-1] if x not in (".", "")]
    for depth in range(1, len(parts) + 1):
        cand = "_".join(parts[-depth:] + [base]).replace(" ", "_")
        if cand.lower() not in _taken[key]:
            _taken[key].add(cand.lower())
            return cand
    i = 2
    while f"{base}.{i}".lower() in _taken[key]:
        i += 1
    _taken[key].add(f"{base}.{i}".lower())
    return f"{base}.{i}"


def cp(rel: str, group: str, note: str = "", flat: bool = True, dest: Path | None = None) -> bool:
    src = REPO / rel
    if not src.is_file():
        skipped.append({"src": rel, "reason": "NOT_FOUND", "group": group})
        return False
    if SECRET_RE.search(src.name):
        skipped.append({"src": rel, "reason": "SECRET_FILTER", "group": group})
        print(f"  SECRET-SKIP {rel}")
        return False
    if dest is not None:
        out = dest
    elif flat:
        d = GROUP_DIR[group]
        d.mkdir(parents=True, exist_ok=True)
        out = d / flat_name(rel, d)
    else:
        out = GROUP_DIR[group] / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    copied.append({"package_path": str(out.relative_to(PKG)).replace("\\", "/"),
                   "repo_path": rel, "size": src.stat().st_size, "sha256": sha256(out),
                   "group": group, "note": note})
    return True


def cp_tree(reldir: str, group: str, note: str, max_size: int | None = None,
            skip_names: set[str] | None = None) -> int:
    root = REPO / reldir
    n = 0
    if not root.is_dir():
        skipped.append({"src": reldir, "reason": "DIR_NOT_FOUND", "group": group})
        return 0
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if max_size is not None and f.stat().st_size > max_size:
            skipped.append({"src": str(f.relative_to(REPO)), "reason": "OVERSIZE", "group": group})
            continue
        if skip_names and f.name in skip_names:
            skipped.append({"src": str(f.relative_to(REPO)), "reason": "EXCLUDED_BY_PLAN", "group": group})
            continue
        if cp(str(f.relative_to(REPO)).replace("\\", "/"), group, note):
            n += 1
    return n


# --------------------------------------------------------------------------- #
def build_core() -> None:
    print("== 2.1 manuscript authority ==")
    for f, note in [
        ("3/docx/ZN_TIFS_CN_R10.docx", "R10 working draft: 55 remaining-evidence markers"),
        ("3/docx/ZN_TIFS_CN_R11_SUBMISSION_READY.docx", "R11 submission-ready: 4 remaining markers incl. PENDING-1"),
        ("3/docx/ZN_TIFS_CN_R9_CONDENSED.docx", "R9 predecessor"),
        ("3/docx/ZN_TIFS_CN_R8_WITH_FIGURES.docx", "R8 predecessor"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/full_manuscript_final.md", "REQUIRED BY SPEC 2.1B"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/04_experiments.md", "R5C experiments EN"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/supplement_CN_S1_covered_quotient.md", "S.1 CN"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/supplement_EN_S1_covered_quotient.md", "S.1 EN"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/supplement_EN_S2_ci_sensitivity.md", "S.2"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/supplement_EN_S3_provenance_adequacy.md", "S.3"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/supplement_EN_S4_scalability.md", "S.4"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/ZN_TIFS_CN_R5_DRAFT.docx", "R5 CN draft docx"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/R5C_FINAL_RETURN.md", "R5C return"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/manuscript/CURRENT_R5C/R5C_AUTHOR_VISUAL_QA_CHECKLIST.md", "R5C QA gate"),
        ("manuscript_final/appendix_B_connector_native_diagnostic.md", "Appendix B connector native diagnostic"),
        ("3/final/FINAL_CN_MANUSCRIPT_MANIFEST.json", "frozen CN manifest"),
        ("3/final/FINAL_CHINESE_MOTHER_MANUSCRIPT_LOCK.json", "CN mother manuscript lock"),
        ("3/final/FINAL_AUTHOR_DECISION_REPORT.md", "author decision closure report"),
        ("3/final/FINAL_AUTHOR_BLOCKERS.md", "final blockers v1"),
        ("3/final/FINAL_AUTHOR_BLOCKERS_V2.md", "final blockers v2"),
        ("3/final/AUTHOR_FINAL_ACTION_LIST.md", "author action list"),
        ("3/final/ZN_TIFS_FINAL_REVISION_REPORT.md", "revision report"),
        ("3/final/ZN_TIFS_FINAL_CHANGELOG.md", "changelog"),
    ]:
        cp(f, "manuscript_authority", note)

    print("== 2.1 reference sources (all variants; none renumbered) ==")
    for f, note in [
        ("3/references/references.bib", "bib co-located with REFERENCE_AUTHOR_ACTIONS (R5C/final line)"),
        ("3/references/REFERENCE_AUTHOR_ACTIONS.md", "author actions on references"),
        ("3/stage2/submission/TIFS_SUBMISSION_PACKAGE/references.bib", "Stage2 EN submission bib"),
        ("3/stage2/submission/TIFS_INITIAL_SUBMISSION_PACKAGE/references.bib", "Stage2 initial submission bib"),
        ("3/stage2/references/references.bib", "Stage2 references dir bib"),
        ("3/stage2/references/references_raw.bib", "Stage2 pre-resolution raw bib"),
        ("3/stage2/latex/references.bib", "Stage2 latex bib"),
        ("3/chinese_rewrite_r4/phase3/references/references_phase3.bib", "R4 phase3 bib"),
        ("3/tifs_reference_rebuild_r3/references/references.bib", "R3 rebuild bib"),
        ("3/tifs_reference_rebuild_r3/latex/references.bib", "R3 latex bib"),
        ("out/paper_full_pipeline_run/manuscript_final/references.bib", "older pipeline bib"),
        ("out/paper_full_pipeline_run/manuscript_final/references_index.md", "older pipeline reference index"),
        ("2/TIFS/references.bib", "early TIFS bib"),
        ("2/table_reference/references.bib", "early table reference bib"),
        ("1/table_reference/references.bib", "earliest table reference bib"),
        ("3/chinese_rewrite_r4/phase3/references/R4_REFERENCE_VERIFICATION_MATRIX.md", "reference verification matrix"),
        ("3/chinese_rewrite_r4/phase3/references/R4_REFERENCE_GAP_MAP.md", "reference gap map"),
        ("3/chinese_rewrite_r5/audit/R5_REFERENCE_MASTER_AUDIT.csv", "R5 reference master audit"),
        ("3/chinese_rewrite_r5/audit/R5_REFERENCE_FIX_REPORT.md", "R5 reference fix report"),
        ("3/chinese_rewrite_r5/audit/R5C_CN_EN_CLAIM_PARITY_AUDIT.md", "CN/EN claim parity audit"),
        ("docs/paper_readiness_checklist.md", "paper readiness checklist"),
    ]:
        cp(f, "pending6_references", note)

    print("== 2.2 PENDING-1 cost weights ==")
    for f, note in [
        ("scripts/multi_bridge/dev_candidate/af_common.py", "DEFINES primary_weights()/ablation_weights()/KEPT_ABS_WEIGHTS (time .25 route .15 risk .15 evidence .05 novelty .05, sum 0.65)"),
        ("scripts/multi_bridge/decoder_audit/da_common.py", "DEFINES FROZEN_PARAMS (uot_reg .05, uot_reg_m .5, lambda_risk .25, thr 1e-9, max_delay 21600, causal pen 5.0, backend pot) + THRESHOLD_MM_CUTOFF"),
        ("scripts/multi_bridge/baseline_mechanism/common.py", "SECOND definition of FROZEN_PARAMS (same values) + frozen cost loader"),
        ("scripts/multi_bridge/run_faithful_flow_structural.py", "THIRD definition of FROZEN_PARAMS incl. cost_weights=default_cost_weights()"),
        ("src/cross/domain/uot/cost_matrix.py", "DEFINES default_cost_weights(): amount .35 time .25 route .15 risk .15 graph .05 evidence .05 novelty .05 (7 keys)"),
        ("src/cross/domain/uot/uot_solver.py", "consumes weights / builds marginals a,b"),
        ("src/cross/domain/uot/uot_solver_numpy.py", "numpy solver path"),
        ("src/cross/domain/uot/delay_policy.py", "time-cost / delay policy"),
        ("src/cross/domain/uot/decode_transport.py", "transport decoding"),
        ("src/cross/application/standalone_flow_uot.py", "standalone UOT entry used by the generator runs"),
        ("scripts/multi_bridge/run_amount_free_dev.py", "amount-free cost construction caller"),
        ("scripts/multi_bridge/run_conditional_plan_dev.py", "conditional-plan cost construction caller"),
        ("scripts/multi_bridge/run_dev_stress_candidates.py", "stress cost construction caller"),
        ("scripts/multi_bridge/run_af_analysis.py", "amount-free analysis"),
        ("scripts/multi_bridge/verify_amount_free_dev.py", "asserts primary_weights == v/0.65 and ablation == KEPT_ABS_WEIGHTS"),
        ("scripts/multi_bridge/verify_conditional_plan_dev.py", "asserts weights/FROZEN_PARAMS unchanged"),
        ("scripts/multi_bridge/verify_transport_diagnosis.py", "asserts weights unchanged"),
        ("scripts/multi_bridge/verify_decoder_attribution_audit.py", "asserts FROZEN_PARAMS == expected dict"),
        ("scripts/multi_bridge/verify_cost_transport_diagnosis.py", "asserts FROZEN_PARAMS unchanged"),
        ("scripts/multi_bridge/diag/ctd_common.py", "cost/transport diagnosis shared loader"),
        ("scripts/multi_bridge/run_cost_forensic_audit.py", "cost component forensic audit"),
        ("scripts/multi_bridge/audit_compare_cost_components.py", "component comparison audit"),
        ("scripts/multi_bridge/audit_compare_frozen_plan.py", "frozen plan comparison audit"),
        ("scripts/multi_bridge/audit_frozen_inputs_current_code.py", "frozen-input reproducibility audit"),
        ("scripts/multi_bridge/audit_freeze_checksums.py", "freeze checksum audit"),
        ("scripts/_run_rc_uot_independent.py", "independent runner with a SECOND DEFAULT_COST_WEIGHTS literal"),
        ("src/cross/application/experiments/real_celer_ablation_io.py", "THIRD DEFAULT_COST_WEIGHTS literal"),
        ("config/defaults.json", "frozen run defaults"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/NEXT_CANDIDATE_SPEC.md", "HASH-LOCKED spec naming the exact fractions"),
    ]:
        cp(f, "pending1_cost_weights", note)

    print("== 2.3 PENDING-2 coverage tiers ==")
    for f, note in [
        ("tools/cross_aml/cross_aml/coverage.py", "OPEN-SOURCE RULE: qualify_coverage(); transfer_key_match<0.85, bridge_event_score<0.5, token_consistency<0.5, min_evidence_score 0.6"),
        ("tools/cross_aml/cross_aml/quotient_builder.py", "OPEN-SOURCE RULE: evidence_tier_for_pair(); transfer_key_match>=1.0 -> A; evidence>=0.5 AND match>=0.85 -> B; else C"),
        ("tools/cross_aml/cross_aml/feature_builder.py", "OPEN-SOURCE RULE: transfer_key_match / evidence_score construction"),
        ("tools/cross_aml/cross_aml/bridge_parser.py", "OPEN-SOURCE RULE: build_transfer_key()"),
        ("tools/cross_aml/cross_aml/explanation.py", "OPEN-SOURCE RULE: 0.85 threshold reuse"),
        ("tools/cross_aml/cross_aml/schemas.py", "OPEN-SOURCE RULE: CoverageDecision / PairFeatures"),
        ("tools/cross_aml/cross_aml/abstention.py", "OPEN-SOURCE RULE: abstention"),
        ("tools/cross_aml/cross_aml/rcuotq_matcher.py", "OPEN-SOURCE RULE: coverage-qualified matcher"),
        ("tools/cross_aml/cross_aml/config.py", "OPEN-SOURCE RULE: coverage config defaults"),
        ("tools/cross_aml/cross_aml/cli.py", "OPEN-SOURCE RULE: pipeline entry"),
        ("tools/cross_aml/cross_aml/test_coverage.py", "OPEN-SOURCE RULE: coverage unit tests"),
        ("tools/cross_aml/README.md", "open-source prototype readme (contains an 'Anonymous Authors' bibtex placeholder)"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/cross_aml/coverage.py", "FROZEN-PKG copy of coverage.py"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/cross_aml/quotient_builder.py", "FROZEN-PKG copy of quotient_builder.py"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/cross_aml/feature_builder.py", "FROZEN-PKG copy of feature_builder.py"),
        ("3/chinese_rewrite_r5/final/R5C_repro_bundle/cross_aml/coverage.py", "R5C bundle copy of coverage.py"),
        ("3/chinese_rewrite_r5/final/R5C_repro_bundle/cross_aml/quotient_builder.py", "R5C bundle copy of quotient_builder.py"),
        ("3/chinese_rewrite_r5/final/R5C_repro_bundle/cross_aml/feature_builder.py", "R5C bundle copy of feature_builder.py"),
        ("scripts/run_phase29_robust_rcuot_superiority.py", "CONFIRMATORY RULE: _decoder_coverage_aware tier_a>=0.8 / tier_b rc_score>=0.3 / tier_c base_score>=0.5, allow_tier_c False"),
        ("scripts/run_phase25_coverage_qualified_training.py", "phase25 coverage-qualified gate thresholds"),
        ("scripts/run_phase24_coverage_qualified_training_gate.py", "phase24 gate thresholds"),
        ("scripts/run_phase27_precision_repair_coverage_expansion.py", "phase27 tier_abc coverage expansion"),
        ("scripts/run_phase20_quotient_integrity_repair.py", "phase20 quotient / bridge_transfer_key definition"),
        ("scripts/run_phase19_event_incidence_quotient_layer.py", "phase19 event-incidence quotient layer"),
        ("scripts/run_phase23_source_event_coverage_expansion.py", "phase23 source-event coverage expansion"),
        ("scripts/run_phase22_event_coverage_repair.py", "phase22 coverage repair"),
        ("scripts/run_phase21_quotient_oracle_score_repair.py", "phase21 corrected quotient scoring"),
        ("scripts/audit_phase20_quotient_integrity_repair.py", "phase20 integrity audit"),
        ("out/multi_bridge_expansion/tifs_temporal_external_validation_preregistration_v3/TEMPORAL_EXTERNAL_PROVENANCE_TIERS.md", "FROZEN tier A/B/C definitions (external validation line)"),
        ("out/multi_bridge_expansion/tifs_temporal_external_validation_preregistration_v4/V4_PROVENANCE_TIERS.md", "FROZEN v4 tier definitions"),
        ("3/tifs_reference_rebuild_r3/chinese/insert_tables_figures.py", "CN table/figure insert script with coverage/ABCTracer rows"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/C_baseline_table4/coverage_tier_report.csv", "frozen tier counts (254 Tier_A_exact_bridge_key / 4942 Uncovered_abstained)"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/C_baseline_table4/coverage_qualified_summary.json", "covered-quotient summary + full_scope_claim_allowed=false"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/C_baseline_table4/full_scope_claim_gate.json", "full-scope gate false"),
        ("out/multi_bridge_expansion/tifs_final_consolidation/FINAL_122_PAIR_QUOTIENT_RULE_AUDIT.md", "frozen audit of the 122-pair quotient rule"),
        ("scripts/audit_phase25_coverage_qualified_training.py", "audits the 122-pair holdout (total 122 / pos 44 / neg 78 assertion)"),
        ("scripts/run_phase26_balanced_superiority.py", "PHASE25_HOLDOUT_SEEDS = range(212,232) -- identifies the covered holdout seed block"),
    ]:
        cp(f, "pending2_coverage_tiers", note)
    n = cp_tree("out/chapter4_data_package/frozen_outputs/coverage_quotient",
                "pending2_coverage_tiers",
                "frozen coverage/quotient output (covered holdout pairs, gates, curves)")
    print(f"   coverage_quotient frozen outputs: {n}")

    print("== 2.4 PENDING-3 baselines ==")
    for f, note in [
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/tables/FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md", "REQUIRED BY SPEC 2.4"),
        ("out/multi_bridge_expansion/tifs_final_consolidation/FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md", "working-tree copy of the same audit"),
        ("scripts/multi_bridge/baseline_mechanism/common.py", "DEFINES decode_connector (per-source argmin amount) / decode_abctracer (argmin .75*amount+.25*time)"),
        ("scripts/multi_bridge/run_main_baseline_study.py", "Main-study caller; header states per-source top-1 rules"),
        ("scripts/multi_bridge/run_locked_test.py", "locked test calling both decoders"),
        ("scripts/multi_bridge/run_capability_tests.py", "capability table + unmatched semantics audit for both styles"),
        ("scripts/multi_bridge/run_rc_uot_q_multi_bridge.py", "run_connector_adapted / predict_abctracer_style adapters"),
        ("scripts/multi_bridge/run_structural_baselines.py", "one-to-one hard matchers"),
        ("scripts/multi_bridge/run_structural_three_bridges.py", "three-bridge structural baselines"),
        ("scripts/multi_bridge/run_one_to_one_flow_baselines.py", "one-to-one flow baselines"),
        ("scripts/multi_bridge/verify_baseline_mechanism_study.py", "asserts split-exact all-zero for both styles"),
        ("scripts/multi_bridge/make_study_report.py", "Table-4-line report generator"),
        ("scripts/multi_bridge/make_study_figures.py", "baseline figure generator"),
        ("scripts/multi_bridge/make_final_figure.py", "final figure generator"),
        ("scripts/multi_bridge/make_audit_figures.py", "audit figure generator"),
        ("scripts/multi_bridge/make_study_diagnostics.py", "study diagnostics"),
        ("scripts/multi_bridge/run_coverage_precision_curve.py", "coverage/precision curve source"),
        ("scripts/run_baseline_compare_phase1_connector.py", "ORIGINAL Connector via external core.dst_chain.WithdrawLocator (top-1); F1=0.974 diagnostic"),
        ("scripts/run_baseline_compare_phase0.py", "phase0 probe; 'no top-k' statement"),
        ("scripts/run_baseline_compare_phase0_2.py", "phase0.2 probe"),
        ("scripts/run_baseline_compare_phase1_5_audit.py", "phase1.5 audit of connector core"),
        ("scripts/run_baseline_compare_phase1_6_package.py", "phase1.6 package"),
        ("scripts/run_baseline_compare_phase1_7_qa.py", "phase1.7 QA"),
        ("scripts/run_baseline_compare_phase2_2_package.py", "phase2.2 package"),
        ("scripts/run_baseline_compare_phase2_connector_anchor_masked.py", "anchor-masked connector run"),
        ("scripts/run_baseline_compare_phase2_fair_package.py", "phase2 fair package"),
        ("scripts/run_routeA_symmetric_masking.py", "routeA symmetric masking (connector)"),
        ("scripts/run_routeA_symmetric_masking_v2.py", "routeA v2 symmetric masking (connector)"),
        ("scripts/run_routeA_bridge_semantic_ablation_v3.py", "routeA v3 bridge-semantic ablation"),
        ("scripts/run_routeA_1_consistency_audit.py", "routeA consistency audit; search_withdraw top-1 note"),
        ("scripts/run_open_pool_baseline.py", "OPEN POOL: connector/abctracer adapters; ABCTracer BLOCKED_MISSING_CHECKPOINT"),
        ("src/cross/domain/locator/withdraw_locator.py", "LOCAL WithdrawLocator.search_withdraw (in-repo reimplementation, NOT the original Connector)"),
        ("src/cross/domain/path_b/legacy_connector.py", "legacy in-repo connector path"),
        ("src/cross/domain/path_b/env.py", "env toggles read by search_withdraw"),
        ("src/cross/baseline_compare/connector_preflight.py", "connector preflight"),
        ("src/cross/baseline_compare/open_pool.py", "open-pool construction"),
        ("src/cross/domain/path_b/unlabeled_io.py", "Connector-style column export"),
        ("src/cross/shared/connector_decimals.py", "connector decimals handling"),
        ("scripts/run_phase7_5_external_baselines.py", "phase7.5 style baselines + feasibility note"),
        ("scripts/run_phase10s_same_scope_baseline_superiority.py", "phase10s same-scope adapted baselines"),
        ("scripts/run_phase10t_rcuot_arch_optimization.py", "phase10t arch optimization"),
        ("scripts/run_phase10u_balanced_tradeoff_profile.py", "phase10u balanced profile"),
        ("scripts/run_phase10v_pair_f1_precision_rcuot.py", "phase10v pair-F1 precision"),
        ("scripts/run_phase26_balanced_superiority.py", "phase26 balanced superiority"),
        ("scripts/run_phase28_multiobjective_rcuot_superiority.py", "phase28 multiobjective"),
        ("scripts/run_phase29_robust_rcuot_superiority.py", "phase29 robust superiority (Table 4 generator)"),
        ("tools/paper_artifact/scripts/run_phase29_robust_rcuot_superiority.py", "paper_artifact copy of phase29"),
        ("1/_connector_adapter/adapter.py", "column-mapping adapter to the Connector core"),
        ("1/_connector_adapter/_patch.py", "adapter patch script"),
        ("1/Connector Enhancing the Traceability of Decentralized Bridge Applications via Automatic Cross-Chain Transaction Association.pdf", "Connector paper PDF (provenance of core.dst_chain expectations)"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/tables/Table6_connector_degradation_curve.json", "Table 6 source data"),
    ]:
        cp(f, "pending3_baselines", note)

    n = cp_tree("3/chinese_rewrite_r5/final/paper_experiments_results/connector_native",
                "pending3_baselines", "cached Connector-native prediction/eval artifact")
    print(f"   connector_native cached artifacts: {n}")
    n = cp_tree("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/tables", "pending3_baselines",
                "packaged table source / audit")
    print(f"   packaged tables: {n}")
    for f, note in [
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/figures/FIGURE_INDEX.md", "figure index"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/figures/_value_audit/R4_FIGURE_DATA_PROVENANCE.md", "figure data provenance"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/TABLE_FIGURE_PROVENANCE_audit.md", "table/figure provenance audit"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/TABLE_PROVENANCE_AUDIT.md", "table provenance audit"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/FIGURE_PROVENANCE_AUDIT.md", "figure provenance audit"),
        ("3/version_audit/TABLE_PROVENANCE_AUDIT.md", "working-tree table provenance audit"),
        ("3/version_audit/FIGURE_PROVENANCE_AUDIT.md", "working-tree figure provenance audit"),
        ("3/chinese_rewrite_r5/final/paper_experiments_results/baseline_table4/phase29_config.json", "phase29 config as used for Table 4"),
    ]:
        cp(f, "pending3_baselines", note)
    n = cp_tree("3/chinese_rewrite_r5/final/paper_experiments_results/baseline_table4",
                "pending3_baselines", "Table 4 / Figure 4 source artifact")
    print(f"   baseline_table4 artifacts: {n}")

    print("== 2.5 frozen holdout / D4 evidence ==")
    for f, note in [
        ("scripts/multi_bridge/holdout/holdout_common.py", "REQUIRED BY SPEC 2.5: frozen machinery, METHODS tuple, EXPECTED_DEV_ANCHORS, gates"),
        ("scripts/multi_bridge/holdout/run_locked_holdout.py", "REQUIRED BY SPEC 2.5: runner (NOT executed)"),
        ("scripts/multi_bridge/holdout/verify_locked_holdout.py", "independent verifier; independently redefines the five D4 methods"),
        ("scripts/multi_bridge/holdout/run_holdout_preflight_tests.py", "preflight tests"),
        ("scripts/multi_bridge/holdout/__init__.py", "package marker"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/FINAL_CONFIRMATORY_HOLDOUT_REPORT.md", "REQUIRED BY SPEC 2.5"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/statistics.json", "frozen statistics"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/verification.json", "independent verification result"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/anchor_overlap.json", "anchor overlap"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/ci_sensitivity_resampling_audit_results.json", "CI sensitivity"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/dependence_reaudit_results.json", "dependence reaudit"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/NEXT_CANDIDATE_SPEC.md", "REQUIRED BY SPEC 2.6 (hash-locked)"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/next_candidate_spec_lock.json", "spec hash lock"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/FINAL_COST_TRANSPORT_DIAGNOSIS.md", "cost/transport diagnosis"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/RUN_MANIFEST.md", "run manifest"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/preaudit_hashes.json", "preaudit hashes"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/verification_report.json", "verification"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/verification_report.md", "verification md"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/dual_scaling/FINAL_TRANSPORT_DIAGNOSIS.md", "dual scaling diagnosis"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/dual_scaling/HASH_MANIFEST.json", "hash manifest"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/dual_scaling/RUN_MANIFEST.md", "run manifest"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/dual_scaling/mechanism_summary.json", "mechanism summary"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/dual_scaling/NEXT_TRANSPORT_CANDIDATE_OPTIONS.md", "transport candidate options"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/dual_scaling/verification_report.json", "verification"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/dual_scaling/verification_report.md", "verification md"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/decoder_comparison/admissible_decoding_manifest.json", "decoder comparison manifest"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/decoder_comparison/admissible_decoding_summary.json", "decoder comparison summary"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/decoder_comparison/admissible_decoding_tradeoff.csv", "decoder tradeoff"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/decoder_comparison/paper_admissible_decoding_main_table_clean.json", "main table"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/decoder_comparison/paper_admissible_decoding_main_table_clean.md", "main table md"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/decoder_comparison/paper_admissible_decoding_table.md", "decoding table"),
        ("out/multi_bridge_expansion/conditional_plan_holdout_results/FINAL_CONFIRMATORY_HOLDOUT_REPORT.md", "working-tree copy of holdout report"),
        ("out/multi_bridge_expansion/conditional_plan_holdout_results/statistics.json", "working-tree frozen statistics"),
        ("out/multi_bridge_expansion/conditional_plan_holdout_results/verification.json", "working-tree verification"),
        ("out/multi_bridge_expansion/conditional_plan_holdout_results/CONFIRMATORY_CLAIM_MAPPING.md", "claim mapping"),
        ("3/chinese_rewrite_r5/final/R5C_repro_bundle/confirmatory_artifacts/run_locked_holdout.py", "R5C bundle runner copy"),
        ("3/chinese_rewrite_r5/final/R5C_repro_bundle/confirmatory_artifacts/statistics.json", "R5C bundle statistics"),
        ("3/chinese_rewrite_r5/final/R5C_repro_bundle/confirmatory_artifacts/verification.json", "R5C bundle verification"),
        ("3/chinese_rewrite_r5/final/R5C_repro_bundle/confirmatory_artifacts/verify_locked_temporal_external_validation.py", "R5C bundle verifier"),
    ]:
        cp(f, "pending5_holdout_d4", note)

    n = cp_tree("out/multi_bridge_expansion/conditional_plan_holdout_preregistration",
                "pending5_holdout_d4", "preregistration package file", max_size=4_000_000)
    print(f"   preregistration files: {n}")
    n = cp_tree("out/multi_bridge_expansion/conditional_plan_holdout_results/cells",
                "pending5_holdout_d4", "per-cell frozen output (cell_inputs.npz/grid.npz live in e1/)",
                skip_names={"cell_inputs.npz", "grid.npz"})
    print(f"   per-cell frozen outputs: {n}")

    print("== 2.6 generator evidence ==")
    for f, note in [
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/RUN_MANIFEST_structural_benchmark.md", "REQUIRED BY SPEC 2.6"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/A_structural/RUN_MANIFEST_structural_benchmark.md", "A_structural copy"),
        ("scripts/multi_bridge/faithful_flow_features.py", "faithful feature/flow construction"),
        ("src/cross/domain/evaluation/semi_synthetic_flows.py", "GENERATOR: build_semi_synthetic_from_flow_labels (split/merge/decoy/noise)"),
        ("src/cross/domain/evaluation/synthetic_segment_subgraph.py", "GENERATOR: write_synthetic_subgraph_segment_csvs"),
        ("src/cross/domain/evaluation/synthetic_scenario_eval.py", "synthetic scenario evaluation"),
        ("src/cross/domain/evaluation/synthetic_failure_debug.py", "synthetic failure debug"),
        ("scripts/multi_bridge/run_stress_ladders.py", "stress ladders (mass/unmatched/decoy/timestamp)"),
        ("scripts/multi_bridge/run_stress_decoder_regression.py", "stress decoder regression"),
        ("scripts/multi_bridge/run_marginal_pressure_sweeps.py", "marginal pressure sweeps"),
        ("scripts/multi_bridge/run_supplementary_analyses.py", "supplementary analyses"),
        ("scripts/multi_bridge/run_threshold_many_match.py", "Threshold-MM baseline (E1 comparison arm)"),
        ("scripts/multi_bridge/run_dev_sweeps.py", "dev sweeps"),
        ("scripts/multi_bridge/run_kernel_transport_audit.py", "kernel/transport decomposition audit"),
        ("scripts/multi_bridge/run_dual_scaling_decomposition.py", "dual scaling decomposition"),
        ("scripts/multi_bridge/run_plan_quality_audit.py", "plan quality audit"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/PAPER_EXPERIMENTS_RESULTS_MANIFEST.json", "frozen results manifest"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/ARCHIVE_REPRODUCIBILITY_INDEX.md", "reproducibility index"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/INDEX.md", "repro code index"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/run_faithful_flow_structural.py", "packaged generator runner"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/run_amount_free_dev.py", "packaged amount-free dev runner"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/holdout_common.py", "packaged holdout_common"),
        ("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/run_locked_holdout.py", "packaged holdout runner"),
    ]:
        cp(f, "pending4_generator", note)
    n = cp_tree("ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag",
                "pending4_generator", "cost_diag artifact subtree")
    print(f"   cost_diag artifacts: {n}")

    print("== 2.7 repository link status inputs ==")
    for f in [
        "tools/cross_aml/README.md",
        "3/stage2/submission/TIFS_DATA_CODE_AVAILABILITY_FINAL.md",
        "3/stage2/submission/TIFS_REPRODUCIBILITY_STATEMENT_FINAL.md",
        "3/stage2/submission/TIFS_COVER_LETTER_FINAL.md",
        "3/stage2/submission/AUTHOR_METADATA_REQUIRED.md",
        "ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/PAPER_EXPERIMENTS_RESULTS_MANIFEST.json",
    ]:
        cp(f, "pending5_repository_link", "scanned for repository/availability language")


def build_e1() -> None:
    print("== 3.1 holdout cell inputs ==")
    cells = REPO / "out/multi_bridge_expansion/conditional_plan_holdout_results/cells"
    keep = {"cell_inputs.npz", "grid.npz", "labels.csv", "flow_labels_pool.csv",
            "flow_label_stats.json", "synthetic_seed_stratification.json", "solver.json"}
    for bridge in ("Celer", "Multi", "Poly"):
        for seed in range(301, 306):
            d = cells / bridge / f"seed_{seed}"
            found = False
            for f in sorted(d.rglob("*")) if d.is_dir() else []:
                if f.is_file() and f.name in keep:
                    found = True
                    rel = str(f.relative_to(REPO)).replace("\\", "/")
                    cp(rel, "e1", f"E1 raw cell input/metadata ({bridge} seed_{seed})",
                       dest=E1 / "cells_raw" / bridge / f"seed_{seed}" / str(f.relative_to(d)))
            if not found:
                skipped.append({"src": f"cells/{bridge}/seed_{seed}", "reason": "CELL_MISSING", "group": "e1"})
                print(f"   MISSING CELL {bridge}/seed_{seed}")

    print("== 3.2 executable code closure ==")
    for key, sub in (("closure_e1_full.json", "code_closure"),
                     ("closure_baselines.json", "code_closure_baseline")):
        data = json.loads((TOOLS / key).read_text(encoding="utf-8-sig"))
        for e in data["files"]:
            rel = e["path"].replace("\\", "/")
            cp(rel, "e1", f"local import closure depth={e['depth']}",
               dest=E1 / sub / rel)

    print("== 3.3 environment metadata ==")
    for f in ["pyproject.toml", "requirements.txt", "setup.py", "setup.cfg",
              "poetry.lock", "pdm.lock", "uv.lock", "environment.yml", "environment.yaml",
              "config/defaults.json", "run.py"]:
        cp(f, "e1", "environment/build metadata", dest=E1 / "env" / Path(f).name)
    # the authoring package ships its own environment files; keep them distinguishable
    for f in ["ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/pyproject.toml",
              "ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/requirements.txt"]:
        cp(f, "e1", "authoring-package environment metadata",
           dest=E1 / "env" / ("repro_pkg_" + Path(f).name))


def emit_indexes() -> None:
    # SHA256SUMS over core/ + e1/ + the top-level reports
    lines = []
    for p in sorted(PKG.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(PKG)).replace("\\", "/")
        if rel.startswith("_tools/") or rel.endswith(".zip") or rel == "SHA256SUMS.txt":
            continue
        lines.append(f"{sha256(p)}  {rel}")
    (PKG / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # E1_CELL_INDEX.csv
    rows = []
    for p in sorted((E1 / "cells_raw").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(E1 / "cells_raw")
        rows.append({
            "bridge": rel.parts[0], "seed": rel.parts[1].replace("seed_", ""),
            "unit_subdir": "cells/" + "/".join(rel.parts[:2]),
            "file": rel.parts[-1],
            "relative_path": str(p.relative_to(PKG)).replace("\\", "/"),
            "file_size": p.stat().st_size, "sha256": sha256(p),
            "origin_repo_path": "out/multi_bridge_expansion/conditional_plan_holdout_results/cells/"
                                + "/".join(rel.parts),
        })
    if rows:
        with (E1 / "E1_CELL_INDEX.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # LOCAL_IMPORT_CLOSURE.txt
    name_of = {e["repo_path"]: e["package_path"] for e in copied}
    out = ["# LOCAL_IMPORT_CLOSURE.txt",
           "# Static AST closure (ast.parse only; no target module was imported).",
           "# caller -> imported local module -> resolved repo path -> packaged path -> sha256", "#"]
    for key, label in (("closure_e1_full.json",
                        "E1 CLOSURE (seeds: holdout runner/verifier/common, faithful generator, "
                        "threshold-MM, cost/solver, af/da/diag)"),
                       ("closure_baselines.json",
                        "BASELINE CLOSURE (seeds: connector phase1, open-pool, main baseline study, "
                        "capability tests, rc-uot-q multi-bridge, structural baselines, verifier, phase29)")):
        data = json.loads((TOOLS / key).read_text(encoding="utf-8-sig"))
        out += ["=" * 100, label, "=" * 100]
        for ed in sorted(data["edges"], key=lambda x: (x["caller"], x["module"], x["resolved"])):
            out.append(f"{ed['caller']}\n    -> {ed['module']}\n    -> {ed['resolved']}\n"
                       f"    -> {name_of.get(ed['resolved'], 'NOT_PACKAGED')}\n    -> {ed['sha256']}")
        out += ["", f"LOCAL FILES IN CLOSURE: {data['n_local_files']}",
                f"UNRESOLVED LOCAL CANDIDATES: {data['unresolved_local_candidates']}", ""]
    (E1 / "LOCAL_IMPORT_CLOSURE.txt").write_text("\n".join(out), encoding="utf-8")

    # TREE.txt
    t = []
    for root, title in ((CORE, "core/"), (E1, "e1/")):
        t.append(title)
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(root)
                t.append("    " * (len(rel.parts) - 1) + rel.name +
                         f"  ({p.stat().st_size} B)")
        t.append("")
    (PKG / "TREE.txt").write_text("\n".join(t) + "\n", encoding="utf-8")

    (TOOLS / "copy_manifest.json").write_text(json.dumps(
        {"copied": copied, "skipped": skipped, "n_copied": len(copied),
         "n_skipped": len(skipped)}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    # clean rebuild of the destination only
    for sub in ("core", "e1"):
        if (PKG / sub).exists():
            shutil.rmtree(PKG / sub)
    TOOLS.mkdir(parents=True, exist_ok=True)
    CORE.mkdir(parents=True, exist_ok=True)
    E1.mkdir(parents=True, exist_ok=True)
    build_core()
    build_e1()
    emit_indexes()
    print()
    print("copied:", len(copied), "skipped:", len(skipped))
    for s in skipped:
        if s["reason"] in ("NOT_FOUND", "DIR_NOT_FOUND", "CELL_MISSING"):
            print("  MISSING", s["reason"], s["src"])
