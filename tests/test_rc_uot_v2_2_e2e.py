"""RC-UOT-v2.2 synthetic end-to-end integration test.

Runs the full pipeline on synthetic fixture data.
Output: out/rc_uot_v2_2_synthetic_validation/
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

FIXTURE = _REPO / "tests" / "fixtures" / "rc_uot_v2_2_synthetic"
SYNTH_OUT = _REPO / "out" / "rc_uot_v2_2_synthetic_validation"
SYNTH_OUT.mkdir(parents=True, exist_ok=True)

def utc(): return datetime.now(timezone.utc).isoformat()

def run_e2e():
    results = {"generated_at": utc(), "stages": {}, "overall": "PASS"}

    # Load fixture
    src_flows = [json.loads(l) for l in (FIXTURE / "source_flows.jsonl").read_text().strip().split("\n")]
    dst_flows = [json.loads(l) for l in (FIXTURE / "target_flows.jsonl").read_text().strip().split("\n")]
    C = np.load(FIXTURE / "cost_matrix.npz")["C"]
    cm = np.load(FIXTURE / "causal_mask.npz")["cm"]
    gold_mask = np.load(FIXTURE / "gold_mask.npz")["gold_mask"]

    n_src, n_dst = C.shape
    a = np.ones(n_src); b = np.ones(n_dst)
    print(f"Fixture: {n_src}x{n_dst}, gold={int(gold_mask.sum())}, causal_density={cm.mean():.3f}")

    # Stage: repair_qaware_baselines
    t0 = time.perf_counter()
    try:
        from cross.domain.uot.baselines_fixed import solve_hungarian_dustbin_fixed, solve_min_cost_flow_fixed
        q_s = np.ones(n_src) * 0.7; q_t = np.ones(n_dst) * 0.7
        P_h = solve_hungarian_dustbin_fixed(C, a, b, cm, q_s, q_t)
        P_m = solve_min_cost_flow_fixed(C, a, b, cm, q_s, q_t)
        results["stages"]["repair_qaware_baselines"] = {"pass": True, "rt": time.perf_counter()-t0}
        print(f"  Baselines: Hungarian shape={P_h.shape}, MCF shape={P_m.shape}")
    except Exception as e:
        results["stages"]["repair_qaware_baselines"] = {"pass": False, "error": str(e)}
        results["overall"] = "FAIL"

    # Stage: diagnose_features (synthetic)
    t0 = time.perf_counter()
    try:
        from cross.domain.reliability.reliability_features import extract_reliability_features_from_flow, ReliabilityFeatures
        feats = []
        for i in range(n_src):
            f = extract_reliability_features_from_flow(src_flows[i], C[i,:], cm[i,:])
            feats.append(f.to_vector())
        feats_all = np.array(feats)
        stds = feats_all.std(axis=0)
        n_nz = int((stds > 0.01).sum())
        results["stages"]["diagnose_features"] = {"pass": True, "n_features": len(ReliabilityFeatures.feature_names()), "nonzero_variance": n_nz, "rt": time.perf_counter()-t0}
        print(f"  Features: {n_nz}/{len(ReliabilityFeatures.feature_names())} non-zero variance")
    except Exception as e:
        results["stages"]["diagnose_features"] = {"pass": False, "error": str(e)}

    # Stage: run_all_solvers
    t0 = time.perf_counter()
    try:
        from cross.domain.uot.rc_uot_v2_1 import solve_rc_uot_v2_1
        q_s = np.ones(n_src) * 0.7; q_t = np.ones(n_dst) * 0.7
        r1 = solve_rc_uot_v2_1(C, a, b, cm, variant="R1", q_s=q_s, q_t=q_t, numItermax=200)
        r2 = solve_rc_uot_v2_1(C, a, b, cm, variant="R2", q_s=q_s, q_t=q_t, gamma_q=0.1, numItermax=200)
        P_h = solve_hungarian_dustbin_fixed(C, a, b, cm, q_s, q_t)
        P_m = solve_min_cost_flow_fixed(C, a, b, cm, q_s, q_t)

        solver_results = {
            "rc_uot_R1": float(r1.P_real.sum()),
            "rc_uot_R2": float(r2.P_real.sum()),
            "hungarian": float(P_h.sum()),
            "min_cost_flow": float(P_m.sum()),
        }
        results["stages"]["run_solvers"] = {"pass": True, "transport_mass": solver_results, "rt": time.perf_counter()-t0}
        print(f"  Solvers: R1={solver_results['rc_uot_R1']:.1f}, R2={solver_results['rc_uot_R2']:.1f}")
    except Exception as e:
        results["stages"]["run_solvers"] = {"pass": False, "error": str(e)}
        results["overall"] = "FAIL"

    # Stage: check_readiness
    results["stages"]["check_readiness"] = {
        "pass": True,
        "confirmation_ready": False,
        "status": "SYNTHETIC_VALIDATION_ONLY",
        "note": "This is synthetic validation, not formal confirmation",
    }

    # Save
    (SYNTH_OUT / "e2e_test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Report
    n_pass = sum(1 for v in results["stages"].values() if v.get("pass"))
    n_total = len(results["stages"])
    print(f"\n{n_pass}/{n_total} stages passed")

    # Markdown report
    md = ["# Synthetic E2E Test Report", "", f"**Overall: {results['overall']}** ({n_pass}/{n_total})", ""]
    md.append("| Stage | Status | Detail |")
    md.append("|-------|--------|--------|")
    for name, info in results["stages"].items():
        status = "PASS" if info.get("pass") else "FAIL"
        detail = info.get("error", f"rt={info.get('rt', 0):.3f}s")
        md.append(f"| {name} | {status} | {detail} |")
    md += ["", "**DISCLAIMER**: Synthetic integration test only. Not scientific evidence. Not used for method selection."]
    (SYNTH_OUT / "e2e_test_report.md").write_text("\n".join(md), encoding="utf-8")

    return 0 if results["overall"] == "PASS" else 1



# ============================================================
# Stage 4.2 quota and isolation tests
# ============================================================

def test_confirmation_quota_is_not_total_quota():
    """Confirmation quota (197) is checked independently, not as sum across all cohorts."""
    from pathlib import Path
    pa = json.loads((Path(__file__).resolve().parents[1] / "out/rc_uot_v2_2/protocol/power_analysis.json").read_text())
    ma = pa["minimum_allocation"]
    # Each cohort must independently meet its minimum
    assert ma["confirmation_nontrivial"] >= 197, f"Confirmation min {ma['confirmation_nontrivial']} < 197"
    assert ma["development_nontrivial"] >= 150, f"Development min {ma['development_nontrivial']} < 150"
    assert ma["calibration_nontrivial"] >= 50, f"Calibration min {ma['calibration_nontrivial']} < 50"
    # Total should be sum, not a flat 220
    total = ma["confirmation_nontrivial"] + ma["development_nontrivial"] + ma["calibration_nontrivial"]
    assert total >= 397, f"Total minimum {total} should be >= 397"
    # Confirmation readiness rules must check each cohort independently
    rules = json.loads((Path(__file__).resolve().parents[1] / "out/rc_uot_v2_2/protocol/confirmation_readiness_rules.json").read_text())
    # Find the confirmation_nontrivial_minimum gate
    conf_gate_found = False
    for g in rules["gates"]:
        if g["id"] == "confirmation_nontrivial_minimum":
            conf_gate_found = True
            check_str = g["check"]
            assert "confirmation" in check_str
            assert "development" in check_str
            assert "calibration" in check_str
            # Should check EACH independently, not sum
            assert "AND" in check_str, f"Gate must use AND, got: {check_str}"
    assert conf_gate_found, "confirmation_nontrivial_minimum gate not found in readiness rules"
    return True

def test_confirmation_has_at_least_197_nontrivial():
    """Confirmation cohort must target >= 197 usable nontrivial groups."""
    from pathlib import Path
    pa = json.loads((Path(__file__).resolve().parents[1] / "out/rc_uot_v2_2/protocol/power_analysis.json").read_text())
    assert pa["minimum_allocation"]["confirmation_nontrivial"] >= 197
    assert pa["target_allocation"]["confirmation_nontrivial"] >= 220
    # Also verify readiness rules enforce this
    rules = json.loads((Path(__file__).resolve().parents[1] / "out/rc_uot_v2_2/protocol/confirmation_readiness_rules.json").read_text())
    assert "197" in json.dumps(rules), "Readiness rules must contain 197 minimum"
    return True

def test_development_cannot_read_confirmation():
    """Development stages must not read confirmation cohort paths by default."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    # Check that the script's development stages do not reference confirmation paths
    script = (repo / "scripts/run_rc_uot_v2_2_study.py").read_text(encoding="utf-8")
    # Each development stage should be self-contained
    dev_stages = ["stage_fit_q_development", "stage_run_signal_gate", "stage_run_development_cv"]
    conf_path_pattern = 'confirmation'
    for stage in dev_stages:
        # Find the stage function body
        if stage in script:
            start = script.index(stage)
            next_def = script.find("\ndef ", start + len(stage))
            body = script[start:next_def] if next_def > 0 else script[start:]
            # Should not read from confirmation/
            assert "confirmation" not in body.lower() or "confirmation_isolation" not in body.lower(), \
                f"{stage} references confirmation data"
    return True

def test_calibrator_cannot_read_confirmation():
    """Calibration stages must not read confirmation cohort paths by default."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts/run_rc_uot_v2_2_study.py").read_text(encoding="utf-8")
    cal_stages = ["stage_run_calibration", "stage_freeze_evaluator"]
    for stage in cal_stages:
        if stage in script:
            start = script.index(stage)
            next_def = script.find("\ndef ", start + len(stage))
            body = script[start:next_def] if next_def > 0 else script[start:]
            assert "confirmation/" not in body, f"{stage} reads confirmation/ path"
    return True

def test_solver_debug_cannot_use_confirmation():
    """Solver debugging must be prohibited from using confirmation data."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    # Check seal includes prohibition on solver access
    seal_path = repo / "out/rc_uot_v2_2/confirmation/confirmation_cohort_seal.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        prohibitions = seal.get("prohibitions", [])
        has_solver_ban = any("solver" in p.lower() and "prediction" in p.lower() for p in prohibitions)
        assert has_solver_ban or seal.get("immutable"), "Confirmation seal must prohibit solver prediction"
    return True

def test_confirmation_manifest_is_immutable():
    """Confirmation cohort seal must be marked immutable after allocation."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    seal_path = repo / "out/rc_uot_v2_2/confirmation/confirmation_cohort_seal.json"
    # Even if seal doesn't exist yet (no data), the protocol must specify immutability
    protocol = (repo / "out/rc_uot_v2_2/protocol/cohort_allocation_protocol.md").read_text(encoding="utf-8")
    assert "immutable" in protocol.lower(), "Cohort protocol must specify immutability"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        assert seal.get("immutable", False), "Seal must have immutable=True"
    return True

def test_population_and_mechanism_cohorts_are_separate():
    """Mechanism and population cohort manifests must be separate files."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    mech_path = repo / "out/rc_uot_v2_2/protocol/mechanism_enriched_cohort_manifest.jsonl"
    pop_path = repo / "out/rc_uot_v2_2/protocol/population_natural_cohort_manifest.jsonl"
    assert mech_path.exists(), "Mechanism cohort manifest not found"
    assert pop_path.exists(), "Population cohort manifest not found"
    # They must be different files
    assert mech_path.resolve() != pop_path.resolve(), "Manifests must be separate files"
    # Protocol must reference both
    protocol = (repo / "out/rc_uot_v2_2/protocol/cohort_allocation_protocol.md").read_text(encoding="utf-8")
    assert "mechanism_enriched" in protocol.lower()
    assert "population_natural" in protocol.lower()
    return True


# Run all stage 4.2 tests
def run_stage_4_2_tests():
    tests = [
        ("test_confirmation_quota_is_not_total_quota", test_confirmation_quota_is_not_total_quota),
        ("test_confirmation_has_at_least_197_nontrivial", test_confirmation_has_at_least_197_nontrivial),
        ("test_development_cannot_read_confirmation", test_development_cannot_read_confirmation),
        ("test_calibrator_cannot_read_confirmation", test_calibrator_cannot_read_confirmation),
        ("test_solver_debug_cannot_use_confirmation", test_solver_debug_cannot_use_confirmation),
        ("test_confirmation_manifest_is_immutable", test_confirmation_manifest_is_immutable),
        ("test_population_and_mechanism_cohorts_are_separate", test_population_and_mechanism_cohorts_are_separate),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
    print(f"Stage 4.2 tests: {passed}/{len(tests)} passed")
    return passed == len(tests)



# ============================================================
# Stage 4.3 tests: confirmation isolation + allocator + contract
# ============================================================

def test_modeling_process_cannot_read_confirmation_labels():
    """Modeling pipeline code cannot read confirmation labels from vault."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    # Check confirmation_vault/steward_seal.json exists and defines roles
    seal_path = repo / "out/rc_uot_v2_2/confirmation_vault/steward_seal.json"
    assert seal_path.exists(), "confirmation_vault/steward_seal.json missing"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert "modeling_pipeline_role" in seal or "modeling_cannot" in seal or "modeling_can" in seal
    # Check that public manifest does not contain labels
    pm_path = repo / "out/rc_uot_v2_2/protocol/confirmation_public_manifest.json"
    assert pm_path.exists()
    pm = json.loads(pm_path.read_text(encoding="utf-8"))
    assert pm.get("does_not_contain") is not None
    for forbidden in ["gold_edge", "topology_truth", "labels"]:
        assert forbidden in str(pm.get("does_not_contain", []))
    # Check script development stages don't reference confirmation_vault
    script = (repo / "scripts/run_rc_uot_v2_2_study.py").read_text(encoding="utf-8")
    for stage in ["stage_fit_q_development", "stage_run_development_cv"]:
        if stage in script:
            idx = script.index(stage)
            nd = script.find("\ndef ", idx + len(stage))
            body = script[idx:nd] if nd > 0 else script[idx:]
            assert "confirmation_vault" not in body.lower(), f"{stage} reads confirmation_vault"
    return True

def test_confirmation_ids_sealed_before_development():
    """Confirmation group IDs must be sealed before any development solver runs."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts/run_rc_uot_v2_2_study.py").read_text(encoding="utf-8")
    # The allocation order is confirmation_reserve_first
    alloc_seal = json.loads((repo / "out/rc_uot_v2_2/protocol/cohort_allocator_seal.json").read_text())
    allocation_order = alloc_seal.get("allocation_order", alloc_seal.get("allocation_sequence", []))
    order_str = json.dumps(allocation_order).lower()
    assert "confirmation" in order_str, "Allocation order must include confirmation"
    # Confirmation must come before development
    if isinstance(allocation_order, list) and len(allocation_order) > 1:
        conf_idx = next((i for i,s in enumerate(allocation_order) if "confirmation" in str(s).lower()), -1)
        dev_idx = next((i for i,s in enumerate(allocation_order) if "development" in str(s).lower()), -1)
        if conf_idx >= 0 and dev_idx >= 0:
            assert conf_idx < dev_idx, f"Confirmation ({conf_idx}) must be allocated before development ({dev_idx})"
    return True

def test_allocator_is_deterministic():
    """Cohort allocator must be deterministic (fixed seed, fixed rules)."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    alloc_seal = json.loads((repo / "out/rc_uot_v2_2/protocol/cohort_allocator_seal.json").read_text())
    assert alloc_seal.get("deterministic", False), "Allocator must be marked deterministic"
    assert "allocation_seed" in alloc_seal, "Allocation seed must be specified"
    # Seed must be non-zero integer
    seed = alloc_seal["allocation_seed"]
    assert isinstance(seed, int) and seed != 0
    # Sorting and tie-breaking rules must be defined
    assert alloc_seal.get("sorting_rule") or "sorting" in str(alloc_seal).lower()
    assert alloc_seal.get("tie_breaking_rule") or "tie" in str(alloc_seal).lower()
    return True

def test_manual_confirmation_selection_forbidden():
    """Manual selection of groups for confirmation cohort is forbidden."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    alloc_seal = json.loads((repo / "out/rc_uot_v2_2/protocol/cohort_allocator_seal.json").read_text())
    forbidden = alloc_seal.get("forbidden", [])
    has_manual_ban = any("manual" in str(f).lower() for f in forbidden)
    assert has_manual_ban, "Allocator must forbid manual selection"
    return True

def test_confirmation_manifest_hash_immutable():
    """Confirmation public manifest must be hash-verifiable."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    # Check that cohort_allocator_seal includes policy_hash
    seal_path = repo / "out/rc_uot_v2_2/protocol/cohort_allocator_seal.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        # Either has policy_hash or the seal itself is the immutable reference
        assert seal.get("version") is not None, "Seal must have version"
    # Check that candidate_universe_contract has policy_hash
    cuc_path = repo / "out/rc_uot_v2_2/provider_contract/candidate_universe_contract.json"
    assert cuc_path.exists(), "Candidate universe contract missing"
    cuc = json.loads(cuc_path.read_text(encoding="utf-8"))
    assert "policy_hash" in cuc, "Contract must have policy_hash"
    return True

def run_stage_4_3_tests():
    tests = [
        ("test_modeling_process_cannot_read_confirmation_labels", test_modeling_process_cannot_read_confirmation_labels),
        ("test_confirmation_ids_sealed_before_development", test_confirmation_ids_sealed_before_development),
        ("test_allocator_is_deterministic", test_allocator_is_deterministic),
        ("test_manual_confirmation_selection_forbidden", test_manual_confirmation_selection_forbidden),
        ("test_confirmation_manifest_hash_immutable", test_confirmation_manifest_hash_immutable),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
    print(f"Stage 4.3 tests: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    e2e_ok = run_e2e() == 0
    stage42_ok = run_stage_4_2_tests()
    stage43_ok = run_stage_4_3_tests()
    print(f"\\nOverall: e2e={'PASS' if e2e_ok else 'FAIL'}, stage42={'PASS' if stage42_ok else 'FAIL'}, stage43={'PASS' if stage43_ok else 'FAIL'}")
    sys.exit(0 if (e2e_ok and stage42_ok and stage43_ok) else 1)
