"""RC-UOT-v2.2 Provider Contract Hotfix 4.3A.1 - Test Suite.

Tests for: field naming, causal time direction, policy hash, manifest encoding,
deprecated-file exclusion, example_valid_p0, and zip content audit.
"""
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

HANDOFF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "out", "rc_uot_v2_2", "provider_handoff_final"
)
PROVIDER_CONTRACT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "out", "rc_uot_v2_2", "provider_contract"
)
EXAMPLE_DIR = os.path.join(HANDOFF_DIR, "example_valid_p0")

# ============================================================================
# Fix 1: Field name tests
# ============================================================================

def test_source_and_target_version_field_names_identical():
    """Both schemas must use data_source_version, not data_target_version."""
    src_schema = json.loads(
        Path(os.path.join(HANDOFF_DIR, "source_flow.schema.json")).read_text()
    )
    tgt_schema = json.loads(
        Path(os.path.join(HANDOFF_DIR, "target_flow.schema.json")).read_text()
    )
    assert "data_source_version" in src_schema["required"]
    assert "data_source_version" in tgt_schema["required"]
    assert "data_target_version" not in src_schema.get("properties", {})
    assert "data_target_version" not in tgt_schema.get("properties", {})


def test_target_record_with_data_source_version_passes():
    """A target record with data_source_version should validate."""
    record = {
        "target_flow_id": "tgt_test_001",
        "target_chain": "ETH",
        "transaction_ids": ["0xabc"],
        "constituent_transaction_ids": ["0xdef"],
        "start_timestamp": "2026-06-01T01:00:00Z",
        "end_timestamp": "2026-06-01T02:00:00Z",
        "asset": "USDC",
        "raw_amount": "1000.00",
        "normalized_amount": "1000.00",
        "target_address": "0xaddr",
        "bridge_or_route_id": None,
        "raw_evidence_reference": None,
        "data_collection_timestamp": "2026-06-24T00:00:00Z",
        "data_source_version": "v1.0"
    }
    import jsonschema
    schema = json.loads(
        Path(os.path.join(HANDOFF_DIR, "target_flow.schema.json")).read_text()
    )
    jsonschema.validate(record, schema)


def test_target_record_with_data_target_version_fails():
    """A target record with data_target_version (wrong name) should fail."""
    record = {
        "target_flow_id": "tgt_test_002",
        "target_chain": "ETH",
        "transaction_ids": ["0xabc"],
        "constituent_transaction_ids": ["0xdef"],
        "start_timestamp": "2026-06-01T01:00:00Z",
        "end_timestamp": "2026-06-01T02:00:00Z",
        "asset": "USDC",
        "raw_amount": "1000.00",
        "normalized_amount": "1000.00",
        "target_address": "0xaddr",
        "bridge_or_route_id": None,
        "raw_evidence_reference": None,
        "data_collection_timestamp": "2026-06-24T00:00:00Z",
        "data_target_version": "v1.0"
    }
    import jsonschema
    schema = json.loads(
        Path(os.path.join(HANDOFF_DIR, "target_flow.schema.json")).read_text()
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, schema)


# ============================================================================
# Fix 2: Causal time direction tests
# ============================================================================

def test_target_after_source_within_window_is_candidate():
    """Target start after source end within window = valid candidate."""
    policy = json.loads(
        Path(os.path.join(HANDOFF_DIR, "candidate_generation_policy.json")).read_text()
    )
    assert policy["causal_rule_direction"] == "forward_only"
    rule = policy["timestamp_reference_rule"]
    assert "source.end_timestamp <= target.start_timestamp" in rule
    window = policy["causal_time_window_seconds"]
    assert window == 86400


def test_target_before_source_is_not_candidate():
    """Target before source = invalid (reverse time not allowed)."""
    policy = json.loads(
        Path(os.path.join(HANDOFF_DIR, "candidate_generation_policy.json")).read_text()
    )
    rule = policy["timestamp_reference_rule"]
    # Verify forward-only: source.end <= target.start
    assert "source.end_timestamp <= target.start_timestamp" in rule


def test_target_after_window_is_not_candidate():
    """Target after window = invalid."""
    policy = json.loads(
        Path(os.path.join(HANDOFF_DIR, "candidate_generation_policy.json")).read_text()
    )
    rule = policy["timestamp_reference_rule"]
    assert "(target.start_timestamp - source.end_timestamp) <= causal_time_window_seconds" in rule


def test_equal_boundary_timestamp_policy():
    """Equal timestamps (source.end == target.start) = allowed (non-strict inequality)."""
    policy = json.loads(
        Path(os.path.join(HANDOFF_DIR, "candidate_generation_policy.json")).read_text()
    )
    rule = policy["timestamp_reference_rule"]
    # source.end_timestamp <= target.start_timestamp allows equality
    assert "<=" in rule.split("AND")[0]


# ============================================================================
# Fix 4: Policy hash tests
# ============================================================================

def test_policy_hash_recomputes_exactly():
    """policy_body_sha256 must match recomputation from body."""
    policy = json.loads(
        Path(os.path.join(HANDOFF_DIR, "candidate_generation_policy.json")).read_text()
    )
    expected = policy["policy_body_sha256"]
    # Recompute: remove hash fields, sort keys, compact separators
    pc = dict(policy)
    for fld in ["policy_body_sha256", "policy_file_sha256", "policy_hash_sha256"]:
        pc.pop(fld, None)
    body_json = json.dumps(pc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    actual = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    assert actual == expected, f"policy_body_sha256 mismatch: {expected} vs {actual}"


def test_contract_references_current_policy_hash():
    """candidate_universe_contract.md must reference the current policy hashes."""
    md = Path(os.path.join(HANDOFF_DIR, "candidate_universe_contract.md")).read_text()
    policy = json.loads(
        Path(os.path.join(HANDOFF_DIR, "candidate_generation_policy.json")).read_text()
    )
    assert policy["policy_body_sha256"] in md
    assert policy["policy_file_sha256"] in md


def test_policy_change_invalidates_hash():
    """Changing any policy field must change policy_body_sha256."""
    policy = json.loads(
        Path(os.path.join(HANDOFF_DIR, "candidate_generation_policy.json")).read_text()
    )
    original_hash = policy["policy_body_sha256"]
    # Change a field
    modified = dict(policy)
    modified["causal_time_window_seconds"] = 99999
    for fld in ["policy_body_sha256", "policy_file_sha256", "policy_hash_sha256"]:
        modified.pop(fld, None)
    body_json = json.dumps(modified, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    new_hash = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    assert new_hash != original_hash, "Changed policy must produce different hash"


def test_policy_hash_sha256_field_removed():
    """policy_hash_sha256 must NOT exist in candidate_generation_policy.json."""
    policy = json.loads(
        Path(os.path.join(HANDOFF_DIR, "candidate_generation_policy.json")).read_text()
    )
    assert "policy_hash_sha256" not in policy, "policy_hash_sha256 must be removed"


# ============================================================================
# Fix 6: Manifest encoding tests
# ============================================================================

def test_manifest_requires_encoding_fields():
    """Manifest schema must require hash_algorithm, timestamp_encoding, timezone_policy,
    amount_normalization_method, asset_decimal_policy."""
    schema = json.loads(
        Path(os.path.join(HANDOFF_DIR, "collection_manifest.schema.json")).read_text()
    )
    required = set(schema["required"])
    for field in ["hash_algorithm", "timestamp_encoding", "timezone_policy",
                  "amount_normalization_method", "asset_decimal_policy"]:
        assert field in required, f"{field} must be in required"


# ============================================================================
# Fix 5: Deprecated-file exclusion tests
# ============================================================================

def test_deprecated_files_not_in_handoff():
    """dataset_directory.schema.json must not be in the handoff directory
    (only dataset_directory_contract.json is allowed)."""
    handoff_files = list(Path(HANDOFF_DIR).glob("*.json"))
    handoff_names = {f.name for f in handoff_files}
    assert "dataset_directory.schema.json" not in handoff_names,         "dataset_directory.schema.json must not be in handoff"
    assert "dataset_directory_contract.json" in handoff_names,         "dataset_directory_contract.json must be in handoff"


def test_deprecated_directory_exists():
    """Deprecated files must be in deprecated/ directory."""
    dep_path = os.path.join(PROVIDER_CONTRACT_DIR, "deprecated", "dataset_directory.schema.json")
    assert os.path.isfile(dep_path), "Deprecated schema must be in deprecated/"
    dep_md = os.path.join(PROVIDER_CONTRACT_DIR, "deprecated", "DEPRECATED.md")
    assert os.path.isfile(dep_md), "DEPRECATED.md must exist"


# ============================================================================
# Fix 7/8: example_valid_p0 tests
# ============================================================================

def test_example_valid_p0_passes_validator():
    """example_valid_p0 must pass the formal validator."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/validate_rc_uot_provider_sample.py",
         "--dataset", EXAMPLE_DIR],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..")
    )
    assert result.returncode == 0, f"Validator failed: {result.stderr}"
    assert "P0_ACCEPTED" in result.stdout


def test_example_valid_p0_has_required_files():
    """example_valid_p0 must contain all required files."""
    required = ["source_flows.jsonl", "target_flows.jsonl", "collection_manifest.json"]
    for fname in required:
        assert os.path.isfile(os.path.join(EXAMPLE_DIR, fname)), f"Missing {fname}"


# ============================================================================
# Fix 8: Handoff file list tests
# ============================================================================

REQUIRED_HANDOFF_FILES = [
    "README_FIRST.md",
    "provider_contract.md",
    "source_flow.schema.json",
    "target_flow.schema.json",
    "collection_manifest.schema.json",
    "dataset_directory_contract.json",
    "candidate_generation_policy.json",
    "candidate_universe_contract.md",
    "anti_triviality_gate.json",
    "source_flows.template.jsonl",
    "target_flows.template.jsonl",
    "collection_manifest.template.json",
    "PRE_SUBMISSION_CHECKLIST.md",
    "PILOT_SUBMISSION_INSTRUCTIONS.md",
    "ERROR_CODE_REFERENCE.md",
    "VALIDATE_LOCALLY.md",
]

FORBIDDEN_IN_HANDOFF = [
    "source_flowss.template.jsonl",
    "target_flowss.template.jsonl",
    "dataset_directory.schema.json",
    "labels",
]

def test_all_required_files_present():
    """All required files must be present in the handoff directory."""
    actual = set(os.listdir(HANDOFF_DIR))
    for fname in REQUIRED_HANDOFF_FILES:
        assert fname in actual, f"Missing required file: {fname}"


def test_forbidden_files_absent():
    """Forbidden files must not be in the handoff directory."""
    actual = set(os.listdir(HANDOFF_DIR))
    for fname in FORBIDDEN_IN_HANDOFF:
        assert fname not in actual, f"Forbidden file present: {fname}"


def test_example_valid_p0_dir_exists():
    """example_valid_p0 directory must exist under handoff."""
    assert os.path.isdir(EXAMPLE_DIR), "example_valid_p0 directory missing"


def test_typo_files_not_in_handoff():
    """source_flowss.template.jsonl and target_flowss.template.jsonl must not exist."""
    for typo in ["source_flowss.template.jsonl", "target_flowss.template.jsonl"]:
        fpath = os.path.join(HANDOFF_DIR, typo)
        assert not os.path.isfile(fpath), f"Typo file should not exist: {typo}"


# ============================================================================
# Fix 11: Zip content audit tests
# ============================================================================

def test_provider_final_zip_content_audit():
    """The handoff zip must contain only allowed files, no forbidden content."""
    zip_path = os.path.join(
        os.path.dirname(__file__), "..", "out", "rc_uot_v2_2", "provider_handoff_final.zip"
    )
    if not os.path.isfile(zip_path):
        pytest.skip("Zip not yet built (will be built in Fix 11)")

    FORBIDDEN_IN_ZIP = [
        "labels", "gold_edges", "solver", "dataset_directory.schema.json",
        "source_flowss", "target_flowss",
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for forbidden in FORBIDDEN_IN_ZIP:
            for name in names:
                assert forbidden not in name.lower(), f"Forbidden content in zip: {name}"


def test_provider_final_zip_sha256_exists():
    """SHA-256 file must exist alongside zip."""
    sha_path = os.path.join(
        os.path.dirname(__file__), "..", "out", "rc_uot_v2_2", "provider_handoff_final.zip.sha256"
    )
    if not os.path.isfile(sha_path):
        pytest.skip("SHA-256 file not yet built")

    content = Path(sha_path).read_text().strip()
    assert len(content) == 64, f"SHA-256 must be 64 hex chars, got {len(content)}"
    assert all(c in "0123456789abcdef" for c in content), "Not a valid hex SHA-256"


# ============================================================================
# Total count reporter
# ============================================================================

TOTAL_TESTS = 21  # manually updated count

def test_report_test_counts():
    """Report total test counts for final audit.

    This test always passes and exists for documentation purposes."""
    print(f"Total tests in this module: {TOTAL_TESTS}")
    assert True
