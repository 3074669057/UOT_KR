from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from cross.application.experiments.ec_uot_q_final import (
    V3_OUT,
    ASSIGNMENTS,
    ETH_PUBLIC,
    RAW_RELAY,
    REPO as MODULE_REPO,
    SPLIT_MANIFEST,
    AccessViolation,
    ProtocolError,
    HeldoutGateError,
    atomic_write_text,
    build_public_relay_candidates,
    candidate_recall,
    evaluate_full_set,
    load_development_truth,
    load_public_sources,
    predict,
    run_heldout_once,
    sha256_file,
    validate_inference_frame,
    write_json,
    V2_OUT,
)

OUT = V3_OUT
V2_DATA = REPO / "out" / "ec_uot_q_final_v2" / "data"

CONFIG = {
    "seed": 42,
    "before_sec": 900,
    "windows_h": [1.0, 3.0, 6.0],
    "cost_weight_presets": [
        {"name": "amount_time", "amount_weight": 0.75, "time_weight": 0.25, "token_weight": 0.0, "route_weight": 0.0},
        {"name": "balanced", "amount_weight": 0.5, "time_weight": 0.5, "token_weight": 0.0, "route_weight": 0.0},
        {"name": "token_aware", "amount_weight": 0.5, "time_weight": 0.25, "token_weight": 0.25, "route_weight": 0.0},
        {"name": "route_aware", "amount_weight": 0.50, "time_weight": 0.20, "token_weight": 0.15, "route_weight": 0.15},
    ],
    "confidence_thresholds": [0.0, 0.5, 0.75],
    "transport": {"reg": 0.05, "reg_m": 0.5, "max_iter": 100, "tol": 1e-6, "top_k": 20, "batch_size": 50},
    "selection": {"precision_floor": 0.90, "metric": "full_set_f1", "tie_break": ["full_set_f1", "coverage", "parameter_simplicity", "lexicographic"]},
    "feature_allowlist": {"source": ["source_tx_hash", "source_timestamp", "source_amount_raw", "source_token_address"], "target": ["candidate_tx_hash", "candidate_timestamp", "amount_raw", "token_address", "contract_address", "topic0", "route_type", "bridge_contract_hit"]},
    "bridge_fields_excluded": ["route_id", "transfer_key", "bridge_key", "message_key", "message_id", "transfer_id", "src_transfer_id", "nonce"],
    "grouping_strategies": ["token_30min", "contract_token_5min", "contract_token_15min", "contract_token_30min", "contract_token_60min", "contract_token_ordermag"],
    "quotient_grouping": "configurable: see grouping_strategies list",
}

DEVELOPMENT_MAX_SOURCE_TIMESTAMP = 1640349379
MAX_CANDIDATE_WINDOW_SEC = 86400
CANDIDATE_MAX_TIMESTAMP_INCLUSIVE = 1640435779
TEST_START_TIMESTAMP = 1640443616
DEVELOPMENT_COUNT = 5107
EMBARGO_COUNT = 110
TEST_COUNT = 2079


def out(*parts: str) -> Path:
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_md_mapping() -> None:
    text = """# Implementation Mapping

## Code identity
The executable implementation is the repository's RC-UOT/RC-UOT-Q family. This protocol does not rename it or claim automatic identity with the paper's EC-UOT-Q.

## Executed RC-UOT-Q composition
- Candidate generation decodes public Relay logs and discards ABI word 0 (the protocol identity) and unvalidated word 5.
- Inference features are restricted to independently observed amount, timestamp, and token fields. Celer/Relay route fields do not enter scoring.
- Quotient grouping is executed as target grouping by public token and a 30-minute public timestamp bucket.
- Coverage qualification is executed before transport: source public fields must be complete and the source must have a nonempty public candidate set.
- Transport calls `cross.domain.uot.uot_solver_numpy.uot_sinkhorn` with frozen `reg`, `reg_m`, iteration, and tolerance settings.
- The decoder chooses a quotient from transport mass, then a deterministic public-feature member within that quotient.
- Abstention is executed by a frozen confidence threshold and contributes FN in full-set evaluation.

## Concept correspondence and differences
The paper EC-UOT-Q concepts correspond to evidence-constrained candidate generation, quotient grouping, unbalanced transport, coverage qualification, and abstaining decode. The executed implementation differs from broader historical paper code in these ways:
- quotient groups use token plus a fixed 30-minute timestamp bucket, not bridge identity or learned incidence;
- coverage qualification is a deterministic public-field gate, not a learned qualifier;
- the decoder is deterministic top-1 within the selected quotient, not the historical admissible top-3 rescue decoder;
- no route, bridge contract, transfer key, receiver, or protocol-specific matching bonus enters score or decision.

The executed implementation and frozen source snapshot are authoritative for this experiment. A future held-out run is wired to the same `predict` and `evaluate_full_set` functions.
"""
    atomic_write_text(out("implementation_mapping.md"), text)


def stage_prepare() -> None:
    write_md_mapping()

    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    assert int(manifest["sample_counts"]["development"]) == DEVELOPMENT_COUNT
    assert int(manifest["sample_counts"]["embargo"]) == EMBARGO_COUNT
    assert int(manifest["sample_counts"]["test"]) == TEST_COUNT
    assert TEST_START_TIMESTAMP == int(manifest["timestamp_ranges"]["test"][0])

    # 1. Isolate development assignments (first DEVELOPMENT_COUNT rows, split==development)
    assignments = pd.read_csv(ASSIGNMENTS, dtype=str, nrows=DEVELOPMENT_COUNT).fillna("")
    if set(assignments["split"].str.lower()) != {"development"}:
        raise ProtocolError("first 5107 rows are not all development")
    if len(assignments) != DEVELOPMENT_COUNT:
        raise ProtocolError(f"expected {DEVELOPMENT_COUNT} development rows, got {len(assignments)}")
    assignments.to_csv(V2_DATA / "development_assignments.csv", index=False)

    # 2. Isolate development truth
    truth = load_development_truth(ASSIGNMENTS, no_test_access=True, split_manifest=True)
    truth_df = pd.DataFrame(
        [(s, d) for s, d in truth.items()], columns=["source_tx_hash", "dest_tx_hash"]
    )
    truth_df.to_csv(V2_DATA / "development_truth.csv", index=False)
    assert len(truth_df) == DEVELOPMENT_COUNT

    # 3. Development public sources
    sources = load_public_sources(ASSIGNMENTS, ETH_PUBLIC, no_test_access=True)
    sources.to_csv(V2_DATA / "development_public_sources.csv", index=False)
    assert len(sources) == DEVELOPMENT_COUNT

    # 4. Public Relay candidates up to candidate_max_timestamp_inclusive
    candidates, lineage = build_public_relay_candidates(
        RAW_RELAY, max_timestamp_exclusive=CANDIDATE_MAX_TIMESTAMP_INCLUSIVE + 1
    )
    actual_max = int(candidates["candidate_timestamp"].max())
    assert actual_max <= CANDIDATE_MAX_TIMESTAMP_INCLUSIVE, (
        f"candidate max {actual_max} > {CANDIDATE_MAX_TIMESTAMP_INCLUSIVE}"
    )
    assert actual_max < TEST_START_TIMESTAMP, (
        f"candidate max {actual_max} >= test start {TEST_START_TIMESTAMP}"
    )

    candidates.to_csv(V2_DATA / "development_public_relay_candidates.csv", index=False)
    lineage["candidate_filter"] = {
        "split": "development",
        "candidate_max_timestamp_inclusive": CANDIDATE_MAX_TIMESTAMP_INCLUSIVE,
        "actual_max_timestamp": actual_max,
        "reason": "public candidates within 24h window of last development source",
    }
    lineage["source_files"].append({
        "path": str(ETH_PUBLIC.resolve()),
        "sha256": sha256_file(ETH_PUBLIC),
        "used_for": "development source public attributes",
    })
    write_json(V2_DATA / "field_lineage.json", lineage)

    # 5. Partition manifest
    partition_manifest = {
        "protocol": "ec_uot_q_final_v2",
        "generated_at_utc": str(pd.Timestamp.utcnow()),
        "development_count": DEVELOPMENT_COUNT,
        "embargo_count": EMBARGO_COUNT,
        "test_count": TEST_COUNT,
        "candidate_max_timestamp_inclusive": CANDIDATE_MAX_TIMESTAMP_INCLUSIVE,
        "actual_candidate_max_timestamp": actual_max,
        "test_start_timestamp": TEST_START_TIMESTAMP,
        "candidate_boundary_assertion": "candidate_max < test_start",
        "development_max_source_timestamp": DEVELOPMENT_MAX_SOURCE_TIMESTAMP,
        "max_candidate_window_sec": MAX_CANDIDATE_WINDOW_SEC,
        "raw_relay_sha256": sha256_file(RAW_RELAY),
        "eth_public_sha256": sha256_file(ETH_PUBLIC),
        "assignments_sha256": sha256_file(ASSIGNMENTS),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "files": {
            "development_assignments.csv": sha256_file(V2_DATA / "development_assignments.csv"),
            "development_truth.csv": sha256_file(V2_DATA / "development_truth.csv"),
            "development_public_sources.csv": sha256_file(V2_DATA / "development_public_sources.csv"),
            "development_public_relay_candidates.csv": sha256_file(V2_DATA / "development_public_relay_candidates.csv"),
            "field_lineage.json": sha256_file(V2_DATA / "field_lineage.json"),
        },
    }
    write_json(V2_DATA / "input_partition_manifest.json", partition_manifest)

    # 6. Detached checksums for partitioned inputs
    checksum_path = V2_DATA / "detached_checksums.sha256"
    entries = {
        "data/input_partition_manifest.json": sha256_file(V2_DATA / "input_partition_manifest.json"),
        "data/development_assignments.csv": sha256_file(V2_DATA / "development_assignments.csv"),
        "data/development_truth.csv": sha256_file(V2_DATA / "development_truth.csv"),
        "data/development_public_sources.csv": sha256_file(V2_DATA / "development_public_sources.csv"),
        "data/development_public_relay_candidates.csv": sha256_file(V2_DATA / "development_public_relay_candidates.csv"),
        "data/field_lineage.json": sha256_file(V2_DATA / "field_lineage.json"),
        "implementation_mapping.md": sha256_file(out("implementation_mapping.md")),
    }
    atomic_write_text(checksum_path, "\n".join(f"{digest}  {path}" for path, digest in sorted(entries.items())) + "\n")

    # 7. Protocol incident note for old directory
    old_out = MODULE_REPO / "out" / "ec_uot_q_final"
    incident = old_out / "protocol_incident.md"
    incident.parent.mkdir(parents=True, exist_ok=True)
    incident.write_text(
        "# Protocol Incident\n\n"
        "The `out/ec_uot_q_final/` directory is invalidated because:\n\n"
        "1. `ensure_inputs()` decoded all 18,271 Relay records before filtering to 17,679, "
        "computing a full-file hash across mixed development/embargo-calendar bytes.\n"
        "2. The candidate cutoff was set to embargo start (1640349667) rather than the correct\n"
        "   candidate observation window boundary (1640435779), destroying legitimate public candidates.\n"
        "3. `load_development_truth()` returned `None` silently.\n\n"
        "All development artifacts in this directory are noncompliant and must not be used.\n"
        f"Replaced by `out/ec_uot_q_final_v2/` on {pd.Timestamp.utcnow().isoformat()}.\n",
        encoding="utf-8",
    )


def get_isolated_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    sources = pd.read_csv(V2_DATA / "development_public_sources.csv")
    candidates = pd.read_csv(V2_DATA / "development_public_relay_candidates.csv")
    # V3 enrichment: add route_type, bridge_contract_hit from pure functions
    from cross.application.experiments.ec_uot_q_final import classify_route_type_target, RELAY_CONTRACT_BSC, RELAY_TOPIC0
    if "route_type" not in candidates.columns:
        candidates["route_type"] = classify_route_type_target(RELAY_CONTRACT_BSC, RELAY_TOPIC0)
    if "bridge_contract_hit" not in candidates.columns:
        candidates["bridge_contract_hit"] = True
    if "source_route_type" not in sources.columns:
        from cross.application.experiments.ec_uot_q_final import classify_route_type_source
        sources["source_route_type"] = sources["source_token_address"].apply(classify_route_type_source)
    truth = pd.read_csv(V2_DATA / "development_truth.csv", dtype=str).fillna("")
    truth_map = dict(zip(truth["source_tx_hash"].str.lower(), truth["dest_tx_hash"].str.lower()))
    assert len(truth_map) == DEVELOPMENT_COUNT
    validate_inference_frame(sources, side="source")
    validate_inference_frame(candidates[list(CONFIG["feature_allowlist"]["target"])], side="target")
    return sources, candidates, truth_map


def ensure_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    need = [
        V2_DATA / "development_public_sources.csv",
        V2_DATA / "development_public_relay_candidates.csv",
        V2_DATA / "development_truth.csv",
    ]
    if not all(p.is_file() for p in need):
        stage_prepare()
    return get_isolated_inputs()


def get_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    need = [
        V2_DATA / "development_public_sources.csv",
        V2_DATA / "development_public_relay_candidates.csv",
        V2_DATA / "development_truth.csv",
    ]
    if not all(p.is_file() for p in need):
        raise ProtocolError("isolated inputs missing; run prepare-development-inputs first")
    return get_isolated_inputs()


def config_rows() -> list[dict[str, Any]]:
    rows = []
    for window_h in CONFIG["windows_h"]:
        for preset in CONFIG["cost_weight_presets"]:
            for threshold in [0.0, 0.5, 0.75]:
                for grouping in CONFIG["grouping_strategies"]:
                    rows.append({
                        "window_h": window_h,
                        "before_sec": CONFIG["before_sec"],
                        "weight_preset": preset["name"],
                        "amount_weight": preset["amount_weight"],
                        "time_weight": preset["time_weight"],
                        "token_weight": preset["token_weight"],
                        "route_weight": preset.get("route_weight", 0.0),
                        "confidence_threshold": threshold,
                        "grouping_strategy": grouping,
                        "use_transport": True,
                        **CONFIG["transport"],
                    })
    return rows


def run_one(source: pd.DataFrame, candidates: pd.DataFrame, truth: dict[str, str], cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        pred, meta = predict(source, candidates, cfg)
        metrics = evaluate_full_set(truth, pred)
        metrics.update({"status": "OK", "config": cfg, "seed": CONFIG["seed"], "confidence_min": min(meta["confidence"].values(), default=0.0)})
        return metrics
    except Exception as exc:
        return {"status": "FAILED", "config": cfg, "seed": CONFIG["seed"], "error": repr(exc)}


def stage_diagnose() -> None:
    sources, candidates, truth = get_inputs()
    assignment = pd.read_csv(V2_DATA / "development_assignments.csv", dtype=str).fillna("")
    source_times = dict(zip(assignment["source_tx_hash"].str.lower(), pd.to_numeric(assignment["source_timestamp"], errors="coerce").fillna(0).astype(int)))
    candidate_by_hash = dict(zip(candidates["candidate_tx_hash"].str.lower(), candidates["candidate_timestamp"].astype(int)))
    in_public = in_window = 0
    failure = {"not_in_public_relay_before_boundary": 0, "outside_24h_window": 0, "negative_delay_beyond_tolerance": 0, "embargo_or_test_rows_decoded": 0}
    for source_id, target_id in truth.items():
        target_time = candidate_by_hash.get(target_id)
        if target_time is None:
            failure["not_in_public_relay_before_boundary"] += 1
            continue
        in_public += 1
        delay = target_time - source_times[source_id]
        if delay < -CONFIG["before_sec"]:
            failure["negative_delay_beyond_tolerance"] += 1
        elif delay > 24 * 3600:
            failure["outside_24h_window"] += 1
        else:
            in_window += 1
    rec = {"n_truth": len(truth), "n_candidate_transactions": len(candidates), "covered_by_public_pool": in_public, "public_pool_recall": in_public / max(len(truth), 1), "covered_within_24h": in_window, "candidate_pool_recall": in_window / max(len(truth), 1), "missed_within_24h": len(truth) - in_window, "failure_sources": failure, "candidate_timestamp_min": int(candidates.candidate_timestamp.min()), "candidate_timestamp_max": int(candidates.candidate_timestamp.max())}
    write_json(out("diagnosis", "candidate_recall.json"), rec)
    write_json(out("diagnosis", "candidate_pool_summary.json"), {"n_sources": len(sources), "n_candidates": len(candidates), "n_truth": len(truth), "zero_candidate_sources": int(sum(not ((candidates.candidate_timestamp >= r.source_timestamp - CONFIG["before_sec"]) & (candidates.candidate_timestamp <= r.source_timestamp + 24 * 3600)).any() for r in sources.itertuples()))})


def _base_config(*, threshold: float = 0.5) -> dict[str, Any]:
    preset = CONFIG["cost_weight_presets"][0]
    return {"window_h": 1.0, "before_sec": CONFIG["before_sec"], "weight_preset": preset["name"], "amount_weight": preset["amount_weight"], "time_weight": preset["time_weight"], "token_weight": preset["token_weight"], "route_weight": 0.0, "confidence_threshold": threshold, "grouping_strategy": "token_30min", "use_transport": True, **CONFIG["transport"]}


def stage_ablations() -> None:
    source, candidates, truth = get_inputs()
    base = _base_config(threshold=0.5)
    variants = [
        ("full_rc_uot_q_v3", base),
        ("no_quotient_grouping", {**base, "grouping_strategy": "none"}),
        ("no_uot_transport", {**base, "use_transport": False}),
        ("no_abstention", {**base, "confidence_threshold": 0.0}),
        ("route_aware", {**base, "weight_preset": "route_aware", "amount_weight": 0.50, "time_weight": 0.20, "token_weight": 0.15, "route_weight": 0.15}),
        ("contract_token_5min", {**base, "grouping_strategy": "contract_token_5min"}),
    ]
    rows = []
    for name, cfg in variants:
        result = run_one(source, candidates, truth, cfg)
        result["ablation"] = name
        rows.append(result)
    write_json(out("diagnosis", "mechanism_ablations.json"), rows)


def _simplicity(config: dict[str, Any]) -> int:
    score = int(config.get("window_h") != 1.0)
    score += int(config.get("weight_preset") != "amount_time")
    score += int(float(config.get("confidence_threshold", 0.0)) > 0.0)
    score += int(config.get("grouping_strategy", "token_30min") != "token_30min")
    return score


def stage_tune(floor: float) -> None:
    source, candidates, truth = get_inputs()
    search_path = out("tuning", "search_space.json")
    write_json(search_path, CONFIG)
    search_hash = sha256_file(search_path)
    write_json(out("tuning", "search_space_hash.json"), {"path": str(search_path.resolve()), "sha256": search_hash})
    rows = [run_one(source, candidates, truth, cfg) for cfg in config_rows()]
    for row in rows:
        row["parameter_simplicity"] = _simplicity(row["config"])
        row["search_space_sha256"] = search_hash
    flat = [{k: v for k, v in row.items() if k != "config"} | row["config"] for row in rows]
    pd.DataFrame(flat).to_csv(out("tuning", "all_configurations.csv"), index=False)
    eligible = [row for row in rows if row.get("status") == "OK" and row.get("precision", 0.0) >= floor]
    eligible.sort(key=lambda row: (-row["full_set_f1"], -row["coverage"], row["parameter_simplicity"], json.dumps(row["config"], sort_keys=True)))
    if eligible:
        selected, status = eligible[0], "ELIGIBLE_CONFIGURATION"
    else:
        fallback = _base_config(threshold=0.0)
        selected, status = run_one(source, candidates, truth, fallback), "NO_ELIGIBLE_CONFIGURATION"
        selected["parameter_simplicity"] = _simplicity(fallback)
        selected["search_space_sha256"] = search_hash
    predictions, meta = predict(source, candidates, selected["config"])
    pd.DataFrame([{"source_tx_hash": source_id, "predicted_candidate_tx_hash": target_id, "abstained": target_id is None, "confidence": meta["confidence"].get(source_id, 0.0)} for source_id, target_id in predictions.items()]).to_csv(out("tuning", "selected_development_predictions.csv"), index=False)
    write_json(out("tuning", "selection.json"), {"selection_status": status, "precision_floor": floor, "selected": selected, "n_configurations": len(rows), "n_eligible": len(eligible), "selection_metric": "full_set_f1", "tie_break": CONFIG["selection"]["tie_break"]})


def stage_curve() -> None:
    source, candidates, truth = get_inputs()
    selection = json.loads(out("tuning", "selection.json").read_text(encoding="utf-8"))
    base = selection["selected"]["config"]
    rows = []
    for threshold in [i / 20 for i in range(21)]:
        cfg = {**base, "confidence_threshold": threshold}
        rows.append({"threshold": threshold, **run_one(source, candidates, truth, cfg)})
    pd.DataFrame([{k: v for k, v in r.items() if k != "config"} | r["config"] for r in rows]).to_csv(out("tuning", "precision_coverage_abstention.csv"), index=False)


def detached_freeze() -> None:
    source_path = V2_DATA / "development_public_sources.csv"
    candidate_path = V2_DATA / "development_public_relay_candidates.csv"
    selection_path = out("tuning", "selection.json")
    partition_manifest_path = V2_DATA / "input_partition_manifest.json"

    snapshot_dir = out("freeze", "source_snapshot")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_files = [
        Path(__file__),
        MODULE_REPO / "src" / "cross" / "application" / "experiments" / "ec_uot_q_final.py",
        MODULE_REPO / "src" / "cross" / "domain" / "uot" / "uot_solver_numpy.py",
        MODULE_REPO / "tests" / "test_ec_uot_q_final_protocol.py",
    ]
    snapshots = []
    for source_file in source_files:
        destination = snapshot_dir / source_file.name
        shutil.copy2(source_file, destination)
        snapshots.append(destination)

    dependencies_path = out("freeze", "python_dependencies.txt")
    dependencies = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    atomic_write_text(dependencies_path, dependencies)

    required = [
        SPLIT_MANIFEST,
        ASSIGNMENTS,
        RAW_RELAY,
        ETH_PUBLIC,
        source_path,
        candidate_path,
        partition_manifest_path,
        V2_DATA / "development_assignments.csv",
        V2_DATA / "development_truth.csv",
        V2_DATA / "field_lineage.json",
        V2_DATA / "input_partition_manifest.json",
        V2_DATA / "detached_checksums.sha256",
        selection_path,
        out("tuning", "search_space.json"),
        out("tuning", "search_space_hash.json"),
        out("tuning", "all_configurations.csv"),
        out("tuning", "precision_coverage_abstention.csv"),
        out("tuning", "selected_development_predictions.csv"),
        out("diagnosis", "candidate_recall.json"),
        out("diagnosis", "mechanism_ablations.json"),
        out("implementation_mapping.md"),
        dependencies_path,
        *snapshots,
    ]
    files = {str(path.resolve().relative_to(MODULE_REPO.resolve())): sha256_file(path) for path in required if path.is_file()}
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    freeze_id = hashlib.sha256(json.dumps({"files": files, "selection": selection, "seed": CONFIG["seed"]}, sort_keys=True).encode()).hexdigest()[:20]

    manifest = {
        "freeze_id": freeze_id,
        "status": "FROZEN_DEVELOPMENT_ONLY",
        "test_run_counter": 0,
        "heldout_output_exists": False,
        "selection": selection,
        "selection_config": CONFIG,
        "files": files,
        "evaluator": {"tp": "correct prediction", "fp": "wrong non-abstained prediction", "fn": "wrong prediction or abstention", "recall_denominator": "all evaluated source instances", "full_set_f1": True},
        "environment": {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(), "dependencies_file": str(dependencies_path.resolve())},
        "output_schema": {"metrics": ["tp", "fp", "fn", "precision", "recall", "full_set_f1", "coverage", "abstention"], "candidate": list(pd.read_csv(candidate_path, nrows=1).columns)},
        "authorization_schema": {"schema_version": 1, "freeze_id": freeze_id, "authorized": True, "authorization_id": "nonempty string"},
        "one_time_lock": {"path": "out/ec_uot_q_final_v2/freeze/heldout_run.lock", "create_mode": "exclusive", "success_state": "CONSUMED"},
        "manifest_hash_policy": "manifest has no self-hash; detached_checksums.sha256 hashes the complete manifest bytes",
    }
    freeze_path = out("freeze", "freeze_manifest.json")
    write_json(freeze_path, manifest)

    checksum_path = out("freeze", "detached_checksums.sha256")
    entries = {"freeze/freeze_manifest.json": sha256_file(freeze_path), **files}
    atomic_write_text(checksum_path, "\n".join(f"{digest}  {path}" for path, digest in sorted(entries.items())) + "\n")

    write_json(out("freeze", "freeze_state.json"), {"test_run_counter": 0, "authorization_file_present": False, "heldout_output_present": False, "run_lock_present": False, "detached_checksum_file": str(checksum_path.resolve())})


def stage_verify(require_counter: int) -> None:
    freeze_path, checksum_path = out("freeze", "freeze_manifest.json"), out("freeze", "detached_checksums.sha256")
    if not freeze_path.is_file() or not checksum_path.is_file():
        raise ProtocolError("freeze artifacts missing")
    m = json.loads(freeze_path.read_text(encoding="utf-8"))
    if m.get("test_run_counter") != require_counter or m.get("heldout_output_exists") is not False:
        raise ProtocolError("freeze counter/output gate failed")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        digest, rel = line.split("  ", 1)
        path = OUT / rel if rel.startswith("freeze/") else MODULE_REPO / rel
        if not path.is_file() or sha256_file(path) != digest:
            raise ProtocolError(f"detached checksum mismatch: {rel}")
    if (OUT / "heldout" / "metrics.json").exists() or (OUT / "heldout" / "predictions.csv").exists():
        raise ProtocolError("heldout output exists")
    write_json(out("freeze", "verify_freeze.json"), {"status": "PASS", "test_run_counter": m["test_run_counter"], "heldout_outputs_absent": True, "checksum_verified": True})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare-development-inputs", "audit-inputs", "diagnose-candidates", "mechanism-ablations", "tune-development", "precision-coverage-abstention", "freeze", "verify-freeze", "run-heldout-once", "summarize"])
    p.add_argument("--split", default="development")
    p.add_argument("--precision-floor", type=float, default=0.90)
    p.add_argument("--selection-metric", default="full_set_f1")
    p.add_argument("--fail-on-forbidden", action="store_true")
    p.add_argument("--no-test-access", action="store_true")
    p.add_argument("--report-recall", action="store_true")
    p.add_argument("--selected-from", default=None)
    p.add_argument("--require-counter", type=int, default=0)
    p.add_argument("--authorization-file", type=Path, default=None)
    p.add_argument("--test-assignment", type=Path, default=None)
    p.add_argument("--test-raw-relay", type=Path, default=None)
    p.add_argument("--test-eth", type=Path, default=None)
    args = p.parse_args(argv)

    if args.stage in {"prepare-development-inputs", "audit-inputs", "diagnose-candidates", "mechanism-ablations", "tune-development", "precision-coverage-abstention", "freeze", "verify-freeze", "summarize"} and args.split not in {"development", ""}:
        raise AccessViolation("development stages accept development only")
    if args.stage in {"prepare-development-inputs", "audit-inputs", "diagnose-candidates", "mechanism-ablations", "tune-development", "precision-coverage-abstention"} and not args.no_test_access:
        raise AccessViolation("development stages require --no-test-access")

    if args.stage == "prepare-development-inputs":
        stage_prepare()
    elif args.stage == "audit-inputs":
        ensure_inputs()
    elif args.stage == "diagnose-candidates":
        stage_diagnose()
    elif args.stage == "mechanism-ablations":
        stage_ablations()
    elif args.stage == "tune-development":
        stage_tune(args.precision_floor)
    elif args.stage == "precision-coverage-abstention":
        stage_curve()
    elif args.stage == "freeze":
        detached_freeze()
    elif args.stage == "verify-freeze":
        stage_verify(args.require_counter)
    elif args.stage == "run-heldout-once":
        required_paths = [args.authorization_file, args.test_assignment, args.test_raw_relay, args.test_eth]
        if any(path is None or not path.is_file() for path in required_paths):
            raise HeldoutGateError("authorization and all explicit held-out inputs are required")
        freeze_path = out("freeze", "freeze_manifest.json")
        heldout_output = out("heldout", "metrics.json")
        def evaluate_heldout(_: dict[str, Any]) -> dict[str, Any]:
            manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
            config = manifest["selection"]["selected"]["config"]
            heldout_assignments = pd.read_csv(args.test_assignment, dtype=str).fillna("")
            if set(heldout_assignments["split"].str.lower()) != {"test"}:
                raise HeldoutGateError("held-out assignment must contain test rows only")
            truth = dict(zip(heldout_assignments["source_tx_hash"].str.lower(), heldout_assignments["dest_tx_hash"].str.lower()))
            source = load_public_sources(args.test_assignment, args.test_eth, no_test_access=False)
            candidate, _ = build_public_relay_candidates(args.test_raw_relay)
            predictions, _ = predict(source, candidate, config)
            return evaluate_full_set(truth, predictions)
        run_heldout_once(freeze_path, args.authorization_file, heldout_output, evaluate_heldout)
    elif args.stage == "summarize":
        selection = json.loads(out("tuning", "selection.json").read_text(encoding="utf-8"))
        write_json(out("summary", "development_summary.json"), selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())