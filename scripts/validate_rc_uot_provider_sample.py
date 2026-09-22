"""RC-UOT-v2.2 Provider Sample Validator (v4.3a.1)"""
import argparse, csv, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import jsonschema
    from jsonschema import Draft202012Validator, FormatChecker
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

EXIT_ACCEPTED = 0
EXIT_REJECTED_DIRECTORY = 10
EXIT_REJECTED_SCHEMA = 10
EXIT_REJECTED_SEMANTICS = 11
EXIT_REJECTED_LABEL_LEAKAGE = 12
EXIT_REJECTED_OVERLAP = 13
EXIT_REJECTED_TRIVIAL = 14
EXIT_REJECTED_INCOMPLETE = 15
EXIT_REJECTED_FEATURE = 16

FORBIDDEN_NAMES = ["labels", "gold_edges.jsonl", "gold_edges.csv", "matches.jsonl",
                   "predictions.jsonl", "solver_scores.jsonl"]


class ValidationReport:
    def __init__(self):
        self.schema_errors = []
        self.semantic_errors = []
        self.decision = "P0_ACCEPTED"
        self.exit_code = EXIT_ACCEPTED
        self.checks = []
        self.manifest = None
        self.source_count = 0
        self.target_count = 0

    def add_check(self, name, passed, detail=""):
        self.checks.append({"check": name, "passed": passed, "detail": detail})

    def reject(self, code_name, exit_code):
        self.decision = code_name
        self.exit_code = exit_code


def _find_contract_dir():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "out", "rc_uot_v2_2", "provider_handoff_final"),
        os.path.join(os.getcwd(), "out", "rc_uot_v2_2", "provider_handoff_final"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "provider_contract.md")):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_rfc3339_utc(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _parse_unix_timestamp(ts, unit):
    try:
        if unit == "milliseconds":
            ts = float(ts) / 1000.0
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None


def validate_directory_structure(dataset_root, report):
    root = Path(dataset_root)
    if not root.is_dir():
        report.reject("P0_REJECTED_DIRECTORY", EXIT_REJECTED_DIRECTORY)
        report.add_check("directory_exists", False, "Not found: " + str(dataset_root))
        return False
    report.add_check("directory_exists", True)
    for fname in ["source_flows.jsonl", "target_flows.jsonl", "collection_manifest.json"]:
        fpath = root / fname
        if not fpath.is_file():
            report.reject("P0_REJECTED_DIRECTORY", EXIT_REJECTED_DIRECTORY)
            report.add_check("required_file_" + fname, False, "Missing: " + str(fpath))
            return False
        report.add_check("required_file_" + fname, True)
    for forbidden_name in FORBIDDEN_NAMES:
        fpath = root / forbidden_name
        if fpath.is_dir():
            report.reject("P0_REJECTED_LABEL_LEAKAGE", EXIT_REJECTED_LABEL_LEAKAGE)
            report.add_check("no_label_leakage", False, "Forbidden directory: " + str(fpath))
            return False
        if "." in forbidden_name and fpath.is_file():
            report.reject("P0_REJECTED_LABEL_LEAKAGE", EXIT_REJECTED_LABEL_LEAKAGE)
            report.add_check("no_label_leakage", False, "Forbidden file: " + str(fpath))
            return False
    report.add_check("no_label_leakage", True)
    return True


def validate_schema(dataset_root, contract_dir, report):
    if not HAS_JSONSCHEMA:
        report.add_check("schema_validation", False, "jsonschema library not installed")
        report.reject("P0_REJECTED_SCHEMA", EXIT_REJECTED_SCHEMA)
        return False
    formatter = FormatChecker()
    schemas = {}
    for name in ["source_flow", "target_flow", "collection_manifest"]:
        schema_path = os.path.join(contract_dir, name + ".schema.json")
        if not os.path.isfile(schema_path):
            report.add_check("schema_load_" + name, False, "Schema not found: " + schema_path)
            report.reject("P0_REJECTED_SCHEMA", EXIT_REJECTED_SCHEMA)
            return False
        schemas[name] = _load_json(schema_path)
        report.add_check("schema_load_" + name, True)

    root = Path(dataset_root)
    manifest = _load_json(str(root / "collection_manifest.json"))
    try:
        jsonschema.validate(manifest, schemas["collection_manifest"],
                          format_checker=formatter, cls=Draft202012Validator)
    except jsonschema.ValidationError as e:
        report.schema_errors.append({"file": "collection_manifest.json",
            "path": "/".join(str(p) for p in e.absolute_path), "message": e.message})
        report.reject("P0_REJECTED_SCHEMA", EXIT_REJECTED_SCHEMA)
        report.add_check("schema_manifest", False, str(e.message)[:120])
        return False
    report.manifest = manifest
    report.add_check("schema_manifest", True)

    # Validate source flows
    source_ids = set()
    sp = root / "source_flows.jsonl"
    with open(sp, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                report.schema_errors.append({"file": "source_flows.jsonl", "line": line_no, "path": "", "message": str(e)})
                report.reject("P0_REJECTED_SCHEMA", EXIT_REJECTED_SCHEMA)
                report.add_check("schema_source", False, "Line " + str(line_no) + ": JSON error")
                return False
            try:
                jsonschema.validate(rec, schemas["source_flow"], format_checker=formatter, cls=Draft202012Validator)
            except jsonschema.ValidationError as e:
                report.schema_errors.append({"file": "source_flows.jsonl", "line": line_no,
                    "path": "/".join(str(p) for p in e.absolute_path), "message": e.message})
                report.reject("P0_REJECTED_SCHEMA", EXIT_REJECTED_SCHEMA)
            sid = rec.get("source_flow_id")
            if sid in source_ids:
                report.semantic_errors.append({"file": "source_flows.jsonl", "line": line_no,
                    "message": "Duplicate source_flow_id: " + str(sid)})
            source_ids.add(sid)
    report.source_count = len(source_ids)
    report.add_check("schema_source", len(report.schema_errors) == 0, str(report.source_count) + " records")

    # Validate target flows
    target_ids = set()
    tp = root / "target_flows.jsonl"
    with open(tp, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                report.schema_errors.append({"file": "target_flows.jsonl", "line": line_no, "path": "", "message": str(e)})
                report.reject("P0_REJECTED_SCHEMA", EXIT_REJECTED_SCHEMA)
                report.add_check("schema_target", False, "Line " + str(line_no) + ": JSON error")
                return False
            try:
                jsonschema.validate(rec, schemas["target_flow"], format_checker=formatter, cls=Draft202012Validator)
            except jsonschema.ValidationError as e:
                report.schema_errors.append({"file": "target_flows.jsonl", "line": line_no,
                    "path": "/".join(str(p) for p in e.absolute_path), "message": e.message})
                report.reject("P0_REJECTED_SCHEMA", EXIT_REJECTED_SCHEMA)
            tid = rec.get("target_flow_id")
            if tid in target_ids:
                report.semantic_errors.append({"file": "target_flows.jsonl", "line": line_no,
                    "message": "Duplicate target_flow_id: " + str(tid)})
            target_ids.add(tid)
    report.target_count = len(target_ids)
    report.add_check("schema_target", len(report.schema_errors) == 0, str(report.target_count) + " records")
    return report.exit_code != EXIT_REJECTED_SCHEMA


def validate_semantics(dataset_root, contract_dir, report):
    if report.manifest is None:
        return True
    m = report.manifest
    ds = m.get("source_flow_count", -1)
    dt = m.get("target_flow_count", -1)
    dr = m.get("raw_record_count", -1)
    if ds >= 0 and ds != report.source_count:
        report.semantic_errors.append({"file": "collection_manifest.json",
            "message": "Declared source_flow_count (" + str(ds) + ") != actual (" + str(report.source_count) + ")"})
        report.reject("P0_REJECTED_SEMANTICS", EXIT_REJECTED_SEMANTICS)
    if dt >= 0 and dt != report.target_count:
        report.semantic_errors.append({"file": "collection_manifest.json",
            "message": "Declared target_flow_count (" + str(dt) + ") != actual (" + str(report.target_count) + ")"})
        report.reject("P0_REJECTED_SEMANTICS", EXIT_REJECTED_SEMANTICS)
    expected_raw = report.source_count + report.target_count
    if dr >= 0 and dr != expected_raw:
        report.semantic_errors.append({"file": "collection_manifest.json",
            "message": "Declared raw_record_count (" + str(dr) + ") != source+target (" + str(expected_raw) + ")"})
        report.reject("P0_REJECTED_SEMANTICS", EXIT_REJECTED_SEMANTICS)

    encoding = m.get("timestamp_encoding", "RFC3339_UTC")
    if encoding in ("unix_seconds", "unix_milliseconds") and "timestamp_unit" not in m:
        report.semantic_errors.append({"file": "collection_manifest.json",
            "message": "timestamp_unit required when timestamp_encoding=" + encoding})
        report.reject("P0_REJECTED_SEMANTICS", EXIT_REJECTED_SEMANTICS)

    tz = m.get("timezone_policy", "")
    if encoding == "RFC3339_UTC" and tz != "UTC":
        report.semantic_errors.append({"file": "collection_manifest.json",
            "message": "timezone_policy must be UTC for RFC3339_UTC, got " + str(tz)})
        report.reject("P0_REJECTED_SEMANTICS", EXIT_REJECTED_SEMANTICS)

    sv = m.get("schema_version", "")
    if not sv.startswith("v4.3"):
        report.semantic_errors.append({"file": "collection_manifest.json",
            "message": "schema_version must be v4.3a, got " + str(sv)})
        report.reject("P0_REJECTED_SEMANTICS", EXIT_REJECTED_SEMANTICS)

    scope = m.get("candidate_universe_scope", "")
    if scope == "matched_pairs_only":
        report.semantic_errors.append({"file": "collection_manifest.json",
            "message": "candidate_universe_scope must not be matched_pairs_only"})
        report.reject("P0_REJECTED_SEMANTICS", EXIT_REJECTED_SEMANTICS)

    # Verify policy hash
    policy_path = os.path.join(contract_dir, "candidate_generation_policy.json")
    if os.path.isfile(policy_path):
        policy = _load_json(policy_path)
        expected_body = policy.get("policy_body_sha256", "")
        pc = dict(policy)
        for fld in ("policy_body_sha256", "policy_file_sha256", "policy_hash_sha256"):
            pc.pop(fld, None)
        body_json = json.dumps(pc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        actual_body = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
        if expected_body and actual_body != expected_body:
            report.semantic_errors.append({"file": "candidate_generation_policy.json",
                "message": "policy_body_sha256 mismatch: expected " + expected_body + ", computed " + actual_body})
            report.reject("P0_REJECTED_SEMANTICS", EXIT_REJECTED_SEMANTICS)

    report.add_check("semantic_validation", len(report.semantic_errors) == 0)
    return len(report.semantic_errors) == 0


def validate_anti_triviality(dataset_root, contract_dir, report):
    gate_path = os.path.join(contract_dir, "anti_triviality_gate.json")
    if not os.path.isfile(gate_path):
        return True
    gate = _load_json(gate_path)
    min_src = gate.get("source_flow_count_min", 20)
    min_tgt = gate.get("target_flow_count_min", 20)
    if report.source_count < min_src:
        report.semantic_errors.append({"file": "source_flows.jsonl",
            "message": "source_flow_count (" + str(report.source_count) + ") < min (" + str(min_src) + ")"})
        report.reject("P0_REJECTED_TRIVIAL", EXIT_REJECTED_TRIVIAL)
    if report.target_count < min_tgt:
        report.semantic_errors.append({"file": "target_flows.jsonl",
            "message": "target_flow_count (" + str(report.target_count) + ") < min (" + str(min_tgt) + ")"})
        report.reject("P0_REJECTED_TRIVIAL", EXIT_REJECTED_TRIVIAL)
    report.add_check("anti_triviality_counts",
        report.source_count >= min_src and report.target_count >= min_tgt,
        "sources=" + str(report.source_count) + ", targets=" + str(report.target_count))
    return report.exit_code == EXIT_ACCEPTED


def write_reports(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rj = {"decision": report.decision, "exit_code": report.exit_code,
          "source_count": report.source_count, "target_count": report.target_count,
          "schema_error_count": len(report.schema_errors),
          "semantic_error_count": len(report.semantic_errors), "checks": report.checks}
    with open(os.path.join(output_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(rj, f, indent=2, ensure_ascii=False)

    lines = ["# Validation Report", "",
             "- **Decision**: " + report.decision,
             "- **Exit Code**: " + str(report.exit_code),
             "- **Sources**: " + str(report.source_count),
             "- **Targets**: " + str(report.target_count),
             "- **Schema Errors**: " + str(len(report.schema_errors)),
             "- **Semantic Errors**: " + str(len(report.semantic_errors)),
             "", "## Checks"]
    for c in report.checks:
        icon = "PASS" if c["passed"] else "FAIL"
        lines.append("- [" + icon + "] **" + c["check"] + "**: " + c.get("detail", ""))
    with open(os.path.join(output_dir, "validation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if report.schema_errors:
        with open(os.path.join(output_dir, "schema_errors.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "line", "path", "message"])
            w.writeheader()
            for e in report.schema_errors:
                w.writerow({"file": e.get("file", ""), "line": e.get("line", ""),
                           "path": e.get("path", ""), "message": e.get("message", "")})

    if report.semantic_errors:
        with open(os.path.join(output_dir, "semantic_errors.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "line", "message"])
            w.writeheader()
            for e in report.semantic_errors:
                w.writerow({"file": e.get("file", ""), "line": e.get("line", ""),
                           "message": e.get("message", "")})

    decision = {"decision": report.decision, "exit_code": report.exit_code,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "note": "P0_ACCEPTED = all validations passed"}
    with open(os.path.join(output_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="RC-UOT-v2.2 Provider Sample Validator")
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory")
    parser.add_argument("--contract", default=None, help="Path to provider handoff directory")
    args = parser.parse_args()
    contract_dir = args.contract or _find_contract_dir()
    dataset_root = os.path.abspath(args.dataset)
    print("Dataset:  " + dataset_root)
    print("Contract: " + contract_dir)
    print()
    report = ValidationReport()

    if not validate_directory_structure(dataset_root, report):
        print("DIRECTORY CHECK FAILED: " + report.decision)
        write_reports(report, dataset_root)
        sys.exit(report.exit_code)

    if not validate_schema(dataset_root, contract_dir, report):
        print("SCHEMA CHECK FAILED: " + report.decision)
        write_reports(report, dataset_root)
        sys.exit(report.exit_code)

    validate_semantics(dataset_root, contract_dir, report)
    if report.exit_code != EXIT_ACCEPTED:
        print("SEMANTIC CHECK FAILED: " + report.decision)
        write_reports(report, dataset_root)
        sys.exit(report.exit_code)

    validate_anti_triviality(dataset_root, contract_dir, report)
    write_reports(report, dataset_root)
    if report.exit_code == EXIT_ACCEPTED:
        print("P0_ACCEPTED")
    else:
        print("REJECTED: " + report.decision)
    sys.exit(report.exit_code)


if __name__ == "__main__":
    main()
